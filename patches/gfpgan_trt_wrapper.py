"""GFPGAN v1.4 StyleGAN2 generator as a TensorRT engine drop-in.

Designed to be installed onto an existing GFPGANer instance:

    from gfpgan import GFPGANer
    from gfpgan_trt_wrapper import GFPGANTRT
    restorer = GFPGANer(model_path=..., upscale=1, arch='clean', channel_multiplier=2)
    restorer.gfpgan = GFPGANTRT('/workspace/trt_work/engines/gfpgan_bf16.trt')

After the swap, `restorer.enhance(img, ...)` keeps using face_helper for
detection, alignment, and paste-back; only the StyleGAN2 generator forward
runs on TensorRT.

Engine I/O (static):
    input  (1, 3, 512, 512) FP32   RGB, normalized to [-1, 1]
    output (1, 3, 512, 512) FP32   RGB, [-1, 1]

Why BF16, not FP16: the StyleGAN2 modulated conv (square-and-sum-over-3x3
kernel) overflows FP16 dynamic range on real face activations and produces
a near-constant output (collapsed range, ~14 dB PSNR vs PyTorch). BF16 has
the same exponent range as FP32 and matches PyTorch to ~51 dB PSNR — both
visually indistinguishable and numerically stable. BF16 is ~22% slower
than FP16 here (~7 ms vs ~5.8 ms per frame on RTX 5080) but still ~5-10x
faster than the PyTorch generator forward, so the trade is worth it.

The wrapper accepts an FP32 input on CUDA (which is what GFPGANer.enhance
hands to self.gfpgan(...)) and returns a tuple `(image_fp32, None)` to
mimic GFPGANv1Clean.forward, which returns `(image, out_rgbs)`.

Env vars:
    GFPGAN_TRT_ENGINE   override default engine path
    GFPGAN_TRT_DEBUG=1  print per-call timing every 50 frames
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import torch

try:
    import tensorrt as trt
except Exception as e:  # pragma: no cover
    raise ImportError(f"tensorrt python binding not available: {e}")


DEFAULT_ENGINE = os.environ.get(
    "GFPGAN_TRT_ENGINE",
    "/workspace/trt_work/engines/gfpgan_bf16.trt",
)


class GFPGANTRT:
    """Drop-in callable replacement for GFPGANer.gfpgan (the StyleGAN2 generator).

    Mimics GFPGANv1Clean.forward(x, return_rgb=..., weight=...) -> (image, out_rgbs).
    """

    # static shapes baked into the engine
    B = 1
    C = 3
    H = 512
    W = 512

    def __init__(self, engine_path: Optional[str] = None, device: str = "cuda"):
        self._engine_path = engine_path or DEFAULT_ENGINE
        if not os.path.isfile(self._engine_path):
            raise FileNotFoundError(self._engine_path)

        self._debug = os.environ.get("GFPGAN_TRT_DEBUG", "0") == "1"
        self._device = torch.device(device)

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        t0 = time.time()
        with open(self._engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {self._engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(
                "create_execution_context returned None. If the log shows "
                "'Inconsistent setting of NVIDIA_TF32_OVERRIDE', the engine was "
                "built with a different value of that env var than is set now. "
                "Build and runtime must match."
            )
        if self._debug:
            sz = os.path.getsize(self._engine_path) / 1e6
            print(f"[GFPGANTRT] engine deserialized in {time.time()-t0:.2f}s ({sz:.0f} MB)")

        # Engine I/O is FP32 (the BuilderFlag.BF16 / .FP16 affects internal
        # kernel precision, not tensor types). Buffers must be FP32 — passing
        # FP16 pointers reinterprets bits and produces nonsense output.
        self._in_buf = torch.empty(
            (self.B, self.C, self.H, self.W), dtype=torch.float32, device=self._device
        )
        self._out_buf = torch.empty(
            (self.B, self.C, self.H, self.W), dtype=torch.float32, device=self._device
        )

        self._in_name, self._out_name = self._discover_io_names()
        self._context.set_input_shape(self._in_name, (self.B, self.C, self.H, self.W))
        self._context.set_tensor_address(self._in_name, self._in_buf.data_ptr())
        self._context.set_tensor_address(self._out_name, self._out_buf.data_ptr())

        self._n_calls = 0
        self._cumul_ms = 0.0

    def _discover_io_names(self) -> Tuple[str, str]:
        in_name = out_name = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and in_name is None:
                in_name = name
            elif mode == trt.TensorIOMode.OUTPUT and out_name is None:
                out_name = name
        if in_name is None or out_name is None:
            raise RuntimeError(
                f"could not find one input + one output tensor in {self._engine_path}"
            )
        return in_name, out_name

    # ---- Compatibility shims so GFPGANer treats this like an nn.Module ----
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def half(self):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def load_state_dict(self, *args, **kwargs):
        return self

    def state_dict(self):
        return {}

    @property
    def training(self) -> bool:
        return False

    # ---- Forward ----
    @torch.no_grad()
    def __call__(self, x: torch.Tensor, return_rgb: bool = False, **kwargs):
        """x: (1, 3, 512, 512) FP32 RGB [-1, 1] on CUDA. Returns (image, None)."""
        if x.dim() != 4 or tuple(x.shape) != (self.B, self.C, self.H, self.W):
            raise ValueError(
                f"GFPGANTRT expects shape ({self.B},{self.C},{self.H},{self.W}); got {tuple(x.shape)}"
            )
        if x.device.type != "cuda":
            x = x.to(self._device, non_blocking=True)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        self._in_buf.copy_(x)

        torch_stream = torch.cuda.current_stream()
        if self._debug:
            t0 = time.time()
        ok = self._context.execute_async_v3(torch_stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")

        out = self._out_buf.clone()

        if self._debug:
            torch_stream.synchronize()
            self._n_calls += 1
            self._cumul_ms += (time.time() - t0) * 1000.0
            if self._n_calls % 50 == 0:
                avg = self._cumul_ms / self._n_calls
                print(f"[GFPGANTRT] {self._n_calls} calls, avg={avg:.2f} ms/frame")

        # Mimic GFPGANv1Clean.forward signature: (image, out_rgbs)
        return out, [] if return_rgb else None
