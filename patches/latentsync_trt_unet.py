"""LatentSync UNet3D as a TensorRT FP16 engine drop-in replacement.

Designed to be `pipeline.unet = TRTUNet(engine_path, pipeline.unet)` after
LipsyncPipeline is constructed. The wrapper preserves the few attributes that
LipsyncPipeline reads from `self.unet` (config, add_audio_layer, training).

The current static engine assumes:
    sample                (B=2, 13, T=16, 64, 64)  fp16
    timestep              ()                        int64
    encoder_hidden_states (B*T=32, 50, 384)         fp16
    -> noise_pred         (2, 4, 16, 64, 64)        fp16

For the last chunk of a video where `T_act < 16`, the wrapper zero-pads to
T=16 and slices the output back to T_act. This avoids re-building a separate
engine for short tails.

CFG batch (B=2) is required (use guidance_scale > 1.0). Mismatch raises.

Env vars:
    LATENTSYNC_TRT_ENGINE=/path/to/unet_fp16.trt   (override default path)
    LATENTSYNC_TRT_DEBUG=1                         (verbose timing)
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tensorrt as trt
except Exception as e:  # pragma: no cover
    raise ImportError(f"tensorrt python binding not available: {e}")

# UNet3DConditionOutput is what diffusers-style pipelines expect when
# return_dict=True. We import lazily so this module does not require LatentSync
# to be on the path at import time.


def _udo():
    from latentsync.models.unet import UNet3DConditionOutput
    return UNet3DConditionOutput


class TRTUNet(nn.Module):
    """Drop-in TRT replacement for UNet3DConditionModel.forward."""

    # static shapes baked into the engine
    ENGINE_B: int = 2
    ENGINE_C_IN: int = 13
    ENGINE_T: int = 16
    ENGINE_H: int = 64
    ENGINE_W: int = 64
    ENGINE_C_OUT: int = 4
    ENGINE_AUDIO_S: int = 50
    ENGINE_AUDIO_D: int = 384

    def __init__(self, engine_path: str, original_unet: Optional[nn.Module] = None):
        super().__init__()

        # === TRTUNet_DEVICE_PATCH ===
        # diffusers DiffusionPipeline.device 가 unet.parameters() 의 첫 device 를 읽음.
        # TRT 엔진은 PyTorch param 이 없으므로 dummy 1-element param 으로 device 노출.
        self._device_marker = nn.Parameter(
            torch.zeros(1, dtype=torch.float16, device="cuda"),
            requires_grad=False,
        )
        # === TRTUNet_DEVICE_PATCH end ===

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(engine_path)

        self._debug = os.environ.get("LATENTSYNC_TRT_DEBUG", "0") == "1"
        self._engine_path = engine_path

        # Borrow attributes that LipsyncPipeline reads
        if original_unet is not None:
            self.config = original_unet.config
            self.add_audio_layer = bool(getattr(original_unet, "add_audio_layer", True))
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace(sample_size=64)
            self.add_audio_layer = True

        # Load TRT engine
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        t0 = time.time()
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._debug:
            print(f"[TRTUNet] engine deserialized in {time.time()-t0:.1f}s ({os.path.getsize(engine_path)/1e6:.0f} MB)")

        # Persistent output buffer (cuda fp16) — re-used across calls.
        # Caller must clone() if it needs to keep the result across iterations.
        self._out_buf = torch.empty(
            (self.ENGINE_B, self.ENGINE_C_OUT, self.ENGINE_T, self.ENGINE_H, self.ENGINE_W),
            dtype=torch.float16,
            device="cuda",
        )

        # Set static shapes once
        self._context.set_input_shape("sample", (self.ENGINE_B, self.ENGINE_C_IN, self.ENGINE_T, self.ENGINE_H, self.ENGINE_W))
        self._context.set_input_shape("timestep", ())
        self._context.set_input_shape("encoder_hidden_states", (self.ENGINE_B * self.ENGINE_T, self.ENGINE_AUDIO_S, self.ENGINE_AUDIO_D))

        self._n_calls = 0
        self._cumul_ms = 0.0

    # ---- Friendly no-ops to absorb pipeline-level configuration calls ----
    def enable_attention_slicing(self, *args, **kwargs):
        return None
    def disable_attention_slicing(self, *args, **kwargs):
        return None
    def enable_xformers_memory_efficient_attention(self, *args, **kwargs):
        return None
    def disable_xformers_memory_efficient_attention(self, *args, **kwargs):
        return None
    def set_attention_slice(self, *args, **kwargs):
        return None
    def set_attn_processor(self, *args, **kwargs):
        return None

    # ---- Forward ----
    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict: bool = True,
    ):
        if class_labels is not None:
            raise NotImplementedError("class_labels path not exported into the TRT engine")
        if attention_mask is not None:
            raise NotImplementedError("attention_mask path not exported into the TRT engine")
        if down_block_additional_residuals is not None or mid_block_additional_residual is not None:
            raise NotImplementedError("controlnet residuals path not exported into the TRT engine")
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states required (engine has add_audio_layer=True)")

        B, C_in, T_act, H, W = sample.shape
        if B != self.ENGINE_B:
            raise ValueError(
                f"TRTUNet engine expects batch={self.ENGINE_B} (CFG). "
                f"Got batch={B}. Run with guidance_scale > 1.0."
            )
        if C_in != self.ENGINE_C_IN or H != self.ENGINE_H or W != self.ENGINE_W:
            raise ValueError(
                f"shape mismatch: engine=({self.ENGINE_C_IN},*,{self.ENGINE_H},{self.ENGINE_W}), "
                f"got=({C_in},{T_act},{H},{W})"
            )
        if T_act > self.ENGINE_T:
            raise ValueError(f"chunk T={T_act} > engine T={self.ENGINE_T}")

        # Pad on T dim to engine size when last chunk is short
        if T_act != self.ENGINE_T:
            pad_T = self.ENGINE_T - T_act
            sample = F.pad(sample, (0, 0, 0, 0, 0, pad_T))  # pads last 3 spatial dims; (W,H,T) -> only T
            eh = encoder_hidden_states
            assert eh.shape[0] == B * T_act, (
                f"audio batch {eh.shape[0]} != B*T_act ({B}*{T_act}={B*T_act})"
            )
            pad_eh = torch.zeros(B * pad_T, eh.shape[1], eh.shape[2], dtype=eh.dtype, device=eh.device)
            encoder_hidden_states = torch.cat([eh, pad_eh], dim=0)

        # Cast / contiguous
        sample_h = sample.contiguous().to(torch.float16)
        eh_h = encoder_hidden_states.contiguous().to(torch.float16)

        # Timestep -> scalar int64 cuda tensor
        if not isinstance(timestep, torch.Tensor):
            ts = torch.tensor(int(timestep), dtype=torch.int64, device=sample.device)
        else:
            ts = timestep.to(torch.int64)
            if ts.numel() > 1:
                ts = ts.flatten()[0]
            ts = ts.contiguous()

        # Bind addresses (shapes were set in __init__)
        self._context.set_tensor_address("sample", sample_h.data_ptr())
        self._context.set_tensor_address("timestep", ts.data_ptr())
        self._context.set_tensor_address("encoder_hidden_states", eh_h.data_ptr())
        self._context.set_tensor_address("noise_pred", self._out_buf.data_ptr())

        # Use the current torch CUDA stream so subsequent torch ops are queued in order
        torch_stream = torch.cuda.current_stream()
        if self._debug:
            t0 = time.time()
        ok = self._context.execute_async_v3(torch_stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        if self._debug:
            torch_stream.synchronize()
            self._n_calls += 1
            self._cumul_ms += (time.time() - t0) * 1000
            if self._n_calls % 20 == 0:
                avg = self._cumul_ms / self._n_calls
                print(f"[TRTUNet] {self._n_calls} calls, avg={avg:.1f}ms")

        out_full = self._out_buf
        if T_act != self.ENGINE_T:
            out = out_full[:, :, :T_act].clone()
        else:
            out = out_full.clone()

        # Match caller dtype (LipsyncPipeline is fp16 in our env, but be defensive)
        if out.dtype != sample.dtype:
            out = out.to(sample.dtype)

        if return_dict:
            return _udo()(sample=out)
        return (out,)
