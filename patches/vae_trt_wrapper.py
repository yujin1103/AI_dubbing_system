"""VAE TRT wrapper — drop-in replacement for diffusers AutoencoderKL.

Usage:
    from vae_trt_wrapper import wrap_vae_with_trt
    pipeline.vae = wrap_vae_with_trt(pipeline.vae)
    # Or inside lipsync_pipeline.py after VAE load.

Requires:
    /workspace/trt_work/engines/vae_encoder_fp16.trt
    /workspace/trt_work/engines/vae_decoder_fp16.trt

Mimics AutoencoderKL.encode() and AutoencoderKL.decode() interfaces.
"""
from __future__ import annotations
import os
import torch
import tensorrt as trt
from types import SimpleNamespace


VAE_ENC_TRT = os.environ.get(
    "LATENTSYNC_VAE_ENC_TRT",
    "/workspace/trt_work/engines/vae_encoder_fp16.trt",
)
VAE_DEC_TRT = os.environ.get(
    "LATENTSYNC_VAE_DEC_TRT",
    "/workspace/trt_work/engines/vae_decoder_fp16.trt",
)


class DiagonalGaussianFromParams:
    """Mimic diffusers DiagonalGaussianDistribution from concatenated mean+logvar."""
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator=None):
        if generator is None:
            noise = torch.randn_like(self.mean)
        else:
            noise = torch.randn(self.mean.shape,
                                generator=generator,
                                device=self.mean.device,
                                dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self):
        return self.mean


class _TRTSession:
    """Single-engine TRT execution context with persistent I/O buffers."""

    def __init__(self, engine_path, in_shape, out_shape, in_dtype=torch.float32,
                 out_dtype=torch.float32):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover I/O names
        self._in_name = None
        self._out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and self._in_name is None:
                self._in_name = name
            elif mode == trt.TensorIOMode.OUTPUT and self._out_name is None:
                self._out_name = name

        self.context.set_input_shape(self._in_name, in_shape)
        self.in_buf = torch.empty(in_shape, dtype=in_dtype, device="cuda")
        self.out_buf = torch.empty(out_shape, dtype=out_dtype, device="cuda")
        self.context.set_tensor_address(self._in_name, self.in_buf.data_ptr())
        self.context.set_tensor_address(self._out_name, self.out_buf.data_ptr())

    @torch.no_grad()
    def infer(self, x):
        if x.shape != self.in_shape:
            raise ValueError(f"shape mismatch: got {tuple(x.shape)}, "
                             f"expected {self.in_shape}")
        if x.device.type != "cuda":
            x = x.to("cuda", non_blocking=True)
        if x.dtype != self.in_buf.dtype:
            x = x.to(self.in_buf.dtype)
        self.in_buf.copy_(x)
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TRT execute_async_v3 failed")
        return self.out_buf.clone()


class VAETRT(torch.nn.Module):
    """Drop-in replacement for diffusers AutoencoderKL with TRT acceleration.

    Wraps:
        encode(image) → AutoencoderKLOutput with .latent_dist
        decode(latent) → DecoderOutput with .sample

    Falls back to PyTorch VAE if TRT engines missing.
    """

    BATCH = 2
    RES = 512
    LATENT_RES = 64

    def __init__(self, original_vae, encoder_engine=VAE_ENC_TRT,
                 decoder_engine=VAE_DEC_TRT):
        super().__init__()
        self._pytorch_vae = original_vae  # fallback
        self.config = original_vae.config  # for compat

        self._enc_session = None
        self._dec_session = None

        if os.path.isfile(encoder_engine):
            try:
                self._enc_session = _TRTSession(
                    encoder_engine,
                    in_shape=(self.BATCH, 3, self.RES, self.RES),
                    out_shape=(self.BATCH, 8, self.LATENT_RES, self.LATENT_RES),
                )
                print(f"[VAE-TRT] encoder loaded: {encoder_engine}", flush=True)
            except Exception as e:
                print(f"[VAE-TRT] encoder load FAILED ({e}); using PyTorch",
                      flush=True)

        if os.path.isfile(decoder_engine):
            try:
                self._dec_session = _TRTSession(
                    decoder_engine,
                    in_shape=(self.BATCH, 4, self.LATENT_RES, self.LATENT_RES),
                    out_shape=(self.BATCH, 3, self.RES, self.RES),
                )
                print(f"[VAE-TRT] decoder loaded: {decoder_engine}", flush=True)
            except Exception as e:
                print(f"[VAE-TRT] decoder load FAILED ({e}); using PyTorch",
                      flush=True)

    @torch.no_grad()
    def encode(self, x, return_dict=True):
        """Mimic AutoencoderKL.encode → AutoencoderKLOutput(latent_dist=...)."""
        # If batch size doesn't match TRT engine, fallback
        if self._enc_session is not None and x.shape[0] == self.BATCH \
                and x.shape[2] == self.RES and x.shape[3] == self.RES:
            params = self._enc_session.infer(x.float())
            # Match original dtype
            params = params.to(x.dtype)
            latent_dist = DiagonalGaussianFromParams(params)
            return SimpleNamespace(latent_dist=latent_dist)
        else:
            return self._pytorch_vae.encode(x, return_dict=return_dict)

    @torch.no_grad()
    def decode(self, z, return_dict=True, generator=None):
        """Mimic AutoencoderKL.decode → DecoderOutput(sample=...)."""
        if self._dec_session is not None and z.shape[0] == self.BATCH \
                and z.shape[2] == self.LATENT_RES \
                and z.shape[3] == self.LATENT_RES:
            sample = self._dec_session.infer(z.float())
            sample = sample.to(z.dtype)
            return SimpleNamespace(sample=sample)
        else:
            return self._pytorch_vae.decode(z, return_dict=return_dict)

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def dtype(self):
        return self._pytorch_vae.dtype

    def to(self, *args, **kwargs):
        # Move inner PyTorch VAE so its fallback path stays consistent.
        # TRT engines themselves are CUDA-only; this just keeps the PT VAE
        # on the same device for fallback.
        self._pytorch_vae = self._pytorch_vae.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        self._pytorch_vae = self._pytorch_vae.cuda(*args, **kwargs)
        return self

    def cpu(self):
        # Don't actually move to CPU (TRT engines are cuda-only)
        return self

    def half(self):
        self._pytorch_vae = self._pytorch_vae.half()
        return self

    def float(self):
        self._pytorch_vae = self._pytorch_vae.float()
        return self

    def eval(self):
        self._pytorch_vae = self._pytorch_vae.eval()
        return self


def wrap_vae_with_trt(original_vae):
    """Convenience function to wrap a diffusers AutoencoderKL.

    Returns VAETRT instance (or original if env disables TRT).
    """
    if os.environ.get("LATENTSYNC_VAE_TRT", "1") != "1":
        print("[VAE-TRT] disabled by env LATENTSYNC_VAE_TRT=0", flush=True)
        return original_vae
    return VAETRT(original_vae)
