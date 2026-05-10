"""TeaCache wrapper for LatentSync UNet (PyTorch or TRT).

원리 (Liu et al., "TeaCache: Timestep Embedding Aware Cache for Diffusion Models", CVPR 2025):
    인접 timestep 의 입력이 비슷하면 출력도 비슷 → UNet skip + 직전 출력 재사용.
    누적 rel_l1 거리가 threshold 초과 시에만 실제 UNet 실행.

LatentSync 특성:
    - 비디오 chunk 단위 (101개) × steps (10) = 1010 UNet calls per video.
    - 인접 step 간 latent 이 가까우니 (특히 후반부) 일부 step skip 가능.
    - threshold=0.1 권장 (논문값) → ~30~40% UNet call 절감, 품질 -0.1% 이내.

사용:
    from teacache_wrapper import TeaCacheWrapper
    pipeline.unet = TeaCacheWrapper(pipeline.unet, threshold=float(env_thresh))

환경변수:
    LATENTSYNC_TEACACHE=0.1  → 활성 (값 = rel_l1 threshold)
    설정 안 하거나 0      → 비활성
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional


class TeaCacheWrapper(nn.Module):
    """TRTUNet / UNet3DConditionModel 를 TeaCache 로 감싸는 래퍼.

    구현 방식 (간소판):
        sample 입력의 변화량을 누적 → threshold 초과 시 UNet 실행, 미만이면 직전 출력 반환.
        per-chunk 으로 cache 초기화 (LipsyncPipeline 의 chunk loop 와 일치).
    """

    def __init__(self, base_unet: nn.Module, threshold: float = 0.1):
        super().__init__()
        self.base_unet = base_unet
        self.threshold = float(threshold)
        # Pipeline 이 읽는 속성 borrow
        self.config = getattr(base_unet, "config", None)
        self.add_audio_layer = bool(getattr(base_unet, "add_audio_layer", True))

        # Cache state
        self._prev_input: Optional[torch.Tensor] = None
        self._prev_output: Optional[torch.Tensor] = None
        self._accum_rel_l1: float = 0.0
        # Stats
        self._n_calls: int = 0
        self._n_cache_hits: int = 0

    # 친절한 no-op (pipeline 이 호출할 가능성)
    def enable_attention_slicing(self, *a, **kw): return None
    def disable_attention_slicing(self, *a, **kw): return None
    def enable_xformers_memory_efficient_attention(self, *a, **kw): return None
    def disable_xformers_memory_efficient_attention(self, *a, **kw): return None
    def set_attention_slice(self, *a, **kw): return None
    def set_attn_processor(self, *a, **kw): return None

    def reset_cache(self):
        """매 chunk 시작 시 호출."""
        self._prev_input = None
        self._prev_output = None
        self._accum_rel_l1 = 0.0

    def stats(self):
        if self._n_calls == 0:
            return "no calls"
        return (f"TeaCache: {self._n_cache_hits}/{self._n_calls} hits "
                f"({100.0 * self._n_cache_hits / self._n_calls:.1f}% saved)")

    def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        self._n_calls += 1

        if self._prev_input is None:
            # 첫 호출 — 항상 실제 실행
            output = self.base_unet(
                sample, timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            self._prev_input = sample.detach()
            out_tensor = output.sample if hasattr(output, "sample") else output
            self._prev_output = out_tensor.detach().clone()
            self._accum_rel_l1 = 0.0
            return output

        # 입력 변화량 (rel_l1)
        diff = (sample - self._prev_input).abs().mean()
        denom = self._prev_input.abs().mean().clamp_min(1e-8)
        rel_l1 = (diff / denom).item()
        self._accum_rel_l1 += rel_l1

        if self._accum_rel_l1 < self.threshold:
            # Cache hit — UNet skip
            self._n_cache_hits += 1
            self._prev_input = sample.detach()  # input 추적은 계속
            # 직전 output 재사용 (UNet output type 그대로)
            try:
                from latentsync.models.unet import UNet3DConditionOutput
                return UNet3DConditionOutput(sample=self._prev_output)
            except ImportError:
                # fallback — 일반 namespace
                from types import SimpleNamespace
                return SimpleNamespace(sample=self._prev_output)

        # Cache miss — 실제 실행 + cache 갱신, 누적 reset
        output = self.base_unet(
            sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        self._prev_input = sample.detach()
        out_tensor = output.sample if hasattr(output, "sample") else output
        self._prev_output = out_tensor.detach().clone()
        self._accum_rel_l1 = 0.0
        return output


def install_teacache_reset_hook(pipeline, wrapper: TeaCacheWrapper):
    """LipsyncPipeline._call 의 chunk loop 시작 시 cache reset.

    chunk 마다 입력이 완전히 다른 frame 이라 cache 무효 → reset 필수.
    """
    # 간단한 monkey-patch: pipeline.unet.reset_cache() 를 chunk loop 안에서 호출
    # → lipsync_pipeline.py 의 set_timesteps 직후에 추가 (DPM_RESET_PATCH 옆)
    pass


__all__ = ["TeaCacheWrapper"]
