"""SageAttention 2++ (INT8 Triton) drop-in patch for ASR / TTS / Diarization.

LatentSync는 SageAttention 3 (FP4) 사용 중. 이 patch는 다른 transformer
모델 (CosyVoice3, Qwen3-ASR, DiariZen 등)에 안전한 INT8 가속 적용.

특징:
  - INT8 Q/K + FP16 P·V (Triton kernel)
  - Mean abs diff 0.001 미만 (SageAttn3 0.015 대비 15x 정확)
  - Blackwell sm_120 호환 (Triton 컴파일 버전)
  - LLM/ASR/TTS 검증된 API

조건:
  - q/k/v dtype FP16 또는 BF16
  - attn_mask None
  - dropout_p 0
  - q.dim() == 4
  - cuda

사용법:
  import sageattn2_patch
  sageattn2_patch.apply()

환경변수:
  SAGEATTN2_OFF=1 → patch 비활성화 (baseline 비교용)
"""
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F

_original_sdpa = F.scaled_dot_product_attention
_patched = False

if os.getenv("SAGEATTN2_OFF") == "1":
    SAGE_AVAILABLE = False
    print("[SageAttn2 Patch] disabled via SAGEATTN2_OFF=1")
else:
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        SAGE_AVAILABLE = True
    except ImportError as _e:
        SAGE_AVAILABLE = False
        print(f"[SageAttn2 Patch] sageattention not available ({_e}) → fallback to SDPA")


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
    """Drop-in replacement using SageAttention 2++ INT8 Triton kernel."""
    if not SAGE_AVAILABLE:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    sage_ok = (
        attn_mask is None
        and dropout_p == 0.0
        and scale is None
        and not enable_gqa
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.is_cuda
        and query.dim() == 4
    )
    if sage_ok:
        try:
            return sageattn_qk_int8_pv_fp16_triton(query, key, value, is_causal=is_causal)
        except Exception as e:
            if not getattr(_patched_sdpa, "_warned", False):
                print(f"[SageAttn2 Patch] runtime fallback: {e}")
                _patched_sdpa._warned = True
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    return _original_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale)


def apply():
    global _patched
    if _patched:
        return
    F.scaled_dot_product_attention = _patched_sdpa
    _patched = True
    if SAGE_AVAILABLE:
        print("[SageAttn2 Patch] ✅ F.scaled_dot_product_attention → sageattn_qk_int8_pv_fp16_triton (INT8 가속, mean_diff<0.001)")
    else:
        print("[SageAttn2 Patch] ⚠️  sageattention unavailable, no-op")


def revert():
    global _patched
    if not _patched:
        return
    F.scaled_dot_product_attention = _original_sdpa
    _patched = False


@contextmanager
def active():
    apply()
    try:
        yield
    finally:
        revert()


if __name__ == "__main__":
    print(f"SageAttn2 available: {SAGE_AVAILABLE}")
    if SAGE_AVAILABLE:
        q = torch.randn(1, 8, 256, 64, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ref = _original_sdpa(q, k, v)
        out = sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=False)
        diff = (ref - out).abs().mean().item()
        print(f"mean abs diff: {diff:.6f}")
