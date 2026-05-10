"""SageAttention 3 (Blackwell FP4) drop-in patch for LatentSync.

LatentSync의 두 attention 레이어:
  - latentsync/models/attention.py:271
  - latentsync/models/motion_module.py:300
모두 F.scaled_dot_product_attention 호출. 본 패치는 그 함수를
sageattn3_blackwell로 교체하여 5x 가속 (RTX 5090 1038 TOPS 기준).

조건:
  - q/k/v dtype FP16 또는 BF16
  - attn_mask None
  - dropout_p 0
위 조건이면 sageattn3 사용, 아니면 원본 SDPA로 fallback (안전).

사용법:
  import sageattn3_patch
  sageattn3_patch.apply()
  # ... LatentSync inference ...
  sageattn3_patch.revert()  # (선택) 다른 모듈 보호 위해

또는 context manager:
  with sageattn3_patch.active():
      output = pipeline(...)
"""
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F

_original_sdpa = F.scaled_dot_product_attention
_patched = False

# 환경변수로 patch 비활성화 가능 (baseline 비교용)
if os.getenv("SAGEATTN3_OFF") == "1":
    SAGE_AVAILABLE = False
    print("[SageAttn3 Patch] disabled via SAGEATTN3_OFF=1 (using original SDPA)")
else:
    try:
        from sageattn3 import sageattn3_blackwell
        SAGE_AVAILABLE = True
    except ImportError as _e:
        SAGE_AVAILABLE = False
        _import_err = str(_e)
        print(f"[SageAttn3 Patch] sageattn3 not available ({_import_err}) → fallback to SDPA")


# 환경변수 SAGEATTN3_SELF_ONLY=1 → self-attention만 SageAttn3 적용 (cross-attention은 SDPA fallback)
# LatentSync 같이 audio cross-attention이 lipsync 결정하는 모델에 안전한 옵션.
SELF_ONLY = os.getenv("SAGEATTN3_SELF_ONLY") == "1"
if SELF_ONLY:
    print("[SageAttn3 Patch] SELF_ONLY mode — cross-attention falls back to SDPA")


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
    """Drop-in replacement for F.scaled_dot_product_attention.

    SageAttention 3가 처리할 수 있는 조건이면 sageattn3 사용,
    아니면 원본 SDPA 호출 (안전한 fallback).

    SAGEATTN3_SELF_ONLY=1이면 q.seq_len != k.seq_len인 cross-attention은 SDPA로.
    """
    if not SAGE_AVAILABLE:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    # SELF_ONLY 모드: cross-attention 감지 → SDPA fallback
    # Self-attention은 q.shape[-2] == k.shape[-2] (같은 sequence)
    # Cross-attention은 q (latent) vs k (encoder feature) seq_len 다름
    if SELF_ONLY and query.dim() >= 2 and key.dim() >= 2 and query.shape[-2] != key.shape[-2]:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    # SageAttn3 supports: FP16/BF16, no mask, no dropout, no scale override, no GQA
    sage_ok = (
        attn_mask is None
        and dropout_p == 0.0
        and scale is None
        and not enable_gqa
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.is_cuda
        and query.dim() == 4   # (B, H, S, D)
    )
    if sage_ok:
        try:
            return sageattn3_blackwell(query, key, value, is_causal=is_causal)
        except Exception as e:
            # 첫 호출 시 1회 경고
            if not getattr(_patched_sdpa, "_warned", False):
                print(f"[SageAttn3 Patch] runtime fallback: {e}")
                _patched_sdpa._warned = True
            # 실패 시 원본 SDPA로 fallback
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    return _original_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale)


def apply():
    """Patch F.scaled_dot_product_attention globally."""
    global _patched
    if _patched:
        return
    F.scaled_dot_product_attention = _patched_sdpa
    _patched = True
    if SAGE_AVAILABLE:
        print("[SageAttn3 Patch] ✅ F.scaled_dot_product_attention → sageattn3_blackwell (5x 가속 활성)")
    else:
        print("[SageAttn3 Patch] ⚠️  sageattn3 unavailable, patch is no-op (SDPA 그대로 사용)")


def revert():
    """Restore original SDPA."""
    global _patched
    if not _patched:
        return
    F.scaled_dot_product_attention = _original_sdpa
    _patched = False
    print("[SageAttn3 Patch] reverted to original SDPA")


@contextmanager
def active():
    """Context manager: SageAttn3 활성화 → 종료 시 자동 revert.

    예:
        with sageattn3_patch.active():
            output = latentsync_pipeline(...)
    """
    apply()
    try:
        yield
    finally:
        revert()


# CLI에서 직접 실행 시 smoke test
if __name__ == "__main__":
    print(f"SageAttn3 available: {SAGE_AVAILABLE}")
    if SAGE_AVAILABLE:
        # smoke test
        q = torch.randn(1, 8, 256, 64, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        # baseline
        ref = _original_sdpa(q, k, v)
        # sageattn3
        out = sageattn3_blackwell(q, k, v, is_causal=False)
        print(f"baseline shape: {ref.shape}, sage shape: {out.shape}")
        diff = (ref - out).abs().mean().item()
        print(f"mean abs diff: {diff:.6f}  (작을수록 정확)")
