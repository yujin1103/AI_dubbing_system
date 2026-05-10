"""LatentSync inference.py에 통합 SageAttention loader 설치.

기존 sageattn3_patch import block을 환경변수 SAGEATTN_BACKEND 기반 dispatcher로 교체:
  SAGEATTN_BACKEND=sage3 (default) → sageattn3_patch (FP4, 5x 가속, video diffusion 검증)
  SAGEATTN_BACKEND=sage2          → sageattn2_patch (INT8 Triton, mean_diff<0.001 안전)
  SAGEATTN_BACKEND=off            → 패치 안 함 (baseline SDPA)
"""
from pathlib import Path
import re

INFERENCE_PY = Path("/opt/LatentSync/scripts/inference.py")
NEW_BLOCK = """
# === SageAttention dispatcher (Blackwell sm_120) ===
# SAGEATTN_BACKEND env var으로 backend 선택:
#   sage3 (default) → FP4 5x 가속 (video diffusion 검증)
#   sage2           → INT8 Triton (안전, mean_diff<0.001)
#   off             → baseline SDPA
import sys as _sage_sys
_sage_sys.path.insert(0, "/workspace/patches")
import os as _sage_os
_sage_backend = _sage_os.getenv("SAGEATTN_BACKEND", "sage3").lower()
try:
    if _sage_backend == "sage3":
        import sageattn3_patch as _sp
        _sp.apply()
    elif _sage_backend == "sage2":
        import sageattn2_patch as _sp
        _sp.apply()
    elif _sage_backend == "off":
        print("[SageAttn dispatcher] disabled (SAGEATTN_BACKEND=off)")
    else:
        print(f"[SageAttn dispatcher] unknown backend '{_sage_backend}', default sage3")
        import sageattn3_patch as _sp
        _sp.apply()
except Exception as _se:
    print(f"[SageAttn dispatcher] init failed: {_se}, fallback to SDPA")

"""


def main():
    src = INFERENCE_PY.read_text()
    # 기존 sageattn3_patch block 또는 dispatcher block 제거
    pattern = re.compile(
        r"# === Sage(Attn3 \(Blackwell FP4\) auto-apply|Attention dispatcher \(Blackwell sm_120\)) ===.*?(?=\n\ndef |\nclass |\Z)",
        re.DOTALL,
    )
    src_clean = pattern.sub("", src)
    if src_clean == src:
        # 패턴 못 찾음 → 그냥 anchor 뒤에 추가
        anchor = "from DeepCache import DeepCacheSDHelper\n"
        if anchor not in src:
            print("[update_sage_loader] anchor not found, abort")
            return 1
        idx = src.find(anchor) + len(anchor)
        src_new = src[:idx] + NEW_BLOCK + src[idx:]
    else:
        # 기존 block 제거됨 → anchor 뒤에 NEW_BLOCK 삽입
        anchor = "from DeepCache import DeepCacheSDHelper\n"
        idx = src_clean.find(anchor) + len(anchor)
        src_new = src_clean[:idx] + NEW_BLOCK + src_clean[idx:]
    INFERENCE_PY.write_text(src_new)
    print("[update_sage_loader] dispatcher installed in inference.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
