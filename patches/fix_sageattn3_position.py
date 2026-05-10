"""inference.py에 잘못 삽입된 SageAttn3 patch 제거 + 올바른 위치 재삽입."""
from pathlib import Path

P = Path("/opt/LatentSync/scripts/inference.py")
ANCHOR = "from DeepCache import DeepCacheSDHelper\n"
MARKER = "# === SageAttn3 (Blackwell FP4) auto-apply ==="

NEW_BLOCK = """
# === SageAttn3 (Blackwell FP4) auto-apply ===
# F.scaled_dot_product_attention -> sageattn3_blackwell (5x 가속)
# 실패 시 자동 fallback to SDPA (안전)
import sys as _sage_sys
_sage_sys.path.insert(0, "/workspace/patches")
try:
    import sageattn3_patch as _sage_patch
    _sage_patch.apply()
except Exception as _sage_err:
    print(f"[SageAttn3] patch failed: {_sage_err}, fallback to SDPA")

"""


def main():
    src = P.read_text()
    # 1. 잘못 삽입된 블록 제거
    if MARKER in src:
        start = src.find(MARKER)
        # 끝 라인 (fallback to SDPA) 다음 줄까지
        end_marker = 'fallback to SDPA")'
        end = src.find(end_marker, start)
        if end != -1:
            end = src.find("\n", end) + 1
            src = src[:start] + src[end:]
            print("[fix] removed wrongly-placed block")
    # 2. 올바른 위치 (top-level import 끝)에 재삽입
    if ANCHOR not in src:
        print(f"[fix] anchor not found: {ANCHOR!r}")
        return 1
    idx = src.find(ANCHOR) + len(ANCHOR)
    src = src[:idx] + NEW_BLOCK + src[idx:]
    P.write_text(src)
    print(f"[fix] re-inserted at offset {idx} (after DeepCache import)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
