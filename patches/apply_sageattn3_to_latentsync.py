"""LatentSync inference.py에 SageAttn3 patch import 자동 삽입.

Dockerfile build 또는 수동 실행 시:
    /opt/venv_lipsync/bin/python /workspace/patches/apply_sageattn3_to_latentsync.py

효과: /opt/LatentSync/scripts/inference.py 상단 import 블록 끝에
SageAttn3 patch import + apply() 호출 추가. 멱등(중복 실행 안전).
"""
from pathlib import Path

INFERENCE_PY = Path("/opt/LatentSync/scripts/inference.py")
MARKER = "# === SageAttn3 (Blackwell FP4) auto-apply ==="

PATCH_BLOCK = '''
# === SageAttn3 (Blackwell FP4) auto-apply ===
# F.scaled_dot_product_attention → sageattn3_blackwell (5x 가속)
# 실패 시 자동 fallback to SDPA (안전)
import sys as _sys
_sys.path.insert(0, "/workspace/patches")
try:
    import sageattn3_patch as _sage_patch
    _sage_patch.apply()
except Exception as _sage_err:
    print(f"[SageAttn3] patch failed: {_sage_err}, fallback to SDPA")
'''


def main():
    if not INFERENCE_PY.is_file():
        print(f"[apply_sageattn3] inference.py not found: {INFERENCE_PY}")
        return 1
    src = INFERENCE_PY.read_text()
    if MARKER in src:
        print("[apply_sageattn3] already patched (idempotent skip)")
        return 0
    # import 블록 끝나는 지점 찾기 (마지막 'from ... import ...' 또는 'import ...' 다음 빈 줄)
    lines = src.splitlines(keepends=True)
    last_import_idx = 0
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith("import ") or s.startswith("from "):
            last_import_idx = i
    insert_idx = last_import_idx + 1
    new_src = "".join(lines[:insert_idx]) + PATCH_BLOCK + "".join(lines[insert_idx:])
    INFERENCE_PY.write_text(new_src)
    print(f"[apply_sageattn3] patched inference.py at line {insert_idx + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
