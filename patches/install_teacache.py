"""TeaCache 를 LatentSync inference.py + lipsync_pipeline.py 에 설치.

inference.py:
    TRT_PATCH 직후에 LATENTSYNC_TEACACHE 환경변수 보고 wrap.

lipsync_pipeline.py:
    chunk 시작 시 reset_cache() 호출 (DPM_RESET_PATCH 옆).
멱등.
"""
from pathlib import Path
import re

INFERENCE_PY = Path("/opt/LatentSync/scripts/inference.py")
PIPELINE_PY = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")

# === inference.py 패치 ===
INF_MARKER = "# === TEACACHE_PATCH"
INF_ANCHOR = '            print("[TRT_PATCH] TRT UNet active'  # TRT_PATCH 의 주요 print 직후
INF_INSERT_BLOCK = '''
    # === TEACACHE_PATCH ===
    # LATENTSYNC_TEACACHE=0.1 (또는 0.05/0.2) 로 활성화. 0/미설정 시 OFF.
    _tc_thresh = float(os.environ.get("LATENTSYNC_TEACACHE", "0"))
    if _tc_thresh > 0:
        try:
            import sys as _tc_sys
            _tc_sys.path.insert(0, "/workspace/patches")
            from teacache_wrapper import TeaCacheWrapper
            pipeline.unet = TeaCacheWrapper(pipeline.unet, threshold=_tc_thresh)
            print(f"[TEACACHE_PATCH] TeaCache enabled (rel_l1 threshold={_tc_thresh})")
        except Exception as _tc_e:
            print(f"[TEACACHE_PATCH] failed: {_tc_e!r}")
    else:
        print("[TEACACHE_PATCH] disabled (LATENTSYNC_TEACACHE=0)")
    # === TEACACHE_PATCH end ===
'''

# === lipsync_pipeline.py 패치 — chunk 시작 시 reset ===
PIPE_MARKER = "# === TEACACHE_RESET_PATCH"
PIPE_ANCHOR = "            # === DPM_RESET_PATCH end ===\n"
PIPE_INSERT_BLOCK = """            # === TEACACHE_RESET_PATCH ===
            # chunk 마다 cache 무효화 (입력 frame 이 완전히 다름)
            if hasattr(self.unet, "reset_cache"):
                self.unet.reset_cache()
            # === TEACACHE_RESET_PATCH end ===
"""


def patch_inference():
    src = INFERENCE_PY.read_text()
    if INF_MARKER in src:
        print("[install_teacache] inference.py already patched")
        return True

    # Anchor: TRT_PATCH 의 마지막 print 다음 줄에 삽입
    # 정확한 anchor 찾기 — TRT_PATCH 블록 종료 후
    # "# === TRT_PATCH ... end ===" 같은 명시적 end 가 없으므로
    # try / except 블록의 except 절 다음을 잡는다
    # 더 안전하게: "USE_TRT = False" 가 있는 except 절 다음에 삽입
    anchor_pat = re.search(
        r'(except Exception as e:\n            print\(f"\[TRT_PATCH\] failed:.*?\n            USE_TRT = False\n)',
        src,
    )
    if not anchor_pat:
        print("[install_teacache] TRT_PATCH end anchor not found")
        return False

    insert_at = anchor_pat.end()
    src_new = src[:insert_at] + INF_INSERT_BLOCK + src[insert_at:]
    INFERENCE_PY.write_text(src_new)
    print("[install_teacache] inference.py patched")
    return True


def patch_pipeline():
    src = PIPELINE_PY.read_text()
    if PIPE_MARKER in src:
        print("[install_teacache] lipsync_pipeline.py already patched")
        return True

    if PIPE_ANCHOR not in src:
        print("[install_teacache] DPM_RESET_PATCH end anchor not found")
        return False

    src_new = src.replace(PIPE_ANCHOR, PIPE_ANCHOR + PIPE_INSERT_BLOCK, 1)
    PIPELINE_PY.write_text(src_new)
    print("[install_teacache] lipsync_pipeline.py patched")
    return True


def main():
    ok1 = patch_inference()
    ok2 = patch_pipeline()
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
