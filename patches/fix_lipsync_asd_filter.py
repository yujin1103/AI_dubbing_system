"""LatentSync `lipsync_pipeline.py` 에 ASD 기반 frame skip 통합.

목적:
    드라마/대화 영상에서 LatentSync 가 비-발화자(listener) 의 얼굴에도 입을
    그리는 문제를 해결. LightASD 가 frame 단위로 "발화자가 화면에 있는가?"
    를 판정해서 발화자 없는 frame 은 lipsync 를 건너뛰고 원본 frame 유지.

활성화:
    orchestrator (Step 4-3 AV-Fusion) 가 chunk 별 ASD pickle 을 생성한 후
    `runs/<run_id>/meta/asd_filter_index.json` 을 작성하고 환경변수
    `LATENTSYNC_ASD_FILTER_RUN_DIR` 를 설정. LatentSync subprocess 는 자동
    감지 → 비-발화자 frame 을 skip.

수정 범위:
    `latentsync/pipelines/lipsync_pipeline.py` 의 두 함수
      1. `affine_transform_video` — 시작 부분에 lazy-init + skip 로직 추가
      2. `_chunked_call` — chunk_start_frame 을 ASD filter 에 전달

전제:
    - `latentsync/utils/asd_filter.py` 가 이미 존재 (patches/asd_filter.py 에서 install)
    - dark_skip / FACE_KEEP_ORIG_PATCH 등 기존 BRIGHTNESS_CHECK / FACE_SKIP 패치는
      적용된 상태 (`for fi, frame in enumerate(tqdm.tqdm(video_frames))` 루프 존재).

호환성:
    - 환경변수 미설정 시 ASD 필터는 비활성 → 기존 동작 그대로.
    - LightASD pkl 이 chunk 에 없으면 해당 chunk 는 "의견 없음" → 모든 frame allow.
"""
from pathlib import Path

PIPELINE_PY = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")
ASD_FILTER_PY_SRC = Path(__file__).resolve().parent / "asd_filter.py"
ASD_FILTER_PY_DST = Path("/opt/LatentSync/latentsync/utils/asd_filter.py")

MARKER = "ASD_FILTER_PATCH:lazy-init"


LAZY_INIT_BLOCK_OLD = """    def affine_transform_video(self, video_frames: np.ndarray):
        # FACE_KEEP_ORIG_PATCH + STABILITY_CHECK: face 위치/크기 outlier 제거"""

LAZY_INIT_BLOCK_NEW = """    def affine_transform_video(self, video_frames: np.ndarray):
        # === ASD_FILTER_PATCH:lazy-init ===
        # Initialize filter on first call (subprocess env var read here).
        if not hasattr(self, "_asd_filter"):
            try:
                from latentsync.utils.asd_filter import maybe_load_filter
                self._asd_filter = maybe_load_filter()
            except Exception as _e:
                import traceback as _tb
                print(f"[ASD-Filter] init failed (continuing without filter): {_e}")
                _tb.print_exc()
                self._asd_filter = None
            self._asd_global_frame_offset = 0
            self._asd_skip_count = 0
        # === ASD_FILTER_PATCH:lazy-init end ===
        # FACE_KEEP_ORIG_PATCH + STABILITY_CHECK: face 위치/크기 outlier 제거"""


SKIP_BLOCK_OLD = """            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:"""

SKIP_BLOCK_NEW = """            # === ASD_FILTER_PATCH ===
            # When ASD says no active speaker is visible at this frame
            # (e.g., over-the-shoulder shot, listener-only frame), skip
            # lipsync — restore_video will keep the original frame.
            _asd_flt = getattr(self, "_asd_filter", None)
            if _asd_flt is not None:
                _g = getattr(self, "_asd_global_frame_offset", 0) + fi
                if _asd_flt.should_skip(_g):
                    valid_mask.append(False)
                    skipped += 1
                    if not hasattr(self, "_asd_skip_count"):
                        self._asd_skip_count = 0
                    self._asd_skip_count += 1
                    faces.append(None); boxes.append(None); affine_matrices.append(None)
                    continue
            # === ASD_FILTER_PATCH end ===

            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:"""


SUMMARY_BLOCK_OLD = """        if skipped > 0:
            print(f"[Face Skip] {skipped}/{len(video_frames)} frames face 미감지 → 원본 유지 (paste skip)")"""

SUMMARY_BLOCK_NEW = """        if skipped > 0:
            print(f"[Face Skip] {skipped}/{len(video_frames)} frames face 미감지 → 원본 유지 (paste skip)")
        # === ASD_FILTER_PATCH end:summary ===
        _asd_n = getattr(self, "_asd_skip_count", 0)
        if _asd_n > 0:
            print(f"[ASD-Filter] {_asd_n}/{len(video_frames)} frames non-speaker face skipped (lipsync bypass)")
            self._asd_skip_count = 0"""


CHUNK_OFFSET_OLD = """            # 3. loop_video + face transform
            video_frames, faces, boxes, affine_matrices = self.loop_video(chunk_whisper, video_frames)"""

CHUNK_OFFSET_NEW = """            # 3. loop_video + face transform
            # === ASD_FILTER_PATCH:chunk-offset ===
            self._asd_global_frame_offset = chunk_start_frame
            # === ASD_FILTER_PATCH:chunk-offset end ===
            video_frames, faces, boxes, affine_matrices = self.loop_video(chunk_whisper, video_frames)"""


def main():
    # 1) ensure asd_filter.py is present
    if not ASD_FILTER_PY_DST.is_file():
        if ASD_FILTER_PY_SRC.is_file():
            ASD_FILTER_PY_DST.write_text(ASD_FILTER_PY_SRC.read_text())
            print(f"[fix_lipsync_asd_filter] installed {ASD_FILTER_PY_DST}")
        else:
            print(f"[fix_lipsync_asd_filter] no asd_filter.py source at {ASD_FILTER_PY_SRC}")
            return 1

    src = PIPELINE_PY.read_text()
    if MARKER in src:
        print("[fix_lipsync_asd_filter] already patched")
        return 0

    if LAZY_INIT_BLOCK_OLD not in src:
        print("[fix_lipsync_asd_filter] anchor (lazy-init) not found — abort")
        return 1
    src = src.replace(LAZY_INIT_BLOCK_OLD, LAZY_INIT_BLOCK_NEW, 1)

    if SKIP_BLOCK_OLD not in src:
        print("[fix_lipsync_asd_filter] anchor (skip-block) not found — abort")
        return 1
    src = src.replace(SKIP_BLOCK_OLD, SKIP_BLOCK_NEW, 1)

    if SUMMARY_BLOCK_OLD not in src:
        print("[fix_lipsync_asd_filter] anchor (summary) not found — abort")
        return 1
    src = src.replace(SUMMARY_BLOCK_OLD, SUMMARY_BLOCK_NEW, 1)

    if CHUNK_OFFSET_OLD not in src:
        print("[fix_lipsync_asd_filter] anchor (chunk-offset) not found — abort")
        return 1
    src = src.replace(CHUNK_OFFSET_OLD, CHUNK_OFFSET_NEW, 1)

    PIPELINE_PY.write_text(src)
    print("[fix_lipsync_asd_filter] patched — ASD filter wired into lipsync_pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
