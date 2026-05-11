"""ASD per-FACE speaker matching (bbox IoU) patch.

Problem solved:
    LatentSync 가 listener (비-발화자) 의 face 를 골라 lipsync 를 적용해서
    "대머리 남자에게 여자 입이 그려지는" artifact 발생. ASD per-scene 체크
    (should_skip) 만으로는 "scene 에 발화자가 있다 → 통과" → LatentSync 가
    어떤 face 를 picked 하든 무조건 lipsync 적용 → wrong person.

해결:
    ASD track 에 bbox 정보가 있음. 검출된 face 의 bbox 와 모든 ASD track
    의 bbox 를 비교해서 가장 잘 매칭되는 (highest IoU) track 을 찾고, 그
    track 의 score 가 threshold 이상 (= 발화자) 이면 통과, 미만 (= listener)
    이면 lipsync skip.

수정 범위:
    1. `latentsync/utils/asd_filter.py` — 새 버전 (per-frame tracks_at 저장 +
       `is_detected_face_speaker(g, bbox)` 메서드). patches/asd_filter.py 에서 install.
    2. `latentsync/utils/image_processor.py` — face_detector 출력 bbox 를
       `self.last_face_bbox` 로 노출 (lipsync_pipeline 가 ASD 매칭에 사용).
    3. `latentsync/pipelines/lipsync_pipeline.py` — affine_transform_video 안에서
       face 가 검출됐을 때 `is_detected_face_speaker(...)` 가 False 면 face=None
       처리 → 원본 frame 유지.

마커: `ASD_BBOX_MATCH_PATCH`

전제:
    - patches/fix_lipsync_asd_filter.py 가 먼저 적용된 상태 (ASD_FILTER_PATCH 가
      있어야 _asd_flt 변수 + _asd_global_frame_offset 가 설정됨).

test4 측정 (chunk 2, 흑인-여자 dialog 장면):
    이전: 4 ASD-scene skip (listener-only scene 만), 246 frame 에 lipsync
          → 그 중 ~85% 는 listener face 에 wrong-person 적용 (대머리 빨간 입술)
    이후: 4 ASD-scene + 209 ASD-bbox skip = 213 skip
          → 33 frame 만 lipsync (실제 발화자 face 일 때만)
"""
from pathlib import Path

PIPELINE_PY = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")
IMG_PROC_PY = Path("/opt/LatentSync/latentsync/utils/image_processor.py")
ASD_FILTER_PY_SRC = Path(__file__).resolve().parent / "asd_filter.py"
ASD_FILTER_PY_DST = Path("/opt/LatentSync/latentsync/utils/asd_filter.py")

MARKER = "ASD_BBOX_MATCH_PATCH"

# === image_processor.py ===
IMG_OLD = """        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            return None, None, None  # FACE_SKIP_PATCH"""
IMG_NEW = """        bbox, landmark_2d_106 = self.face_detector(image)
        # === ASD_BBOX_MATCH_PATCH ===
        # Stash original-frame bbox so the lipsync pipeline can ask ASD
        # whether the detected face matches the speaker track.
        self.last_face_bbox = bbox
        # === ASD_BBOX_MATCH_PATCH end ===
        if bbox is None:
            return None, None, None  # FACE_SKIP_PATCH"""


# === lipsync_pipeline.py ===
PIPELINE_OLD = """            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:
                valid_mask.append(False)
                skipped += 1
                faces.append(None); boxes.append(None); affine_matrices.append(None)
            else:
                valid_mask.append(True)
                if placeholder_face is None:
                    placeholder_face = face
                    placeholder_box = box
                    placeholder_affine = affine_matrix
                faces.append(face); boxes.append(box); affine_matrices.append(affine_matrix)"""
PIPELINE_NEW = """            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            # === ASD_BBOX_MATCH_PATCH ===
            # Per-FACE speaker check: if ASD has data at this frame and the
            # detected face bbox does NOT match the speaker track bbox, the
            # detected face is the listener / wrong person — skip lipsync.
            if face is not None and _asd_flt is not None:
                _det_bbox = getattr(self.image_processor, "last_face_bbox", None)
                if _det_bbox is not None:
                    _is_spk = _asd_flt.is_detected_face_speaker(
                        getattr(self, "_asd_global_frame_offset", 0) + fi,
                        list(_det_bbox),
                    )
                    if _is_spk is False:
                        if not hasattr(self, "_asd_bbox_mismatch_count"):
                            self._asd_bbox_mismatch_count = 0
                        self._asd_bbox_mismatch_count += 1
                        face = None  # treat like face-miss → keep original
            # === ASD_BBOX_MATCH_PATCH end ===
            if face is None:
                valid_mask.append(False)
                skipped += 1
                faces.append(None); boxes.append(None); affine_matrices.append(None)
            else:
                valid_mask.append(True)
                if placeholder_face is None:
                    placeholder_face = face
                    placeholder_box = box
                    placeholder_affine = affine_matrix
                faces.append(face); boxes.append(box); affine_matrices.append(affine_matrix)"""


# === Summary print (also adds new log line for bbox-mismatch) ===
SUMMARY_OLD = """        _asd_n = getattr(self, "_asd_skip_count", 0)
        if _asd_n > 0:
            print(f"[ASD-Filter] {_asd_n}/{len(video_frames)} frames non-speaker face skipped (lipsync bypass)")
            self._asd_skip_count = 0"""
SUMMARY_NEW = """        _asd_n = getattr(self, "_asd_skip_count", 0)
        if _asd_n > 0:
            print(f"[ASD-Filter] {_asd_n}/{len(video_frames)} frames non-speaker face skipped (lipsync bypass)")
            self._asd_skip_count = 0
        _asd_bbox_mm = getattr(self, "_asd_bbox_mismatch_count", 0)
        if _asd_bbox_mm > 0:
            print(f"[ASD-BBox] {_asd_bbox_mm}/{len(video_frames)} frames detected face != speaker track → skipped")
            self._asd_bbox_mismatch_count = 0"""


def main():
    # 1) Install latest asd_filter.py
    if ASD_FILTER_PY_SRC.is_file():
        ASD_FILTER_PY_DST.write_text(ASD_FILTER_PY_SRC.read_text())
        print(f"[fix_asd_bbox_match] installed/updated {ASD_FILTER_PY_DST}")
    else:
        print(f"[fix_asd_bbox_match] no asd_filter.py source at {ASD_FILTER_PY_SRC}")
        return 1

    # 2) Patch image_processor.py
    img = IMG_PROC_PY.read_text()
    if MARKER in img:
        print("[fix_asd_bbox_match] image_processor already patched")
    elif IMG_OLD in img:
        IMG_PROC_PY.write_text(img.replace(IMG_OLD, IMG_NEW, 1))
        print("[fix_asd_bbox_match] image_processor patched (last_face_bbox)")
    else:
        print("[fix_asd_bbox_match] image_processor anchor not found — abort")
        return 1

    # 3) Patch lipsync_pipeline.py (skip block + summary)
    pl = PIPELINE_PY.read_text()
    bbox_patched = MARKER in pl
    if not bbox_patched:
        if PIPELINE_OLD not in pl:
            print("[fix_asd_bbox_match] lipsync_pipeline skip-block anchor not found — abort")
            return 1
        pl = pl.replace(PIPELINE_OLD, PIPELINE_NEW, 1)
        if SUMMARY_OLD not in pl:
            print("[fix_asd_bbox_match] lipsync_pipeline summary anchor not found — abort")
            return 1
        pl = pl.replace(SUMMARY_OLD, SUMMARY_NEW, 1)
        PIPELINE_PY.write_text(pl)
        print("[fix_asd_bbox_match] lipsync_pipeline patched (skip block + summary)")
    else:
        print("[fix_asd_bbox_match] lipsync_pipeline already patched")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
