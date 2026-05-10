"""affine_transform_video에 face stability check 추가.

문제:
  insightface가 title card의 grunge 텍스처를 face로 오감지 (det_score 0.5+).
  threshold만 올리면 진짜 어두운 face도 못 잡음.

해법 (size-based filter):
  1. 모든 face detection의 size (width × height) 수집
  2. median size 계산
  3. 각 face가 median 대비 너무 작거나 (< 0.5×) 너무 크면 (> 2.0×) invalid
  4. + position stability: 이전 valid face 대비 거리 > 2× face_width 면 invalid

이 방식은 같은 인물이 chunk 내에서 비슷한 크기/위치에 있다는 가정.
title card frame은 face size 다르거나 위치가 점프 → 자동 filter.
"""
import re
from pathlib import Path

p = Path('/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py')
src = p.read_text()

old = '''    def affine_transform_video(self, video_frames: np.ndarray):
        # FACE_KEEP_ORIG_PATCH: face=None인 frame은 placeholder 유지 (paste back에서 원본 사용)
        faces = []
        boxes = []
        affine_matrices = []
        valid_mask = []  # True = face 있음, False = face 없음 (원본 유지)
        skipped = 0
        # placeholder face: 첫 valid face로 채움 (inference 통과용, 결과는 paste 안 함)
        placeholder_face = None
        placeholder_box = None
        placeholder_affine = None
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:
                valid_mask.append(False)
                skipped += 1
                # placeholder는 나중에 채움
                faces.append(None); boxes.append(None); affine_matrices.append(None)
            else:
                valid_mask.append(True)
                if placeholder_face is None:
                    placeholder_face = face
                    placeholder_box = box
                    placeholder_affine = affine_matrix
                faces.append(face); boxes.append(box); affine_matrices.append(affine_matrix)

        if placeholder_face is None:
            raise RuntimeError("FACE_KEEP_ORIG_PATCH: 영상 전체에서 face 미감지")

        # None 자리에 placeholder 채움 (inference batch 통과용)
        for i in range(len(faces)):
            if faces[i] is None:
                faces[i] = placeholder_face
                boxes[i] = placeholder_box
                affine_matrices[i] = placeholder_affine

        if skipped > 0:
            print(f"[Face Skip] {skipped}/{len(video_frames)} frames face 미감지 → 원본 유지 (paste skip)")

        faces = torch.stack(faces)
        # valid_mask는 instance attribute로 저장 (restore_video에서 사용)
        self._valid_face_mask = valid_mask
        return faces, boxes, affine_matrices'''

new = '''    def affine_transform_video(self, video_frames: np.ndarray):
        # FACE_KEEP_ORIG_PATCH + STABILITY_CHECK: face 위치/크기 outlier 제거
        faces = []
        boxes = []
        affine_matrices = []
        valid_mask = []  # True = face 있음 (paste OK), False = invalid (원본 유지)
        skipped = 0
        placeholder_face = None
        placeholder_box = None
        placeholder_affine = None
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
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
                faces.append(face); boxes.append(box); affine_matrices.append(affine_matrix)

        if placeholder_face is None:
            raise RuntimeError("FACE_KEEP_ORIG_PATCH: 영상 전체에서 face 미감지")

        # === STABILITY_CHECK: face size/position outlier 제거 ===
        # title card 등 false positive는 일반적으로 size/position이 normal range 벗어남
        import numpy as _np
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        if len(valid_indices) >= 5:  # 통계 가능한 frame 수
            sizes = []
            positions = []
            for i in valid_indices:
                x1, y1, x2, y2 = boxes[i]
                w = x2 - x1; h = y2 - y1
                sizes.append((w, h))
                positions.append(((x1 + x2) / 2, (y1 + y2) / 2))

            sizes_arr = _np.array(sizes, dtype=_np.float32)
            positions_arr = _np.array(positions, dtype=_np.float32)

            # median size로 reference 잡기
            med_w = float(_np.median(sizes_arr[:, 0]))
            med_h = float(_np.median(sizes_arr[:, 1]))
            med_cx = float(_np.median(positions_arr[:, 0]))
            med_cy = float(_np.median(positions_arr[:, 1]))

            # 허용 범위: size는 ±50%, position은 face width × 2 이내
            min_w, max_w = med_w * 0.5, med_w * 2.0
            min_h, max_h = med_h * 0.5, med_h * 2.0
            max_pos_dist = max(med_w, med_h) * 2.0

            outliers = 0
            for i in valid_indices:
                x1, y1, x2, y2 = boxes[i]
                w = x2 - x1; h = y2 - y1
                cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                # size outlier
                size_ok = (min_w <= w <= max_w) and (min_h <= h <= max_h)
                # position outlier (median 대비 거리)
                dist = ((cx - med_cx) ** 2 + (cy - med_cy) ** 2) ** 0.5
                pos_ok = dist <= max_pos_dist
                if not (size_ok and pos_ok):
                    valid_mask[i] = False
                    outliers += 1

            if outliers > 0:
                print(f"[Face Stability] {outliers}/{len(valid_indices)} frames outlier 감지 → invalid 처리")

        # None 자리에 placeholder 채움 (inference batch 통과용)
        for i in range(len(faces)):
            if faces[i] is None:
                faces[i] = placeholder_face
                boxes[i] = placeholder_box
                affine_matrices[i] = placeholder_affine

        if skipped > 0:
            print(f"[Face Skip] {skipped}/{len(video_frames)} frames face 미감지 → 원본 유지 (paste skip)")

        faces = torch.stack(faces)
        self._valid_face_mask = valid_mask
        return faces, boxes, affine_matrices'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: face stability check added")
else:
    print("NOT FOUND - check if patch was already modified")
