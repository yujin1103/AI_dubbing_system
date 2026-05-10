"""Face Skip Patch 비활성화 (원본 동작 복원).

원래 face_skip_patch가 한 변경:
  - image_processor.py: raise RuntimeError → return None, None, None
  - lipsync_pipeline.py: 원본 affine_transform_video → 강화된 버전

revert:
  두 파일을 원본 동작으로 되돌림
"""
from pathlib import Path

# === image_processor.py revert ===
p1 = Path("/opt/LatentSync/latentsync/utils/image_processor.py")
src1 = p1.read_text(encoding="utf-8")

if "FACE_SKIP_PATCH" not in src1:
    print("[image_processor] face_skip 이미 비활성")
else:
    old = """        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            # FACE_SKIP_PATCH: face 못 찾으면 None 반환 (caller가 reuse 처리)
            return None, None, None"""
    new = """        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")"""
    if old in src1:
        src1 = src1.replace(old, new)
        p1.write_text(src1, encoding="utf-8")
        print("[image_processor] face_skip 비활성화 ✅")

# === lipsync_pipeline.py revert ===
p2 = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")
src2 = p2.read_text(encoding="utf-8")

if "FACE_SKIP_PATCH" not in src2:
    print("[lipsync_pipeline] face_skip 이미 비활성")
else:
    # 강화된 버전 (face_skip patch 적용된)
    old = """    def affine_transform_video(self, video_frames: np.ndarray):
        # FACE_SKIP_PATCH: face 못 찾는 frame은 마지막 valid 결과 reuse (전체 중단 방지)
        faces = []
        boxes = []
        affine_matrices = []
        last_valid = None  # (face, box, affine_matrix)
        skipped = 0
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:
                if last_valid is None:
                    # 첫 frame부터 face 없음 → 다음 valid frame까지 일단 None
                    faces.append(None)
                    boxes.append(None)
                    affine_matrices.append(None)
                    skipped += 1
                    continue
                # 이전 valid frame 결과 reuse
                face, box, affine_matrix = last_valid
                skipped += 1
            else:
                last_valid = (face, box, affine_matrix)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        # 시작 부분에 None 있으면 첫 valid 결과로 채우기
        first_valid_idx = next((i for i, f in enumerate(faces) if f is not None), None)
        if first_valid_idx is None:
            raise RuntimeError("FACE_SKIP_PATCH: 영상 전체에서 face 감지 실패")
        for i in range(first_valid_idx):
            faces[i] = faces[first_valid_idx]
            boxes[i] = boxes[first_valid_idx]
            affine_matrices[i] = affine_matrices[first_valid_idx]

        if skipped > 0:
            print(f"[Face Skip] {skipped}/{len(video_frames)} frames에서 face 미감지 → 이전 valid frame reuse")

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices"""

    # 원본 (단순)
    new = """    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices"""

    if old in src2:
        src2 = src2.replace(old, new)
        p2.write_text(src2, encoding="utf-8")
        print("[lipsync_pipeline] face_skip 비활성화 ✅")

print("[Face Skip Revert] 완료. face 못 찾는 frame이 있으면 RuntimeError 발생.")
