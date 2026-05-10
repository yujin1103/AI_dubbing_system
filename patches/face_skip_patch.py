"""Face Skip Patch — face 못 찾는 frame 처리.

기존: image_processor.affine_transform이 face 못 찾으면 RuntimeError → 추론 전체 중단
패치: face 못 찾으면 None 반환 + caller가 마지막 valid frame 결과 reuse

수정 위치:
  1. /opt/LatentSync/latentsync/utils/image_processor.py
     - "raise RuntimeError" → "return None, None, None"
  2. /opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py
     - affine_transform_video: face가 None이면 last_valid reuse
"""
from pathlib import Path

# === Patch 1: image_processor.py ===
p1 = Path("/opt/LatentSync/latentsync/utils/image_processor.py")
src1 = p1.read_text(encoding="utf-8")

if "FACE_SKIP_PATCH" in src1:
    print("[Face Skip] image_processor 이미 적용됨")
else:
    old1 = """        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")"""
    new1 = """        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            # FACE_SKIP_PATCH: face 못 찾으면 None 반환 (caller가 reuse 처리)
            return None, None, None"""
    if old1 in src1:
        src1 = src1.replace(old1, new1)
        p1.write_text(src1, encoding="utf-8")
        print("[Face Skip] image_processor.py 패치 적용 ✅")
    else:
        print("[Face Skip] image_processor 패턴 못 찾음")
        raise SystemExit(1)

# === Patch 2: lipsync_pipeline.py ===
p2 = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")
src2 = p2.read_text(encoding="utf-8")

if "FACE_SKIP_PATCH" in src2:
    print("[Face Skip] lipsync_pipeline 이미 적용됨")
else:
    old2 = """    def affine_transform_video(self, video_frames: np.ndarray):
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

    new2 = """    def affine_transform_video(self, video_frames: np.ndarray):
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

    if old2 in src2:
        src2 = src2.replace(old2, new2)
        p2.write_text(src2, encoding="utf-8")
        print("[Face Skip] lipsync_pipeline.py 패치 적용 ✅")
    else:
        print("[Face Skip] lipsync_pipeline 패턴 못 찾음")
        raise SystemExit(1)

print("\n[Face Skip] 모든 패치 완료 — face 못 찾는 frame은 이전 valid frame reuse")
