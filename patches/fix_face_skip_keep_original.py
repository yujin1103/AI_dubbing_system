"""face 없는 frame은 원본 그대로 유지 (paste 안 함).

기존 face_skip_patch 문제:
  face=None일 때 last_valid를 재사용 → 옛 affine_matrix 그대로 paste
  → 화자 위치 어긋남 + 화면 다른 위치에 작은 face crop 합성 (사용자 screenshot)

새 동작:
  face=None인 frame은 inference 단계에서 마스크 처리
  paste back 시 원본 video_frame을 그대로 유지 (lipsync 효과 없음)
  → 위치 mismatch 해결 + 메모리 절약 (face 없는 frame 처리 안 함)
"""
from pathlib import Path

p = Path('/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py')
src = p.read_text()

# 1. affine_transform_video를 face=None 보존 형태로 (last_valid 재사용 X)
old1 = '''    def affine_transform_video(self, video_frames: np.ndarray):
        # FACE_SKIP_PATCH: face 못 찾는 frame은 last_valid 재사용
        faces = []
        boxes = []
        affine_matrices = []
        last_valid = None
        skipped = 0
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            if face is None:
                if last_valid is None:
                    faces.append(None); boxes.append(None); affine_matrices.append(None)
                    skipped += 1; continue
                face, box, affine_matrix = last_valid
                skipped += 1
            else:
                last_valid = (face, box, affine_matrix)
            faces.append(face); boxes.append(box); affine_matrices.append(affine_matrix)
        # 시작 None 채우기
        first_valid = next((i for i, f in enumerate(faces) if f is not None), None)
        if first_valid is None:
            raise RuntimeError("FACE_SKIP_PATCH: 영상 전체에서 face 미감지")
        for i in range(first_valid):
            faces[i] = faces[first_valid]
            boxes[i] = boxes[first_valid]
            affine_matrices[i] = affine_matrices[first_valid]
        if skipped > 0:
            print(f"[Face Skip] {skipped}/{len(video_frames)} frames reuse")
        faces = torch.stack(faces)
        return faces, boxes, affine_matrices'''

new1 = '''    def affine_transform_video(self, video_frames: np.ndarray):
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

if old1 in src:
    src = src.replace(old1, new1)
    print('[1] affine_transform_video valid_mask 적용')
else:
    print('[1] not found')

# 2. restore_video에서 valid_mask가 False인 frame은 원본 유지
# restore_video 함수 찾기
import re
restore_match = re.search(r'def restore_video\(self.*?\):\n', src)
if restore_match:
    print(f'[2] restore_video 위치 확인 OK')
else:
    print('[2] restore_video 못 찾음')

# 가장 단순: restore_video 함수 마지막에 valid_mask 적용
# 일단 restore_video 본문 출력 (수동 확인 위함)
m = re.search(r'(    def restore_video\(self.*?\n(?:.*?\n)*?    def )', src)
if m:
    print('--- restore_video 본문 (handle 처리 위해) ---')
    body = m.group(1)
    print(body[:2000])

p.write_text(src)
print('[Done] affine_transform_video patch 적용 (restore_video는 별도 확인)')
