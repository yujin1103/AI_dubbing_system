"""affine_transform_video에 frame brightness check 추가.

문제:
  title card frame은 거의 black인데, "WALKING DEAD" 텍스처 노이즈에
  insightface가 face를 오감지 (det_score 0.5+).
  Stability check도 못 잡음 (noise face 위치가 우연히 정상 face와 비슷).

해법:
  frame 전체 평균 brightness 계산.
  특정 frame이 너무 어두우면 (mean < 25 of 255) → 의미 있는 scene 없음 →
  face detect 결과 무시하고 valid_mask = False (원본 유지).

  Walking Dead 같은 어두운 scene도 mean ~40+ 정도라 30 정도 임계값이면
  진짜 black title card만 걸러짐.
"""
from pathlib import Path

p = Path('/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py')
src = p.read_text()

# 기존 stability_check 위쪽에 brightness check 추가
old = '''        print(f"Affine transforming {len(video_frames)} faces...")
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
                faces.append(face); boxes.append(box); affine_matrices.append(affine_matrix)'''

new = '''        # BRIGHTNESS_CHECK: 너무 어두운 frame은 face detect 무시 (title card 등)
        import numpy as _np_b
        print(f"Affine transforming {len(video_frames)} faces...")
        dark_skip = 0
        for fi, frame in enumerate(tqdm.tqdm(video_frames)):
            # frame 평균 brightness (0~255)
            mean_brightness = float(_np_b.array(frame).mean())
            if mean_brightness < 25.0:
                # 거의 black frame → face가 있어도 무시 (title card 등)
                valid_mask.append(False)
                skipped += 1
                dark_skip += 1
                faces.append(None); boxes.append(None); affine_matrices.append(None)
                continue

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
        if dark_skip > 0:
            print(f"[Brightness] {dark_skip} 어두운 frame skip (mean<25)")'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: brightness check added")
else:
    print("NOT FOUND - check pipeline state")
