"""restore_video: valid_mask가 False인 frame은 원본 그대로 (lipsync skip)."""
from pathlib import Path

p = Path('/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py')
src = p.read_text()

old = '''    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)'''

new = '''    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        # FACE_KEEP_ORIG_PATCH: valid_mask가 False면 원본 frame 유지 (잘못된 paste 방지)
        video_frames = video_frames[: len(faces)]
        valid_mask = getattr(self, "_valid_face_mask", None)
        if valid_mask is None or len(valid_mask) != len(faces):
            valid_mask = [True] * len(faces)
        out_frames = []
        kept_original = 0
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            if not valid_mask[index]:
                # face=None이었던 frame → 원본 video_frame 그대로 (lipsync 적용 X)
                out_frames.append(video_frames[index])
                kept_original += 1
                continue
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        if kept_original > 0:
            print(f"[Restore] {kept_original}/{len(faces)} frames 원본 유지 (face 미감지)")
        return np.stack(out_frames, axis=0)'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print('OK: restore_video valid_mask 적용')
else:
    print('NOT FOUND')
