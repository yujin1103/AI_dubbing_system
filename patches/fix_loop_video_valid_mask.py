"""loop_video가 valid_mask도 함께 ping-pong 연장하도록 patch.

진짜 bug:
  - affine_transform_video: valid_mask 250개 (input frames) 생성, self._valid_face_mask 저장
  - loop_video: faces 250 → 302 ping-pong 연장 (audio가 video보다 길면)
  - 하지만 _valid_face_mask는 250개로 stale
  - restore_video: len(valid_mask)=250 ≠ len(faces)=302 → fallback → 전체 True
  → 모든 frame에 face paste (title card 포함)

해결:
  loop_video에서 faces 연장과 동시에 valid_mask도 연장하고 self._valid_face_mask 갱신.
"""
from pathlib import Path

p = Path('/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py')
src = p.read_text()

old = '''    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices'''

new = '''    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # LOOP_VALID_MASK_FIX: loop 연장 시 valid_mask도 같이 ping-pong 해야 함
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            base_valid_mask = list(getattr(self, "_valid_face_mask", [True] * len(faces)))
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_valid_mask = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_valid_mask += base_valid_mask
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_valid_mask += base_valid_mask[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            # 연장된 valid_mask 갱신
            self._valid_face_mask = loop_valid_mask[: len(whisper_chunks)]
            print(f"[Loop] valid_mask {len(base_valid_mask)} → {len(self._valid_face_mask)} 확장")
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: loop_video valid_mask 연장 패치 적용")
else:
    print("NOT FOUND - check pipeline")
