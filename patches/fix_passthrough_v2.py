"""Proper fix: chunk 전체 face 없을 때 3-tuple 으로 return 하고 valid_mask 저장."""
from pathlib import Path
import re

P = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")

# 깨진 patch 또는 원래 raise 둘 다 매칭하는 regex
PATTERN = re.compile(
    r"        if placeholder_face is None:\n"
    r"(?:            # === CHUNKED_NO_FACE_PASSTHROUGH ===.*?return faces, boxes, affine_matrices, valid_mask\n)"
    r"|        if placeholder_face is None:\n            raise RuntimeError\(.+\)\n",
    re.DOTALL,
)

NEW = '''        if placeholder_face is None:
            # === CHUNKED_NO_FACE_PASSTHROUGH ===
            # chunk 전체 face 없음 (drama 어두운/풍경 chunk) → 모든 frame 원본 유지.
            # restore_video 가 valid_mask=False 인 frame 은 원본 그대로 두므로
            # 빈 placeholder (검은 face) 를 set 하고 valid_mask 모두 False 로.
            print(f"[FACE_PASSTHROUGH] chunk 전체에 face 없음 → 원본 그대로 출력")
            import torch as _torch_pf
            import numpy as _np_pf
            placeholder_face = _torch_pf.zeros(3, 512, 512, dtype=_torch_pf.uint8)
            placeholder_box = [0, 0, 512, 512]
            placeholder_affine = _np_pf.eye(2, 3, dtype=_np_pf.float32)
            # 모든 frame invalid → restore_video 가 원본 frame 유지
            valid_mask = [False] * len(video_frames)
            # faces 빈 자리 placeholder 채움 (inference batch 통과용)
            faces = [placeholder_face] * len(video_frames)
            boxes = [placeholder_box] * len(video_frames)
            affine_matrices = [placeholder_affine] * len(video_frames)
            self._valid_face_mask = valid_mask
            faces_stacked = _torch_pf.stack(faces)
            return faces_stacked, boxes, affine_matrices
'''


def main():
    src = P.read_text()
    new_src, n = PATTERN.subn(NEW, src, count=1)
    if n == 0:
        print("[fix_passthrough_v2] anchor not found")
        return 1
    P.write_text(new_src)
    print(f"[fix_passthrough_v2] OK — {n} block replaced (passthrough 3-tuple)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
