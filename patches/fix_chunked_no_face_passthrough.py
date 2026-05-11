"""LatentSync `_chunked_call` 의 empty-face chunk 처리 fix.

기존 `affine_transform_video` 는 video 전체에 face 가 하나도 없으면 RuntimeError.
이건 짧은 영상엔 OK 지만 장편 drama 에서는 한 chunk 가 통째로 face 없을 수
있음 (어두운 장면, 풍경 cut 등). 이 경우 chunk 전체를 원본 그대로 통과하면 됨.

수정: RuntimeError 대신 special return (모든 frame valid_mask=False) 으로 처리.
호출자 (`_chunked_call`, `affine_transform_video` 의 caller) 가 받아서 처리.

호환성: strict mode 가 아닐 때 (placeholder 가 있을 때) 는 기존 동작 그대로.
"""
from pathlib import Path

PIPELINE_PY = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")

OLD = '''        if placeholder_face is None:
            raise RuntimeError("FACE_KEEP_ORIG_PATCH: 영상 전체에서 face 미감지")'''

NEW = '''        if placeholder_face is None:
            # === CHUNKED_NO_FACE_PASSTHROUGH ===
            # chunk 전체에 face 없으면 (drama 의 어두운/풍경 chunk) 원본 그대로
            # 출력하도록 dummy placeholder 생성. lipsync 적용은 안 됨.
            print(f"[FACE_PASSTHROUGH] chunk 전체에 face 없음 → 원본 그대로 출력")
            import torch as _torch_pf
            placeholder_face = _torch_pf.zeros(3, 512, 512, dtype=_torch_pf.uint8)
            placeholder_box = [0, 0, 512, 512]
            placeholder_affine = _torch_pf.eye(2, 3, dtype=_torch_pf.float32).numpy() if hasattr(_torch_pf.eye(2, 3, dtype=_torch_pf.float32), "numpy") else None
            # 모든 frame 을 invalid 로 채워서 원본 유지 흐름 타게 함
            faces = [None] * len(video_frames)
            boxes = [None] * len(video_frames)
            affine_matrices = [None] * len(video_frames)
            valid_mask = [False] * len(video_frames)
            return faces, boxes, affine_matrices, valid_mask'''


def main():
    src = PIPELINE_PY.read_text()
    if "CHUNKED_NO_FACE_PASSTHROUGH" in src:
        print("[fix_chunked_no_face_passthrough] already patched")
        return 0
    if OLD not in src:
        print("[fix_chunked_no_face_passthrough] anchor not found")
        return 1
    PIPELINE_PY.write_text(src.replace(OLD, NEW, 1))
    print("[fix_chunked_no_face_passthrough] OK — empty-face chunks now pass through")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
