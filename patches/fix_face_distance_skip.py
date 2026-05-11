"""LatentSync image_processor 에 face distance (face 크기) skip 로직 추가.

작은 face (멀리 있거나 화면 비중 작은) 는 LatentSync 가 512×512 로 over-zoom
하면서 입 위치 부정확 + paste-back 시 마스크 자국 튀어나옴.

해결: bbox area 가 threshold 미만이면 face=None 반환 (원본 frame 유지).
profile_skip 과 비슷한 방식.

판정 공식:
    bbox = [x1, y1, x2, y2] (face_detector 출력)
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    face_diag = sqrt(bbox_w**2 + bbox_h**2)
    distance_ratio = face_diag / frame_diag  (frame_diag 는 1080p 영상 기준 ~2202)

    정면 close-up (TED 같음): 0.25~0.5
    중간 거리: 0.15~0.25
    멀리: 0.05~0.15 ← skip 권장
    아주 작음: <0.05 ← 무조건 skip

환경변수:
    LATENTSYNC_FACE_DIAG_MIN_RATIO=0.10  (default 0 = 비활성, 권장 0.10~0.15)
    설정 0 또는 환경변수 없으면 비활성화 (기존 동작 유지).
"""
from pathlib import Path

IMG_PROC = Path("/opt/LatentSync/latentsync/utils/image_processor.py")

MARKER = "# === FACE_DISTANCE_SKIP_PATCH"
ANCHOR = """        # === FACE_PROFILE_SKIP_PATCH end ===

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])"""
INJECT = """        # === FACE_PROFILE_SKIP_PATCH end ===

        # === FACE_DISTANCE_SKIP_PATCH ===
        # 작은 (멀리 있는) face 는 over-zoom 으로 입 위치 부정확 → 원본 유지
        import os as _os_dist
        _dist_min = float(_os_dist.environ.get("LATENTSYNC_FACE_DIAG_MIN_RATIO", "0"))
        if _dist_min > 0 and bbox is not None:
            try:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                bbox_diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                # image is torch tensor (H, W, C) — get frame diagonal
                if hasattr(image, "shape") and len(image.shape) >= 2:
                    H, W = int(image.shape[0]), int(image.shape[1])
                    frame_diag = (H ** 2 + W ** 2) ** 0.5
                    if frame_diag > 0:
                        dist_ratio = bbox_diag / frame_diag
                        if dist_ratio < _dist_min:
                            return None, None, None  # 너무 작은 face → skip
            except Exception:
                pass  # 측정 실패 시 패스
        # === FACE_DISTANCE_SKIP_PATCH end ===

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])"""


def main():
    src = IMG_PROC.read_text()
    if MARKER in src:
        print("[fix_face_distance_skip] already patched")
        return 0
    if ANCHOR not in src:
        print("[fix_face_distance_skip] anchor not found — abort (run fix_face_profile_skip.py first)")
        return 1
    IMG_PROC.write_text(src.replace(ANCHOR, INJECT, 1))
    print("[fix_face_distance_skip] patched (set LATENTSYNC_FACE_DIAG_MIN_RATIO=0.10 to enable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
