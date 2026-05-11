"""LatentSync image_processor 에 face profile (측면 얼굴) skip 로직 추가.

LatentSync 는 정면 얼굴에 학습되어 있어서 측면 (90도 가까이 회전된 face)에서는
입 landmark 위치가 부정확해진다 → 입이 "떠다니는" artifact.

해결: 106-point landmark 로 face yaw 추정 → threshold 이상 회전이면 face=None 반환
      (FACE_SKIP_PATCH 흐름 타고 원본 frame 그대로 유지).

Yaw 추정 방식:
    pt_left_eye  = mean(landmark[43, 48..51])  (왼쪽 눈썹 중앙)
    pt_right_eye = mean(landmark[101..105])    (오른쪽 눈썹 중앙)
    pt_nose      = mean(landmark[74, 77, 83, 86]) (코 중앙)

    eye_distance = ||left_eye - right_eye||
    eye_midpoint = (left_eye + right_eye) / 2
    nose_offset  = |nose.x - eye_midpoint.x|
    yaw_ratio    = nose_offset / eye_distance

    정면 face: ratio ≈ 0.05~0.15
    측면 30도: ratio ≈ 0.25
    측면 60도: ratio ≈ 0.5
    측면 90도: ratio → 1+ (eye_distance 도 작아짐)

환경변수:
    LATENTSYNC_PROFILE_THRESHOLD=0.35  (default, 약 45도 이상이면 skip)
    설정 0 또는 환경변수 없으면 비활성화 (기존 동작 유지).
"""
from pathlib import Path

IMG_PROC = Path("/opt/LatentSync/latentsync/utils/image_processor.py")

MARKER = "# === FACE_PROFILE_SKIP_PATCH"
OLD_BLOCK = '''    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            return None, None, None  # FACE_SKIP_PATCH

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])'''

NEW_BLOCK = '''    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            return None, None, None  # FACE_SKIP_PATCH

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        # === FACE_PROFILE_SKIP_PATCH ===
        # 측면 face (yaw > threshold) 는 LatentSync 가 입 위치를 잘못 잡아서
        # "입이 떠다니는" artifact 발생 → 검출만 하고 원본 frame 유지.
        import os as _os_pf
        _profile_thresh = float(_os_pf.environ.get("LATENTSYNC_PROFILE_THRESHOLD", "0"))
        if _profile_thresh > 0:
            eye_dist = float(np.linalg.norm(pt_right_eye - pt_left_eye))
            if eye_dist > 1.0:  # eye_dist=0 이면 div by zero 회피
                eye_midpoint_x = float((pt_left_eye[0] + pt_right_eye[0]) / 2)
                nose_offset_x = abs(float(pt_nose[0]) - eye_midpoint_x)
                yaw_ratio = nose_offset_x / eye_dist
                if yaw_ratio > _profile_thresh:
                    return None, None, None  # profile face → skip (원본 유지)
        # === FACE_PROFILE_SKIP_PATCH end ===

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])'''


def main():
    src = IMG_PROC.read_text()
    if MARKER in src:
        print("[fix_face_profile_skip] already patched")
        return 0
    if OLD_BLOCK not in src:
        print("[fix_face_profile_skip] anchor not found — abort")
        return 1
    IMG_PROC.write_text(src.replace(OLD_BLOCK, NEW_BLOCK, 1))
    print("[fix_face_profile_skip] patched (set LATENTSYNC_PROFILE_THRESHOLD=0.35 to enable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
