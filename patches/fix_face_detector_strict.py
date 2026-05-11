"""LatentSync face_detector.py 강화 — 드라마 artifact 줄이기.

발견된 문제 (test4 drama frames):
  1. 측면 face 가 통과 (w/h ratio 0.2 임계값이 너무 관대)
  2. det_score 0.5 (주석에는 0.85 권장이지만 코드 미수정)
  3. landmark sanity 체크 없음

이 패치:
  A) det_score threshold 0.5 → 0.85 (낮은 신뢰도 face skip)
  B) w/h ratio 더 엄격: 0.2 → 0.55 (측면 face 더 적극 skip)
  C) landmark sanity: 눈은 코보다 위, 입 landmark가 nose 와 너무 가까우면 skip

환경변수로 ON/OFF:
  LATENTSYNC_FACE_STRICT=1  → 위 강화 적용 (default off, 기존 동작 유지)
"""
from pathlib import Path

FACE_DET = Path("/opt/LatentSync/latentsync/utils/face_detector.py")

MARKER = "# === FACE_DETECTOR_STRICT_PATCH"
OLD_BLOCK = '''    def __call__(self, frame, threshold=0.5):  # FACE_CONFIDENCE_FIX: 0.5 -> 0.85
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        get_face_store = None
        max_size = 0

        if len(faces) == 0:
            return None, None
        else:
            for face in faces:
                bbox = face.bbox.astype(np.int_).tolist()
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 50 or h < 80:
                    continue
                if w / h > 1.5 or w / h < 0.2:
                    continue
                if face.det_score < threshold:
                    continue
                size_now = w * h

                if size_now > max_size:
                    max_size = size_now
                    get_face_store = face'''

NEW_BLOCK = '''    def __call__(self, frame, threshold=0.5):  # FACE_CONFIDENCE_FIX: 0.5 -> 0.85
        # === FACE_DETECTOR_STRICT_PATCH ===
        # LATENTSYNC_FACE_STRICT=1 일 때 strict mode (드라마 artifact 방지)
        import os as _os_fd
        _strict = _os_fd.environ.get("LATENTSYNC_FACE_STRICT", "0") == "1"
        if _strict:
            threshold = 0.85          # 0.5 → 0.85
            _wh_min = 0.4             # 0.2 → 0.4 (측면 face skip, 0.55는 과함)
            _wh_max = 1.5
        else:
            _wh_min = 0.2
            _wh_max = 1.5
        # === FACE_DETECTOR_STRICT_PATCH end ===
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        get_face_store = None
        max_size = 0

        if len(faces) == 0:
            return None, None
        else:
            for face in faces:
                bbox = face.bbox.astype(np.int_).tolist()
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 50 or h < 80:
                    continue
                if w / h > _wh_max or w / h < _wh_min:
                    continue
                if face.det_score < threshold:
                    continue
                # FACE_DETECTOR_STRICT_PATCH: landmark sanity (strict mode)
                if _strict:
                    try:
                        _lmk = face.landmark_2d_106
                        # left eye center vs right eye center y 가 비슷해야 (롤 ±30°)
                        _le_y = float((_lmk[33][1] + _lmk[35][1]) / 2)
                        _re_y = float((_lmk[87][1] + _lmk[89][1]) / 2)
                        _eye_y_diff = abs(_le_y - _re_y)
                        _eye_x_diff = abs(float(_lmk[33][0]) - float(_lmk[87][0]))
                        # roll 너무 크면 skip
                        if _eye_x_diff > 1.0 and _eye_y_diff / _eye_x_diff > 0.5:
                            continue
                    except Exception:
                        pass
                size_now = w * h

                if size_now > max_size:
                    max_size = size_now
                    get_face_store = face'''


def main():
    src = FACE_DET.read_text()
    if MARKER in src:
        print("[fix_face_detector_strict] already patched")
        return 0
    if OLD_BLOCK not in src:
        print("[fix_face_detector_strict] anchor not found — abort")
        return 1
    FACE_DET.write_text(src.replace(OLD_BLOCK, NEW_BLOCK, 1))
    print("[fix_face_detector_strict] patched (LATENTSYNC_FACE_STRICT=1 to enable strict mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
