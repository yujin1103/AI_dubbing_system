from insightface.app import FaceAnalysis
import numpy as np
import torch

INSIGHTFACE_DETECT_SIZE = 512


class FaceDetector:
    def __init__(self, device="cuda"):
        # === SPEAKER_PROFILE_PATCH: auto-enable face embedding ===
        # Recognition is enabled if EITHER explicit flag is set, OR the
        # speaker profile path is set (implying the caller will need
        # embeddings to match faces against profiles).
        import os as _os_sp
        _explicit = _os_sp.environ.get("LATENTSYNC_ENABLE_FACE_RECOGNITION", "0") == "1"
        _profile_path = _os_sp.environ.get("LATENTSYNC_SPEAKER_PROFILES_PATH")
        _profile_set = bool(_profile_path and _os_sp.path.isfile(_profile_path))
        _enable_emb = _explicit or _profile_set
        _modules = ["detection", "landmark_2d_106"]
        if _enable_emb:
            _modules.append("recognition")
            print(f"[FaceDetector] recognition ENABLED (explicit={_explicit}, profile_set={_profile_set})", flush=True)
        self._embedding_enabled = _enable_emb
        self.last_embedding = None
        # === SPEAKER_PROFILE_PATCH end ===
        self.app = FaceAnalysis(
            allowed_modules=_modules,
            root="checkpoints/auxiliary",
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    def __call__(self, frame, threshold=0.5):  # FACE_CONFIDENCE_FIX: 0.5 -> 0.85
        # === FACE_DETECTOR_STRICT_PATCH ===
        # LATENTSYNC_FACE_STRICT=1 일 때 strict mode (드라마 artifact 방지)
        import os as _os_fd
        _strict = _os_fd.environ.get("LATENTSYNC_FACE_STRICT", "0") == "1"
        if _strict:
            threshold = 0.85          # 0.5 → 0.85
            _wh_min = 0.4             # 0.2 → 0.4 (측면 face skip)
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
            self.last_embedding = None  # SPEAKER_PROFILE_PATCH
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
                    get_face_store = face

        if get_face_store is None:
            self.last_embedding = None  # SPEAKER_PROFILE_PATCH
            return None, None
        else:
            face = get_face_store
            lmk = np.round(face.landmark_2d_106).astype(np.int_)

            halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)  # lmk[73]

            sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
            halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
            upper_bond = halk_face_coord[1] - halk_face_dist  # *0.94

            x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

            y2 += int((x2 - x1) * 0.1)
            x1 -= int((x2 - x1) * 0.05)
            x2 += int((x2 - x1) * 0.05)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)

            # === SPEAKER_PROFILE_PATCH: stash embedding of chosen face ===
            if self._embedding_enabled:
                self.last_embedding = getattr(face, "normed_embedding", None)
            else:
                self.last_embedding = None
            # === SPEAKER_PROFILE_PATCH end ===
            return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
