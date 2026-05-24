"""Patched face_detector.py with:

  Phase 1: Track-level person ID locking
    LATENTSYNC_TRACK_LOCK=1  → prefer face overlapping previous frame's bbox

  Phase 3: Overlap detection (multi-face) skip
    LATENTSYNC_OVERLAP_SKIP=1 → when ≥2 faces pass strict filters, skip lipsync
                                 (returns None, None) — original frame preserved

  Embedding-aware face match (when speaker profiles enabled)
"""
from insightface.app import FaceAnalysis
import numpy as np
import torch

INSIGHTFACE_DETECT_SIZE = 512


def _iou(b1, b2):
    """IoU between two bboxes (x1,y1,x2,y2)."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class FaceDetector:
    def __init__(self, device="cuda"):
        # === SPEAKER_PROFILE_PATCH: auto-enable face embedding ===
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

        # === PHASE_1_TRACK_LOCK_PATCH ===
        # Buffer last frame's chosen face bbox for tracking continuity.
        # Reset every K frames if no detection (avoid stale lock).
        self._track_lock_enabled = _os_sp.environ.get("LATENTSYNC_TRACK_LOCK", "0") == "1"
        self._last_chosen_bbox = None       # tuple of int (x1,y1,x2,y2) or None
        self._frames_since_lock = 0
        self._track_lock_iou_thresh = float(_os_sp.environ.get(
            "LATENTSYNC_TRACK_LOCK_IOU", "0.30"))
        self._track_lock_reset_after = int(_os_sp.environ.get(
            "LATENTSYNC_TRACK_LOCK_RESET_AFTER", "8"))
        if self._track_lock_enabled:
            print(f"[FaceDetector] TRACK_LOCK enabled (IoU≥{self._track_lock_iou_thresh}, "
                  f"reset after {self._track_lock_reset_after} no-detect frames)",
                  flush=True)
        # === PHASE_1_TRACK_LOCK_PATCH end ===

        # === PHASE_3_OVERLAP_SKIP_PATCH ===
        # Skip lipsync when ≥2 valid faces detected (multi-speaker scene).
        self._overlap_skip_enabled = _os_sp.environ.get(
            "LATENTSYNC_OVERLAP_SKIP", "0") == "1"
        if self._overlap_skip_enabled:
            print("[FaceDetector] OVERLAP_SKIP enabled (skip lipsync when ≥2 faces)",
                  flush=True)
        # === PHASE_3_OVERLAP_SKIP_PATCH end ===

        self.app = FaceAnalysis(
            allowed_modules=_modules,
            root="checkpoints/auxiliary",
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    def __call__(self, frame, threshold=0.5):
        # === FACE_DETECTOR_STRICT_PATCH ===
        import os as _os_fd
        _strict = _os_fd.environ.get("LATENTSYNC_FACE_STRICT", "0") == "1"
        if _strict:
            threshold = 0.85
            _wh_min = 0.4
            _wh_max = 1.5
        else:
            _wh_min = 0.2
            _wh_max = 1.5
        # === FACE_DETECTOR_STRICT_PATCH end ===
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        if len(faces) == 0:
            self.last_embedding = None
            # PHASE_1: nothing to lock; let lock decay
            self._frames_since_lock += 1
            if (self._track_lock_enabled
                    and self._frames_since_lock > self._track_lock_reset_after):
                self._last_chosen_bbox = None
            return None, None

        # First pass: collect all faces passing strict filters
        valid_faces = []
        for face in faces:
            bbox = face.bbox.astype(np.int_).tolist()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < 50 or h < 80:
                continue
            if w / h > _wh_max or w / h < _wh_min:
                continue
            if face.det_score < threshold:
                continue
            if _strict:
                try:
                    _lmk = face.landmark_2d_106
                    _le_y = float((_lmk[33][1] + _lmk[35][1]) / 2)
                    _re_y = float((_lmk[87][1] + _lmk[89][1]) / 2)
                    _eye_y_diff = abs(_le_y - _re_y)
                    _eye_x_diff = abs(float(_lmk[33][0]) - float(_lmk[87][0]))
                    if _eye_x_diff > 1.0 and _eye_y_diff / _eye_x_diff > 0.5:
                        continue
                except Exception:
                    pass
            valid_faces.append((face, bbox, w * h))

        if not valid_faces:
            self.last_embedding = None
            self._frames_since_lock += 1
            if (self._track_lock_enabled
                    and self._frames_since_lock > self._track_lock_reset_after):
                self._last_chosen_bbox = None
            return None, None

        # === PHASE_3_OVERLAP_SKIP_PATCH ===
        if self._overlap_skip_enabled and len(valid_faces) >= 2:
            # ≥2 speakers visible → ambiguous, skip lipsync
            self.last_embedding = None
            return None, None
        # === PHASE_3_OVERLAP_SKIP_PATCH end ===

        # === PHASE_1_TRACK_LOCK_PATCH: prefer face matching previous track ===
        get_face_store = None
        if self._track_lock_enabled and self._last_chosen_bbox is not None:
            # find best IoU with prev
            best_iou = 0.0
            best_face = None
            for face, bbox, size in valid_faces:
                cur_iou = _iou(bbox, self._last_chosen_bbox)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_face = face
                    best_bbox = bbox
            if best_iou >= self._track_lock_iou_thresh:
                get_face_store = best_face
                self._last_chosen_bbox = best_bbox
                self._frames_since_lock = 0
        # === PHASE_1_TRACK_LOCK_PATCH end ===

        # Default (and fallback if no track match): pick largest
        if get_face_store is None:
            max_size = 0
            for face, bbox, size in valid_faces:
                if size > max_size:
                    max_size = size
                    get_face_store = face
                    chosen_bbox = bbox
            if self._track_lock_enabled and get_face_store is not None:
                self._last_chosen_bbox = chosen_bbox
                self._frames_since_lock = 0

        face = get_face_store
        lmk = np.round(face.landmark_2d_106).astype(np.int_)

        halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)
        sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
        halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
        upper_bond = halk_face_coord[1] - halk_face_dist

        x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond),
                          np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

        y2 += int((x2 - x1) * 0.1)
        x1 -= int((x2 - x1) * 0.05)
        x2 += int((x2 - x1) * 0.05)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(f_w, x2)
        y2 = min(f_h, y2)

        # SPEAKER_PROFILE_PATCH: stash embedding
        if self._embedding_enabled:
            self.last_embedding = getattr(face, "normed_embedding", None)
        else:
            self.last_embedding = None

        return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1, 10, 12, 14, 16, 3, 5, 7, 0, 23,
    21, 19, 32, 30, 28, 26, 17, 43, 48, 49,
    51, 50, 102, 103, 104, 105, 101, 73, 74, 86,
]
