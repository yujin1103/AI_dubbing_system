"""Installer: applies speaker-profile patches to a LatentSync container.

Idempotent. Backs up originals as `.pre_speaker_profile` if not done already.

Patches applied:
  1. `latentsync/utils/face_detector.py` — enable insightface 'recognition' if
     env var `LATENTSYNC_ENABLE_FACE_RECOGNITION=1` + stash `last_embedding`.
  2. `latentsync/utils/image_processor.py` — stash `last_face_embedding`
     from face_detector onto the ImageProcessor instance.
  3. `latentsync/utils/asd_filter.py` — REPLACE with the version that
     includes `SpeakerFaceProfiles` + `AudioGenderTimeline` classes.
     (Assumes patches/asd_filter.py source file is alongside this script.)
  4. `latentsync/pipelines/lipsync_pipeline.py` — wire profile + gender
     gates after the ASD bbox match gate.

Run inside the container OR via docker exec:
    python /workspace/patches/fix_speaker_profile.py
"""
from __future__ import annotations
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent

LATENTSYNC_ROOT = Path("/opt/LatentSync")
FILES = {
    "face_detector": LATENTSYNC_ROOT / "latentsync" / "utils" / "face_detector.py",
    "image_processor": LATENTSYNC_ROOT / "latentsync" / "utils" / "image_processor.py",
    "asd_filter": LATENTSYNC_ROOT / "latentsync" / "utils" / "asd_filter.py",
    "lipsync_pipeline": LATENTSYNC_ROOT / "latentsync" / "pipelines" / "lipsync_pipeline.py",
}
MARKER = "SPEAKER_PROFILE_PATCH"


def backup(p: Path) -> None:
    b = p.with_suffix(p.suffix + ".pre_speaker_profile")
    if not b.exists():
        shutil.copyfile(p, b)
        print(f"backed up -> {b}")


def patch_face_detector():
    p = FILES["face_detector"]
    src = p.read_text()
    if MARKER in src:
        print("face_detector already patched")
        return
    backup(p)

    old1 = """class FaceDetector:
    def __init__(self, device=\"cuda\"):
        self.app = FaceAnalysis(
            allowed_modules=[\"detection\", \"landmark_2d_106\"],
            root=\"checkpoints/auxiliary\",
            providers=[\"CUDAExecutionProvider\"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))"""
    new1 = """class FaceDetector:
    def __init__(self, device=\"cuda\"):
        # === SPEAKER_PROFILE_PATCH: optionally enable face embedding ===
        import os as _os_sp
        _modules = [\"detection\", \"landmark_2d_106\"]
        _enable_emb = _os_sp.environ.get(\"LATENTSYNC_ENABLE_FACE_RECOGNITION\", \"0\") == \"1\"
        if _enable_emb:
            _modules.append(\"recognition\")
        self._embedding_enabled = _enable_emb
        self.last_embedding = None
        # === SPEAKER_PROFILE_PATCH end ===
        self.app = FaceAnalysis(
            allowed_modules=_modules,
            root=\"checkpoints/auxiliary\",
            providers=[\"CUDAExecutionProvider\"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))"""
    assert old1 in src, "face_detector init anchor not found"
    src = src.replace(old1, new1, 1)

    old2 = """            return (x1, y1, x2, y2), lmk


def cuda_to_int"""
    new2 = """            # === SPEAKER_PROFILE_PATCH: stash embedding of chosen face ===
            if self._embedding_enabled:
                self.last_embedding = getattr(face, \"normed_embedding\", None)
            else:
                self.last_embedding = None
            # === SPEAKER_PROFILE_PATCH end ===
            return (x1, y1, x2, y2), lmk


def cuda_to_int"""
    assert old2 in src, "face_detector return anchor not found"
    src = src.replace(old2, new2, 1)

    old3 = """        if len(faces) == 0:
            return None, None"""
    new3 = """        if len(faces) == 0:
            self.last_embedding = None  # SPEAKER_PROFILE_PATCH
            return None, None"""
    assert old3 in src, "face_detector no-faces anchor not found"
    src = src.replace(old3, new3, 1)

    old4 = """        if get_face_store is None:
            return None, None"""
    new4 = """        if get_face_store is None:
            self.last_embedding = None  # SPEAKER_PROFILE_PATCH
            return None, None"""
    assert old4 in src, "face_detector no-store anchor not found"
    src = src.replace(old4, new4, 1)

    p.write_text(src)
    print("face_detector patched")


def patch_image_processor():
    p = FILES["image_processor"]
    src = p.read_text()
    if "SPEAKER_PROFILE_PATCH: face embedding" in src:
        print("image_processor already patched")
        return
    backup(p)

    old = """        bbox, landmark_2d_106 = self.face_detector(image)
        # === ASD_BBOX_MATCH_PATCH ===
        # Stash original-frame bbox so the lipsync pipeline can ask ASD
        # whether the detected face matches the speaker track.
        self.last_face_bbox = bbox
        # === ASD_BBOX_MATCH_PATCH end ==="""
    new = """        bbox, landmark_2d_106 = self.face_detector(image)
        # === ASD_BBOX_MATCH_PATCH ===
        # Stash original-frame bbox so the lipsync pipeline can ask ASD
        # whether the detected face matches the speaker track.
        self.last_face_bbox = bbox
        # === ASD_BBOX_MATCH_PATCH end ===
        # === SPEAKER_PROFILE_PATCH: face embedding ===
        # Stash face embedding (if recognition module enabled) so the lipsync
        # pipeline can verify the detected face matches the current speaker.
        self.last_face_embedding = getattr(self.face_detector, \"last_embedding\", None)
        # === SPEAKER_PROFILE_PATCH end ==="""
    assert old in src, "image_processor anchor not found"
    src = src.replace(old, new, 1)
    p.write_text(src)
    print("image_processor patched")


def install_asd_filter():
    src = HERE / "asd_filter.py"
    if not src.is_file():
        print(f"ERROR: source asd_filter.py not at {src}")
        return False
    dst = FILES["asd_filter"]
    backup(dst)
    shutil.copyfile(src, dst)
    print(f"asd_filter.py installed -> {dst}")
    return True


def patch_lipsync_pipeline():
    p = FILES["lipsync_pipeline"]
    src = p.read_text()
    if "SPEAKER_PROFILE_PATCH:lazy-init" in src:
        print("lipsync_pipeline already speaker_profile patched")
        return
    backup(p)

    old1 = """        # === ASD_FILTER_PATCH:lazy-init ===
        # Initialize filter on first call (subprocess env var read here).
        if not hasattr(self, \"_asd_filter\"):
            try:
                from latentsync.utils.asd_filter import maybe_load_filter
                self._asd_filter = maybe_load_filter()
            except Exception as _e:
                import traceback as _tb
                print(f\"[ASD-Filter] init failed (continuing without filter): {_e}\")
                _tb.print_exc()
                self._asd_filter = None
            self._asd_global_frame_offset = 0
            self._asd_skip_count = 0
        # === ASD_FILTER_PATCH:lazy-init end ==="""
    new1 = """        # === ASD_FILTER_PATCH:lazy-init ===
        # Initialize filter on first call (subprocess env var read here).
        if not hasattr(self, \"_asd_filter\"):
            try:
                from latentsync.utils.asd_filter import maybe_load_filter
                self._asd_filter = maybe_load_filter()
            except Exception as _e:
                import traceback as _tb
                print(f\"[ASD-Filter] init failed (continuing without filter): {_e}\")
                _tb.print_exc()
                self._asd_filter = None
            self._asd_global_frame_offset = 0
            self._asd_skip_count = 0
        # === ASD_FILTER_PATCH:lazy-init end ===
        # === SPEAKER_PROFILE_PATCH:lazy-init ===
        if not hasattr(self, \"_speaker_profiles\"):
            try:
                from latentsync.utils.asd_filter import (
                    maybe_load_speaker_profiles, maybe_load_audio_gender,
                )
                self._speaker_profiles = maybe_load_speaker_profiles()
                self._audio_gender = maybe_load_audio_gender()
            except Exception as _e:
                import traceback as _tb
                print(f\"[SpeakerProfile] init failed (continuing without profiles): {_e}\")
                _tb.print_exc()
                self._speaker_profiles = None
                self._audio_gender = None
            self._sp_mismatch_count = 0
            self._sp_gender_mismatch_count = 0
        # === SPEAKER_PROFILE_PATCH:lazy-init end ==="""
    assert old1 in src
    src = src.replace(old1, new1, 1)

    old2 = """            # === ASD_BBOX_MATCH_PATCH ===
            # Per-FACE speaker check: if ASD has data at this frame and the
            # detected face bbox does NOT match the speaker track bbox, the
            # detected face is the listener / wrong person — skip lipsync.
            if face is not None and _asd_flt is not None:
                _det_bbox = getattr(self.image_processor, \"last_face_bbox\", None)
                if _det_bbox is not None:
                    _is_spk = _asd_flt.is_detected_face_speaker(
                        getattr(self, \"_asd_global_frame_offset\", 0) + fi,
                        list(_det_bbox),
                    )
                    if _is_spk is False:
                        if not hasattr(self, \"_asd_bbox_mismatch_count\"):
                            self._asd_bbox_mismatch_count = 0
                        self._asd_bbox_mismatch_count += 1
                        face = None  # treat like face-miss → keep original
            # === ASD_BBOX_MATCH_PATCH end ==="""
    new2 = """            # === ASD_BBOX_MATCH_PATCH ===
            # Per-FACE speaker check: if ASD has data at this frame and the
            # detected face bbox does NOT match the speaker track bbox, the
            # detected face is the listener / wrong person — skip lipsync.
            if face is not None and _asd_flt is not None:
                _det_bbox = getattr(self.image_processor, \"last_face_bbox\", None)
                if _det_bbox is not None:
                    _is_spk = _asd_flt.is_detected_face_speaker(
                        getattr(self, \"_asd_global_frame_offset\", 0) + fi,
                        list(_det_bbox),
                    )
                    if _is_spk is False:
                        if not hasattr(self, \"_asd_bbox_mismatch_count\"):
                            self._asd_bbox_mismatch_count = 0
                        self._asd_bbox_mismatch_count += 1
                        face = None  # treat like face-miss → keep original
            # === ASD_BBOX_MATCH_PATCH end ===
            # === SPEAKER_PROFILE_PATCH: profile + gender check ===
            if face is not None:
                _g = getattr(self, \"_asd_global_frame_offset\", 0) + fi
                _profiles = getattr(self, \"_speaker_profiles\", None)
                if _profiles is not None:
                    _det_emb = getattr(self.image_processor, \"last_face_embedding\", None)
                    if _det_emb is not None:
                        _is_correct = _profiles.is_detected_face_correct_speaker(_g, _det_emb)
                        if _is_correct is False:
                            self._sp_mismatch_count = getattr(self, \"_sp_mismatch_count\", 0) + 1
                            face = None  # different person than the diarized speaker
                _audio_gender = getattr(self, \"_audio_gender\", None)
                if face is not None and _audio_gender is not None and _profiles is not None:
                    _aud_g = _audio_gender.gender_at_frame(_g)
                    _spk = _profiles.speaker_at_frame(_g)
                    _spk_g = _profiles.gender_hint(_spk) if _spk else None
                    if _aud_g in (\"male\", \"female\") and _spk_g in (\"male\", \"female\") and _aud_g != _spk_g:
                        self._sp_gender_mismatch_count = getattr(self, \"_sp_gender_mismatch_count\", 0) + 1
                        face = None
            # === SPEAKER_PROFILE_PATCH end ==="""
    assert old2 in src
    src = src.replace(old2, new2, 1)

    old3 = """        _asd_bbox_mm = getattr(self, \"_asd_bbox_mismatch_count\", 0)
        if _asd_bbox_mm > 0:
            print(f\"[ASD-BBox] {_asd_bbox_mm}/{len(video_frames)} frames detected face != speaker track → skipped\")
            self._asd_bbox_mismatch_count = 0"""
    new3 = """        _asd_bbox_mm = getattr(self, \"_asd_bbox_mismatch_count\", 0)
        if _asd_bbox_mm > 0:
            print(f\"[ASD-BBox] {_asd_bbox_mm}/{len(video_frames)} frames detected face != speaker track → skipped\")
            self._asd_bbox_mismatch_count = 0
        _sp_mm = getattr(self, \"_sp_mismatch_count\", 0)
        if _sp_mm > 0:
            print(f\"[SpeakerProfile] {_sp_mm}/{len(video_frames)} frames detected face != diarized speaker profile → skipped\")
            self._sp_mismatch_count = 0
        _sp_gmm = getattr(self, \"_sp_gender_mismatch_count\", 0)
        if _sp_gmm > 0:
            print(f\"[SpeakerProfile-Gender] {_sp_gmm}/{len(video_frames)} frames audio gender != face gender → skipped\")
            self._sp_gender_mismatch_count = 0"""
    assert old3 in src
    src = src.replace(old3, new3, 1)

    p.write_text(src)
    print("lipsync_pipeline patched (3 blocks)")


def main():
    patch_face_detector()
    patch_image_processor()
    if not install_asd_filter():
        return 1
    patch_lipsync_pipeline()
    print("\n=== speaker profile patch install complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
