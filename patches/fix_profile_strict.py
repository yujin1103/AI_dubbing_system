"""Quality fix: tighten wrong-person lipsync gate.

Two new env vars (read at runtime, no rebuild needed):

  LATENTSYNC_PROFILE_MATCH_THRESHOLD=0.55
    Override JSON's match_threshold (default 0.5). Higher = stricter.
    Recommended 0.55-0.60 for drama with similar-looking people.

  LATENTSYNC_PROFILE_STRICT_NONE=1
    When the profile gate returns None (no diarization OR no embedding),
    treat as FALSE (skip lipsync). Default off (None means "no opinion").
    Use when wrong-person lipsync is worse than no-lipsync.

Applies to:
  /opt/LatentSync/latentsync/utils/asd_filter.py
  /workspace/patches/asd_filter.py
"""
import os
import re
import shutil

TARGETS = [
    "/opt/LatentSync/latentsync/utils/asd_filter.py",
    "/workspace/patches/asd_filter.py",
]


def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "PROFILE_MATCH_THRESHOLD_PATCH" in content:
        print(f"[skip] already patched: {path}")
        return False

    # --- patch 1: override match_threshold from env ---
    old_init = """    def __init__(self, fps: float, match_threshold: float,
                 speakers: Dict, timeline: List[Dict]) -> None:
        self.fps = fps
        self.match_threshold = match_threshold"""

    new_init = """    def __init__(self, fps: float, match_threshold: float,
                 speakers: Dict, timeline: List[Dict]) -> None:
        self.fps = fps
        # === PROFILE_MATCH_THRESHOLD_PATCH (5/13) ===
        # Env override for stricter matching on drama with similar faces.
        import os as _os_pmt
        _env_thr = _os_pmt.environ.get("LATENTSYNC_PROFILE_MATCH_THRESHOLD")
        if _env_thr:
            try:
                match_threshold = float(_env_thr)
                print(f"[PROFILE] match_threshold override -> {match_threshold}",
                      flush=True)
            except Exception:
                pass
        # === PROFILE_MATCH_THRESHOLD_PATCH end ===
        self.match_threshold = match_threshold"""

    if old_init not in content:
        print(f"[fail] match_threshold init pattern not found in {path}")
        return False
    content = content.replace(old_init, new_init)

    # --- patch 2: STRICT_NONE — treat None return as False ---
    # Locate is_detected_face_correct_speaker (also called is_match_for_profile
    # depending on version). Look for the "return None" lines and add a strict
    # check upfront.
    # The function name varies by version; look for the body pattern.

    # The pattern we want to wrap:
    # ```
    # if detected_embedding is None:
    #     return None
    # speaker = self.speaker_at_frame(global_frame_idx)
    # if speaker is None:
    #     return None
    # target = self.expected_embedding(speaker)
    # if target is None:
    #     return False
    # sim = _cosine_sim(...)
    # return sim >= self.match_threshold
    # ```
    # In strict-none mode, the first two "return None" become "return False".

    old_body = '''        if detected_embedding is None:
            return None  # no embedding -> can't compare; let other gates handle
        speaker = self.speaker_at_frame(global_frame_idx)
        if speaker is None:
            return None  # no diarization data at this frame -> no opinion
        target = self.expected_embedding(speaker)'''

    new_body = '''        # === PROFILE_STRICT_NONE_PATCH (5/13) ===
        import os as _os_psn
        _strict_none = _os_psn.environ.get("LATENTSYNC_PROFILE_STRICT_NONE", "0") == "1"
        _none_result = False if _strict_none else None
        # === PROFILE_STRICT_NONE_PATCH end ===
        if detected_embedding is None:
            return _none_result  # no embedding -> strict: skip, lenient: opinion
        speaker = self.speaker_at_frame(global_frame_idx)
        if speaker is None:
            return _none_result  # no diarization -> strict: skip, lenient: opinion
        target = self.expected_embedding(speaker)'''

    if old_body not in content:
        print(f"[fail] is_match body pattern not found in {path}")
        return False
    content = content.replace(old_body, new_body)

    # Backup
    bk = path + ".pre_profile_strict"
    shutil.copy(path, bk)
    print(f"[backup] {bk}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[patched] {path}")
    return True


if __name__ == "__main__":
    n_patched = 0
    for t in TARGETS:
        if os.path.isfile(t):
            if patch_file(t):
                n_patched += 1
        else:
            print(f"[miss] {t} not found")
    print(f"[done] patched {n_patched} file(s)")
