"""ASD-based lipsync filter runtime (loaded inside the LatentSync subprocess).

Goal:
    Per-frame "is there any active speaker visible to the camera at this frame?"
    -> returned as a `should_skip(global_frame_idx)` boolean.

Activation:
    The host orchestrator sets the env var
        LATENTSYNC_ASD_FILTER_RUN_DIR=/workspace/media/runs/<run_id>/
    before launching `scripts/inference`. The patched LatentSync pipeline
    constructs a `LipsyncASDFilter` from that path on first use.

Discovery (in order):
    1. <run_dir>/meta/asd_filter_index.json
       (canonical; written by the orchestrator after per-chunk ASD runs)
    2. media/cache/lightasd/<chunk_hash>.pkl, matched per-chunk by content hash
       (fallback for runs predating the index file)

Index format (JSON):
    {
      "version": 1,
      "fps": 25.0,
      "score_threshold": 0.0,
      "chunks": [
        {"stem": "...", "asd_path": "/abs/path/<hash>.pkl", "n_frames": 2732,
         "score_threshold": 0.0}
      ]
    }
    The chunk order MUST match the concat order fed to LatentSync.

Skip rule (scene-level, the simplest correct default):
    For each global frame index `g`:
      max_score_at[g] = max(track.scores at g across ALL face tracks active at g)
      should_skip(g) = (max_score_at[g] < threshold)
    Frames that no chunk covers, or that are inside a chunk but outside every
    track's active window, default to "do not skip" (preserves current
    behaviour where ASD has no opinion).

Threshold:
    score_threshold defaults to env LATENTSYNC_ASD_THRESHOLD or 0.0
    (LightASD scores are roughly in [-3, +6]; > 0 ~= speaking).

Caveats:
    * This is a per-frame *scene* check, not a per-face match. If frame has
      both speaker A and listener B and the LatentSync face detector picks B,
      we still let it through because *someone* is speaking on screen.
      Listener-only frames (over-the-shoulder shots, reaction shots) are
      what we actually want to catch — those have zero positive scores.
    * LightASD always normalises to 25fps internally. The orchestrator feeds
      LatentSync the same 25fps-rendered chunks, so frame indices align 1:1.
      If a future caller changes either side, the index file's `fps` field
      will diverge and we log a warning.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def _log(msg: str) -> None:
    print(f"[ASD-Filter] {msg}", flush=True)


class LipsyncASDFilter:
    """Per-global-frame skip lookup over a concatenated set of ASD chunks."""

    def __init__(
        self,
        chunks: List[Dict],
        score_threshold: float = 0.0,
        fps: float = 25.0,
    ) -> None:
        # chunks: [{"stem", "asd_path", "n_frames", "score_threshold"?}]
        # Concat them into one global timeline.
        self.fps = fps
        self.score_threshold = score_threshold
        self.chunk_specs = list(chunks)
        # offset[i] = first global frame index for chunk i
        self.chunk_offsets: List[int] = []
        offset = 0
        for c in self.chunk_specs:
            self.chunk_offsets.append(offset)
            offset += int(c.get("n_frames", 0))
        self.total_frames = offset
        # score_at[g] = float, or None if no ASD data (treat as "allow")
        # Lazy-fill: load each chunk's pickle on first access.
        self._score_at: List[Optional[float]] = [None] * self.total_frames
        self._loaded_chunks: set = set()
        self._covered: List[bool] = [False] * self.total_frames
        # tracks_at[g] = [{"bbox":[x1,y1,x2,y2], "score":float}, ...]  per-face info
        # for bbox-match speaker identification (per-FACE check, not just per-scene).
        self._tracks_at: List[List[Dict]] = [[] for _ in range(self.total_frames)]

    @classmethod
    def from_index(cls, index_path: Path) -> "LipsyncASDFilter":
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 1:
            _log(f"warning: index version {data.get('version')} != 1, parsing anyway")
        return cls(
            chunks=data.get("chunks", []),
            score_threshold=float(data.get("score_threshold", 0.0)),
            fps=float(data.get("fps", 25.0)),
        )

    @classmethod
    def from_run_dir(cls, run_dir: str) -> Optional["LipsyncASDFilter"]:
        rd = Path(run_dir)
        if not rd.is_dir():
            _log(f"run dir not found: {run_dir}")
            return None
        idx = rd / "meta" / "asd_filter_index.json"
        if idx.is_file():
            try:
                return cls.from_index(idx)
            except Exception as e:
                _log(f"failed to read index {idx}: {e}")
        # Fallback: try to assemble from cache.
        chunks_dir = rd / "chunks"
        if not chunks_dir.is_dir():
            _log(f"no chunks dir at {chunks_dir} -> filter disabled")
            return None
        # Order must match concat_chunks() in orchestrator: sorted by name,
        # filtered to "_final.mp4". We use stems of the *original* chunks
        # (no _final suffix) for cache hash matching.
        finals = sorted(
            p for p in chunks_dir.iterdir()
            if p.is_file() and p.name.endswith("_final.mp4")
        )
        if not finals:
            _log(f"no _final.mp4 chunks in {chunks_dir} -> filter disabled")
            return None
        cache_dir = Path("/workspace/media/cache/lightasd")
        chunks: List[Dict] = []
        threshold = float(os.environ.get("LATENTSYNC_ASD_THRESHOLD", "0.0"))
        for final in finals:
            stem = final.name[: -len("_final.mp4")]
            orig = chunks_dir / f"{stem}.mp4"
            asd_path = None
            n_frames = 0
            if orig.is_file():
                try:
                    h = _chunk_content_hash(str(orig))
                    candidate = cache_dir / f"{h}.pkl"
                    if candidate.is_file():
                        asd_path = str(candidate)
                        try:
                            import pickle
                            with open(candidate, "rb") as f:
                                pkl = pickle.load(f)
                            n_frames = int(pkl.get("n_frames", 0))
                        except Exception as e:
                            _log(f"can't read frames from {candidate}: {e}")
                except Exception as e:
                    _log(f"hash failed for {orig}: {e}")
            if asd_path is None:
                _log(f"no ASD cache for {stem} -> chunk passes through")
            chunks.append({
                "stem": stem,
                "asd_path": asd_path,
                "n_frames": n_frames,
                "score_threshold": threshold,
            })
        return cls(chunks=chunks, score_threshold=threshold)

    # ─── lazy chunk loading ───
    def _load_chunk(self, chunk_idx: int) -> None:
        if chunk_idx in self._loaded_chunks:
            return
        spec = self.chunk_specs[chunk_idx]
        offset = self.chunk_offsets[chunk_idx]
        n_frames = int(spec.get("n_frames", 0))
        self._loaded_chunks.add(chunk_idx)
        asd_path = spec.get("asd_path")
        if not asd_path or not os.path.isfile(asd_path):
            # No data for this chunk -> leave as None (do not skip).
            return
        try:
            import pickle
            with open(asd_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            _log(f"can't load {asd_path}: {e}")
            return
        # Mark every frame in this chunk's window as "covered" by an attempt;
        # un-covered frames within this window stay None and won't skip.
        for f in range(offset, offset + n_frames):
            if 0 <= f < self.total_frames:
                self._covered[f] = True
        # Per local frame, take max score across all tracks active at that frame.
        # Also record every (bbox, score) pair so bbox-match can find which
        # track corresponds to a given detected face.
        local_max: List[Optional[float]] = [None] * n_frames
        for t in data.get("tracks", []):
            frames = t.get("frames", [])
            scores = t.get("scores", [])
            bboxes = t.get("bboxes", [])
            for idx, (fi, sc) in enumerate(zip(frames, scores)):
                fi = int(fi)
                if 0 <= fi < n_frames:
                    cur = local_max[fi]
                    if cur is None or float(sc) > cur:
                        local_max[fi] = float(sc)
                    # store per-track entry for bbox match (bbox may be missing
                    # for older caches — degrade gracefully).
                    if idx < len(bboxes) and bboxes[idx] is not None:
                        g_idx = offset + fi
                        if 0 <= g_idx < self.total_frames:
                            self._tracks_at[g_idx].append({
                                "bbox": [float(x) for x in bboxes[idx]],
                                "score": float(sc),
                            })
        # Project into global timeline.
        for fi, val in enumerate(local_max):
            g = offset + fi
            if 0 <= g < self.total_frames:
                self._score_at[g] = val

    def _which_chunk(self, global_frame_idx: int) -> Optional[int]:
        if global_frame_idx < 0 or global_frame_idx >= self.total_frames:
            return None
        # binary search would be nicer; the lists are typically tiny (<20).
        last = -1
        for i, off in enumerate(self.chunk_offsets):
            if off > global_frame_idx:
                break
            last = i
        return last if last >= 0 else None

    def should_skip(self, global_frame_idx: int) -> bool:
        """Return True iff this frame should be lipsync-skipped per ASD."""
        chunk_idx = self._which_chunk(global_frame_idx)
        if chunk_idx is None:
            return False  # outside known chunks -> no opinion -> allow
        self._load_chunk(chunk_idx)
        if not self._covered[global_frame_idx]:
            return False
        sc = self._score_at[global_frame_idx]
        if sc is None:
            return False  # frame not in any ASD track window -> allow
        return sc < self.score_threshold

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        """IoU between two bboxes [x1,y1,x2,y2]."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def is_detected_face_speaker(
        self,
        global_frame_idx: int,
        detected_bbox: List[float],
        iou_match: float = 0.2,
    ) -> Optional[bool]:
        """Per-FACE speaker check (not per-scene like should_skip).

        Policy (STRICT by default, 5/11 update):
          - If the chunk has NO ASD data at all → return None (allow).
          - If the chunk has ASD data but the *frame* has no active tracks
            and `should_skip(g)` says no speaker either → return False (skip).
            Most "listener-only / no-face" frames hit this path.
          - If tracks are active at this frame:
              * Match detected_bbox to the best-IoU track.
              * IoU < iou_match → detected face is an *untracked* face in a
                tracked scene → wrong person → return False (skip).
              * IoU >= iou_match → return (best_track.score >= threshold).
                True = speaker, False = listener.

        Why strict by default: drama screenshots showed lipsync still being
        applied to listener / far-away non-speaker faces when ASD had no
        opinion on a specific bbox. With STRICT default, those frames keep
        the original.
        """
        chunk_idx = self._which_chunk(global_frame_idx)
        if chunk_idx is None:
            return None  # no ASD chunk here → caller may default-allow
        self._load_chunk(chunk_idx)
        # Chunk has no usable ASD pickle?  → no opinion (allow).
        spec = self.chunk_specs[chunk_idx]
        if not spec.get("asd_path"):
            return None
        tracks = self._tracks_at[global_frame_idx]
        if not tracks:
            # Chunk has ASD data but no track at this exact frame.
            # If the per-scene check said "skip" (no speaker visible in
            # this scene anyway), confirm skip. Otherwise be conservative
            # and still skip — the detected face is untracked, meaning
            # ASD didn't think it was a real face worth tracking (too
            # small / too profile / blurred). Apply lipsync to it would
            # mostly fall on listeners/extras.
            return False
        best = None
        best_iou = 0.0
        for t in tracks:
            iou = self._iou(detected_bbox, t["bbox"])
            if iou > best_iou:
                best_iou = iou
                best = t
        if best is None or best_iou < iou_match:
            # Detected face at a position no ASD track covers (in a frame
            # where other faces ARE tracked) → unknown face → skip.
            return False
        return float(best["score"]) >= self.score_threshold

    # ─── stats helpers (smoke test / report) ───
    def stats(self) -> Dict:
        # Force load all chunks then count.
        for i in range(len(self.chunk_specs)):
            self._load_chunk(i)
        n_total = self.total_frames
        n_covered = sum(1 for c in self._covered if c)
        n_with_score = sum(1 for s in self._score_at if s is not None)
        n_skip = sum(
            1 for g in range(n_total)
            if self._covered[g] and self._score_at[g] is not None
            and self._score_at[g] < self.score_threshold
        )
        return {
            "total_frames": n_total,
            "covered_frames": n_covered,
            "frames_with_score": n_with_score,
            "skip_frames": n_skip,
            "skip_pct": (100.0 * n_skip / n_total) if n_total else 0.0,
            "threshold": self.score_threshold,
        }


def _chunk_content_hash(path: str) -> str:
    """Same hash the orchestrator uses to cache ASD results."""
    import hashlib
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(0)
        head = f.read(1024 * 1024)
        f.seek(-min(1024 * 1024, size), 2)
        tail = f.read()
    return hashlib.md5(head + tail + str(size).encode()).hexdigest()[:16]


def maybe_load_filter() -> Optional[LipsyncASDFilter]:
    """Read env var, return a configured filter or None."""
    run_dir = os.environ.get("LATENTSYNC_ASD_FILTER_RUN_DIR")
    if not run_dir:
        return None
    flt = LipsyncASDFilter.from_run_dir(run_dir)
    if flt is not None:
        _log(
            f"enabled run_dir={run_dir} "
            f"chunks={len(flt.chunk_specs)} "
            f"total_frames={flt.total_frames} "
            f"threshold={flt.score_threshold}"
        )
    return flt


# ────────────────────────────────────────────────────────────────
# Speaker face profiles  (per-VIDEO PERSON identity, vs per-frame
# `LipsyncASDFilter` which is per-FRAME/per-FACE).  Built offline by
# `patches/build_face_profiles.py` from ASD pickles + pyannote diarization,
# read here at lipsync time to verify the detected face is the audio's
# speaker.
# ────────────────────────────────────────────────────────────────


def _cosine_sim(a, b):
    import numpy as _np
    a = _np.asarray(a, dtype=_np.float32)
    b = _np.asarray(b, dtype=_np.float32)
    na = float(_np.linalg.norm(a))
    nb = float(_np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float((a @ b) / (na * nb))


class SpeakerFaceProfiles:
    """Maps diarization SPEAKER_XX → reference face embedding.

    JSON schema (`speaker_face_profiles.json`):
        {
          "version": 1, "fps": 25.0, "embedding_dim": 512,
          "match_threshold": 0.5,
          "speakers": {
              "SPEAKER_01": {
                  "embedding": [...512 floats...],
                  "face_cluster_id": "C0",
                  "n_track_samples": 42,
                  "associated_track_ids": [9, 22, 25, 31],
                  "gender_hint": "male" | "female" | "unknown"
              }
          },
          "speaker_timeline": [
              {"start_sec": 0.0, "end_sec": 5.2, "speaker": "SPEAKER_01"}
          ]
        }
    """

    def __init__(self, fps: float, match_threshold: float,
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
        self.match_threshold = match_threshold
        self.speakers = speakers  # name -> {embedding, ...}
        self.timeline = sorted(timeline, key=lambda e: float(e.get("start_sec", 0)))
        # Precompute parallel arrays for O(log n) binary search by start time.
        self._tl_starts = [float(e.get("start_sec", 0)) for e in self.timeline]
        self._tl_ends = [float(e.get("end_sec", 0)) for e in self.timeline]
        self._tl_speakers = [str(e.get("speaker", "")) for e in self.timeline]
        # Pre-resolve expected embeddings as numpy arrays (avoid per-call asarray).
        import numpy as _np
        self._embedding_by_speaker = {}
        for spk_name, info in self.speakers.items():
            emb = info.get("embedding")
            if emb is not None:
                self._embedding_by_speaker[spk_name] = _np.asarray(emb, dtype=_np.float32)

    @classmethod
    def from_json(cls, path) -> Optional["SpeakerFaceProfiles"]:
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _log(f"speaker profiles load failed: {e}")
            return None
        speakers = data.get("speakers", {}) or {}
        if not speakers:
            _log("speaker profiles empty -> disabled")
            return None
        return cls(
            fps=float(data.get("fps", 25.0)),
            match_threshold=float(data.get("match_threshold", 0.5)),
            speakers=speakers,
            timeline=data.get("speaker_timeline", []) or [],
        )

    def speaker_at_frame(self, global_frame_idx: int) -> Optional[str]:
        """Return the SPEAKER_XX label whose pyannote segment covers this frame,
        or None if no segment covers it / no timeline data.

        O(log n) via bisect on sorted start times, then validate against the
        candidate segment's end time.
        """
        n = len(self._tl_starts)
        if n == 0:
            return None
        t = global_frame_idx / max(self.fps, 1e-6)
        import bisect
        # Largest i such that starts[i] <= t. bisect_right gives the
        # insertion point AFTER such i.
        idx = bisect.bisect_right(self._tl_starts, t) - 1
        if idx < 0:
            return None
        if t < self._tl_ends[idx]:
            return self._tl_speakers[idx]
        return None

    def expected_embedding(self, speaker_label: str):
        if not speaker_label:
            return None
        # Pre-resolved at init.
        return self._embedding_by_speaker.get(speaker_label)

    def gender_hint(self, speaker_label: Optional[str]) -> Optional[str]:
        if not speaker_label:
            return None
        spk = self.speakers.get(speaker_label)
        return spk.get("gender_hint") if spk else None

    def is_detected_face_correct_speaker(
        self,
        global_frame_idx: int,
        detected_embedding,
    ) -> Optional[bool]:
        """Per-PROFILE check.

        Policy (5/11 strict-on-unmapped update):
          True  -> detected face matches the expected speaker (apply lipsync)
          False -> EITHER detected face is a different person from the
                   expected speaker, OR the speaker is in the diarization
                   timeline but we have no face profile mapped for them
                   (e.g. over-split pyannote labels, short utterances with
                   no clear face track). Either way -> skip.
          None  -> only when there is NO diarization coverage at this frame
                   (genuinely no opinion). Caller may default-allow.

        Rationale: user explicitly said "wrong-face lipsync is worse than
        no-lipsync". When we don't have a profile for the current diarized
        speaker, we can't safely allow lipsync on any face.
        """
        # === PROFILE_STRICT_NONE_PATCH (5/13) ===
        import os as _os_psn
        _strict_none = _os_psn.environ.get("LATENTSYNC_PROFILE_STRICT_NONE", "0") == "1"
        _none_result = False if _strict_none else None
        # === PROFILE_STRICT_NONE_PATCH end ===
        if detected_embedding is None:
            return _none_result  # no embedding -> strict: skip, lenient: opinion
        speaker = self.speaker_at_frame(global_frame_idx)
        if speaker is None:
            return _none_result  # no diarization -> strict: skip, lenient: opinion
        target = self.expected_embedding(speaker)
        if target is None:
            # Speaker exists in diarization timeline but is NOT mapped to a
            # face cluster (e.g. pyannote over-split, no good ASD track in
            # their segments).  Without an expected embedding we cannot
            # verify the detected face matches -> conservative: skip.
            return False
        sim = _cosine_sim(detected_embedding, target)
        return sim >= self.match_threshold


def maybe_load_speaker_profiles() -> Optional[SpeakerFaceProfiles]:
    """Read env var, return loaded profiles or None."""
    p = os.environ.get("LATENTSYNC_SPEAKER_PROFILES_PATH")
    if not p:
        return None
    prof = SpeakerFaceProfiles.from_json(p)
    if prof is not None:
        _log(
            f"speaker profiles loaded: {p} "
            f"speakers={list(prof.speakers.keys())} "
            f"timeline_segs={len(prof.timeline)} "
            f"threshold={prof.match_threshold}"
        )
    return prof


# ────────────────────────────────────────────────────────────────
# Audio F0 gender lookup  (auxiliary check: face gender vs audio gender)
# ────────────────────────────────────────────────────────────────


class AudioGenderTimeline:
    """Per-frame male/female/unknown lookup from F0 estimation."""

    def __init__(self, fps: float, gender_per_frame: List[str]) -> None:
        self.fps = fps
        self.gender_per_frame = gender_per_frame

    @classmethod
    def from_json(cls, path) -> Optional["AudioGenderTimeline"]:
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _log(f"audio gender load failed: {e}")
            return None
        gpf = data.get("gender_per_frame")
        if not gpf:
            return None
        return cls(fps=float(data.get("fps", 25.0)), gender_per_frame=list(gpf))

    def gender_at_frame(self, global_frame_idx: int) -> str:
        if 0 <= global_frame_idx < len(self.gender_per_frame):
            return str(self.gender_per_frame[global_frame_idx])
        return "unknown"


def maybe_load_audio_gender() -> Optional[AudioGenderTimeline]:
    p = os.environ.get("LATENTSYNC_AUDIO_F0_GENDER_PATH")
    if not p:
        return None
    ag = AudioGenderTimeline.from_json(p)
    if ag is not None:
        _log(f"audio gender loaded: {p} (n={len(ag.gender_per_frame)})")
    return ag
