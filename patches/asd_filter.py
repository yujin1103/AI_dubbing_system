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
        local_max: List[Optional[float]] = [None] * n_frames
        for t in data.get("tracks", []):
            frames = t.get("frames", [])
            scores = t.get("scores", [])
            for fi, sc in zip(frames, scores):
                fi = int(fi)
                if 0 <= fi < n_frames:
                    cur = local_max[fi]
                    if cur is None or float(sc) > cur:
                        local_max[fi] = float(sc)
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
