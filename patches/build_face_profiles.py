"""Build per-speaker face profiles from ASD pickles + pyannote diarization.

Output: `runs/<run_id>/meta/speaker_face_profiles.json` consumed by
`latentsync/utils/asd_filter.py:SpeakerFaceProfiles`.

Algorithm:
  1. For each ASD track, sample N frames (default 3) — pick the highest
     ASD-score frames so embeddings come from clearest sync moments.
  2. Run insightface (recognition module) on those samples → 512-D embedding.
  3. Average per-track → one embedding per track.
  4. Cluster all track embeddings via agglomerative (cosine ≥ 0.5) →
     unique persons.
  5. For each pyannote SPEAKER_XX timeline segment, look up which ASD tracks
     are active in that segment with positive ASD score.  Tally cluster votes
     → the cluster with the most votes is that speaker's face cluster.
  6. Aggregate per-cluster embedding (mean of contributing tracks), attach
     gender hint (from genderage if available, else from F0 of associated
     diarized segments), write JSON.

Usage:
    python build_face_profiles.py \\
        --run-dir /workspace/media/runs/20260508_102603_test4_10ed0e/ \\
        [--diarization-json <path>] \\
        [--out <path>]

If `--diarization-json` omitted, runs pyannote on the dubbed.wav (slow).
For the orchestrator we'll wire this so diarization is passed in directly.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _log(msg):
    print(f"[FaceProfile] {msg}", flush=True)


def _get_face_analysis():
    """Lazy-load insightface FaceAnalysis with recognition + genderage."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106", "recognition", "genderage"],
        root="/opt/LatentSync/checkpoints/auxiliary",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(512, 512))
    return app


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float((a @ b) / (na * nb))


def _bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def _bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def sample_track_embeddings(
    app,
    video_path: str,
    track: Dict,
    n_samples: int = 10,
    bbox_iou_min: float = 0.3,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Sample frames from a track and extract face embedding + gender.

    Sampling strategy (5/11 improved coverage):
        Try frames in this priority order, deduped:
          1. Top-`n_samples` highest-ASD-score frames
          2. Top-`n_samples` largest-bbox frames (closest to camera =
             best for recognition)
          3. Uniformly-spaced frames across the track
        Return as soon as we accumulate `n_samples` valid embeddings
        (whichever combination works).

    Returns (mean_embedding[512], gender_str|None).
    """
    frames_arr = track.get("frames", [])
    scores = track.get("scores", [])
    bboxes = track.get("bboxes", [])
    if not frames_arr or not scores or not bboxes:
        return None, None
    n = len(frames_arr)
    # Build candidate ordering: top-score then top-bbox-area then uniform.
    order_score = sorted(range(n), key=lambda i: scores[i], reverse=True)
    order_area = sorted(range(n), key=lambda i: _bbox_area(bboxes[i]), reverse=True)
    order_uniform = list(range(0, n, max(1, n // max(1, n_samples * 2)))) if n > 0 else []
    seen = set()
    candidates = []
    for lst in (order_score, order_area, order_uniform):
        for i in lst:
            if i in seen:
                continue
            seen.add(i)
            candidates.append(i)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _log(f"can't open video {video_path}")
        return None, None
    embs = []
    genders = []
    tried = 0
    for idx in candidates:
        if len(embs) >= n_samples:
            break
        # Cap on tries so we don't run forever on bad tracks
        tried += 1
        if tried > max(20, n_samples * 4):
            break
        fnum = int(frames_arr[idx])
        track_bbox = bboxes[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = cap.read()
        if not ret:
            continue
        # insightface app.get expects BGR
        faces = app.get(frame)
        if not faces:
            continue
        # Pick face closest to track bbox (handles multi-face frames)
        faces.sort(key=lambda f: _bbox_iou(list(f.bbox.tolist()), track_bbox), reverse=True)
        best = faces[0]
        if _bbox_iou(list(best.bbox.tolist()), track_bbox) < bbox_iou_min:
            continue
        emb = getattr(best, "normed_embedding", None)
        if emb is None:
            continue
        embs.append(np.asarray(emb, dtype=np.float32))
        # genderage outputs: sex ('M' or 'F') + age
        sex = getattr(best, "sex", None)
        if sex == "M":
            genders.append("male")
        elif sex == "F":
            genders.append("female")
    cap.release()
    if not embs:
        return None, None
    mean_emb = np.mean(embs, axis=0)
    mean_emb /= max(np.linalg.norm(mean_emb), 1e-6)
    if not genders:
        gender = None
    else:
        gm = {"male": genders.count("male"), "female": genders.count("female")}
        gender = max(gm, key=gm.get) if max(gm.values()) > 0 else None
    return mean_emb, gender


def cluster_tracks(
    track_embeddings: List[np.ndarray],
    sim_threshold: float = 0.5,
) -> List[int]:
    """Greedy agglomerative clustering by cosine similarity.
    Returns cluster_id per input (same length as input)."""
    n = len(track_embeddings)
    if n == 0:
        return []
    cluster_ids = [-1] * n
    # Centroids (one per cluster)
    centroids: List[np.ndarray] = []
    for i, e in enumerate(track_embeddings):
        if e is None:
            cluster_ids[i] = -1
            continue
        # Find existing cluster with highest similarity
        best_c = -1
        best_sim = -1.0
        for ci, c in enumerate(centroids):
            sim = _cosine_sim(e, c)
            if sim > best_sim:
                best_sim = sim
                best_c = ci
        if best_sim >= sim_threshold and best_c >= 0:
            # Merge into that cluster (running mean approximation)
            cluster_ids[i] = best_c
            # Update centroid
            centroids[best_c] = 0.5 * (centroids[best_c] + e)
            centroids[best_c] /= max(np.linalg.norm(centroids[best_c]), 1e-6)
        else:
            cluster_ids[i] = len(centroids)
            centroids.append(e.copy())
    return cluster_ids


def load_diarization_segments(path: str, fps: float = 25.0) -> List[Dict]:
    """Load pyannote-style diarization JSON.

    Two accepted formats:
      A) {"segments": [{"start": float, "end": float, "speaker": "SPEAKER_01"}, ...]}
      B) {"segments": [{"start_sec":..., "end_sec":..., "speaker":...}]}  (our format)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments", []) or []
    out = []
    for s in segs:
        start = float(s.get("start_sec", s.get("start", 0)))
        end = float(s.get("end_sec", s.get("end", 0)))
        spk = str(s.get("speaker", s.get("label", "")))
        if end > start and spk:
            out.append({"start_sec": start, "end_sec": end, "speaker": spk})
    return out


def run_diarization_inline(audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
    """Fallback: run pyannote on the audio if no diarization.json available.

    Tries `DiariZen` daemon first (faster, more accurate), falls back to
    pyannote if available. If neither works, returns empty list.

    `num_speakers`: if known, hint to constrain clustering (avoids over-split).
    """
    import urllib.request, urllib.error, json as _json
    # Try DiariZen worker subprocess first (matches orchestrator)
    worker = "/workspace/scripts/diarize_worker_diarizen.py"
    diarizen_python = "/opt/venv_diarizen/bin/python"
    if os.path.exists(worker) and os.path.exists(diarizen_python):
        import subprocess
        _log(f"running DiariZen on {audio_path} (may take 30-60s)"
             + (f" with num_speakers={num_speakers}" if num_speakers else ""))
        try:
            cmd = [diarizen_python, worker, audio_path]
            if num_speakers:
                cmd += ["--num-speakers", str(num_speakers)]
            res = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600,
            )
            if res.returncode == 0:
                # The worker emits progress logs first, then the JSON dict
                # on the last line. Find that JSON line robustly.
                data = None
                # Try last-non-empty-line first
                for line in reversed(res.stdout.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        try:
                            data = _json.loads(line)
                            break
                        except Exception:
                            continue
                if data is None:
                    # last-ditch: search any { ... } block
                    import re as _re
                    m = _re.search(r'\{"segments"\s*:.*\}\s*$', res.stdout, _re.S)
                    if m:
                        try:
                            data = _json.loads(m.group(0))
                        except Exception:
                            pass
                if data is None:
                    _log(f"DiariZen stdout parse failed (last 200 chars): {res.stdout[-200:]!r}")
                    return []
                segs = []
                for s in data.get("segments", []):
                    segs.append({
                        "start_sec": float(s["start"]),
                        "end_sec": float(s["end"]),
                        "speaker": str(s["speaker"]),
                    })
                if segs:
                    _log(f"DiariZen done: {len(segs)} segments, "
                         f"n_speakers={data.get('n_speakers', '?')}")
                    return segs
            else:
                _log(f"DiariZen rc={res.returncode}, stderr (last 200): {res.stderr[-200:]!r}")
        except Exception as e:
            _log(f"DiariZen exception: {e}")
    _log("no fallback diarization available")
    return []


def map_speakers_to_clusters(
    diarization: List[Dict],
    tracks: List[Dict],
    cluster_ids: List[int],
    track_score_threshold: float = 0.0,
    fps: float = 25.0,
) -> Dict[str, Dict]:
    """For each SPEAKER_XX, find which face cluster has the most votes.

    Vote rule: for each (track t, frame f) where t is active at f with
    ASD score > threshold, vote for cluster of t if f is within any
    SPEAKER_X segment.
    """
    n_clusters = max(cluster_ids, default=-1) + 1
    spk_votes: Dict[str, np.ndarray] = {}  # spk -> [n_clusters]
    spk_track_assoc: Dict[str, set] = {}   # spk -> {track_idx}
    for t_idx, track in enumerate(tracks):
        cid = cluster_ids[t_idx]
        if cid < 0:
            continue
        frames = track.get("frames", [])
        scores = track.get("scores", [])
        for fnum, sc in zip(frames, scores):
            if sc <= track_score_threshold:
                continue
            t_sec = float(fnum) / max(fps, 1e-6)
            # Find which diarization segment covers this time
            for seg in diarization:
                if seg["start_sec"] <= t_sec < seg["end_sec"]:
                    spk = seg["speaker"]
                    if spk not in spk_votes:
                        spk_votes[spk] = np.zeros(max(1, n_clusters), dtype=np.int32)
                        spk_track_assoc[spk] = set()
                    if cid < n_clusters:
                        spk_votes[spk][cid] += 1
                    spk_track_assoc[spk].add(t_idx)
                    break
    mapping = {}
    for spk, votes in spk_votes.items():
        if votes.sum() == 0:
            continue
        best_cluster = int(np.argmax(votes))
        mapping[spk] = {
            "face_cluster_id": f"C{best_cluster}",
            "_face_cluster_idx": best_cluster,
            "vote_total": int(votes.sum()),
            "vote_best": int(votes[best_cluster]),
            "associated_track_ids": sorted(list(spk_track_assoc[spk])),
        }
    return mapping


def build_profiles(
    run_dir: str,
    diarization_json: Optional[str] = None,
    out_path: Optional[str] = None,
    n_samples_per_track: int = 10,
    cluster_threshold: float = 0.4,
    match_threshold: float = 0.5,
    num_speakers_hint: Optional[int] = None,
) -> Optional[str]:
    rd = Path(run_dir)
    chunks_dir = rd / "chunks"
    if not chunks_dir.is_dir():
        _log(f"no chunks/ dir at {rd}")
        return None
    # Load ASD index (we already have one from previous run)
    idx_path = rd / "meta" / "asd_filter_index.json"
    if not idx_path.is_file():
        _log(f"no asd_filter_index.json at {idx_path}")
        return None
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    fps = float(idx.get("fps", 25.0))
    chunks = idx.get("chunks", [])
    if not chunks:
        _log("no chunks in index")
        return None

    # Load diarization
    if diarization_json and os.path.isfile(diarization_json):
        diarization = load_diarization_segments(diarization_json, fps=fps)
        _log(f"diarization loaded: {len(diarization)} segments from {diarization_json}")
    else:
        # Prefer ORIGINAL (English) vocals over dubbed — original actors have
        # distinctive voices, dubbed TTS voices can sound similar to each
        # other and confuse diarization.
        vocals_dir = rd / "vocals"
        candidates = []
        if vocals_dir.exists():
            # clean_vocals (BSRoformer) > vocals (demucs) > anything *_vocals.wav
            candidates += sorted(vocals_dir.glob("*_clean_vocals.wav"))
            candidates += sorted(vocals_dir.glob("*_vocals.wav"))
        if not candidates:
            dubbed_dir = rd / "dubbed"
            if dubbed_dir.exists():
                candidates += sorted(dubbed_dir.glob("*_dubbed.wav"))
                _log("WARNING: no original vocals found, falling back to DUBBED audio "
                     "(may give worse diarization due to similar TTS voices)")
        if candidates:
            chosen = candidates[0]
            _log(f"running diarization on {chosen}")
            diarization = run_diarization_inline(str(chosen), num_speakers=num_speakers_hint)
        else:
            diarization = []
    if not diarization:
        _log("no diarization data - cannot map speakers to faces. abort.")
        return None

    _log("initializing insightface (recognition + genderage)...")
    app = _get_face_analysis()

    # Step 1: per-track embeddings
    all_tracks: List[Dict] = []
    track_chunk_offsets: List[int] = []  # for translating local→global frame
    track_video_path: List[str] = []
    chunk_global_offset = 0
    for ch in chunks:
        asd_path = ch.get("asd_path")
        n_frames = int(ch.get("n_frames", 0))
        stem = ch.get("stem")
        # find the chunk video file
        ch_video = chunks_dir / f"{stem}.mp4"
        if not ch_video.is_file():
            ch_video = chunks_dir / f"{stem}_final.mp4"
        if not asd_path or not os.path.isfile(asd_path) or not ch_video.is_file():
            chunk_global_offset += n_frames
            continue
        with open(asd_path, "rb") as f:
            chdata = pickle.load(f)
        for t in chdata.get("tracks", []):
            all_tracks.append(t)
            track_chunk_offsets.append(chunk_global_offset)
            track_video_path.append(str(ch_video))
        chunk_global_offset += n_frames

    _log(f"loaded {len(all_tracks)} tracks across {len(chunks)} chunks")
    if not all_tracks:
        return None

    # Step 2-3: embeddings
    t0 = time.time()
    track_embs: List[Optional[np.ndarray]] = [None] * len(all_tracks)
    track_genders: List[Optional[str]] = [None] * len(all_tracks)
    for i, (t, vp) in enumerate(zip(all_tracks, track_video_path)):
        emb, gender = sample_track_embeddings(app, vp, t, n_samples=n_samples_per_track)
        track_embs[i] = emb
        track_genders[i] = gender
    _log(f"embeddings: {sum(1 for e in track_embs if e is not None)}/{len(track_embs)} ok "
         f"({time.time()-t0:.1f}s)")

    # Step 4: cluster
    valid_idx = [i for i, e in enumerate(track_embs) if e is not None]
    valid_embs = [track_embs[i] for i in valid_idx]
    valid_cluster_ids = cluster_tracks(valid_embs, sim_threshold=cluster_threshold)
    cluster_ids_full: List[int] = [-1] * len(all_tracks)
    for vi, full_i in enumerate(valid_idx):
        cluster_ids_full[full_i] = valid_cluster_ids[vi]
    n_clusters = max(cluster_ids_full, default=-1) + 1
    _log(f"clustering: {n_clusters} unique persons identified")

    # Translate per-chunk frame numbers to global for tracks (so map_speakers
    # uses global frames). Build a fake global-frame array.
    global_tracks: List[Dict] = []
    for t, offset in zip(all_tracks, track_chunk_offsets):
        global_tracks.append({
            "frames": [int(f) + offset for f in t.get("frames", [])],
            "scores": t.get("scores", []),
            "bboxes": t.get("bboxes", []),
        })

    # Step 5: SPEAKER_XX → cluster mapping
    spk_map = map_speakers_to_clusters(
        diarization, global_tracks, cluster_ids_full,
        track_score_threshold=0.0, fps=fps,
    )
    _log(f"speaker mapping: {len(spk_map)} speakers mapped to clusters")
    for spk, info in spk_map.items():
        _log(f"  {spk} -> {info['face_cluster_id']} "
             f"(votes={info['vote_best']}/{info['vote_total']}, "
             f"tracks={len(info['associated_track_ids'])})")

    # Step 6: aggregate per-speaker embedding (mean of constituent tracks)
    speakers_out = {}
    for spk, info in spk_map.items():
        c_idx = info["_face_cluster_idx"]
        member_tracks = [i for i in valid_idx if cluster_ids_full[i] == c_idx]
        if not member_tracks:
            continue
        member_embs = np.stack([track_embs[i] for i in member_tracks])
        mean_emb = member_embs.mean(axis=0)
        mean_emb /= max(np.linalg.norm(mean_emb), 1e-6)
        # Gender vote
        ms_genders = [track_genders[i] for i in member_tracks if track_genders[i]]
        if ms_genders:
            counts = {"male": ms_genders.count("male"), "female": ms_genders.count("female")}
            best_g = max(counts, key=counts.get)
            gender_hint = best_g if counts[best_g] > 0 else "unknown"
        else:
            gender_hint = "unknown"
        speakers_out[spk] = {
            "face_cluster_id": info["face_cluster_id"],
            "embedding": [float(x) for x in mean_emb.tolist()],
            "n_track_samples": len(member_tracks),
            "associated_track_ids": info["associated_track_ids"],
            "gender_hint": gender_hint,
            "vote_best": info["vote_best"],
            "vote_total": info["vote_total"],
        }

    profile = {
        "version": 1,
        "fps": fps,
        "embedding_dim": 512,
        "match_threshold": match_threshold,
        "cluster_threshold": cluster_threshold,
        "speakers": speakers_out,
        "speaker_timeline": diarization,
        "stats": {
            "n_tracks": len(all_tracks),
            "n_clusters": n_clusters,
            "n_speakers_mapped": len(speakers_out),
        },
    }

    if out_path is None:
        out_dir = rd / "meta"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "speaker_face_profiles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    _log(f"wrote {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--diarization-json", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--cluster-threshold", type=float, default=0.4)
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Hint for DiariZen num speakers when known (drops over-split risk)")
    args = p.parse_args()
    out = build_profiles(
        run_dir=args.run_dir,
        diarization_json=args.diarization_json,
        out_path=args.out,
        n_samples_per_track=args.n_samples,
        cluster_threshold=args.cluster_threshold,
        match_threshold=args.match_threshold,
        num_speakers_hint=args.num_speakers,
    )
    if out is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
