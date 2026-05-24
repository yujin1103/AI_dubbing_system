"""Phase 2: Post-lipsync validation (lightweight motion + StableSyncNet sim).

The original SyncNet checkpoint (`stable_syncnet.pt`) uses a different
architecture (`audio_encoder` + `visual_encoder`) than the joonson SyncNet
in `eval/syncnet`, so we can't directly call `SyncNetEval.evaluate()`.

Instead we provide two metrics that work without external dependencies:

  1) lip_motion_score
       mean abs-diff in the lower-third (mouth area) of consecutive frames.
       Higher = more lip motion. Expected ~0.5-3.0 for active speech.
       <0.5 = static (likely no lipsync applied or silent video).

  2) face_pixel_changed_frac
       fraction of pixels in the face region that differ from the original
       (pre-lipsync) video.  Useful when paired with --original to confirm
       lipsync actually modified the face region.

Usage:
  python syncnet_validate.py video1.mp4 [video2.mp4 ...] \\
         [--original /path/to/pre_lipsync.mp4]
"""
from __future__ import annotations
import os
import sys
import time
import argparse

import numpy as np
import cv2


def analyze_video(video_path: str, original_path: str | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap_o = None
    if original_path and os.path.isfile(original_path):
        cap_o = cv2.VideoCapture(original_path)

    prev_lower = None
    lip_motion = []
    pixel_changed_frac = []

    sampled = 0
    max_sample = min(n_frames, 500)  # cap at 500 frames for speed
    step = max(1, n_frames // max_sample)

    for fi in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # lower-third = mouth region (rough heuristic)
        lower = gray[int(h * 0.55):, :]
        if prev_lower is not None and lower.shape == prev_lower.shape:
            lip_motion.append(float(np.mean(np.abs(lower - prev_lower)) * 100.0))
        prev_lower = lower

        if cap_o is not None:
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok_o, frame_o = cap_o.read()
            if ok_o and frame_o is not None:
                # Center crop face region (rough estimate, middle 50%)
                cy0, cy1 = int(h * 0.25), int(h * 0.85)
                cx0, cx1 = int(w * 0.3), int(w * 0.7)
                a = frame[cy0:cy1, cx0:cx1].astype(np.float32)
                if frame_o.shape[:2] != frame.shape[:2]:
                    frame_o = cv2.resize(frame_o, (w, h))
                b = frame_o[cy0:cy1, cx0:cx1].astype(np.float32)
                if a.shape == b.shape:
                    diff = np.abs(a - b)
                    pixel_changed_frac.append(float((diff.mean(-1) > 5).mean() * 100))

        sampled += 1

    cap.release()
    if cap_o is not None:
        cap_o.release()

    return {
        "frames": n_frames,
        "fps": fps,
        "sampled": sampled,
        "lip_motion_score": float(np.mean(lip_motion)) if lip_motion else 0.0,
        "lip_motion_max": float(np.max(lip_motion)) if lip_motion else 0.0,
        "face_changed_pct": float(np.mean(pixel_changed_frac)) if pixel_changed_frac else None,
    }


def rate_lipsync(stats):
    """Heuristic rating based on lip motion + face change."""
    if stats is None:
        return "ERROR"
    score = stats["lip_motion_score"]
    if stats["face_changed_pct"] is not None:
        if stats["face_changed_pct"] < 5:
            return "NO_CHANGE (lipsync may not have applied)"
    if score < 0.3:
        return "STATIC (low lip motion)"
    elif score < 0.8:
        return "OK (some motion)"
    elif score < 2.0:
        return "GOOD (active speech)"
    else:
        return "VERY_ACTIVE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--original", default=None,
                    help="Pre-lipsync video for face-diff comparison")
    args = ap.parse_args()

    print(f"{'video':<48} | {'lip_motion':>10} | {'face_chg%':>9} | {'rating'}")
    print("-" * 95)
    for v in args.videos:
        t0 = time.time()
        stats = analyze_video(v, args.original)
        if stats is None:
            print(f"{os.path.basename(v):<48} | FAILED to read")
            continue
        fc = f"{stats['face_changed_pct']:.1f}" if stats["face_changed_pct"] is not None else "—"
        rating = rate_lipsync(stats)
        elapsed = time.time() - t0
        print(f"{os.path.basename(v):<48} | "
              f"{stats['lip_motion_score']:>10.2f} | "
              f"{fc:>9} | {rating} ({elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
