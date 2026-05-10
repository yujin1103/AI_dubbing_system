"""tracks.pckl 자세한 구조 분석 (face track별 bbox + frame index)."""
import pickle
import numpy as np
from pathlib import Path

p = Path("/opt/Light-ASD/demo/test3/pywork")

with open(p / "tracks.pckl", "rb") as fh:
    tracks = pickle.load(fh)

with open(p / "scores.pckl", "rb") as fh:
    scores = pickle.load(fh)

print(f"전체 face track 수: {len(tracks)}")
print()
for i, t in enumerate(tracks):
    print(f"=== Track {i} ===")
    print(f"keys: {list(t.keys())}")
    track = t["track"]
    proc = t["proc_track"]
    print(f"  track keys: {list(track.keys())}")
    print(f"  proc_track keys: {list(proc.keys())}")

    # frame indices
    if "frame" in track:
        frames = track["frame"]
        print(f"  frames: shape={np.array(frames).shape}, range={frames[0]}~{frames[-1]} ({len(frames)} frames)")

    # bbox info (s = scale, x, y)
    if "bbox" in track:
        bbox = np.array(track["bbox"])
        print(f"  bbox shape: {bbox.shape}")
        print(f"  bbox sample (first 3): {bbox[:3]}")

    # score
    s = np.array(scores[i])
    print(f"  scores: shape={s.shape}, mean={s.mean():.2f}, max={s.max():.2f}, "
          f"speaking ratio (>0): {(s > 0).mean()*100:.1f}%")

    print()
