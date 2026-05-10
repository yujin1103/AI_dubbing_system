"""LightASD 출력 구조 분석."""
import pickle
import numpy as np
from pathlib import Path

p = Path("/opt/Light-ASD/demo/test3/pywork")

for f in ["scores.pckl", "tracks.pckl", "faces.pckl", "scene.pckl"]:
    fp = p / f
    if not fp.exists():
        print(f"{f}: NOT FOUND")
        continue
    with open(fp, "rb") as fh:
        data = pickle.load(fh)
    print(f"\n=== {f} ===")
    print(f"type: {type(data).__name__}")
    if isinstance(data, list):
        print(f"length: {len(data)}")
        if len(data) > 0:
            print(f"first element type: {type(data[0]).__name__}")
            if isinstance(data[0], dict):
                print(f"first element keys: {list(data[0].keys())}")
                for k, v in data[0].items():
                    if isinstance(v, np.ndarray):
                        print(f"  {k}: array shape={v.shape} dtype={v.dtype}")
                    elif isinstance(v, list):
                        print(f"  {k}: list len={len(v)}")
                    else:
                        print(f"  {k}: {type(v).__name__}={v if isinstance(v,(int,float,str)) else '...'}")
            elif isinstance(data[0], (np.ndarray, list)):
                print(f"first element shape/len: {len(data[0]) if hasattr(data[0],'__len__') else '?'}")
                if isinstance(data[0], np.ndarray):
                    print(f"  shape={data[0].shape} dtype={data[0].dtype}")
                    print(f"  sample values: {data[0][:5]}")
    elif isinstance(data, dict):
        print(f"keys: {list(data.keys())}")

# scores 자세히
print("\n\n=== scores 분석 ===")
with open(p / "scores.pckl", "rb") as fh:
    scores = pickle.load(fh)
print(f"전체 face track 수: {len(scores)}")
for i, track_scores in enumerate(scores[:3]):
    arr = np.array(track_scores)
    print(f"track {i}: {len(arr)} frames, "
          f"mean={arr.mean():.2f}, max={arr.max():.2f}, "
          f"speaking ratio (>0.5): {(arr > 0.5).mean()*100:.1f}%")
