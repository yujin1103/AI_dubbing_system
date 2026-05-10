"""LightASD chunk runner.

기능:
  chunk video → face tracks + per-frame ASD scores 추출.

출력:
  list of {
    "frames": [int, ...],          # frame indices (0-based)
    "bboxes": [[x1,y1,x2,y2], ...], # face bbox per frame
    "scores": [float, ...],         # ASD speaking score per frame (>0 = speaking)
    "fps": float,
  }

LightASD 자체 demo는 여러 stage 거침 (scene detect → face detect → track → ASD).
우리는 결과 (tracks.pckl, scores.pckl)만 필요.
"""
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

LIGHT_ASD_DIR = "/opt/Light-ASD"
PYTHON = "/opt/venv_lipsync/bin/python"
ASD_WEIGHT = "weight/finetuning_TalkSet.model"


def run_asd(video_path: str, work_dir: Optional[str] = None,
            cleanup: bool = True) -> Optional[Dict]:
    """chunk video에 LightASD 실행. 결과 리턴.

    Args:
      video_path: 입력 chunk video (mp4)
      work_dir: 작업 디렉토리 (None이면 tempdir 사용)
      cleanup: 끝나면 work_dir 삭제

    Returns:
      {
        "fps": float,
        "n_frames": int,
        "tracks": [
          {"frames": [int,...], "bboxes": [[x1,y1,x2,y2],...], "scores": [float,...]},
          ...
        ]
      }
      또는 None (실패 시)
    """
    if not os.path.exists(video_path):
        print(f"[ASD] video not found: {video_path}")
        return None

    video_name = Path(video_path).stem
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix=f"asd_{video_name}_")

    # LightASD demo 폴더 구조:
    #   demo/<videoName>.mp4 (input)
    #   demo/<videoName>/pyavi/, pywork/, pycrop/, pyframes/ (outputs)
    demo_dir = os.path.join(work_dir, "demo")
    os.makedirs(demo_dir, exist_ok=True)

    # input copy (LightASD가 같은 폴더에서 처리)
    input_copy = os.path.join(demo_dir, f"{video_name}.mp4")
    if not os.path.exists(input_copy):
        shutil.copy(video_path, input_copy)

    # Columbia_test.py 실행
    cmd = [
        PYTHON, "Columbia_test.py",
        "--videoName", video_name,
        "--videoFolder", demo_dir,
        "--pretrainModel", ASD_WEIGHT,
    ]
    print(f"[ASD] Running on {video_name}...")
    try:
        result = subprocess.run(
            cmd, cwd=LIGHT_ASD_DIR,
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            print(f"[ASD] FAIL: {result.stderr[-500:]}")
            return None
    except subprocess.TimeoutExpired:
        print("[ASD] TIMEOUT (30min)")
        return None
    except Exception as e:
        print(f"[ASD] EXCEPTION: {e}")
        return None

    # 결과 로드
    pywork = os.path.join(demo_dir, video_name, "pywork")
    tracks_pkl = os.path.join(pywork, "tracks.pckl")
    scores_pkl = os.path.join(pywork, "scores.pckl")

    if not os.path.exists(tracks_pkl) or not os.path.exists(scores_pkl):
        print(f"[ASD] output files missing in {pywork}")
        return None

    with open(tracks_pkl, "rb") as f:
        tracks_raw = pickle.load(f)
    with open(scores_pkl, "rb") as f:
        scores_raw = pickle.load(f)

    # FPS는 LightASD 내부 가정 25fps (변환됨)
    # video.avi (25fps로 변환된)에서 frame count 측정 (cv2가 .avi 못 읽으면 fallback)
    video_25fps = os.path.join(demo_dir, video_name, "pyavi", "video.avi")
    fps = 25.0
    n_frames = 0
    try:
        import cv2
        cap = cv2.VideoCapture(video_25fps)
        fps_cv = cap.get(cv2.CAP_PROP_FPS)
        n_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps_cv > 0:
            fps = fps_cv
        if n_frames_cv > 0:
            n_frames = n_frames_cv
    except Exception:
        pass

    # ffprobe fallback (cv2가 .avi 못 읽는 경우)
    if n_frames == 0:
        try:
            import subprocess as _sp
            r = _sp.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate",
                 "-of", "default=nw=1", video_25fps],
                capture_output=True, text=True, timeout=60,
            )
            for line in r.stdout.splitlines():
                if line.startswith("nb_read_frames="):
                    n_frames = int(line.split("=")[1])
                elif line.startswith("r_frame_rate="):
                    num, den = line.split("=")[1].split("/")
                    if int(den) > 0:
                        fps = int(num) / int(den)
        except Exception:
            pass

    # track frame index에서 derive (마지막 fallback)
    if n_frames == 0:
        max_f = 0
        for t in tracks_raw:
            track = t.get("track", {})
            for f in track.get("frame", []):
                if f > max_f:
                    max_f = f
        n_frames = max_f + 1
        print(f"[ASD] WARNING: n_frames derived from tracks ({n_frames})")

    # 정리
    out_tracks = []
    for i, t in enumerate(tracks_raw):
        track = t.get("track", {})
        frames = list(track.get("frame", []))
        bboxes = [list(b) for b in track.get("bbox", [])]
        scores = list(scores_raw[i]) if i < len(scores_raw) else []
        # frames와 scores 길이가 다를 수 있음 (ASD 윈도우 크기 영향)
        # 간단히 짧은 길이로 trim
        n = min(len(frames), len(bboxes), len(scores))
        frames = frames[:n]
        bboxes = bboxes[:n]
        scores = scores[:n]
        out_tracks.append({
            "frames": frames,
            "bboxes": bboxes,
            "scores": scores,
        })

    if cleanup:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

    return {
        "fps": fps,
        "n_frames": n_frames,
        "tracks": out_tracks,
    }


def summarize_asd(asd_result: Dict) -> str:
    """ASD 결과 요약 (디버깅용)."""
    if not asd_result:
        return "[ASD] no result"
    lines = [f"FPS={asd_result['fps']:.1f}, n_frames={asd_result['n_frames']}, "
             f"tracks={len(asd_result['tracks'])}"]
    for i, t in enumerate(asd_result["tracks"]):
        if not t["frames"]:
            lines.append(f"  Track {i}: empty")
            continue
        import numpy as np
        sc = np.array(t["scores"])
        f0, f1 = t["frames"][0], t["frames"][-1]
        speaking = float((sc > 0).mean()) * 100
        lines.append(f"  Track {i}: frames {f0}~{f1} ({len(t['frames'])}), "
                     f"speaking {speaking:.1f}% (max score={sc.max():.2f})")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: asd_runner.py <video.mp4>")
        sys.exit(1)
    result = run_asd(sys.argv[1])
    if result:
        print(summarize_asd(result))
    else:
        print("ASD failed")
        sys.exit(1)
