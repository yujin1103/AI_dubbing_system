"""
AIHub 립리딩 데이터 전처리 — center-crop 방식 (단순/빠름).

INPUT: 1920x1080 30fps mp4 + 44.1kHz stereo audio
OUTPUT: 256x256 25fps mp4 + 16kHz mono audio (LatentSync 학습용)

Center crop 전략:
  - 모든 영상이 동일 셋업 (정면 + 카메라 고정 + 얼굴 중앙) 이므로
    1920x1080 가운데 1080x1080 crop만으로 얼굴 영역 충분히 커버
  - face_alignment 추론 불필요 → 60 영상 5분 안에 처리

JSON 라벨 활용:
  - Audio_env.Noise: 환경 정보 (참고용)
  - Sentence_info: 발화 timestamp (필요 시 미래 사용)
  - Face_bounding_box: 좌표계 불일치로 무시 (portrait/landscape 혼재)

사용:
  /opt/venv_lipsync/bin/python /workspace/_aihub_face_crop.py \
    --json_root /workspace/media/aihub_extracted/labels_train \
    --video_root /workspace/media/aihub_extracted/video_train \
    --out_dir /workspace/media/aihub_processed/train \
    --max_videos 60
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def collect_videos(json_root: str, video_root: str,
                   filter_angle: str = "A",
                   filter_noise: Optional[int] = None) -> list:
    """JSON ↔ mp4 페어 수집. Angle 필터 적용."""
    pairs = []
    json_files = list(Path(json_root).rglob("*.json"))
    print(f"[Collect] JSON 파일 {len(json_files)}개 발견")

    for jp in json_files:
        try:
            with open(jp, encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, list):
                meta = meta[0]

            angle = meta.get("Video_env", {}).get("Angle") or meta.get("video_env", {}).get("angle", "")
            noise = meta.get("Audio_env", {}).get("Noise", -1)
            if filter_angle and angle != filter_angle:
                continue
            if filter_noise is not None and noise != filter_noise:
                continue

            mp4_name = (meta.get("Video_info", {}).get("video_Name")
                       or meta.get("video_info", {}).get("video_name")
                       or jp.stem + ".mp4")

            # video_root에서 같은 basename mp4 찾기
            cands = list(Path(video_root).rglob(mp4_name))
            if not cands:
                continue

            pairs.append({
                "json": jp,
                "mp4": cands[0],
                "angle": angle,
                "noise": noise,
            })
        except Exception:
            continue

    print(f"[Collect] 필터 통과 페어 {len(pairs)}개 (angle={filter_angle}, noise={filter_noise})")
    return pairs


def preprocess_one(mp4_in: Path, mp4_out: Path,
                   target_size: int = 256,
                   target_fps: int = 25,
                   target_sr: int = 16000) -> bool:
    """ffmpeg 한 번에: center crop → resize → fps/sr 변환."""
    # 1. 영상 해상도 확인
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
         "-select_streams", "v:0", "-of", "default=noprint_wrappers=1:nokey=1",
         str(mp4_in)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return False
    try:
        w, h = [int(x) for x in probe.stdout.strip().split()[:2]]
    except ValueError:
        return False

    # 2. center square crop (짧은 변 기준)
    side = min(w, h)
    crop_x = (w - side) // 2
    crop_y = (h - side) // 2

    vf = (f"crop={side}:{side}:{crop_x}:{crop_y},"
          f"scale={target_size}:{target_size}:flags=lanczos,"
          f"fps={target_fps}")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp4_in),
        "-vf", vf,
        "-ar", str(target_sr), "-ac", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        str(mp4_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_root", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--filter_angle", default="A")
    parser.add_argument("--filter_noise", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fileslist = out_dir.parent / f"fileslist_{out_dir.name}.txt"

    pairs = collect_videos(args.json_root, args.video_root,
                           filter_angle=args.filter_angle,
                           filter_noise=args.filter_noise)
    if args.max_videos:
        pairs = pairs[:args.max_videos]
    if not pairs:
        print("[ERROR] 처리할 페어 없음")
        return 1

    success = []
    t0 = time.time()
    for i, p in enumerate(pairs):
        out_path = out_dir / (p["mp4"].stem + ".mp4")
        if out_path.exists() and out_path.stat().st_size > 1024:
            success.append(str(out_path))
            continue

        if preprocess_one(p["mp4"], out_path):
            success.append(str(out_path))
            elapsed = time.time() - t0
            eta = (len(pairs) - i - 1) * elapsed / (i + 1)
            print(f"[{i+1}/{len(pairs)}] ✅ {out_path.name} (ETA {eta:.0f}s)",
                  flush=True)
        else:
            print(f"[{i+1}/{len(pairs)}] ❌ {out_path.name}")

    with open(fileslist, "w", encoding="utf-8") as f:
        for path in success:
            f.write(f"{path}\n")

    elapsed = time.time() - t0
    print(f"\n=== 완료 ===")
    print(f"성공: {len(success)}/{len(pairs)} 영상")
    print(f"시간: {elapsed:.0f}초 ({elapsed/max(1,len(success)):.1f}초/영상)")
    print(f"fileslist: {fileslist} ({len(success)} 라인)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
