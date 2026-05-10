"""GFPGAN 후처리: lipsync 결과의 face quality 향상.

사용:
    /opt/venv_gfpgan/bin/python gfpgan_postprocess.py \
        --input /workspace/media/output/lipsync.mp4 \
        --output /workspace/media/output/lipsync_gfpgan.mp4 \
        --upscale 1

- input: LatentSync lipsync 결과 mp4
- output: GFPGAN 후처리된 mp4
- upscale: 1 (해상도 유지) / 2 (2x upscale)

처리:
    1. ffmpeg로 frame 추출
    2. 각 frame에 GFPGAN restore_face 적용
    3. ffmpeg로 frame 재조합 + audio merge
"""
import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
import tqdm


def extract_frames(video_path: str, frame_dir: str, fps: float = None):
    """ffmpeg로 frame 추출 (PNG sequence)."""
    os.makedirs(frame_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        os.path.join(frame_dir, "%08d.png")
    ]
    subprocess.run(cmd, check=True)


def get_video_info(video_path: str):
    """ffprobe로 fps + duration 확인."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration",
        "-of", "csv=p=0",
        video_path
    ]
    out = subprocess.check_output(cmd).decode().strip()
    parts = out.split(",")
    fps_str = parts[0]
    if "/" in fps_str:
        n, d = fps_str.split("/")
        fps = float(n) / float(d)
    else:
        fps = float(fps_str)
    duration = float(parts[1]) if len(parts) > 1 else 0
    return fps, duration


def assemble_video(frame_dir: str, output_path: str, fps: float, audio_source: str):
    """ffmpeg로 frame 재조합 + audio merge."""
    temp_video = output_path + ".tmp.mp4"
    cmd_video = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "%08d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18",  # 높은 품질
        temp_video
    ]
    subprocess.run(cmd_video, check=True)
    # audio merge
    cmd_merge = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", temp_video,
        "-i", audio_source,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd_merge, check=True)
    os.remove(temp_video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/opt/gfpgan_models/GFPGANv1.4.pth")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--bg-upsampler", default="none",
                        help="none / realesrgan (배경도 SR)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[GFPGAN] input not found: {args.input}", flush=True)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[GFPGAN] model not found: {args.model}", flush=True)
        sys.exit(1)

    fps, duration = get_video_info(args.input)
    print(f"[GFPGAN] input: {args.input} (fps={fps:.2f}, dur={duration:.1f}s)")

    # === GFPGAN 모델 로드 ===
    print(f"[GFPGAN] loading model: {args.model}")
    bg_upsampler = None
    if args.bg_upsampler == "realesrgan":
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            model=bg_model,
            tile=800,            # 400→800 (boundary 자국 감소 + 속도 향상)
            tile_pad=20,         # padding도 같이 증가 (boundary 자연스러움)
            pre_pad=0,
            half=True,
        )
    restorer = GFPGANer(
        model_path=args.model,
        upscale=args.upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
    )
    print("[GFPGAN] model loaded")

    # === Frame 추출 ===
    with tempfile.TemporaryDirectory(prefix="gfpgan_") as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        out_frame_dir = os.path.join(tmpdir, "out_frames")
        os.makedirs(out_frame_dir, exist_ok=True)
        print(f"[GFPGAN] extracting frames...")
        extract_frames(args.input, frame_dir)

        # 각 frame 처리
        frame_files = sorted(Path(frame_dir).glob("*.png"))
        print(f"[GFPGAN] processing {len(frame_files)} frames...")
        kept = 0
        restored = 0
        for f in tqdm.tqdm(frame_files):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            try:
                _, _, output = restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                if output is not None and output.size > 0:
                    cv2.imwrite(str(out_frame_dir + "/" + f.name), output)
                    restored += 1
                else:
                    cv2.imwrite(str(out_frame_dir + "/" + f.name), img)
                    kept += 1
            except Exception as e:
                # face 미감지 등 → 원본 유지
                cv2.imwrite(str(out_frame_dir + "/" + f.name), img)
                kept += 1
        print(f"[GFPGAN] {restored} restored, {kept} original kept")

        # === 영상 재조합 + 오디오 merge ===
        print("[GFPGAN] assembling video...")
        assemble_video(out_frame_dir, args.output, fps, args.input)
        print(f"[GFPGAN] output: {args.output}")


if __name__ == "__main__":
    main()
