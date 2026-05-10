"""GFPGAN async I/O 후처리 — frame I/O를 GPU 작업과 병렬화.

핵심 개선:
    기존: read → enhance → write (sequential, GPU idle 시간 많음)
    개선: read N+1 (background) | enhance N (GPU) | write N-1 (background)
          → GPU 가동률 ↑, ~25-35% 시간 절감

GFPGAN.enhance() 자체는 1 frame씩 처리 (배치 미지원).
하지만 frame I/O (cv2.imread, cv2.imwrite, ffmpeg)를 GPU 작업과 겹쳐서 가속.

사용:
    /opt/venv_gfpgan/bin/python gfpgan_async_postprocess.py \\
        --input /workspace/media/output/test15_v60.mp4 \\
        --output /workspace/media/output/test15_v60_FINAL.mp4 \\
        --upscale 2
"""
import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
import tqdm


def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        os.path.join(frame_dir, "%08d.png")
    ], check=True)


def get_video_info(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,duration",
           "-of", "csv=p=0", video_path]
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


def assemble_video(frame_dir, output_path, fps, audio_source):
    temp_video = output_path + ".tmp.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "%08d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", temp_video
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", temp_video, "-i", audio_source,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path
    ], check=True)
    os.remove(temp_video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/opt/gfpgan_models/GFPGANv1.4.pth")
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=8,
                         help="Number of frames to prefetch (default: 8 — was 4)")
    parser.add_argument("--write-workers", type=int, default=4,
                         help="Number of parallel write threads (default: 4 — was 2)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[GFPGAN-async] input not found: {args.input}")
        sys.exit(1)

    fps, duration = get_video_info(args.input)
    print(f"[GFPGAN-async] input: {args.input} (fps={fps:.2f}, dur={duration:.1f}s)")
    print(f"[GFPGAN-async] prefetch: {args.prefetch} frames")

    print(f"[GFPGAN-async] loading model: {args.model}")
    restorer = GFPGANer(
        model_path=args.model,
        upscale=args.upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    print("[GFPGAN-async] model loaded")

    with tempfile.TemporaryDirectory(prefix="gfpgan_async_") as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        out_frame_dir = os.path.join(tmpdir, "out_frames")
        os.makedirs(out_frame_dir, exist_ok=True)
        print(f"[GFPGAN-async] extracting frames...")
        extract_frames(args.input, frame_dir)

        frame_files = sorted(Path(frame_dir).glob("*.png"))
        n = len(frame_files)
        print(f"[GFPGAN-async] processing {n} frames (async I/O)...")

        # ─── ASYNC I/O PIPELINE ─────────────────────────────────
        # Producer: pre-read N+prefetch frames into queue
        # Main: GFPGAN.enhance() on GPU (sequential)
        # Writer pool: cv2.imwrite in background threads
        read_queue = Queue(maxsize=args.prefetch)
        SENTINEL = (None, None, None)

        def reader():
            for i, fpath in enumerate(frame_files):
                img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
                read_queue.put((i, fpath.name, img))
            read_queue.put(SENTINEL)

        write_pool = ThreadPoolExecutor(max_workers=args.write_workers)

        kept = 0
        restored = 0
        progress = tqdm.tqdm(total=n)

        # Start reader thread
        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        # Main loop: enhance + dispatch write
        while True:
            i, name, img = read_queue.get()
            if name is None:
                break
            try:
                _, _, output = restorer.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                if output is not None and output.size > 0:
                    write_pool.submit(cv2.imwrite,
                                       os.path.join(out_frame_dir, name),
                                       output)
                    restored += 1
                else:
                    write_pool.submit(cv2.imwrite,
                                       os.path.join(out_frame_dir, name),
                                       img)
                    kept += 1
            except Exception:
                write_pool.submit(cv2.imwrite,
                                   os.path.join(out_frame_dir, name),
                                   img)
                kept += 1
            progress.update(1)

        progress.close()
        write_pool.shutdown(wait=True)
        reader_thread.join()
        print(f"[GFPGAN-async] {restored} restored, {kept} original kept")

        print("[GFPGAN-async] assembling video...")
        assemble_video(out_frame_dir, args.output, fps, args.input)
        print(f"[GFPGAN-async] output: {args.output}")


if __name__ == "__main__":
    main()
