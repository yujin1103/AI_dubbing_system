"""GFPGAN async I/O 후처리 + TensorRT (StyleGAN2 forward).

기존 gfpgan_async_postprocess.py 와 동일한 async I/O 파이프라인을 유지하되,
StyleGAN2 generator forward만 TRT 엔진으로 대체한다. face detection /
alignment / paste-back / 비디오 I/O는 그대로 PyTorch + face_helper 사용.

핵심 차이:
    PyTorch:  enhance() 한 프레임당 ~22min/1920frames ≈ 688 ms
    TRT(BF16): StyleGAN forward만 ~7-9 ms (PT 대비 수십 ms 단축)

모델 vs 엔진 일관성:
    엔진 입력 dtype = FP32 (BuilderFlag.BF16 는 internal kernel만 BF16).
    GFPGANer.enhance() 는 FP32 [-1,1] CHW 텐서를 self.gfpgan(...) 으로 넘김
    → 그대로 TRT 엔진에 바인딩.

사용:
    /opt/venv_gfpgan/bin/python gfpgan_async_postprocess_trt.py \\
        --input  /workspace/media/output/test15_v60.mp4 \\
        --output /workspace/media/output/test15_v60_FINAL.mp4 \\
        --upscale 1
"""
import argparse
import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

import cv2
import numpy as np
import torch
from gfpgan import GFPGANer
import tqdm

# import the TRT wrapper that lives next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfpgan_trt_wrapper import GFPGANTRT  # noqa: E402


def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        os.path.join(frame_dir, "%08d.png"),
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
        "-crf", "18", temp_video,
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", temp_video, "-i", audio_source,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path,
    ], check=True)
    os.remove(temp_video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/opt/gfpgan_models/GFPGANv1.4.pth")
    parser.add_argument("--engine", default="/workspace/trt_work/engines/gfpgan_bf16.trt",
                        help="Path to TRT engine (BF16 recommended; FP16 has overflow).")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--write-workers", type=int, default=4)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[GFPGAN-trt] input not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.engine):
        print(f"[GFPGAN-trt] engine not found: {args.engine}")
        sys.exit(1)

    fps, duration = get_video_info(args.input)
    print(f"[GFPGAN-trt] input: {args.input} (fps={fps:.2f}, dur={duration:.1f}s)")
    print(f"[GFPGAN-trt] engine: {args.engine}")
    print(f"[GFPGAN-trt] prefetch={args.prefetch}, write_workers={args.write-workers if False else args.write_workers}")

    print(f"[GFPGAN-trt] loading GFPGANer (face_helper + PyTorch generator for warm-up)...")
    restorer = GFPGANer(
        model_path=args.model,
        upscale=args.upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    # Free the PyTorch StyleGAN2 weights and swap in TRT.
    print(f"[GFPGAN-trt] swapping PyTorch generator -> TRT engine")
    del restorer.gfpgan
    torch.cuda.empty_cache()
    restorer.gfpgan = GFPGANTRT(engine_path=args.engine)
    print("[GFPGAN-trt] TRT generator ready")

    with tempfile.TemporaryDirectory(prefix="gfpgan_trt_") as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        out_frame_dir = os.path.join(tmpdir, "out_frames")
        os.makedirs(out_frame_dir, exist_ok=True)
        print("[GFPGAN-trt] extracting frames...")
        t_extract0 = time.time()
        extract_frames(args.input, frame_dir)
        t_extract = time.time() - t_extract0

        frame_files = sorted(Path(frame_dir).glob("*.png"))
        n = len(frame_files)
        print(f"[GFPGAN-trt] processing {n} frames (async I/O + TRT)...")

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
        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        t_loop0 = time.time()
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
                                      os.path.join(out_frame_dir, name), output)
                    restored += 1
                else:
                    write_pool.submit(cv2.imwrite,
                                      os.path.join(out_frame_dir, name), img)
                    kept += 1
            except Exception as e:
                if i < 5:
                    # surface early failures so we don't silently fall back to copy on every frame
                    print(f"[GFPGAN-trt] frame {i} ({name}) enhance failed: {e}")
                write_pool.submit(cv2.imwrite,
                                  os.path.join(out_frame_dir, name), img)
                kept += 1
            progress.update(1)
        progress.close()
        write_pool.shutdown(wait=True)
        reader_thread.join()
        t_loop = time.time() - t_loop0
        print(f"[GFPGAN-trt] enhance loop: {t_loop:.1f}s ({1000*t_loop/max(n,1):.1f} ms/frame)")
        print(f"[GFPGAN-trt] {restored} restored, {kept} original kept")

        print("[GFPGAN-trt] assembling video...")
        t_asm0 = time.time()
        assemble_video(out_frame_dir, args.output, fps, args.input)
        t_asm = time.time() - t_asm0

    total = t_extract + t_loop + t_asm
    print(f"[GFPGAN-trt] timing: extract={t_extract:.1f}s, enhance={t_loop:.1f}s, assemble={t_asm:.1f}s, total={total:.1f}s")
    print(f"[GFPGAN-trt] output: {args.output}")


if __name__ == "__main__":
    main()
