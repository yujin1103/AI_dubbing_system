"""GFPGAN async post-process v3 — safe combo + optional downscale-detect.

v2 (gfpgan_async_postprocess_trt_v2.py) 에서 추가:
  A) RetinaFace BF16 엔진 자동 사용 (retinaface_trt_wrapper.py 가 BF16 우선)
  D) NMS GPU 패치 (torchvision.ops.nms) 시작 시 자동 적용
  F) --downscale-detect N: face detection 만 N배 작은 해상도로 (default 1=원본)
     - 1080p → 540p 검출 후 좌표 ×2 scale-up → 원본 해상도에서 crop+paste
     - 품질은 1080p detection 대비 lipsync 영역 거의 동일, frame 디테일 무관

사용:
    # Phase 1: safe combo (A+D)
    /opt/venv_gfpgan/bin/python gfpgan_async_postprocess_trt_v3.py \
        --input  ... --output ... --upscale 1 --retinaface-trt

    # Phase 2: + 540p detect downscale (F)
    /opt/venv_gfpgan/bin/python gfpgan_async_postprocess_trt_v3.py \
        --input  ... --output ... --upscale 1 --retinaface-trt \
        --downscale-detect 2
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import torch
import tqdm
from gfpgan import GFPGANer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfpgan_trt_wrapper import GFPGANTRT       # noqa: E402
from gpu_face_aligner import GPUFaceAligner    # noqa: E402

try:
    from retinaface_trt_wrapper import wrap_face_helper_detector
    HAS_RETINAFACE_TRT = True
except Exception:
    HAS_RETINAFACE_TRT = False

# D) NMS GPU patch — 시작 시 자동 적용 (zero quality risk, ~10s/run 절감)
try:
    from retinaface_postprocess_gpu import patch_postprocess as _patch_nms_gpu
    _patch_nms_gpu(None)  # 함수 안의 face_det 인자 사용 안 함, 그냥 global swap
    HAS_NMS_GPU = True
except Exception as e:
    print(f"[v3] NMS GPU patch failed: {e}", file=sys.stderr)
    HAS_NMS_GPU = False


# ---------------------------------------------------------------------------
# ffmpeg helpers (v2 와 동일)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-frame GPU enhance (with optional downscale-detect)
# ---------------------------------------------------------------------------

@torch.no_grad()
def enhance_frame_gpu(img_bgr_u8: np.ndarray,
                      face_det,
                      gfpgan_trt: GFPGANTRT,
                      aligner: GPUFaceAligner,
                      device: torch.device,
                      eye_dist_threshold: float = 5.0,
                      conf_threshold: float = 0.97,
                      downscale_detect: int = 1,
                      timing: dict = None) -> np.ndarray:
    """입력 BGR uint8 -> GFPGAN 처리된 BGR uint8.

    downscale_detect>1: detection 만 1/N 해상도 입력 사용, 결과 좌표는 ×N 스케일.
    crop+paste 는 원본 해상도 그대로.
    """
    H, W = img_bgr_u8.shape[:2]

    # ---- 1) detect (numpy in -> numpy out) ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()

    if downscale_detect > 1:
        # F) detect 만 다운스케일된 입력으로
        dh, dw = H // downscale_detect, W // downscale_detect
        det_img = cv2.resize(img_bgr_u8, (dw, dh), interpolation=cv2.INTER_AREA)
        bboxes_landm = face_det.detect_faces(det_img, conf_threshold)
        # 좌표 scale up
        if len(bboxes_landm) > 0:
            bboxes_landm = np.array(bboxes_landm, dtype=np.float32).copy()
            # bbox (cols 0-3) + landmarks (cols 5-14)
            for c in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                if c < bboxes_landm.shape[1]:
                    bboxes_landm[:, c] *= downscale_detect
            bboxes_landm = bboxes_landm.tolist()
    else:
        bboxes_landm = face_det.detect_faces(img_bgr_u8, conf_threshold)

    if timing is not None:
        torch.cuda.synchronize()
        timing['detect'] += time.time() - t

    # filter by eye distance
    landmarks_list = []
    for bbox in bboxes_landm:
        eye_dist = np.linalg.norm([bbox[5] - bbox[7], bbox[6] - bbox[8]])
        if eye_dist < eye_dist_threshold:
            continue
        landmarks_list.append(np.array(
            [[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)], dtype=np.float32))

    if not landmarks_list:
        return img_bgr_u8

    # ---- 2) upload ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    img_t = torch.from_numpy(img_bgr_u8).to(device, non_blocking=True)
    img_chw_bgr_01 = img_t.permute(2, 0, 1).contiguous().to(torch.float32).mul_(1.0 / 255.0)
    if timing is not None:
        torch.cuda.synchronize()
        timing['upload'] += time.time() - t

    # ---- 3) GPU align ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    faces_rgb_norm, affines = aligner.align(img_chw_bgr_01, landmarks_list)
    if timing is not None:
        torch.cuda.synchronize()
        timing['align'] += time.time() - t

    # ---- 4) GFPGAN TRT ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    N = faces_rgb_norm.shape[0]
    restored_list = []
    for i in range(N):
        out = gfpgan_trt(faces_rgb_norm[i:i+1].float())[0]
        restored_list.append(out)
    restored = torch.cat(restored_list, dim=0) if N > 1 else restored_list[0]
    if timing is not None:
        torch.cuda.synchronize()
        timing['gfpgan'] += time.time() - t

    # ---- 5) GPU paste back ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    out_chw = aligner.paste(img_chw_bgr_01, restored, affines,
                            parse_input_faces_rgb_norm=restored)
    if timing is not None:
        torch.cuda.synchronize()
        timing['paste'] += time.time() - t

    # ---- 6) download ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    out_u8 = (out_chw.permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8)
    out_np = out_u8.cpu().numpy()
    if timing is not None:
        torch.cuda.synchronize()
        timing['download'] += time.time() - t

    return out_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/opt/gfpgan_models/GFPGANv1.4.pth")
    parser.add_argument("--engine", default="/workspace/trt_work/engines/gfpgan_bf16.trt")
    parser.add_argument("--retinaface-engine-dir",
                        default="/workspace/trt_work/engines")
    parser.add_argument("--retinaface-trt", action="store_true",
                        help="Use BF16 RetinaFace TRT (auto-fallback to FP16)")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--detail-timing", action="store_true")
    parser.add_argument("--downscale-detect", type=int, default=1,
                        help="(F) Reduce detection input by this factor (1=original, 2=540p for 1080p)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[v3] input not found: {args.input}"); sys.exit(1)
    if not os.path.isfile(args.engine):
        print(f"[v3] engine not found: {args.engine}"); sys.exit(1)

    fps, duration = get_video_info(args.input)
    print(f"[v3] input: {args.input} (fps={fps:.2f}, dur={duration:.1f}s)")
    print(f"[v3] gfpgan engine: {args.engine}")
    print(f"[v3] NMS GPU patch: {'enabled' if HAS_NMS_GPU else 'disabled'}")
    print(f"[v3] downscale_detect: {args.downscale_detect}x")

    restorer = GFPGANer(
        model_path=args.model, upscale=args.upscale, arch="clean",
        channel_multiplier=2, bg_upsampler=None,
    )
    del restorer.gfpgan
    torch.cuda.empty_cache()
    gfpgan_trt = GFPGANTRT(engine_path=args.engine)
    print("[v3] GFPGAN TRT ready")

    if args.retinaface_trt and HAS_RETINAFACE_TRT:
        try:
            wrap_face_helper_detector(
                restorer.face_helper, args.retinaface_engine_dir, preload=["fhd"],
            )
            print("[v3] RetinaFace TRT ready (BF16 preferred)")
        except Exception as e:
            print(f"[v3] WARN: RetinaFace TRT swap failed ({e}); using PyTorch detector")
    else:
        print("[v3] RetinaFace: PyTorch detector")

    device = torch.device("cuda")
    aligner = GPUFaceAligner(
        device=device, face_template=restorer.face_helper.face_template,
        face_size=restorer.face_helper.face_size[0], use_parse=True,
        face_parse_module=restorer.face_helper.face_parse,
    )
    aligner.face_parse.eval()
    print("[v3] GPUFaceAligner ready")

    with tempfile.TemporaryDirectory(prefix="gfpgan_trt_v3_") as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        out_frame_dir = os.path.join(tmpdir, "out_frames")
        os.makedirs(out_frame_dir, exist_ok=True)
        print("[v3] extracting frames...")
        t_extract0 = time.time()
        extract_frames(args.input, frame_dir)
        t_extract = time.time() - t_extract0

        frame_files = sorted(Path(frame_dir).glob("*.png"))
        if args.limit > 0:
            frame_files = frame_files[:args.limit]
        n = len(frame_files)
        print(f"[v3] processing {n} frames...")

        read_queue: Queue = Queue(maxsize=args.prefetch)
        SENTINEL = (None, None, None)

        def reader():
            for i, fpath in enumerate(frame_files):
                img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
                read_queue.put((i, fpath.name, img))
            read_queue.put(SENTINEL)

        write_pool = ThreadPoolExecutor(max_workers=args.write_workers)

        timing = {
            'detect': 0.0, 'upload': 0.0, 'align': 0.0, 'gfpgan': 0.0,
            'paste': 0.0, 'download': 0.0,
        } if args.detail_timing else None

        kept = 0
        restored = 0
        progress = tqdm.tqdm(total=n)
        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        # warm-up
        if n > 0:
            warm_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
            try:
                _ = enhance_frame_gpu(
                    warm_img, restorer.face_helper.face_det,
                    gfpgan_trt, aligner, device,
                    downscale_detect=args.downscale_detect, timing=None,
                )
                torch.cuda.synchronize()
                print("[v3] warm-up done")
            except Exception as e:
                print(f"[v3] WARN: warm-up failed: {e}")

        t_loop0 = time.time()
        while True:
            i, name, img = read_queue.get()
            if name is None:
                break
            try:
                out = enhance_frame_gpu(
                    img, restorer.face_helper.face_det,
                    gfpgan_trt, aligner, device,
                    downscale_detect=args.downscale_detect, timing=timing,
                )
                if out is img:
                    write_pool.submit(cv2.imwrite,
                                      os.path.join(out_frame_dir, name), img)
                    kept += 1
                else:
                    write_pool.submit(cv2.imwrite,
                                      os.path.join(out_frame_dir, name), out)
                    restored += 1
            except Exception as e:
                if i < 5:
                    print(f"[v3] frame {i} ({name}) failed: {e}")
                write_pool.submit(cv2.imwrite,
                                  os.path.join(out_frame_dir, name), img)
                kept += 1
            progress.update(1)
        progress.close()
        write_pool.shutdown(wait=True)
        reader_thread.join()
        torch.cuda.synchronize()
        t_loop = time.time() - t_loop0
        print(f"[v3] enhance loop: {t_loop:.1f}s ({1000*t_loop/max(n,1):.1f} ms/frame)")
        print(f"[v3] {restored} restored, {kept} original kept")
        if timing is not None:
            for k, v in timing.items():
                print(f"  {k:>10s}: {v:.1f}s ({1000*v/max(n,1):.2f} ms/frame)")

        print("[v3] assembling video...")
        t_asm0 = time.time()
        assemble_video(out_frame_dir, args.output, fps, args.input)
        t_asm = time.time() - t_asm0

    total = t_extract + t_loop + t_asm
    print(f"[v3] timing: extract={t_extract:.1f}s, enhance={t_loop:.1f}s, "
          f"assemble={t_asm:.1f}s, total={total:.1f}s")
    print(f"[v3] output: {args.output}")


if __name__ == "__main__":
    main()
