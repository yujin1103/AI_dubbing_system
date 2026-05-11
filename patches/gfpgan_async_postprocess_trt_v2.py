"""GFPGAN async post-process v2: GPU-resident pipeline.

v1 (gfpgan_async_postprocess_trt.py) 은 face crop / paste-back 을 cv2.warpAffine
으로 CPU 에서 했다.  v2 는 그 두 단계를 grid_sample 로 GPU 에 옮기고, 입력 영상을
한 번만 GPU 로 업로드해서 detection 빼고 모든 것이 GPU resident 가 되게 한다.

흐름 (per frame):
    1) reader 스레드: cv2.imread -> uint8 numpy
    2) main:
        - upload (H, W, 3) BGR uint8 -> (3, H, W) FP32 [0, 1] cuda
        - face_det.detect_faces(numpy)  (얼굴 검출은 facexlib 그대로)
            → numpy landmarks (N, 5, 2)
        - GPUFaceAligner.align(...) -> faces (N, 3, 512, 512) RGB [-1, 1]
        - GFPGAN TRT forward (per face, B=1) -> restored faces
        - GPUFaceAligner.paste(...) -> output (3, H, W) BGR [0, 1]
        - download (3, H, W) -> uint8 numpy
    3) writer pool: cv2.imwrite

Fallback: 얼굴이 검출되지 않은 프레임은 그대로 원본을 저장 (v1 동작과 동일).

사용:
    /opt/venv_gfpgan/bin/python gfpgan_async_postprocess_trt_v2.py \
        --input  /workspace/media/output/test_I_trt_dpm10_tea.mp4 \
        --output /workspace/media/output/test_I_trt_dpm10_tea_v2.mp4 \
        --upscale 1
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


# ---------------------------------------------------------------------------
# ffmpeg helpers (v1 과 동일)
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
# Per-frame GPU enhance
# ---------------------------------------------------------------------------

@torch.no_grad()
def enhance_frame_gpu(img_bgr_u8: np.ndarray,
                      face_det,
                      gfpgan_trt: GFPGANTRT,
                      aligner: GPUFaceAligner,
                      device: torch.device,
                      eye_dist_threshold: float = 5.0,
                      conf_threshold: float = 0.97,
                      timing: dict = None) -> np.ndarray:
    """입력 BGR uint8 -> GFPGAN 처리된 BGR uint8.

    timing dict 가 주어지면 각 단계 시간을 누적한다.
    """
    H, W = img_bgr_u8.shape[:2]

    # ---- 1) detect (numpy in -> numpy out) ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    bboxes_landm = face_det.detect_faces(img_bgr_u8, conf_threshold)
    if timing is not None:
        torch.cuda.synchronize()
        timing['detect'] += time.time() - t

    # filter by eye distance (facexlib FaceRestoreHelper.get_face_landmarks_5 와 동일)
    landmarks_list = []
    for bbox in bboxes_landm:
        eye_dist = np.linalg.norm([bbox[5] - bbox[7], bbox[6] - bbox[8]])
        if eye_dist < eye_dist_threshold:
            continue
        landmarks_list.append(np.array(
            [[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)], dtype=np.float32))

    if not landmarks_list:
        # 얼굴 미검출 → 원본 그대로 반환
        return img_bgr_u8

    # ---- 2) upload (H, W, 3) BGR uint8 -> (3, H, W) FP32 [0, 1] cuda ----
    # uint8 transfer (3x cheaper PCIe) then on-GPU permute+convert.
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    img_t = torch.from_numpy(img_bgr_u8).to(device, non_blocking=True)  # (H, W, 3) uint8
    img_chw_bgr_01 = img_t.permute(2, 0, 1).contiguous().to(torch.float32).mul_(1.0 / 255.0)
    if timing is not None:
        torch.cuda.synchronize()
        timing['upload'] += time.time() - t

    # ---- 3) GPU align -> faces (N, 3, 512, 512) RGB [-1, 1] ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    faces_rgb_norm, affines = aligner.align(img_chw_bgr_01, landmarks_list)
    if timing is not None:
        torch.cuda.synchronize()
        timing['align'] += time.time() - t

    # ---- 4) GFPGAN TRT (B=1 static engine) -> restored ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    N = faces_rgb_norm.shape[0]
    restored_list = []
    for i in range(N):
        out = gfpgan_trt(faces_rgb_norm[i:i+1].float())[0]  # (1, 3, 512, 512)
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

    # ---- 6) download (3, H, W) -> (H, W, 3) uint8 ----
    if timing is not None:
        torch.cuda.synchronize()
        t = time.time()
    out_u8 = (out_chw.permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8)
    out_np = out_u8.cpu().numpy()
    if timing is not None:
        torch.cuda.synchronize()
        timing['download'] += time.time() - t

    return out_np


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/opt/gfpgan_models/GFPGANv1.4.pth")
    parser.add_argument("--engine", default="/workspace/trt_work/engines/gfpgan_bf16.trt")
    parser.add_argument("--retinaface-engine-dir",
                        default="/workspace/trt_work/engines",
                        help="Directory with retinaface_r50_<res>_fp16.trt files")
    parser.add_argument("--retinaface-trt", action="store_true",
                        help=("Swap face_det.forward to TRT (default: off). The "
                              "checked-in retinaface_r50_*_fp16.trt engines were "
                              "exported with strict=False weight loading without "
                              "stripping the 'module.' prefix, so they have random "
                              "init weights and produce ~40k spurious detections "
                              "per frame. Don't enable until those engines are "
                              "rebuilt with the correct weights."))
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N frames (0 = all)")
    parser.add_argument("--detail-timing", action="store_true",
                        help="Per-stage timing breakdown")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[v2] input not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.engine):
        print(f"[v2] engine not found: {args.engine}")
        sys.exit(1)

    fps, duration = get_video_info(args.input)
    print(f"[v2] input: {args.input} (fps={fps:.2f}, dur={duration:.1f}s)")
    print(f"[v2] gfpgan engine: {args.engine}")

    print("[v2] loading GFPGANer (face_helper + PyTorch generator for warm-up)...")
    restorer = GFPGANer(
        model_path=args.model,
        upscale=args.upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    # Free PyTorch StyleGAN2; install TRT generator.
    print("[v2] swapping PyTorch generator -> TRT engine")
    del restorer.gfpgan
    torch.cuda.empty_cache()
    gfpgan_trt = GFPGANTRT(engine_path=args.engine)
    print("[v2] GFPGAN TRT ready")

    # Swap RetinaFace to TRT (multi-resolution). Off by default because the
    # currently-shipped engines have broken weights (see arg help).
    if args.retinaface_trt and HAS_RETINAFACE_TRT:
        try:
            wrap_face_helper_detector(
                restorer.face_helper, args.retinaface_engine_dir,
                preload=["fhd"],
            )
            print("[v2] RetinaFace TRT ready (preloaded fhd)")
        except Exception as e:
            print(f"[v2] WARN: RetinaFace TRT swap failed ({e}); falling back to PyTorch detector")
    else:
        print("[v2] RetinaFace: PyTorch detector")

    device = torch.device("cuda")
    aligner = GPUFaceAligner(
        device=device,
        face_template=restorer.face_helper.face_template,
        face_size=restorer.face_helper.face_size[0],
        use_parse=True,
        face_parse_module=restorer.face_helper.face_parse,
    )
    # parse model on GPU eval
    aligner.face_parse.eval()
    print("[v2] GPUFaceAligner ready (use_parse=True)")

    with tempfile.TemporaryDirectory(prefix="gfpgan_trt_v2_") as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        out_frame_dir = os.path.join(tmpdir, "out_frames")
        os.makedirs(out_frame_dir, exist_ok=True)
        print("[v2] extracting frames...")
        t_extract0 = time.time()
        extract_frames(args.input, frame_dir)
        t_extract = time.time() - t_extract0

        frame_files = sorted(Path(frame_dir).glob("*.png"))
        if args.limit > 0:
            frame_files = frame_files[:args.limit]
        n = len(frame_files)
        print(f"[v2] processing {n} frames (async I/O + GPU pipeline)...")

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

        # warm-up: run one frame through detect+align+enhance+paste so kernels
        # and contexts are JIT-built before the real timer.
        if n > 0:
            warm_path = frame_files[0]
            warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
            try:
                _ = enhance_frame_gpu(
                    warm_img, restorer.face_helper.face_det,
                    gfpgan_trt, aligner, device, timing=None,
                )
                torch.cuda.synchronize()
                print("[v2] warm-up frame done")
            except Exception as e:
                print(f"[v2] WARN: warm-up failed: {e}")

        t_loop0 = time.time()
        while True:
            i, name, img = read_queue.get()
            if name is None:
                break
            try:
                out = enhance_frame_gpu(
                    img, restorer.face_helper.face_det,
                    gfpgan_trt, aligner, device, timing=timing,
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
                    print(f"[v2] frame {i} ({name}) enhance failed: {e}")
                write_pool.submit(cv2.imwrite,
                                  os.path.join(out_frame_dir, name), img)
                kept += 1
            progress.update(1)
        progress.close()
        write_pool.shutdown(wait=True)
        reader_thread.join()
        torch.cuda.synchronize()
        t_loop = time.time() - t_loop0
        print(f"[v2] enhance loop: {t_loop:.1f}s ({1000*t_loop/max(n,1):.1f} ms/frame)")
        print(f"[v2] {restored} restored, {kept} original kept")
        if timing is not None:
            for k, v in timing.items():
                print(f"  {k:>10s}: {v:.1f}s ({1000*v/max(n,1):.2f} ms/frame)")

        print("[v2] assembling video...")
        t_asm0 = time.time()
        # If --limit was used, the audio source still has the full duration,
        # ffmpeg -shortest will trim. For a limited run we emit a short video.
        assemble_video(out_frame_dir, args.output, fps, args.input)
        t_asm = time.time() - t_asm0

    total = t_extract + t_loop + t_asm
    print(f"[v2] timing: extract={t_extract:.1f}s, enhance={t_loop:.1f}s, "
          f"assemble={t_asm:.1f}s, total={total:.1f}s")
    print(f"[v2] output: {args.output}")


if __name__ == "__main__":
    main()
