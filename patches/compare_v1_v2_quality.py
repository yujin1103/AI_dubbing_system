"""v1 (CPU warpAffine) vs v2 (GPU grid_sample) 출력 품질 비교.

같은 입력 비디오에 대해
    1) gfpgan_async_postprocess_trt.py 로 만든 v1 출력
    2) gfpgan_async_postprocess_trt_v2.py 로 만든 v2 출력
의 frame-by-frame PSNR / SSIM 을 측정한다. 일부 프레임은 PNG 로 dump.

요구사항: 두 mp4 가 모두 존재하고 동일 길이.
"""
import argparse
import os
import sys
import subprocess
import tempfile

import cv2
import numpy as np


def extract_frames(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path, os.path.join(out_dir, "%08d.png"),
    ], check=True)


def psnr(a, b, data_range=255):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range ** 2 / mse)


def ssim_y(a, b):
    """Quick SSIM on Y channel only. Uses skimage if available, else fallback."""
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except Exception:
        return None
    ay = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[..., 0]
    by = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[..., 0]
    return compare_ssim(ay, by, data_range=255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--sample-frames", default="0,400,800,1200,1500",
                    help="comma-separated frame indices to dump as PNG")
    ap.add_argument("--dump-dir", default="/tmp/quality_dump")
    ap.add_argument("--every", type=int, default=20,
                    help="measure every N-th frame (1 = every frame)")
    args = ap.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    sample_frames = [int(x) for x in args.sample_frames.split(",") if x.strip()]

    with tempfile.TemporaryDirectory() as td:
        d1 = os.path.join(td, "v1")
        d2 = os.path.join(td, "v2")
        print("extracting v1 frames...")
        extract_frames(args.v1, d1)
        print("extracting v2 frames...")
        extract_frames(args.v2, d2)

        files1 = sorted(os.listdir(d1))
        files2 = sorted(os.listdir(d2))
        n = min(len(files1), len(files2))
        print(f"comparing {n} frames (every {args.every})")

        psnrs = []
        ssims = []
        for i in range(0, n, args.every):
            p1 = os.path.join(d1, files1[i])
            p2 = os.path.join(d2, files2[i])
            a = cv2.imread(p1, cv2.IMREAD_COLOR)
            b = cv2.imread(p2, cv2.IMREAD_COLOR)
            if a is None or b is None:
                continue
            if a.shape != b.shape:
                print(f"  frame {i}: shape mismatch v1={a.shape} v2={b.shape}, resizing v1")
                a = cv2.resize(a, (b.shape[1], b.shape[0]))
            psnrs.append(psnr(a, b))
            s = ssim_y(a, b)
            if s is not None:
                ssims.append(s)
            if i in sample_frames:
                cv2.imwrite(os.path.join(args.dump_dir, f"v1_{i:06d}.png"), a)
                cv2.imwrite(os.path.join(args.dump_dir, f"v2_{i:06d}.png"), b)
                cv2.imwrite(os.path.join(args.dump_dir, f"diff_{i:06d}.png"),
                            np.abs(a.astype(np.int32) - b.astype(np.int32)).clip(0, 255).astype(np.uint8))

        psnr_arr = np.array(psnrs)
        print(f"\n=== PSNR (n={len(psnr_arr)}) ===")
        print(f"  mean: {psnr_arr.mean():.2f} dB")
        print(f"  median: {np.median(psnr_arr):.2f} dB")
        print(f"  min: {psnr_arr.min():.2f} dB")
        print(f"  p5: {np.percentile(psnr_arr, 5):.2f} dB")
        print(f"  p95: {np.percentile(psnr_arr, 95):.2f} dB")

        if ssims:
            ssim_arr = np.array(ssims)
            print(f"\n=== SSIM (Y, n={len(ssim_arr)}) ===")
            print(f"  mean: {ssim_arr.mean():.4f}")
            print(f"  median: {np.median(ssim_arr):.4f}")
            print(f"  min: {ssim_arr.min():.4f}")
            print(f"  p5: {np.percentile(ssim_arr, 5):.4f}")

        print(f"\nsamples in {args.dump_dir}")


if __name__ == "__main__":
    main()
