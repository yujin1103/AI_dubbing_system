"""GPU aligner sanity test — cv2.warpAffine 와 grid_sample 결과 비교.

목적: 본격 파이프라인 돌리기 전에 affine 매핑이 픽셀 단위로 일치하는지 확인.
"""
import sys
import os
sys.path.insert(0, "/workspace/patches")
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gpu_face_aligner import _cv_to_theta, GPUFaceAligner


def main():
    device = torch.device("cuda")
    H, W = 1080, 1920
    face_size = 512

    # synthetic landmark (~ 정상적인 얼굴 비율, 가운데 부근)
    cx, cy = W / 2, H / 2 - 50
    landmarks = np.array([
        [cx - 60, cy - 30],   # left eye
        [cx + 60, cy - 30],   # right eye
        [cx,      cy + 20],   # nose
        [cx - 50, cy + 80],   # mouth left
        [cx + 50, cy + 80],   # mouth right
    ], dtype=np.float32)

    face_template = np.array([
        [192.98138, 239.94708], [318.90277, 240.1936],
        [256.63416, 314.01935], [201.26117, 371.41043],
        [313.08905, 371.15118],
    ], dtype=np.float32)

    # synthetic image: gradient + circle markers at the landmarks
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    img[..., 2] = 128
    for (lx, ly) in landmarks:
        cv2.circle(img, (int(lx), int(ly)), 8, (255, 255, 255), -1)

    # ---- cv2 baseline forward warp ----
    M = cv2.estimateAffinePartial2D(landmarks, face_template, method=cv2.LMEDS)[0]
    cv2_face = cv2.warpAffine(img, M, (face_size, face_size),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    # ---- GPU forward warp ----
    img_t = torch.from_numpy(img).to(device).float() / 255.0   # (H, W, 3) BGR
    img_chw = img_t.permute(2, 0, 1).contiguous()              # (3, H, W) BGR
    M_inv = cv2.invertAffineTransform(M)
    theta = _cv_to_theta(M_inv, src_size=(H, W), dst_size=(face_size, face_size)).to(device).unsqueeze(0)
    grid = F.affine_grid(theta, (1, 3, face_size, face_size), align_corners=True)
    gpu_face = F.grid_sample(img_chw.unsqueeze(0), grid,
                             mode='bilinear', padding_mode='zeros',
                             align_corners=True)
    gpu_face_np = (gpu_face[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    diff = np.abs(cv2_face.astype(np.int32) - gpu_face_np.astype(np.int32))
    print(f"forward warp diff: mean={diff.mean():.3f}, max={diff.max()}, p99={np.percentile(diff, 99):.1f}")
    print(f"  cv2_face mean {cv2_face.mean():.1f}, gpu_face mean {gpu_face_np.mean():.1f}")

    # PSNR
    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
    print(f"forward warp PSNR vs cv2: {psnr:.2f} dB")

    # ---- cv2 baseline inverse warp (paste back, no mask) ----
    inv_M = cv2.invertAffineTransform(M)
    paste_cv2 = cv2.warpAffine(cv2_face, inv_M, (W, H))

    # ---- GPU inverse warp ----
    M_out_to_in = cv2.invertAffineTransform(inv_M)  # = M (input->face_512)
    theta_p = _cv_to_theta(M_out_to_in, src_size=(face_size, face_size),
                            dst_size=(H, W)).to(device).unsqueeze(0)
    grid_p = F.affine_grid(theta_p, (1, 3, H, W), align_corners=True)
    gpu_face_chw = torch.from_numpy(cv2_face).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    paste_gpu = F.grid_sample(gpu_face_chw, grid_p, mode='bilinear',
                              padding_mode='zeros', align_corners=True)
    paste_gpu_np = (paste_gpu[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    diff2 = np.abs(paste_cv2.astype(np.int32) - paste_gpu_np.astype(np.int32))
    print(f"inverse warp diff: mean={diff2.mean():.3f}, max={diff2.max()}, p99={np.percentile(diff2, 99):.1f}")
    mse2 = float((diff2.astype(np.float64) ** 2).mean())
    psnr2 = 10 * np.log10(255.0 ** 2 / max(mse2, 1e-9))
    print(f"inverse warp PSNR vs cv2: {psnr2:.2f} dB")

    # ---- microbench ----
    print("\nbenchmarking 100 iterations...")
    img_chw_b = img_chw.unsqueeze(0)
    theta_b = theta.expand(1, 2, 3).contiguous()

    # CPU cv2 forward
    t0 = time.time()
    for _ in range(100):
        _ = cv2.warpAffine(img, M, (face_size, face_size),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    t_cv2_fwd = (time.time() - t0) * 10  # ms per iter

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        grid = F.affine_grid(theta_b, (1, 3, face_size, face_size), align_corners=True)
        _ = F.grid_sample(img_chw_b, grid, mode='bilinear',
                          padding_mode='zeros', align_corners=True)
    torch.cuda.synchronize()
    t_gpu_fwd = (time.time() - t0) * 10

    print(f"  cv2 warpAffine forward (1080p->512): {t_cv2_fwd:.2f} ms")
    print(f"  GPU grid_sample forward         :    {t_gpu_fwd:.2f} ms")

    # CPU cv2 paste
    t0 = time.time()
    for _ in range(100):
        _ = cv2.warpAffine(cv2_face, inv_M, (W, H))
    t_cv2_paste = (time.time() - t0) * 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        grid_p = F.affine_grid(theta_p, (1, 3, H, W), align_corners=True)
        _ = F.grid_sample(gpu_face_chw, grid_p, mode='bilinear',
                          padding_mode='zeros', align_corners=True)
    torch.cuda.synchronize()
    t_gpu_paste = (time.time() - t0) * 10

    print(f"  cv2 warpAffine paste (512->1080p): {t_cv2_paste:.2f} ms")
    print(f"  GPU grid_sample paste           : {t_gpu_paste:.2f} ms")


if __name__ == "__main__":
    main()
