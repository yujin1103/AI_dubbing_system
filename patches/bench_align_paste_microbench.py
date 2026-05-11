"""한 프레임에 대해 cv2.warpAffine vs GPU grid_sample 단계별 시간 비교.

실제 1080p 얼굴 프레임을 사용. detection 결과는 facexlib PT 로 한 번 받아서
같은 landmark 로 두 경로 (cv2 / GPU) 를 동일하게 비교.

실행:
    /opt/venv_gfpgan/bin/python bench_align_paste_microbench.py
"""
import sys
import os
import time
import subprocess
import tempfile

sys.path.insert(0, "/workspace/patches")

import cv2
import numpy as np
import torch

from gfpgan import GFPGANer
from gfpgan_trt_wrapper import GFPGANTRT
from gpu_face_aligner import GPUFaceAligner


def main():
    device = torch.device("cuda")
    print("loading GFPGANer...")
    restorer = GFPGANer(
        model_path="/opt/gfpgan_models/GFPGANv1.4.pth",
        upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
    )
    del restorer.gfpgan
    torch.cuda.empty_cache()
    gfpgan_trt = GFPGANTRT()

    aligner = GPUFaceAligner(
        device=device,
        face_template=restorer.face_helper.face_template,
        face_size=restorer.face_helper.face_size[0],
        use_parse=True,
        face_parse_module=restorer.face_helper.face_parse,
    )

    # frame at t=5s (얼굴이 잡히는 위치)
    print("extracting one frame at t=5s...")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", "5",
            "-i", "/workspace/media/output/test_I_trt_dpm10_tea.mp4",
            "-vframes", "1", f"{td}/f.png",
        ], check=True)
        img = cv2.imread(f"{td}/f.png", cv2.IMREAD_COLOR)
    print(f"  shape={img.shape}")

    # run detection once
    with torch.no_grad():
        bboxes_landm = restorer.face_helper.face_det.detect_faces(img, 0.97)
    print(f"  detected {len(bboxes_landm)} faces")
    landmarks = []
    for bbox in bboxes_landm:
        landmarks.append(np.array(
            [[bbox[i], bbox[i+1]] for i in range(5, 15, 2)], dtype=np.float32))

    # ------------------------------------------------------------------
    # CV2 path (mimic facexlib)
    # ------------------------------------------------------------------
    print("\n=== CV2 baseline ===")
    helper = restorer.face_helper

    N_RUN = 50

    def cv2_align_only():
        helper.clean_all()
        helper.read_image(img.copy())
        for landmark in landmarks:
            helper.all_landmarks_5.append(landmark)
        helper.align_warp_face()
        return helper.cropped_faces

    # warmup
    for _ in range(3):
        _ = cv2_align_only()
    t0 = time.time()
    for _ in range(N_RUN):
        _ = cv2_align_only()
    t_cv2_align = (time.time() - t0) / N_RUN * 1000
    print(f"  cv2 align_warp_face: {t_cv2_align:.2f} ms/frame")

    # GFPGAN forward (TRT, same in both paths)
    cropped = cv2_align_only()  # has cropped face
    cropped_face = cropped[0]
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms.functional import normalize
    t0 = time.time()
    for _ in range(N_RUN):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        with torch.no_grad():
            output = gfpgan_trt(cropped_face_t)[0]
        torch.cuda.synchronize()
    t_cv2_gfpgan = (time.time() - t0) / N_RUN * 1000
    # produce restored uint8 BGR for cv2 paste path
    restored_face_bgr = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    print(f"  cv2 path GFPGAN forward (incl. transfer): {t_cv2_gfpgan:.2f} ms")

    # cv2 paste back
    helper.clean_all()
    helper.read_image(img.copy())
    for landmark in landmarks:
        helper.all_landmarks_5.append(landmark)
    helper.align_warp_face()
    helper.add_restored_face(restored_face_bgr.astype('uint8'))
    helper.get_inverse_affine()
    # warmup
    for _ in range(3):
        helper2 = restorer.face_helper
        helper2.clean_all()
        helper2.read_image(img.copy())
        for landmark in landmarks:
            helper2.all_landmarks_5.append(landmark)
        helper2.align_warp_face()
        helper2.add_restored_face(restored_face_bgr.astype('uint8'))
        helper2.get_inverse_affine()
        _ = helper2.paste_faces_to_input_image()
    t0 = time.time()
    for _ in range(N_RUN):
        helper2 = restorer.face_helper
        helper2.clean_all()
        helper2.read_image(img.copy())
        for landmark in landmarks:
            helper2.all_landmarks_5.append(landmark)
        helper2.align_warp_face()
        helper2.add_restored_face(restored_face_bgr.astype('uint8'))
        helper2.get_inverse_affine()
        _ = helper2.paste_faces_to_input_image()
    t_cv2_paste = (time.time() - t0) / N_RUN * 1000
    print(f"  cv2 paste_faces_to_input_image (full incl. parse): {t_cv2_paste:.2f} ms/frame")

    # ------------------------------------------------------------------
    # GPU path
    # ------------------------------------------------------------------
    print("\n=== GPU grid_sample path ===")

    img_t = torch.from_numpy(img).to(device).float() / 255.0
    img_chw_bgr_01 = img_t.permute(2, 0, 1).contiguous()

    # warmup
    for _ in range(3):
        with torch.no_grad():
            f, _ = aligner.align(img_chw_bgr_01, landmarks)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_RUN):
        with torch.no_grad():
            f, affines = aligner.align(img_chw_bgr_01, landmarks)
        torch.cuda.synchronize()
    t_gpu_align = (time.time() - t0) / N_RUN * 1000
    print(f"  gpu align (grid_sample 1080p->512): {t_gpu_align:.2f} ms/frame")

    # gfpgan trt forward (already on GPU)
    t0 = time.time()
    for _ in range(N_RUN):
        with torch.no_grad():
            out = gfpgan_trt(f.float())[0]
        torch.cuda.synchronize()
    t_gpu_gfpgan = (time.time() - t0) / N_RUN * 1000
    print(f"  gpu GFPGAN forward (no transfer): {t_gpu_gfpgan:.2f} ms")

    # gpu paste (with parse + grid_sample)
    for _ in range(3):
        with torch.no_grad():
            o = aligner.paste(img_chw_bgr_01, out, affines, parse_input_faces_rgb_norm=out)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_RUN):
        with torch.no_grad():
            o = aligner.paste(img_chw_bgr_01, out, affines, parse_input_faces_rgb_norm=out)
        torch.cuda.synchronize()
    t_gpu_paste = (time.time() - t0) / N_RUN * 1000
    print(f"  gpu paste (parse + grid_sample + blend): {t_gpu_paste:.2f} ms/frame")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"  align:         cv2 {t_cv2_align:6.2f} ms  vs  gpu {t_gpu_align:6.2f} ms  "
          f"(speedup {t_cv2_align/max(t_gpu_align,1e-3):.2f}x)")
    print(f"  gfpgan forward (full incl. transfer): cv2 {t_cv2_gfpgan:6.2f} ms  vs  gpu {t_gpu_gfpgan:6.2f} ms")
    print(f"  paste:         cv2 {t_cv2_paste:6.2f} ms  vs  gpu {t_gpu_paste:6.2f} ms  "
          f"(speedup {t_cv2_paste/max(t_gpu_paste,1e-3):.2f}x)")
    delta_align = t_cv2_align - t_gpu_align
    delta_paste = t_cv2_paste - t_gpu_paste
    print(f"\n  per-frame savings (align+paste): {delta_align + delta_paste:.2f} ms")
    print(f"  for 1602 frames: {(delta_align + delta_paste) * 1602 / 1000:.1f}s")


if __name__ == "__main__":
    main()
