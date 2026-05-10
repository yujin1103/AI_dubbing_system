"""GFPGAN + face_helper full enhance() 시간 측정. (TRT detector 사용)"""
import sys
import time
sys.path.insert(0, "/workspace/patches")

import torch
import cv2
import subprocess
import tempfile
from gfpgan import GFPGANer
from retinaface_trt_wrapper import wrap_face_helper_detector

print("loading GFPGANer...", flush=True)
restorer = GFPGANer(
    model_path="/opt/gfpgan_models/GFPGANv1.4.pth",
    upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
)
print("swapping detector to TRT...", flush=True)
wrap_face_helper_detector(
    restorer.face_helper,
    "/workspace/trt_work/engines/retinaface_r50_1080p_fp16.trt",
)

with tempfile.TemporaryDirectory() as td:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", "/workspace/media/output/test_I_trt_dpm10_tea.mp4",
         "-vframes", "1", f"{td}/f.png"],
        check=True,
    )
    img = cv2.imread(f"{td}/f.png", cv2.IMREAD_COLOR)
print(f"frame shape: {img.shape}", flush=True)

print("warmup x3...", flush=True)
for i in range(3):
    _, _, out = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    print(f"  warmup {i}: out={out.shape if out is not None else None}", flush=True)
torch.cuda.synchronize()

print("benchmarking 30 frames...", flush=True)
t0 = time.time()
N = 30
for i in range(N):
    _, _, out = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"[FULL] {elapsed/N*1000:.1f} ms/frame ({N} frames in {elapsed:.2f}s)", flush=True)
print(f"output shape: {out.shape if out is not None else None}", flush=True)
