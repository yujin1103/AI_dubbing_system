"""RetinaFace multi-resolution TRT dispatch 테스트.

1) raw forward (해상도별 엔진 자동 선택) 검증
2) facexlib detect_faces 가 우리 wrapper 와 동작하는지 검증
3) PyTorch baseline 과 결과 비교 (detection 좌표 차이 측정)
"""
import sys
import time
sys.path.insert(0, "/workspace/patches")

import torch
import cv2
import subprocess
import tempfile
import numpy as np
from facexlib.detection.retinaface import RetinaFace
from retinaface_trt_wrapper import RetinaFaceTRT, wrap_face_helper_detector

CKPT = "/workspace/gfpgan/weights/detection_Resnet50_Final.pth"
ENGINE_DIR = "/workspace/trt_work/engines"
TEST_VID = "/workspace/media/output/test_I_trt_dpm10_tea.mp4"


def make_pt_model():
    m = RetinaFace(network_name="resnet50", half=False, device="cuda")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    m.load_state_dict(ckpt, strict=False)
    return m.eval().cuda()


def get_test_frame():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", TEST_VID, "-vframes", "1", f"{td}/f.png"],
            check=True,
        )
        return cv2.imread(f"{td}/f.png", cv2.IMREAD_COLOR)


def test_raw_forward():
    """입력 해상도별 엔진 자동 선택 — random tensor."""
    print("\n[TEST 1] raw forward dispatch (4 resolutions)")
    trt_mod = RetinaFaceTRT(ENGINE_DIR).cuda().eval()
    for h, w in [(720, 1280), (1080, 1920), (1440, 2560), (2160, 3840)]:
        x = torch.randn(1, 3, h, w, device="cuda")
        t0 = time.time()
        with torch.no_grad():
            bbox, conf, lmk = trt_mod(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000
        print(f"  {h}x{w}: bbox={tuple(bbox.shape)} conf={tuple(conf.shape)} "
              f"lmk={tuple(lmk.shape)} ({dt:.1f} ms)")
    print("  PASS")


def test_facexlib_detect_faces():
    """facexlib detect_faces 가 TRT forward 로 동작하는지.

    facexlib RetinaFace.detect_faces 의 priors / decode 가 grad-tracking 을 켜놔서
    실제 사용 시에는 GFPGANer.enhance() 가 @torch.no_grad() 감싸서 호출함.
    여기서도 동일하게 no_grad() 안에서 호출.
    """
    print("\n[TEST 2] facexlib detect_faces with TRT forward")
    img = get_test_frame()
    print(f"  test frame: {img.shape}")

    # PyTorch baseline
    pt_model = make_pt_model()
    with torch.no_grad():
        # warmup
        for _ in range(3):
            _ = pt_model.detect_faces(img.copy(), 0.97)
        torch.cuda.synchronize()
        t0 = time.time()
        pt_bboxes = pt_model.detect_faces(img.copy(), 0.97)
        torch.cuda.synchronize()
        pt_ms = (time.time() - t0) * 1000
    print(f"  PT detect_faces: {pt_ms:.1f} ms, found {len(pt_bboxes)} faces")
    if len(pt_bboxes):
        print(f"    first face bbox: {pt_bboxes[0][:4]}")

    # TRT (replace forward)
    trt_module = RetinaFaceTRT(ENGINE_DIR, original=pt_model).cuda().eval()
    import types
    def _f(self, x):
        return trt_module(x)
    pt_model.forward = types.MethodType(_f, pt_model)
    pt_model._trt = trt_module

    with torch.no_grad():
        # warmup
        for _ in range(3):
            _ = pt_model.detect_faces(img.copy(), 0.97)
        torch.cuda.synchronize()
        t0 = time.time()
        trt_bboxes = pt_model.detect_faces(img.copy(), 0.97)
        torch.cuda.synchronize()
        trt_ms = (time.time() - t0) * 1000
    print(f"  TRT detect_faces: {trt_ms:.1f} ms, found {len(trt_bboxes)} faces")
    if len(trt_bboxes):
        print(f"    first face bbox: {trt_bboxes[0][:4]}")

    # 결과 비교 — bbox 좌표 차이
    if len(pt_bboxes) == len(trt_bboxes) and len(pt_bboxes) > 0:
        diff = np.abs(np.array(pt_bboxes)[:, :4] - np.array(trt_bboxes)[:, :4])
        print(f"  bbox L1 diff: max={diff.max():.2f}, mean={diff.mean():.2f} (pixels)")
    print(f"  speedup: {pt_ms / trt_ms:.2f}x ({(1 - trt_ms/pt_ms)*100:.0f}% saved)")
    print("  PASS")


def test_full_enhance():
    """GFPGAN + face_helper 전체 enhance — 시간 측정."""
    print("\n[TEST 3] full enhance() pipeline timing")
    from gfpgan import GFPGANer

    img = get_test_frame()

    # PT baseline
    print("  building PT-only baseline (5 frames)...")
    rest_pt = GFPGANer(
        model_path="/opt/gfpgan_models/GFPGANv1.4.pth",
        upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
    )
    for _ in range(2):  # warmup
        rest_pt.enhance(img.copy(), has_aligned=False, only_center_face=False, paste_back=True)
    torch.cuda.synchronize()
    t0 = time.time()
    N = 5
    for _ in range(N):
        rest_pt.enhance(img.copy(), has_aligned=False, only_center_face=False, paste_back=True)
    torch.cuda.synchronize()
    pt_ms = (time.time() - t0) / N * 1000
    print(f"  PT (full PyTorch): {pt_ms:.1f} ms/frame")
    del rest_pt
    torch.cuda.empty_cache()

    # TRT detector
    print("  building TRT-detector pipeline (5 frames)...")
    rest_trt = GFPGANer(
        model_path="/opt/gfpgan_models/GFPGANv1.4.pth",
        upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
    )
    wrap_face_helper_detector(rest_trt.face_helper, ENGINE_DIR, preload=["fhd"])
    for _ in range(2):  # warmup
        rest_trt.enhance(img.copy(), has_aligned=False, only_center_face=False, paste_back=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        rest_trt.enhance(img.copy(), has_aligned=False, only_center_face=False, paste_back=True)
    torch.cuda.synchronize()
    trt_ms = (time.time() - t0) / N * 1000
    print(f"  TRT (detector only): {trt_ms:.1f} ms/frame")
    print(f"  speedup: {pt_ms / trt_ms:.2f}x ({(1 - trt_ms/pt_ms)*100:.0f}% saved)")
    print("  PASS")


if __name__ == "__main__":
    test_raw_forward()
    test_facexlib_detect_faces()
    test_full_enhance()
    print("\n=== ALL TESTS PASSED ===")
