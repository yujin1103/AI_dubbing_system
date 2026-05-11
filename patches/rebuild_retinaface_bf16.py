"""RetinaFace FHD TRT engine BF16 재빌드.

FP16 빌드의 conf softmax precision 부족 → 0.97 threshold 통과 실패 → 0 faces detection.
BF16 은 exponent range 가 FP32 와 같아 conf 정확도 보존.

가중치 로딩은 strict (module. prefix 처리) — 이미 rebuild_retinaface_fhd.py 의 방식 그대로.
"""
import os
import time
import torch
import tensorrt as trt

CKPT = "/workspace/gfpgan/weights/detection_Resnet50_Final.pth"
ONNX_DIR = "/workspace/trt_work/onnx/retinaface"
ENGINE_DIR = "/workspace/trt_work/engines"
H, W = 1080, 1920
ONNX = os.path.join(ONNX_DIR, "retinaface_r50_fhd.onnx")  # rebuild_fhd 로 이미 정상 weights
ENGINE_BF16 = os.path.join(ENGINE_DIR, "retinaface_r50_fhd_bf16.trt")


def build_bf16():
    print(f"building TRT BF16 engine from {ONNX}")
    if not os.path.isfile(ONNX):
        raise FileNotFoundError(f"{ONNX} — run rebuild_retinaface_fhd.py first")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(ONNX):
        for i in range(parser.num_errors):
            print(f"ERROR: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
    config.set_flag(trt.BuilderFlag.BF16)  # ★ FP16 대신 BF16
    # FP32 도 허용해서 conf softmax 같은 정밀도 민감 부분은 FP32 유지하게
    config.set_flag(trt.BuilderFlag.FP16)  # mixed precision 허용

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("BF16 engine build failed")
    elapsed = time.time() - t0

    with open(ENGINE_BF16, "wb") as f:
        f.write(serialized)
    print(f"wrote {ENGINE_BF16} ({os.path.getsize(ENGINE_BF16)/1e6:.1f} MB, {elapsed:.0f}s)")


def smoke_test_bf16():
    import sys
    import cv2
    import subprocess
    import tempfile
    import types
    import numpy as np
    sys.path.insert(0, "/workspace/patches")
    from facexlib.detection.retinaface import RetinaFace
    from retinaface_trt_wrapper import _TRTSession

    # PyTorch reference
    m = RetinaFace(network_name="resnet50", half=False, device="cuda")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    m.load_state_dict(cleaned, strict=True)
    m.eval().cuda()

    # Test frame
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-i", "/workspace/media/output/test_I_trt_dpm10_tea.mp4",
                        "-ss", "5", "-vframes", "1", f"{td}/f.png"], check=True)
        img = cv2.imread(f"{td}/f.png", cv2.IMREAD_COLOR)
    print(f"test frame: {img.shape}")

    # PT baseline
    with torch.no_grad():
        pt_bboxes = m.detect_faces(img.copy(), 0.97)
    print(f"PT detect: {len(pt_bboxes)} faces")
    if len(pt_bboxes):
        print(f"  first bbox: {pt_bboxes[0][:4]}")
        print(f"  first conf: {pt_bboxes[0][4] if len(pt_bboxes[0]) > 4 else 'N/A'}")

    # TRT BF16
    sess = _TRTSession(ENGINE_BF16, H, W, 85200)

    def trt_forward(self, x):
        return sess.infer(x)
    m.forward = types.MethodType(trt_forward, m)

    with torch.no_grad():
        trt_bboxes = m.detect_faces(img.copy(), 0.97)
    print(f"TRT BF16 detect: {len(trt_bboxes)} faces")
    if len(trt_bboxes):
        print(f"  first bbox: {trt_bboxes[0][:4]}")
        print(f"  first conf: {trt_bboxes[0][4] if len(trt_bboxes[0]) > 4 else 'N/A'}")

    if len(pt_bboxes) == len(trt_bboxes) and len(pt_bboxes) > 0:
        diff = np.abs(np.array(pt_bboxes)[:, :4] - np.array(trt_bboxes)[:, :4])
        print(f"bbox L1 diff: max={diff.max():.2f} px, mean={diff.mean():.2f} px")


if __name__ == "__main__":
    build_bf16()
    smoke_test_bf16()
