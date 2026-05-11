"""RetinaFace FHD (1080x1920) TRT engine 재빌드 — strict=True 로 weights 로드.

기존 build_retinaface_multires.py 는 strict=False 로 로드해서 'module.' prefix 가
있는 checkpoint 키가 모두 무시되고, 결과 ONNX/TRT 엔진이 random init 가중치만
가지고 있다. 그래서 detection 결과가 ~40k spurious anchors 를 토해낸다 (40,423
faces 검출되는 거 확인). 이걸 고친 빌드.

기존 엔진 백업:
    mv retinaface_r50_fhd_fp16.trt retinaface_r50_fhd_fp16.broken.trt
    rm retinaface_r50_fhd.onnx*
    /opt/venv_gfpgan/bin/python rebuild_retinaface_fhd.py
"""
import os
import sys
import time
import torch
import tensorrt as trt

CKPT = "/workspace/gfpgan/weights/detection_Resnet50_Final.pth"
ONNX_DIR = "/workspace/trt_work/onnx/retinaface"
ENGINE_DIR = "/workspace/trt_work/engines"


def export_onnx_fixed(h: int = 1080, w: int = 1920) -> str:
    from facexlib.detection.retinaface import RetinaFace
    onnx_path = os.path.join(ONNX_DIR, f"retinaface_r50_fhd.onnx")
    print(f"exporting ONNX for fhd ({h}x{w}) with strict=True weight loading...")
    m = RetinaFace(network_name="resnet50", half=False, device="cuda")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    # facexlib's init_detection_model strips 'module.' prefix; do the same.
    cleaned = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    missing, unexpected = m.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    if missing or unexpected:
        # fallback to strict=True to surface the real problem
        m.load_state_dict(cleaned, strict=True)
    print("  weights loaded OK")
    m.eval().cuda()

    x = torch.randn(1, 3, h, w, device="cuda")
    with torch.no_grad():
        out = m(x)
    print(f"  forward OK, shapes: {[tuple(t.shape) for t in out]}")
    print(f"  conf range: [{out[1].min().item():.3f}, {out[1].max().item():.3f}] (should be 0..1 from softmax)")

    os.makedirs(ONNX_DIR, exist_ok=True)
    torch.onnx.export(
        m, x, onnx_path,
        input_names=["images"],
        output_names=["bbox", "conf", "landmarks"],
        opset_version=17,
        dynamic_axes=None,
    )
    sz = os.path.getsize(onnx_path) / 1e6
    data_path = onnx_path + ".data"
    data_sz = os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0
    print(f"  wrote {onnx_path} ({sz:.2f} MB graph + {data_sz:.0f} MB weights)")
    return onnx_path


def build_engine(onnx_path: str) -> str:
    engine_path = os.path.join(ENGINE_DIR, f"retinaface_r50_fhd_fp16.trt")
    if os.path.exists(engine_path):
        backup = engine_path + ".broken_strictFalse"
        os.rename(engine_path, backup)
        print(f"  backed up old engine to {backup}")

    print(f"building TRT FP16 engine -> {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"  ERROR: {parser.get_error(i)}")
        raise RuntimeError(f"ONNX parse failed for {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
    config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    dt = time.time() - t0
    os.makedirs(ENGINE_DIR, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    sz = os.path.getsize(engine_path) / 1e6
    print(f"  wrote {engine_path} ({sz:.1f} MB, {dt:.0f}s)")
    return engine_path


def smoke_test(engine_path: str):
    """엔진 forward + accuracy check (real frame 한 장으로 detect_faces)."""
    print(f"\nsmoke test on real 1080p frame...")
    import subprocess, tempfile, cv2, numpy as np
    from facexlib.detection.retinaface import RetinaFace
    sys.path.insert(0, "/workspace/patches")
    from retinaface_trt_wrapper import RetinaFaceTRT
    import types

    with tempfile.TemporaryDirectory() as td:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", "5",
            "-i", "/workspace/media/output/test_I_trt_dpm10_tea.mp4",
            "-vframes", "1", f"{td}/f.png",
        ], check=True)
        img = cv2.imread(f"{td}/f.png", cv2.IMREAD_COLOR)

    # PT baseline (this is what facexlib actually runs in detect_faces)
    pt = RetinaFace(network_name="resnet50", half=False, device="cuda")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    pt.load_state_dict(cleaned, strict=True)
    pt.eval().cuda()
    with torch.no_grad():
        pt_bboxes = pt.detect_faces(img.copy(), 0.97)
    print(f"  PT detect_faces: found {len(pt_bboxes)} faces")
    if len(pt_bboxes):
        print(f"    bbox: {pt_bboxes[0][:4]}")

    # swap forward to TRT
    trt_module = RetinaFaceTRT(ENGINE_DIR, original=pt, preload=["fhd"]).cuda().eval()
    def _f(self, x):
        return trt_module(x)
    pt.forward = types.MethodType(_f, pt)

    with torch.no_grad():
        trt_bboxes = pt.detect_faces(img.copy(), 0.97)
    print(f"  TRT detect_faces: found {len(trt_bboxes)} faces")
    if len(trt_bboxes):
        print(f"    bbox: {trt_bboxes[0][:4]}")

    if len(pt_bboxes) == len(trt_bboxes) and len(pt_bboxes) > 0:
        diff = np.abs(np.array(pt_bboxes)[:, :4] - np.array(trt_bboxes)[:, :4])
        print(f"  bbox L1 diff: max={diff.max():.2f}, mean={diff.mean():.2f} px")
        if diff.max() > 5:
            print(f"  WARNING: large diff suggests engine and PT disagree")
        else:
            print(f"  ✓ engine matches PT to within 5 px")


def main():
    onnx_path = export_onnx_fixed()
    engine_path = build_engine(onnx_path)
    smoke_test(engine_path)


if __name__ == "__main__":
    main()
