"""RetinaFace TRT 다중 해상도 엔진 빌드 (HD, FHD, QHD, UHD).

각 해상도별로 별도 ONNX export + TRT FP16 엔진 빌드.
런타임에 입력 영상 해상도 보고 적절한 엔진 자동 선택 (별도 wrapper).
"""
import os
import sys
import time
import torch
import tensorrt as trt

CKPT = "/workspace/gfpgan/weights/detection_Resnet50_Final.pth"
ONNX_DIR = "/workspace/trt_work/onnx/retinaface"
ENGINE_DIR = "/workspace/trt_work/engines"

# 해상도 목록 (이름, H, W)
RESOLUTIONS = [
    ("hd", 720, 1280),       # 720p
    ("fhd", 1080, 1920),     # 1080p
    ("qhd", 1440, 2560),     # 1440p
    ("uhd", 2160, 3840),     # 4K
]


def export_onnx(name: str, h: int, w: int) -> str:
    from facexlib.detection.retinaface import RetinaFace
    onnx_path = os.path.join(ONNX_DIR, f"retinaface_r50_{name}.onnx")
    if os.path.exists(onnx_path):
        print(f"  [skip] ONNX already exists: {onnx_path}")
        return onnx_path
    print(f"  exporting ONNX for {name} ({h}x{w})...")
    m = RetinaFace(network_name="resnet50", half=False, device="cuda")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    m.load_state_dict(ckpt, strict=False)
    m.eval().cuda()
    x = torch.randn(1, 3, h, w, device="cuda")
    with torch.no_grad():
        out = m(x)
    print(f"    forward OK, output shapes: {[tuple(t.shape) for t in out]}")
    torch.onnx.export(
        m, x, onnx_path,
        input_names=["images"],
        output_names=["bbox", "conf", "landmarks"],
        opset_version=17,
        dynamic_axes=None,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    data_path = onnx_path + ".data"
    data_size = os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0
    print(f"    wrote {onnx_path} ({size_mb:.2f} MB graph + {data_size:.0f} MB weights)")
    del m
    torch.cuda.empty_cache()
    return onnx_path


def build_engine(name: str, onnx_path: str) -> str:
    engine_path = os.path.join(ENGINE_DIR, f"retinaface_r50_{name}_fp16.trt")
    if os.path.exists(engine_path):
        print(f"  [skip] engine already exists: {engine_path}")
        return engine_path
    print(f"  building TRT engine for {name}...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"    ERROR: {parser.get_error(i)}")
        raise RuntimeError(f"ONNX parse failed for {name}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
    config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {name}")
    build_time = time.time() - t0
    os.makedirs(ENGINE_DIR, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"    wrote {engine_path} ({size_mb:.1f} MB, {build_time:.0f}s)")
    return engine_path


def smoke_test(name: str, h: int, w: int, engine_path: str):
    """엔진 forward + shape 검증."""
    print(f"  smoke test for {name}...")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    ctx.set_input_shape("images", (1, 3, h, w))
    # anchor 개수 계산: 2 * (h*w/64 + h*w/256 + h*w/1024) = h*w * 42 / 1024
    expected_anchors = h * w * 42 // 1024
    inp = torch.randn(1, 3, h, w, device="cuda", dtype=torch.float32)
    # output bindings — query shape from engine
    out_bbox_shape = tuple(ctx.get_tensor_shape("bbox"))
    out_conf_shape = tuple(ctx.get_tensor_shape("conf"))
    out_lmk_shape = tuple(ctx.get_tensor_shape("landmarks"))
    out_bbox = torch.empty(out_bbox_shape, dtype=torch.float32, device="cuda")
    out_conf = torch.empty(out_conf_shape, dtype=torch.float32, device="cuda")
    out_lmk = torch.empty(out_lmk_shape, dtype=torch.float32, device="cuda")
    ctx.set_tensor_address("images", inp.data_ptr())
    ctx.set_tensor_address("bbox", out_bbox.data_ptr())
    ctx.set_tensor_address("conf", out_conf.data_ptr())
    ctx.set_tensor_address("landmarks", out_lmk.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    # warmup
    for _ in range(3):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    # timing
    t0 = time.time()
    for _ in range(20):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) / 20 * 1000
    actual_anchors = out_bbox_shape[1]
    ratio_ok = abs(actual_anchors - expected_anchors) / expected_anchors < 0.05
    print(f"    forward: {fwd_ms:.2f} ms")
    print(f"    output bbox={out_bbox_shape}, conf={out_conf_shape}, lmk={out_lmk_shape}")
    print(f"    anchors {actual_anchors} (expected ~{expected_anchors}, {'OK' if ratio_ok else 'MISMATCH'})")
    return fwd_ms


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(ENGINE_DIR, exist_ok=True)
    results = []
    for name, h, w in RESOLUTIONS:
        print(f"\n=== {name.upper()} ({h}x{w}) ===")
        try:
            onnx_path = export_onnx(name, h, w)
            engine_path = build_engine(name, onnx_path)
            fwd_ms = smoke_test(name, h, w, engine_path)
            results.append((name, h, w, engine_path, fwd_ms, "OK"))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, h, w, None, None, f"FAILED: {e}"))
    print("\n\n=== 결과 요약 ===")
    print(f"{'name':<6} {'shape':<14} {'engine size':<12} {'fwd ms':<10} {'status'}")
    print("-" * 60)
    for name, h, w, eng, fwd_ms, status in results:
        size_str = f"{os.path.getsize(eng)/1e6:.1f} MB" if eng and os.path.exists(eng) else "—"
        fwd_str = f"{fwd_ms:.2f}" if fwd_ms else "—"
        print(f"{name:<6} {h}x{w:<10} {size_str:<12} {fwd_str:<10} {status}")


if __name__ == "__main__":
    main()
