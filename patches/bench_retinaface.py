"""RetinaFace TRT FP16 engine vs PyTorch FP32 속도 벤치마크."""
import time
import torch
import numpy as np
import tensorrt as trt
from facexlib.detection.retinaface import RetinaFace

CKPT = "/workspace/gfpgan/weights/detection_Resnet50_Final.pth"
ENGINE = "/workspace/trt_work/engines/retinaface_r50_fp16.trt"
N_WARMUP = 5
N_ITER = 30
H, W = 720, 1280


def bench_pytorch():
    m = RetinaFace(network_name="resnet50", half=False, device="cuda")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    m.load_state_dict(ckpt, strict=False)
    m.eval().cuda()
    x = torch.randn(1, 3, H, W, device="cuda")
    # warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = m(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        with torch.no_grad():
            _ = m(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / N_ITER * 1000


def bench_trt():
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(ENGINE, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    ctx.set_input_shape("images", (1, 3, H, W))
    # bind buffers
    inp = torch.randn(1, 3, H, W, device="cuda", dtype=torch.float32)
    out_bbox = torch.empty(1, 37840, 4, device="cuda", dtype=torch.float32)
    out_conf = torch.empty(1, 37840, 2, device="cuda", dtype=torch.float32)
    out_lmk = torch.empty(1, 37840, 10, device="cuda", dtype=torch.float32)
    ctx.set_tensor_address("images", inp.data_ptr())
    ctx.set_tensor_address("bbox", out_bbox.data_ptr())
    ctx.set_tensor_address("conf", out_conf.data_ptr())
    ctx.set_tensor_address("landmarks", out_lmk.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    # warmup
    for _ in range(N_WARMUP):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    return (time.time() - t0) / N_ITER * 1000


def main():
    pt_ms = bench_pytorch()
    print(f"PyTorch FP32: {pt_ms:.1f} ms/forward")
    trt_ms = bench_trt()
    print(f"TRT FP16:     {trt_ms:.1f} ms/forward")
    print(f"Speedup:      {pt_ms/trt_ms:.2f}x  ({(1-trt_ms/pt_ms)*100:.0f}% saved)")


if __name__ == "__main__":
    main()
