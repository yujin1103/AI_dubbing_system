"""RetinaFace (ResNet50) → ONNX export.

facexlib 의 RetinaFace 를 static shape (720x1280) 로 ONNX export.
1080p 입력 영상은 720p 로 다운스케일해서 추론하고, 좌표만 원본 해상도로 scale-up.

출력: bbox (1, N, 4), conf (1, N, 2), landmarks (1, N, 10).
"""
import os
import torch
from facexlib.detection.retinaface import RetinaFace

DEVICE = "cuda"
H, W = 1080, 1920  # static input — facexlib feeds original 1080p as-is
CKPT = "/workspace/gfpgan/weights/detection_Resnet50_Final.pth"
OUT_ONNX = "/workspace/trt_work/onnx/retinaface/retinaface_r50_1080p.onnx"

os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)

m = RetinaFace(network_name="resnet50", half=False, device=DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
m.load_state_dict(ckpt, strict=False)
m.eval().to(DEVICE)

x = torch.randn(1, 3, H, W, device=DEVICE)
with torch.no_grad():
    out = m(x)
print("forward OK, shapes:", [tuple(t.shape) for t in out])

torch.onnx.export(
    m, x, OUT_ONNX,
    input_names=["images"],
    output_names=["bbox", "conf", "landmarks"],
    opset_version=17,
    dynamic_axes=None,  # static shape
)
print(f"wrote {OUT_ONNX}")
print(f"size: {os.path.getsize(OUT_ONNX) / 1e6:.1f} MB")
