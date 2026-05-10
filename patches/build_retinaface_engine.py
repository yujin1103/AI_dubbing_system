"""RetinaFace ONNX → TRT FP16 engine builder.

GFPGAN BF16 와 달리 RetinaFace 는 weight^2 sum 같은 overflow 위험이 없어
FP16 사용 가능 (BF16 대비 약간 빠름).
"""
import os
import tensorrt as trt

ONNX = "/workspace/trt_work/onnx/retinaface/retinaface_r50_1080p.onnx"
ENGINE = "/workspace/trt_work/engines/retinaface_r50_1080p_fp16.trt"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)

if not parser.parse_from_file(ONNX):
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    raise RuntimeError("ONNX parse failed")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4 GB
config.set_flag(trt.BuilderFlag.FP16)

print(f"network inputs:")
for i in range(network.num_inputs):
    t = network.get_input(i)
    print(f"  {t.name} dtype={t.dtype} shape={t.shape}")
print(f"network outputs:")
for i in range(network.num_outputs):
    t = network.get_output(i)
    print(f"  {t.name} dtype={t.dtype} shape={t.shape}")

print("building serialized engine...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("engine build returned None")

os.makedirs(os.path.dirname(ENGINE), exist_ok=True)
with open(ENGINE, "wb") as f:
    f.write(serialized)
print(f"wrote {ENGINE} ({os.path.getsize(ENGINE)/1e6:.1f} MB)")
