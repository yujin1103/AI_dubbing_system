# GFPGAN v1.4 → TensorRT (BF16)

**Hardware:** RTX 5080, sm_120 (Blackwell), CUDA 13.0
**Software:** TensorRT 10.16.1.11, PyTorch 2.11.0+cu130, container `dubbing_pipeline`
**Test clip:** `/workspace/media/output/test_I_trt_dpm10_tea.mp4` — 1602 frames, 25 fps, 1080p, 64.06 s

## What runs on TRT

Only the StyleGAN2 generator forward (`GFPGANv1Clean.forward`) is replaced by a TRT engine. Face detection (RetinaFace), 5-point alignment, and paste-back stay on the existing PyTorch `face_helper`. The async I/O pipeline (reader thread + write pool) from `gfpgan_async_postprocess.py` is preserved unchanged.

```
[mp4] → ffmpeg extract → reader thread → cv2.imread → face_helper detect+align (PyTorch)
                                                          ↓
                                                   TRT BF16 generator (replaces PyTorch StyleGAN2)
                                                          ↓
                                                  face_helper paste-back → write pool → ffmpeg assemble
```

## Build pipeline

### 1. PyTorch → ONNX  (FP32, static shape)

[`/workspace/trt_work/onnx/gfpgan/export_gfpgan_onnx.py`](workspace/trt_work/onnx/gfpgan/export_gfpgan_onnx.py)

Wraps `GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)` and forces `randomize_noise=False, return_rgb=False, return_latents=False` so the StyleGAN2 noise tensors become deterministic constants and the graph has one image output. Loads `params_ema` from `GFPGANv1.4.pth`.

```bash
/opt/venv_gfpgan/bin/python /workspace/trt_work/onnx/gfpgan/export_gfpgan_onnx.py
# → /workspace/trt_work/onnx/gfpgan/gfpgan.onnx       (1.0 MB metadata, opset 18)
# → /workspace/trt_work/onnx/gfpgan/gfpgan.onnx.data  (340 MB FP32 external weights)
```

`onnxruntime` CPU (FP32) vs the same PyTorch model on a real face crop: **PSNR 73.4 dB**. The ONNX is a faithful representation.

### 2. ONNX → TRT engine

[`/workspace/trt_work/build_gfpgan_engine.py`](workspace/trt_work/build_gfpgan_engine.py)

```bash
/opt/venv_gfpgan/bin/python /workspace/trt_work/build_gfpgan_engine.py            # BF16 (default)
/opt/venv_gfpgan/bin/python /workspace/trt_work/build_gfpgan_engine.py --precision fp16  # FP16, broken
# → /workspace/trt_work/engines/gfpgan_bf16.trt   (175 MB, ~32 s build)
# → /workspace/trt_work/engines/gfpgan_fp16.trt   (180 MB, ~80 s build)
```

`trtexec` in this image is TensorRT 10.15 but the `tensorrt` Python wheel installed in `venv_gfpgan` is 10.16 — engines built by 10.15 fail to deserialise on 10.16 (`Error 6: API Usage Error … please rebuild`). The build script uses the Python builder API at 10.16 to match the runtime.

### 3. Why BF16, not FP16

Tested on a real 512×512 face crop (PT `randomize_noise=False` vs TRT, both with FP32 I/O buffers, RTX 5080):

| Precision | tensor PSNR | uint8 PSNR | output range | comment |
|---|---|---|---|---|
| ONNX FP32 (ORT-CPU) | 73.4 dB | — | (−1.009, +0.675) | matches PT |
| TRT BF16 | **51.4 dB** | **49.8 dB** | (−1.016, +0.668) | indistinguishable |
| TRT FP16 | 13.9 dB | 13.9 dB | (−0.503, −0.198) | broken |

StyleGAN2's `ModulatedConv2d` computes `weight.pow(2).sum([2,3,4])` then `rsqrt(..)` for demodulation. Real-face activations push intermediate squared sums past the FP16 max and the demodulator collapses; the network output narrows to a near-constant range. BF16 has the same exponent as FP32 and is rock-solid here. Speed cost on RTX 5080: BF16 ~7.1 ms/frame vs FP16 ~5.8 ms/frame — both far below the PyTorch generator forward.

### 4. Engine I/O is FP32

`BuilderFlag.BF16` (or `.FP16`) only affects internal kernel precision — engine input/output bindings stay in the dtype of the ONNX graph (FP32). Passing FP16 tensor pointers reinterprets bits and produces nonsense (we saw output saturate at ±512.0). The wrapper allocates **FP32** persistent I/O buffers.

### 5. NVIDIA_TF32_OVERRIDE consistency

TRT bakes the value of `NVIDIA_TF32_OVERRIDE` into the engine and refuses to deserialise into a process where the value differs (`Error 1: Myelin … Inconsistent setting of NVIDIA_TF32_OVERRIDE env var at build N and at execution M`). The container has `NVIDIA_TF32_OVERRIDE=1` set globally; we leave it alone, build inside the container, and engines deserialise cleanly inside the same container. Don't override the env var when running scripts.

## Wall-clock results (full pipeline, end-to-end)

Both runs used the same input video, the same `--upscale 1`, and the same async-I/O pipeline. The only difference is the StyleGAN2 generator forward (PyTorch vs TRT BF16).

| Path | enhance loop | extract | assemble | TRT-script total | per-frame (enhance) |
|---|---|---|---|---|---|
| PyTorch (`gfpgan_async_postprocess.py`)        | **16:49** (1009 s) | ~31 s | ~38 s | ~17:58 (1078 s) | 630 ms |
| TRT BF16 (`gfpgan_async_postprocess_trt.py`)   | **14:16** (855.6 s) | 30.8 s | 37.6 s | **15:24 (924 s)** | **534 ms** |
| **savings** | **2:33** | — | — | **2:34** | **96 ms / frame (15%)** |

Per-frame enhance speed-up: **1.18x** (PT 630 ms → TRT 534 ms).
Total wall speed-up: **~1.17x**.

This is below the 2x target stated in the brief. Why: the StyleGAN2 generator forward is only a fraction of the per-frame budget. The TRT engine itself runs at ~7 ms/frame, but the per-frame enhance step also includes:

- RetinaFace face detection on 1080p (PyTorch, unchanged, dominant cost)
- 5-point landmark + similarity-warp alignment (CPU)
- Paste-back of restored 512 onto 1080p (CPU)
- I/O (cv2.imread/imwrite, partly threaded)

To get further wall-clock gains, the next-biggest target is RetinaFace; replacing it with a TRT engine would likely double the speed-up. (Out of scope here — the brief was explicit about replacing only the StyleGAN2 generator.)

There is a slow segment in both runs around frames 1153–1290 (~140 frames at ~2 s/frame in TRT, similar pattern in PT) — likely a multi-face / motion-heavy scene where face detection dominates and there are 2 generator calls per frame. The TRT generator stays fast across this segment; the slowdown is in face_helper.

## Quality

PSNR between PyTorch-restored and TRT-restored MP4s, sampled across the clip:

| t | PSNR | mean abs diff |
|---|---|---|
|  1 s | 45.54 dB | 0.72 / 255 |
| 15 s | 48.26 dB | 0.43 / 255 |
| 30 s | 45.82 dB | 0.68 / 255 |
| 45 s | 43.90 dB | 0.84 / 255 |
| 60 s | 46.18 dB | 0.57 / 255 |

44–48 dB is well above the visual-difference threshold. Most of the residual difference is PyTorch's `randomize_noise=True` (default) injecting fresh `torch.randn` noise per frame, while the ONNX export froze the noise. Visually no regression in spot checks (`/tmp/qa/orig_t1.png`, `trt_video_t1.png`, `trt_video_t30.png`).

## Files

| Path | Size | Purpose |
|---|---|---|
| `/workspace/trt_work/onnx/gfpgan/export_gfpgan_onnx.py` | 3 KB | PyTorch → ONNX export |
| `/workspace/trt_work/onnx/gfpgan/gfpgan.onnx` (+`.data`) | 1 MB + 340 MB | FP32 ONNX, external weights |
| `/workspace/trt_work/build_gfpgan_engine.py` | 3 KB | ONNX → TRT builder (Python API, 10.16) |
| `/workspace/trt_work/engines/gfpgan_bf16.trt` | 175 MB | **BF16 engine (default, recommended)** |
| `/workspace/trt_work/engines/gfpgan_fp16.trt` | 180 MB | FP16 engine (broken — kept only as proof) |
| `/workspace/patches/gfpgan_trt_wrapper.py` | 7 KB | `GFPGANTRT` drop-in for `GFPGANer.gfpgan` |
| `/workspace/patches/gfpgan_async_postprocess_trt.py` | 8 KB | TRT-enabled async post-process |
| `/workspace/patches/gfpgan_async_postprocess.py` | 6 KB | original PyTorch path, **untouched** |

## Usage

```bash
/opt/venv_gfpgan/bin/python /workspace/patches/gfpgan_async_postprocess_trt.py \
    --input  /workspace/media/output/test_I_trt_dpm10_tea.mp4 \
    --output /workspace/media/output/test_I_gfpgan_trt.mp4 \
    --upscale 1
```

Override engine by env or CLI:
```bash
GFPGAN_TRT_ENGINE=/path/to/other.trt /opt/venv_gfpgan/bin/python ...
# or
... --engine /path/to/other.trt
```

The PyTorch path remains available as `gfpgan_async_postprocess.py` (unchanged) for fallback or comparison runs.
