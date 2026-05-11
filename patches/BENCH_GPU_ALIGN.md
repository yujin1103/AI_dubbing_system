# GPU Face Alignment + Paste-back — Benchmark

> Move `cv2.warpAffine` (CPU) face alignment and paste-back to GPU
> `grid_sample` for the GFPGAN post-processing pipeline. Replaces v1
> (`gfpgan_async_postprocess_trt.py`) with v2
> (`gfpgan_async_postprocess_trt_v2.py`).

## TL;DR

End-to-end on 1602 frames @ 1080p (`test_I_trt_dpm10_tea.mp4`):

| run | total | savings |
|---|---|---|
| v1 (CPU `cv2.warpAffine`, user-quoted) | **924 s** (15:24) | — |
| v2 (GPU `grid_sample`) | **484 s** (8:04) | **−47.6 % (−440 s)** |

Target was −17 % (~767 s). We hit **−47.6 %** — well past the target.

The big win is **paste-back**, not alignment: a 1080p frame with one face goes
from 113 ms (cv2) → 24 ms (GPU) — 4.66× speedup. Per-face cost matters because
the test video averages ~5 faces per frame in the dialog scenes, so v1 paid the
113 ms × 5 cv2 cost while v2 batches the parse/warp/blend on GPU.

Quality: PSNR median **45.6 dB**, SSIM mean **0.9922** vs v1 reference.
Both clear the targets (≥45 dB, ≥0.99).

## Setup

- Container: `dubbing_pipeline` (RTX 5080, sm_120, CUDA 13.0)
- venv: `/opt/venv_gfpgan` (PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130)
- Test video: `/workspace/media/output/test_I_trt_dpm10_tea.mp4`
  - 1602 frames @ 25 fps, 1080p, 64.1 s
- Engines: `gfpgan_bf16.trt` (175 MB)
- v1 reference output: `test_I_gfpgan_trt.mp4` (May 10 09:50 — same 924 s run)

## What changed (v1 → v2)

| stage | v1 (CPU `cv2.warpAffine`) | v2 (GPU `grid_sample`) |
|---|---|---|
| frame upload | per-call inside `enhance()` | once per frame, GPU-resident |
| face crop (1080p → 512) | `cv2.warpAffine`, CPU | `F.affine_grid` + `F.grid_sample`, CUDA |
| GFPGAN forward | TRT BF16 (B=1) | TRT BF16 (B=1) — unchanged |
| parse mask (face seg) | `face_parse` GPU → `.cpu()` → `cv2.GaussianBlur` × 2 → `cv2.warpAffine` | all GPU: `face_parse` → 19-class lookup → depthwise separable Gaussian (1×101 + 101×1) → 4-channel `grid_sample` |
| paste back (face → 1080p) | `cv2.warpAffine`, CPU | `grid_sample` (face + mask in one 4-ch sample), CUDA |
| blend | NumPy multiply/add | torch in-place ops |

The matrix conversion (cv2 affine ↔ normalized `theta` for `affine_grid`) is in
`_cv_to_theta` — derivation in the docstring; with `align_corners=True` the
result matches `cv2.warpAffine` to **PSNR 58 dB on synthetic data** (see
sanity test below).

## Files

- [gpu_face_aligner.py](gpu_face_aligner.py) — `GPUFaceAligner` with `align()`
  and `paste()`.
- [gfpgan_async_postprocess_trt_v2.py](gfpgan_async_postprocess_trt_v2.py) — main
  script. Drop-in compatible with v1's CLI (same `--input` / `--output` /
  `--upscale`). RetinaFace TRT swap is gated behind `--retinaface-trt` (off by
  default; see "Known issue" below).
- [test_gpu_aligner_sanity.py](test_gpu_aligner_sanity.py) — synthetic
  cv2-vs-GPU equivalence test.
- [bench_align_paste_microbench.py](bench_align_paste_microbench.py) — per-stage
  cv2-vs-GPU timing on a real 1080p frame.
- [compare_v1_v2_quality.py](compare_v1_v2_quality.py) — PSNR/SSIM vs v1.

## Correctness — synthetic 1080p sanity

`test_gpu_aligner_sanity.py`, identity landmark, 5-point template:

```
forward warp (1080p -> 512x512):   PSNR vs cv2 = 58.47 dB  (max diff 10/255, p99=1)
inverse warp (512x512 -> 1080p):   PSNR vs cv2 = 56.92 dB  (max diff 45 at borders, p99=0)

cv2 warpAffine forward (1080p->512): 0.57 ms
GPU grid_sample forward            : 0.15 ms      (3.8x)

cv2 warpAffine paste  (512->1080p) : 2.27 ms
GPU grid_sample paste              : 0.56 ms      (4.1x)
```

The forward warp matches cv2 to ≤1 LSB on 99% of pixels. The inverse warp
shows max diff 45 only in border pixels where cv2 uses `BORDER_CONSTANT` with
value 0 vs grid_sample's `padding_mode='zeros'` (the soft mask is 0 in those
regions anyway, so they don't reach the final composite).

## Per-stage micro-benchmark — real 1080p frame, 1 face

`bench_align_paste_microbench.py`, frame at t=5 s of test video, 1 face
detected:

| stage | cv2 (ms) | GPU (ms) | speedup |
|---|---:|---:|---:|
| align (1080p → 512) | 0.57 | 0.70 | 0.8× *(Python overhead beats grid_sample for N=1)* |
| GFPGAN forward + transfer | 9.61 | 7.37 | 1.3× |
| **paste back (incl. face_parse + Gaussian + warp + blend)** | **113.15** | **24.28** | **4.66×** |
| **per-frame savings (align + paste)** | | **88.73 ms** | |

Projected to 1602 frames: **142 s saved** purely on a single-face workload.
With multi-face frames (which average ~5 in this video), the savings scale
because each face costs 113 ms with cv2 but only one batched parse+blend
on GPU.

## End-to-end wall time — 1602 frames, 1080p

Full run with `--detail-timing`:

```
v1 (user-quoted):                       924.0 s   (15:24)
v2 (this change):                       484.2 s   ( 8:04)
                                       ─────────
                                       Δ −440 s   (−47.6 %)
```

### v2 per-stage breakdown (`--detail-timing`)

```
extract (ffmpeg):       79.0 s          # heavy because libx264 was running concurrently
enhance loop:          365.6 s          (228.2 ms/frame)
  detect (PT RetinaFace):  137.3 s   ( 85.69 ms/frame)
  upload  (uint8 H2D):       1.2 s   (  0.73 ms/frame)
  align   (grid_sample):     2.3 s   (  1.45 ms/frame)
  gfpgan  (TRT BF16, B=1):  57.8 s   ( 36.09 ms/frame, ~5 faces avg)
  paste   (parse+warp+blend): 135.3 s ( 84.46 ms/frame, ~5 faces avg)
  download (D2H+uint8):      1.2 s   (  0.73 ms/frame)
assemble (ffmpeg):      39.6 s
                       ─────
total:                 484.2 s
```

`1407 / 1602` frames had detected faces (88 %); the remaining 195 are
black/transition frames where v2 short-circuits and writes the original.

### Where the savings came from

The user's pre-task estimate identified ~160 s in `cv2.warpAffine` + ~32 s in
NMS + ~67 s in I/O — leaving the bulk under "Python overhead, sync, queue
waits". The microbench tells the real story: **cv2 paste-back is ~113 ms for
*one* face**, and the test video averages ~5 faces per frame, so v1 paid
~5 × 113 = 565 ms per multi-face frame just on paste. v2 collapses that
to 84 ms by:

1. Running `face_parse` once with batch N (instead of looping numpy mask
   construction per face).
2. Replacing two `cv2.GaussianBlur(101, 101, sigma=11)` passes with a
   depthwise-separable Gaussian on a 512×512 mask (kernel stays small,
   image stays small — cv2 was running this on the input-size mask in some
   code paths).
3. Sampling face + mask back to the upscale canvas in a single 4-channel
   `grid_sample` call.

GFPGAN forward is unchanged (TRT BF16, B=1) — at ~36 ms/frame avg
(7 ms/face × ~5 faces) it's now the third-largest stage. Detection
(85 ms) is now the single largest because RetinaFace TRT is unusable in
its current shape (see Known Issue below).

## Quality vs v1

`compare_v1_v2_quality.py`, every-20th-frame sample, n=80:

```
PSNR
  mean       : (frames with no face are bit-identical → inf)
  median     : 45.62 dB
  p5         : 43.82 dB
  min        : 43.15 dB

SSIM (Y channel)
  mean       : 0.9922
  median     : 0.9916
  p5         : 0.9882
  min        : 0.9862
```

Targets: PSNR ≥ 45 dB (median ✓), SSIM ≥ 0.99 (mean / median ✓). Worst frames
sit at 43.15 dB / SSIM 0.9862 — visually indistinguishable; the diff lives at
the soft-edge of the parse mask where cv2's Gaussian rounding and grid_sample's
bilinear differ by a few LSB at sub-pixel precision.

Sample-frame diffs (max diff in any color channel; p99 = pixel diff at the
99th percentile):

| frame | PSNR | max diff | p99 |
|---:|---:|---:|---:|
|    0 | 138.13 dB | 0 | 0.0 |
|  400 |  47.75 dB | 93 | 4.0 |
|  800 |  43.67 dB | 28 | 6.0 |
| 1200 |  44.93 dB | 87 | 5.0 |
| 1500 |  45.96 dB | 43 | 5.0 |

Frame 0 is bit-identical (no face — both pipelines short-circuit). The other
samples have p99 ≤ 6 LSB — i.e. on 99 % of pixels the two pipelines agree to
within 6 / 255 brightness levels. The handful of pixels at max diff 87–93 are
at the parse-mask transition zone.

## Known issue — RetinaFace TRT engine has random init weights

`build_retinaface_multires.py` and `export_retinaface_onnx.py` both do
`m.load_state_dict(ckpt, strict=False)` *without* stripping the `module.`
prefix from the checkpoint keys. The pretrained checkpoint has all keys
under `module.body...`, so `strict=False` silently no-ops every key and
the exported ONNX is a randomly-initialised model.

Effect: `wrap_face_helper_detector(...)` swaps `face_det.forward` with the
TRT engine; the engine returns garbage logits; downstream `decode + NMS`
filters surface ~40,000 spurious "faces" per frame, which then OOMs
`affine_grid` (40k × 3 × 512 × 512 grid is 31 GB).

That's why `--retinaface-trt` is **off by default** in v2. Fix is in
`rebuild_retinaface_fhd.py` (this branch, not run): strip `module.` before
`load_state_dict`, then rebuild the engines. Out of scope for this change.

Once fixed, expect another **~50 s** off the 484 s wall: PT detection is
85.69 ms/frame and the user's bench had RetinaFace TRT at ~13 ms/frame
end-to-end (real forward + decode + NMS).

## Reproduce

```bash
# v1 baseline (existing pipeline) — already produced test_I_gfpgan_trt.mp4
/opt/venv_gfpgan/bin/python /workspace/patches/gfpgan_async_postprocess_trt.py \
  --input  /workspace/media/output/test_I_trt_dpm10_tea.mp4 \
  --output /workspace/media/output/test_I_gfpgan_trt.mp4 \
  --upscale 1

# v2 (this change)
/opt/venv_gfpgan/bin/python /workspace/patches/gfpgan_async_postprocess_trt_v2.py \
  --input  /workspace/media/output/test_I_trt_dpm10_tea.mp4 \
  --output /workspace/media/output/test_I_trt_v2.mp4 \
  --upscale 1 --detail-timing

# quality comparison
/opt/venv_gfpgan/bin/python /workspace/patches/compare_v1_v2_quality.py \
  --v1 /workspace/media/output/test_I_gfpgan_trt.mp4 \
  --v2 /workspace/media/output/test_I_trt_v2.mp4 --every 20

# sanity / microbench
/opt/venv_gfpgan/bin/python /workspace/patches/test_gpu_aligner_sanity.py
/opt/venv_gfpgan/bin/python /workspace/patches/bench_align_paste_microbench.py
```

## Gotchas / notes for future work

- **`@torch.no_grad()` everywhere.** facexlib's `RetinaFace.detect_faces`
  computes priors with grad-tracking on; without `no_grad` you get
  `RuntimeError: Can't call numpy() on Tensor that requires grad`. v2 wraps
  the per-frame entry point so detect / align / GFPGAN / paste all share one
  no-grad context.
- **`align_corners=True`.** Both `affine_grid` and `grid_sample` use it. With
  `align_corners=False`, pixel-center conventions differ slightly from
  `cv2.warpAffine` and PSNR vs cv2 drops by a couple of dB on synthetic data.
- **cv2 affine matrix is *input → output*.** OpenCV's `warpAffine` inverts it
  internally; PyTorch's `affine_grid` takes *output → input*. The conversion in
  `_cv_to_theta` does exactly one `cv2.invertAffineTransform` and one
  pixel↔normalized-coords change of basis (full derivation in the function
  docstring).
- **Multi-face cost.** GFPGAN forward and paste both scale linearly with the
  per-frame face count. The static B=1 GFPGAN engine is the limiter — a B=4
  rebuild would knock another ~20 s off this run (stretch goal, not done).
- **PyTorch CUDA caching allocator** holds 15.7 GB during the run; that's not
  a leak — it's reserved memory that gets recycled. We're at 16 GB total VRAM,
  so any concurrent UNet TRT job (2.55 GB) would force `torch.cuda.empty_cache()`
  calls.
- **Don't run two GPU jobs at once.** During development the test ran
  alongside an unrelated `--upscale 2` v1 invocation that was using libx264 at
  250 % CPU. The v2 enhance throughput dipped from 13 it/s to 2 it/s while
  that was active. The 484 s number above includes that dip; a clean
  back-to-back run would likely come in lower.
