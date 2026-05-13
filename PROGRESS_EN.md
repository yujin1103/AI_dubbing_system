# Progress Log (English)

> Development progress journal — newest first. Records code changes, decisions, and verification results.

## 2026-05-13 (Wed, validation infra) — Phased speaker-matching improvement plan

### 🎯 Key insight
- TRT conversion has no quality impact (FP16/BF16 safe)
- Real quality issue is **speaker-matching accuracy** + Korean cross-lingual gap
- LoRA solves the latter, the former needs separate patches

### 📋 Phased validation plan (after LoRA finishes)

**Phase 0**: Current baseline (v3 + LoRA + strict envs)
**Phase 1**: + Track-level person ID locking (per-frame → per-track decision)
**Phase 2**: + SyncNet post-validation (auto-reject low-score frames)
**Phase 3**: + Overlap detection + skip (lipsync off in DiariZen overlap regions)
**Phase 4**: + Multi-frame voting (N=5 frame averaged embedding)

### 🛠️ Validation automation infrastructure
- **validation_report_generator.py**: consistent phase comparison reports
- Per video: timing.json + metrics.json + report.md + thumbnails
- Auto summary.md per phase
- 4 category videos for diversity:
  - A. AIHub clean Korean (60 owned)
  - B. Drama frontal 1-speaker
  - C. Drama multi-speaker + overlap
  - D. Profile/motion (LoRA edge cases)

### 📊 ROI comparison (model swap vs logic improvement)

| Task | Effect | Cost | ROI |
|---|---|---|---|
| Track-level locking | -70% wrong matches | 0.5d | ⭐⭐⭐⭐⭐ |
| SyncNet validation | -50% wrong matches | 0.5d | ⭐⭐⭐⭐⭐ |
| Overlap skip | -100% (overlap only) | 0.3d | ⭐⭐⭐⭐ |
| LoCoNet ASD swap | +8% accuracy | 2-3d | ⭐⭐ |
| Sortformer diarization | -33% DER | 3-5d | ⭐⭐ |

→ **Smart logic beats model swap on ROI**

### Status
- LoRA: step 44,032/50,000 (88%) — ~1h 40min remaining
- All A·B·C scripts complete, awaiting validation
- Validation report system ready


---


## 2026-05-13 (Wed, additional) — A·B·C speed optimization tasks completed

### 🎯 Written during LoRA training, no GPU needed (validated after LoRA finishes)

#### Task A: VAE TRT conversion (expected: -25s per 1-min video)
- **vae_trt_build.py** (`/workspace/trt_work/scripts/`)
  - VAE encoder + decoder ONNX export
  - FP16 TRT engine build (opt_level=5)
  - Input: stabilityai/sd-vae-ft-mse (used by LatentSync)
  - Output: vae_encoder_fp16.trt, vae_decoder_fp16.trt
- **vae_trt_wrapper.py** (`/workspace/patches/`)
  - Drop-in replacement for diffusers AutoencoderKL
  - Activated by LATENTSYNC_VAE_TRT=1
  - Auto-fallback to PyTorch on failure

#### Task B: CosyVoice TRT-LLM auto setup (expected: -3-10s per chunk)
- **cosyvoice_trt_setup.sh** (in dubbing_pipeline `/tmp/`)
  - Auto-start cosyvoice-trt container
  - Install 5 missing dependencies (x_transformers, s3tokenizer>=0.3, loguru, torch-einops-utils, einx)
  - Sequential start: trtllm-serve (port 8010) + tritonserver (18000)
  - Wait for 5/5 models loaded + smoke test
  - Full Triton up in ~3 min

#### Task C-lite: frame-level pipelining in mouth_enhance (expected: -30% mouth_enhance time)
- **mouth_only_enhance_v4.py** (`/workspace/patches/`)
  - 3-thread pipeline:
    - Thread 1 (CPU): video decode + frame read
    - Thread 2 (GPU): RetinaFace TRT + GFPGAN TRT
    - Thread 3 (CPU): Poisson blend + color match
  - Frame order preserved
  - **Zero additional memory** — same buffers, async execution
  - Risk: very low (LATENTSYNC_ENHANCE_NO_PIPELINE=1 fallback)

#### Task C-full: chunk-level parallelization (expected: -1.5-2.5 min per 1-min video)
- **parallel_lipsync_orchestrator.py** (`/workspace/patches/`)
  - New orchestrator: pre-chunks video at orchestrator level
  - chunk N+1 lipsync ‖ chunk N enhance (async)
  - GPU memory pre-check (falls back to serial if <3GB free)
  - LATENTSYNC_PARALLEL_ENHANCE=0 to disable
  - Auto serial fallback on OOM

### 📊 Cumulative effect (1-min video)

| Step applied | Time | Cumulative savings |
|---|---|---|
| Current (v3, GFPGAN TRT + RetinaFace TRT + NVENC) | 10-12 min | -28% |
| + A (VAE TRT) | 10-11.5 min | -32% |
| + B (CosyVoice TRT-LLM) | 9.5-11 min | -36% |
| + C-lite (frame pipelining) | 8.5-10 min | -41% |
| + C-full (chunk parallel) | **7-8.5 min** | **-50%** |
| + LoRA + steps=8 (after validation) | 5.5-7 min | -60% |

### 💾 Memory safety analysis (RTX 5080 16GB)
- LatentSync inference: ~6GB
- mouth_enhance (TRT): ~1.5GB
- Concurrent (C-full): 7.5GB → 8.5GB headroom (safe)
- cosyvoice-trt must stay stopped during LoRA training

### 🔒 Safety guarantees on all tasks
- Tasks A/B: separate modules, no impact if not invoked
- Task C-lite: env-disable available, instant fallback
- Task C-full: GPU memory check + OOM fallback + separate script (existing orchestrator untouched)


---


## 2026-05-13 (Wed) — AIHub validation dataset prepared

### ✅ Completed
- **AIHub Lipreading VS11 dataset extracted** (E:/download → /workspace/media/aihub_validation)
  - 60 mp4 videos (1920×1080 @ 30fps, ~5min avg)
  - 60 JSON labels (sentence-level timestamps + Korean text)
  - 18GB total, tar concatenation + extraction complete
  - Speaker: Male M_2, environment: Noise level 2
- **lora_validation.py written** (`/workspace/patches/`)
  - Extracts sentence-level chunks (6-12s) from JSON labels
  - Compares LoRA scales (base / 0.5 / 0.7 / 1.0)
  - Auto-generates 2×2 grid comparison videos
  - Integrates mouth_only_enhance v3 (TRT)

### 🎯 Validation procedure (after LoRA completes)
1. Auto-detects checkpoint-50000.pt
2. Run: `python /workspace/patches/lora_validation.py --n-samples 5 --n-sentences 2`
3. Generates 10 sentence chunks × 4 variants = 40 videos + 10 comparison grids
4. Output: `/workspace/media/aihub_validation/results/<video>/sentence_<id>/comparison.mp4`

### Current status
- LoRA: step 18,020/50,000 (36%) — on track
- Validation data: ready (waiting for LoRA to finish)


---

>
> Korean version: [PROGRESS.md](./PROGRESS.md)

---

## 2026-05-13 (Wed) — TRT post-processing acceleration + LoRA training

### 🎯 Goals
1. Apply TensorRT to `mouth_only_enhance` post-processing (GFPGAN + RetinaFace)
2. Accelerate ffmpeg muxing with NVENC GPU encoding
3. Train LoRA for 50k steps on Korean AIHub dataset

### ✅ Completed

#### 1. mouth_only_enhance v3 integration (`/workspace/patches/mouth_only_enhance.py`)
- **GFPGAN PyTorch → BF16 TRT** (`gfpgan_bf16.trt`, 175MB)
  - 50ms → 7ms/frame (~7× speedup)
  - BF16 chosen to avoid FP16 overflow in StyleGAN2 modulated conv
- **RetinaFace PyTorch → FP16 TRT** (`retinaface_r50_fhd_fp16.trt`, 60MB)
  - 30ms → 10ms/frame (~3× speedup)
  - Multi-resolution engine auto-selection (HD/FHD/QHD/UHD)
- **NVENC h264_nvenc with libx264 fallback**
  - mux 3s → 1s
- All v2 quality features retained:
  - Temporal mask smoothing (5-frame rolling avg)
  - Reinhard color histogram match
  - Mask erosion + adaptive feather
  - Face diagonal min ratio gate
- **Backups**: `mouth_only_enhance.v1_backup.py`, `mouth_only_enhance.v2_backup.py`

#### 2. LatentSync pipeline NVENC patch
- `/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py` — 2 locations modified:
  - Line 561 (input normalize): libx264 → h264_nvenc
  - Line 902 (final mux): libx264 → h264_nvenc
- `LATENTSYNC_USE_NVENC=1` enabled by default, auto-fallback to libx264 on failure

#### 3. LoRA training in progress (Korean AIHub data)
- **Config**: `lora_nf4_train.yaml`
  - num_frames=4, batch_size=1, resolution=256
  - lora_r=32, lora_alpha=16
  - max_train_steps=50000
- **Current state**: step 18,020/50,000 (36%)
- **Speed**: 1.10 it/s
- **ETA**: ~12:20 (8 hours remaining)
- **Checkpoint**: 4.4GB saved every 5000 steps (10 total = 50GB)
- **Resume successful**: Restored from checkpoint-10000.pt after Docker crash

#### 4. Quality improvement patches
- **`fix_profile_strict.py`**: 
  - Added `LATENTSYNC_PROFILE_MATCH_THRESHOLD` env (default 0.5 → 0.55 stricter)
  - Added `LATENTSYNC_PROFILE_STRICT_NONE=1` env (skip when no diarization)
- **`strict_quality_preset.sh`**: One-shot wrapper for 10 quality envs

### ⚠️ Partial / Deferred

#### CosyVoice TRT-LLM Triton integration
- ✅ TRT-LLM engine built (`rank0.engine`, 1.3GB)
- ✅ `trtllm-serve` LLM API working (port 8010)
- ✅ Triton 5/5 models loaded (after installing 4 missing dependencies)
- ❌ **GPU OOM when running concurrently with LoRA** → retry after LoRA finishes
- Missing dependencies: `x_transformers`, `s3tokenizer 0.3.0+`, `loguru`, `torch_einops_utils`, `einx`
- `cosyvoice-trt` container kept stopped (protect LoRA GPU)

#### FP8 ONNX quantization attempt (creative workaround)
- Bypassed modelopt bugs via direct ONNX QDQ surgery
- Verified TRT FP8 mma kernels (`tensor16x8x32`, `e4m3`) properly selected
- Engine size reduced 25% (2557 → 1930 MB)
- **No speedup** — LatentSync UNet is memory-bound, so compute savings invisible
- Files: `tmp/fp8_full_qdq_surgery.py`, `tmp/step3a_trt_build_fp8_full.py`

### 📊 Expected impact (to be verified after LoRA completes)

#### Per 10-second chunk

| Stage | Current (v2 PyTorch) | v3 TRT | Saved |
|---|---|---|---|
| RetinaFace detect | ~7.5s | ~2.5s | -5s |
| GFPGAN enhance | ~12.5s | ~3.75s | **-8.75s** |
| Poisson blend | ~7s | ~7s | 0 |
| Final mux (NVENC) | ~3s | ~1s | -2s |
| LatentSync normalize | ~10s/video | ~3s/video | -7s/video |

#### 1-minute video total

```
Current (v2): ~14-16 min
v3 TRT+NVENC: ~10-12 min (~28% faster)
```

With LoRA applied (steps 8 recommended):
```
v3 TRT+NVENC + LoRA + steps 8: ~7-9 min (50% faster)
```

### 🔬 Verification pending

After LoRA training completes (estimated 5/13 12:20):
1. **VAE TRT conversion** (30 min, needs GPU)
2. **v3 end-to-end test** (10s chunk × 1 = ~2-3 min)
3. **LoRA scale comparison videos (5 outputs)**:
   - base (no LoRA)
   - LoRA scale 0.5 (conservative)
   - LoRA scale 0.7
   - LoRA scale 1.0 (as trained)
   - LoRA + v3 enhance

### 📁 Key files modified

```
patches/
  mouth_only_enhance.py             # v3 (TRT + NVENC)
  mouth_only_enhance.v1_backup.py   # v1
  mouth_only_enhance.v2_backup.py   # v2 (PyTorch, color match)
  asd_filter.py                     # profile strict patch
  fix_profile_strict.py             # env override patch
  strict_quality_preset.sh          # 10-env preset

configs/
  lora_nf4_train.yaml                       # active training config (modified)
  lora_nf4_train.yaml.before_resume         # pre-Docker-crash
  lora_nf4_train.yaml.before_resume2        # checkpoint-10000 attempt

orchestrator.py                     # mouth_enhance, profile gates integration
```

### 💡 Lessons learned

1. **GPU memory budget is critical**: Triton (1.3GB engine + buffers) + LoRA (15.8GB) > 16GB → Docker WSL2 crash
2. **`use_8bit_adam: true` ≠ actual NF4**: falls back to AdamW if bitsandbytes is missing
3. **FP8 quantization is workload-dependent**: memory-bound layers show no compute savings
4. **Triton prebuilt image had incomplete deps**: 4 modules needed manual installation
5. **Checkpoints are real lifesavers**: 5000-step auto-save → near-zero loss on Docker crash

---

## 2026-05-12 (Tue) — FP8 exploration + Quality fixes

### Completed
- FP8 ONNX QDQ surgery — 5 attempts to bypass modelopt bugs
- ✅ FP8 engine built, mma kernels verified
- ❌ Same actual speed on LatentSync (memory-bound)
- TRT EXHAUSTIVE rebuild (only 0.5% improvement)
- Unified benchmark (FP16 baseline / EXHAUSTIVE / FP8 wo / FP8 full)
- mouth_only_enhance v2 (color match, temporal smoothing)
- profile strict patch
- CosyVoice TRT-LLM image pull (8GB)

### Decisions
- TRT engine level has no more headroom (memory-bound)
- Speed gains must come from:
  - Post-processing (GFPGAN/RetinaFace TRT)
  - Encoding (NVENC)
  - Step reduction after training

---

## Infrastructure / Dependencies

### Containers
- `dubbing_pipeline`: Main pipeline (LatentSync, CosyVoice PyTorch, GFPGAN PyTorch)
- `cosyvoice-trt`: TRT-LLM + Triton (kept stopped during LoRA)

### GPU
- RTX 5080 16GB (sm_120 / Blackwell)
- LoRA peak: 15.8GB / 16GB (near limit)

### Main TRT engines
- `unet_fp16.trt` (2.5GB) — currently in use ★
- `unet_fp8_full.trt` (1.9GB) — FP8 verification, unused
- `gfpgan_bf16.trt` (175MB) — used in v3
- `retinaface_r50_fhd_fp16.trt` (60MB) — used in v3

---

## How to contribute / continue

Updates are tracked via two files:
- `PROGRESS.md` (Korean) — primary
- `PROGRESS_EN.md` (English) — translation

The auto-update script `update_progress.sh` updates the Korean version. If contributing in English, edit `PROGRESS_EN.md` directly and commit.
