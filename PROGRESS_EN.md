# Progress Log (English)

> Development progress journal — newest first. Records code changes, decisions, and verification results.
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
