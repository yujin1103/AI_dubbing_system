# Progress Log

> 진행사항 / 작업 일지 — 최신순. 코드 변경 + 결정 + 검증 결과 기록.

## 2026-05-13 (수, 추가) — A·B·C 시간 절감 작업 완료

### 🎯 LoRA 학습 동안 GPU 없이 작성 (LoRA 끝난 후 검증)

#### Task A: VAE TRT 변환 (예상 절감: -25s/1분 영상)
- **vae_trt_build.py** (`/workspace/trt_work/scripts/`)
  - VAE encoder + decoder ONNX export
  - FP16 TRT engine 빌드 (opt_level=5)
  - 입력: stabilityai/sd-vae-ft-mse (LatentSync 사용)
  - 출력: vae_encoder_fp16.trt, vae_decoder_fp16.trt
- **vae_trt_wrapper.py** (`/workspace/patches/`)
  - diffusers AutoencoderKL drop-in 교체
  - LATENTSYNC_VAE_TRT=1 로 활성화
  - 실패 시 PyTorch fallback 자동

#### Task B: CosyVoice TRT-LLM 자동 셋업 (예상 절감: -3-10s/chunk)
- **cosyvoice_trt_setup.sh** (`/tmp/` in dubbing_pipeline)
  - cosyvoice-trt 컨테이너 자동 시작
  - 누락 의존성 5개 자동 설치 (x_transformers, s3tokenizer>=0.3, loguru, torch-einops-utils, einx)
  - trtllm-serve (port 8010) + tritonserver (18000) 순차 시작
  - 5/5 모델 로드 대기 + smoke test
  - 한 번에 실행하면 ~3분이면 Triton 가동

#### Task C-lite: mouth_enhance 내부 frame-level pipelining (예상 절감: -30% mouth_enhance)
- **mouth_only_enhance_v4.py** (`/workspace/patches/`)
  - 3-thread 파이프라인:
    - Thread 1 (CPU): video decode + frame read
    - Thread 2 (GPU): RetinaFace TRT + GFPGAN TRT
    - Thread 3 (CPU): Poisson blend + color match
  - Frame order preserved (sequential write)
  - **메모리 추가 0** — 같은 버퍼, 비동기 실행만
  - 위험: 매우 낮음 (LATENTSYNC_ENHANCE_NO_PIPELINE=1 로 fallback)

#### Task C-full: chunk-level 병렬화 (예상 절감: -1.5-2.5분/1분 영상)
- **parallel_lipsync_orchestrator.py** (`/workspace/patches/`)
  - 새 orchestrator: video를 미리 chunk 단위로 분할
  - chunk N+1 lipsync ‖ chunk N enhance (async)
  - GPU 메모리 사전 체크 (3GB 미만이면 serial fallback)
  - LATENTSYNC_PARALLEL_ENHANCE=0 으로 비활성화 가능
  - OOM 시 자동 serial fallback

### 📊 누적 효과 (1분 영상 기준)

| 적용 단계 | 시간 | 누적 절감 |
|---|---|---|
| 현재 (v3, GFPGAN TRT + RetinaFace TRT + NVENC) | 10-12분 | -28% |
| + A (VAE TRT) | 10-11.5분 | -32% |
| + B (CosyVoice TRT-LLM) | 9.5-11분 | -36% |
| + C-lite (frame pipelining) | 8.5-10분 | -41% |
| + C-full (chunk parallel) | **7-8.5분** | **-50%** |
| + LoRA + steps=8 (검증 후) | 5.5-7분 | -60% |

### 💾 메모리 안전 분석 (RTX 5080 16GB)
- LatentSync 추론: ~6GB
- mouth_enhance (TRT): ~1.5GB
- 동시 실행 (C-full): 7.5GB → 8.5GB 여유 (안전)
- LoRA 학습 중에는 cosyvoice-trt stopped 유지 필수

### 🔒 모든 작업 안전 장치
- Task A/B: 별도 모듈, 호출 안 하면 영향 0
- Task C-lite: env로 비활성화 가능, fallback 즉시 가능
- Task C-full: GPU 메모리 체크 + OOM fallback + 별도 스크립트 (기존 orchestrator 미수정)


---


## 2026-05-13 (수) — AIHub 검증 데이터 준비

### ✅ 완료
- **AIHub 립리딩 VS11 데이터셋 추출** (E:/download → /workspace/media/aihub_validation)
  - 60개 mp4 영상 (1920×1080 @ 30fps, 평균 5분)
  - 60개 JSON 라벨 (sentence-level timestamp + 한국어 텍스트)
  - 총 18GB → tar concatenation + extraction 완료
  - 화자: 남성 M(남성)_2, 환경: 소음환경2
- **lora_validation.py 작성** (`/workspace/patches/`)
  - JSON label에서 sentence 단위로 chunk 추출 (6-12초)
  - LoRA scale별 비교 (base / 0.5 / 0.7 / 1.0)
  - 2×2 grid comparison video 자동 생성
  - mouth_only_enhance v3 (TRT) 통합

### 🎯 LoRA 학습 완료 후 검증 절차
1. checkpoint-50000.pt 자동 사용
2. `python /workspace/patches/lora_validation.py --n-samples 5 --n-sentences 2`
3. 10개 sentence chunk × 4 variant = 40개 비디오 + 10개 grid 비교 영상
4. 결과: `/workspace/media/aihub_validation/results/<video>/sentence_<id>/comparison.mp4`

### 진행 상황
- LoRA: step 18,020/50,000 (36%) — 정상 진행
- 검증 데이터: 준비 완료 (대기 중)


---


---

## 2026-05-13 (수) — TRT 후처리 가속 + LoRA 학습

### 🎯 목표
1. mouth_only_enhance 후처리에 TRT 적용 (GFPGAN + RetinaFace)
2. NVENC GPU 인코딩으로 ffmpeg muxing 가속
3. LoRA 50k step 학습 (한국어 AIHub 데이터)

### ✅ 완료된 작업

#### 1. mouth_only_enhance v3 통합 (`/workspace/patches/mouth_only_enhance.py`)
- **GFPGAN PyTorch → BF16 TRT** (`gfpgan_bf16.trt`, 175MB)
  - 50ms → 7ms/frame (~7× 가속)
  - StyleGAN2 modulated conv의 FP16 overflow 회피 위해 BF16
- **RetinaFace PyTorch → FP16 TRT** (`retinaface_r50_fhd_fp16.trt`, 60MB)
  - 30ms → 10ms/frame (~3× 가속)
  - 해상도별 멀티 엔진 자동 선택 (HD/FHD/QHD/UHD)
- **NVENC h264_nvenc fallback to libx264**
  - mux 3s → 1s
- v2 quality features 모두 유지:
  - Temporal mask smoothing (5-frame rolling avg)
  - Reinhard color histogram match
  - Mask erosion + adaptive feather
  - Face diagonal min ratio gate
- **백업**: `mouth_only_enhance.v1_backup.py`, `mouth_only_enhance.v2_backup.py`

#### 2. LatentSync 파이프라인 NVENC 패치
- `/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py` 2곳 수정:
  - Line 561 (input normalize): libx264 → h264_nvenc
  - Line 902 (final mux): libx264 → h264_nvenc
- `LATENTSYNC_USE_NVENC=1` 기본 활성화, 실패 시 libx264 자동 fallback

#### 3. LoRA 학습 진행 중 (한국어 AIHub 데이터)
- **Config**: `lora_nf4_train.yaml`
  - num_frames=4, batch_size=1, resolution=256
  - lora_r=32, lora_alpha=16
  - max_train_steps=50000
- **현재 상태**: step 18,020/50,000 (36%)
- **속도**: 1.10 it/s
- **ETA**: 약 12:20 (8시간 후 완료)
- **체크포인트**: 5000 step마다 4.4GB 저장 (10개 총 50GB)
- **Resume 시도 성공**: Docker 크래시 후 checkpoint-10000.pt에서 재개

#### 4. 품질 개선 패치
- **`fix_profile_strict.py`**: 
  - `LATENTSYNC_PROFILE_MATCH_THRESHOLD` env 추가 (default 0.5 → 0.55 가능)
  - `LATENTSYNC_PROFILE_STRICT_NONE=1` env 추가 (no diarization도 skip)
- **`strict_quality_preset.sh`**: 10개 품질 env 일괄 설정 wrapper

### ⚠️ 부분 완료 / 보류

#### CosyVoice TRT-LLM Triton 통합
- ✅ TRT-LLM 엔진 빌드 완료 (`rank0.engine` 1.3GB)
- ✅ `trtllm-serve` LLM API 작동 (port 8010)
- ✅ Triton 5/5 모델 로드 성공 (의존성 4개 추가 설치 후)
- ❌ **LoRA와 동시 실행 시 GPU OOM** → LoRA 끝난 후 재시도
- 누락 의존성: `x_transformers`, `s3tokenizer 0.3.0+`, `loguru`, `torch_einops_utils`, `einx`
- `cosyvoice-trt` 컨테이너 stopped 상태 유지 (LoRA GPU 보호)

#### FP8 ONNX 양자화 시도 (창조적 우회)
- modelopt 버그 우회를 위해 직접 ONNX QDQ 수술
- TRT FP8 mma 커널 (`tensor16x8x32`, `e4m3`) 정상 선택 검증
- 엔진 크기 25% 감소 (2557 → 1930 MB)
- **속도는 동일** — LatentSync UNet이 memory-bound라 compute 절감 효과 미미
- 파일: `tmp/fp8_full_qdq_surgery.py`, `tmp/step3a_trt_build_fp8_full.py`

### 📊 예상 효과 (LoRA 완료 후 검증)

#### 10초 chunk 기준

| 단계 | 현재 (v2 PyTorch) | v3 TRT | 절감 |
|---|---|---|---|
| RetinaFace 검출 | ~7.5s | ~2.5s | -5s |
| GFPGAN enhance | ~12.5s | ~3.75s | **-8.75s** |
| Poisson blend | ~7s | ~7s | 0 |
| Final mux (NVENC) | ~3s | ~1s | -2s |
| LatentSync normalize | ~10s/video | ~3s/video | -7s/video |

#### 1분 영상 전체

```
현재 (v2): ~14-16분
v3 TRT+NVENC: ~10-12분 (약 28% ↓)
```

LoRA 추가 후 (steps 8 권장):
```
v3 TRT+NVENC + LoRA + steps 8: ~7-9분 (50% ↓)
```

### 🔬 검증 대기

LoRA 학습 끝나면 (예상 5/13 12:20):
1. **VAE TRT 변환** (30분, GPU 필요)
2. **v3 end-to-end 테스트** (10초 chunk × 1 = ~2-3분)
3. **LoRA scale 비교 영상 5개 생성**:
   - base (LoRA 없음)
   - LoRA scale 0.5 (보수)
   - LoRA scale 0.7
   - LoRA scale 1.0 (학습된 그대로)
   - LoRA + v3 enhance

### 📁 핵심 변경 파일

```
patches/
  mouth_only_enhance.py             # v3 (TRT + NVENC)
  mouth_only_enhance.v1_backup.py   # v1
  mouth_only_enhance.v2_backup.py   # v2 (PyTorch, color match)
  asd_filter.py                     # profile strict patch
  fix_profile_strict.py             # env override patch
  strict_quality_preset.sh          # 10개 env 일괄 설정

configs/
  lora_nf4_train.yaml                       # 현재 학습 config (수정됨)
  lora_nf4_train.yaml.before_resume         # Docker 크래시 전
  lora_nf4_train.yaml.before_resume2        # checkpoint-10000 시도

orchestrator.py                     # mouth_enhance, profile gates 통합
```

### 💡 학습된 교훈

1. **GPU 메모리 확인 필수**: Triton (1.3GB engine + 버퍼) + LoRA (15.8GB) = 16GB 초과 → Docker WSL2 크래시
2. **`use_8bit_adam: true` ≠ 실제 NF4 적용**: bitsandbytes 미설치면 폴백
3. **FP8 양자화는 워크로드 의존**: memory-bound면 compute 절감 안 보임
4. **Triton 이미지 의존성 불완전**: prebuilt도 모듈 4개 추가 필요
5. **체크포인트가 진짜 자원**: 5000 step마다 자동 저장 → Docker 크래시도 거의 무손실

---

## 2026-05-12 (화) — FP8 탐색 + Quality fixes

### 완료
- FP8 ONNX QDQ surgery 5번 시도 (modelopt 버그 우회)
- ✅ FP8 엔진 빌드 성공, mma 커널 검증
- ❌ LatentSync에서 실속도 동일
- TRT EXHAUSTIVE 재빌드 (0.5% 개선만)
- 통합 벤치마크 (FP16 baseline / EXHAUSTIVE / FP8 wo / FP8 full)
- mouth_only_enhance v2 (color match, temporal smoothing)
- profile strict patch
- CosyVoice TRT-LLM image pull (8GB)

### 결정
- TRT 엔진 레벨 더 줄일 여지 없음 (memory-bound)
- 속도 향상은 다음 영역에서:
  - 후처리 (GFPGAN/RetinaFace TRT)
  - 인코딩 (NVENC)
  - 학습 후 step 감소

---

## 인프라 / 의존성

### 컨테이너
- `dubbing_pipeline`: 메인 파이프라인 (LatentSync, CosyVoice PyTorch, GFPGAN PyTorch)
- `cosyvoice-trt`: TRT-LLM + Triton (LoRA 동안 stopped 유지)

### GPU
- RTX 5080 16GB (sm_120 / Blackwell)
- LoRA peak: 15.8GB / 16GB (한계 근접)

### 주요 TRT 엔진
- `unet_fp16.trt` (2.5GB) — 현재 사용 ★
- `unet_fp8_full.trt` (1.9GB) — FP8 검증용, 미사용
- `gfpgan_bf16.trt` (175MB) — v3에서 사용
- `retinaface_r50_fhd_fp16.trt` (60MB) — v3에서 사용
