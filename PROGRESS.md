# Progress Log

> 진행사항 / 작업 일지 — 최신순. 코드 변경 + 결정 + 검증 결과 기록.

## 2026-05-13 (수, 13:00 업데이트) — VAE TRT 버그 + 검증 진행

### ⚠️ VAE TRT 통합 이슈
- Smoke test는 통과했지만 실제 LatentSync 추론 통합 시 `CUDA driver error: device not ready`
- 원인: TRT engine 컨텍스트와 PyTorch 추론 stream 간 충돌 추정
- 임시 해결: `LATENTSYNC_VAE_TRT=0` (PyTorch VAE 사용)
- 추후 별도 디버깅 필요 (CUDA stream 관리 검토)

### 🔄 LoRA scale 검증 재시작
- 4 variants (base / lora_0.5 / lora_0.7 / lora_1.0)
- PyTorch UNet (LoRA-merged 체크포인트는 TRT 미지원)
- VAE도 PyTorch (임시)
- 8.7초 chunk × 4 variant ≈ 30-50분
- 완료 후 mouth_enhance v3 (TRT) 적용 + 2×2 grid 영상 생성

### Triton 임시 중단
- LoRA 검증 중 GPU 메모리 충돌 방지
- 검증 완료 후 재시작 예정 (full pipeline 테스트용)

### 다음 단계
1. 4 variants 완료
2. mouth_enhance 4개 적용 (~2분)
3. 2×2 grid comparison.mp4 생성
4. 사용자 시각 비교 → 최적 LoRA scale 결정
5. Phase 0 baseline (선택된 scale로 영상 8-12개)


---


## 2026-05-13 (수, 검증 시작) — VAE TRT 통합 + LoRA scale 비교

### ✅ 완료된 통합 (12:00-12:10)

#### VAE TRT (Task A)
- VAE encoder TRT: 67MB (20.7s 빌드)
- VAE decoder TRT: 96MB (23.4s 빌드)
- 총 빌드 시간: **61초**
- inference.py 패치 완료: `LATENTSYNC_VAE_TRT=1` 기본 활성
- Smoke test 통과 (encode/decode forward 정상)

#### CosyVoice TRT-LLM (Task B)
- trtllm-serve port 8010: HTTP 200 ✅
- tritonserver port 18000: HTTP 200 ✅
- 5/5 모델 로드 완료 (speaker_embedding, cosyvoice3, audio_tokenizer, vocoder, token2wav)
- 의존성 5개 자동 설치 검증

#### LoRA Merge (3 scales)
- merged_scale_0.5.pt (effective scale 0.25) — 4.4GB
- merged_scale_0.7.pt (effective scale 0.35) — 4.4GB
- merged_scale_1.0.pt (effective scale 0.50) — 4.4GB
- 모두 224 LoRA pair + 336 motion_modules 정상 병합

### 🔧 발견된 이슈

#### NVENC 불가 (dubbing_pipeline)
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` — `video` 누락
- 해결: `LATENTSYNC_USE_NVENC=0` 으로 libx264 fallback
- 영향: -5-10s/min 영상 (작은 손실)

#### LoRA Runtime Loading
- inference.py에 LoRA 환경변수 처리 없음
- 해결: merge_lora_into_base.py로 미리 병합한 체크포인트 사용
- 장점: vanilla state_dict → TRT 재export 시 호환

### 🎬 현재 진행
- AIHub VS11 lip_J_2_M_05_C221_A_001.mp4
- Sentence ID 6: "은퇴를 빨리하고 창업을 하는 게 나을까?" (8.7s)
- 4 variants 동시 비교 중 (PyTorch UNet, no TRT for LoRA):
  - base / lora_0.5 / lora_0.7 / lora_1.0
- 예상 시간: ~20-40분 (4 × 5-10min)

### LoRA 학습 최종 상태
- 중단 step: 45,115 / 50,000
- 사용 체크포인트: **checkpoint-45000.pt**
- 모든 다른 체크포인트 보존 (15k/20k/25k/30k/35k/40k/45k)


---


## 2026-05-13 (수, 검증 인프라) — 단계별 화자 매칭 개선 계획

### 🎯 핵심 인식
- TRT 변환은 품질 영향 없음 (FP16/BF16 안전)
- 진짜 품질 문제는 **화자 매칭 정확도** + 한국어 cross-lingual gap
- LoRA가 후자 해결, 전자는 별도 패치 필요

### 📋 Phase별 검증 계획 (LoRA 완료 후)

**Phase 0**: 현재 baseline (v3 + LoRA + strict envs)
**Phase 1**: + Track-level person ID locking (per-frame → per-track 결정)
**Phase 2**: + SyncNet post-validation (잘못 매칭된 frame 자동 거부)
**Phase 3**: + Overlap detection + skip (DiariZen overlap 구간 lipsync off)
**Phase 4**: + Multi-frame voting (N=5 frame 평균 임베딩)

### 🛠️ 검증 자동화 인프라
- **validation_report_generator.py**: phase별 일관된 비교 report 생성
- 각 영상마다: timing.json + metrics.json + report.md + thumbnails
- Phase별 summary.md 자동 생성
- 4개 카테고리 영상으로 다양성 확보:
  - A. AIHub 깨끗 한국어 (60개 보유)
  - B. 드라마 정면 1인
  - C. 드라마 다인 + overlap
  - D. 측면/움직임 (LoRA 한계 케이스)

### 📊 ROI 비교 (모델 교체 vs 로직 개선)

| 작업 | 효과 | 비용 | ROI |
|---|---|---|---|
| Track-level locking | -70% | 0.5d | ⭐⭐⭐⭐⭐ |
| SyncNet validation | -50% | 0.5d | ⭐⭐⭐⭐⭐ |
| Overlap skip | -100% (overlap만) | 0.3d | ⭐⭐⭐⭐ |
| LoCoNet ASD 교체 | +8% | 2-3d | ⭐⭐ |
| Sortformer diarization | -33% DER | 3-5d | ⭐⭐ |

→ **새 모델 통합보다 똑똑한 로직이 훨씬 가성비 좋음**

### 진행 상황
- LoRA: step 44,032/50,000 (88%) — ~1h 40min 남음
- 모든 A·B·C 스크립트 작성 완료, 검증 대기
- 검증 리포트 시스템 준비 완료


---


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
