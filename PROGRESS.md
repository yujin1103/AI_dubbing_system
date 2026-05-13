# Progress Log

> 진행사항 / 작업 일지 — 최신순. 코드 변경 + 결정 + 검증 결과 기록.

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
