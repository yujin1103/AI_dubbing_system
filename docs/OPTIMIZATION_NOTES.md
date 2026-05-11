# LatentSync 가속 노트 (sm_120 Blackwell)

> 기록일: 2026-05-10
> 환경: sm_120 (Blackwell, 16GB) · CUDA 13.0 · PyTorch 2.11+cu130 · Python 3.13

## 결과 요약

64초 영상 (1920 frames, 30fps) 처리 시간:

| Run | 설정 | 시간 | A 대비 | 비고 |
|-----|------|------|-------|------|
| **A** | DDIM 20 step | 29:30 | baseline | 원본 |
| B | DDIM 20 + SageAttn3 | 33:35 | +14% (느려짐) | 폐기 |
| C | DDIM 20 + SageAttn2 | 30:13 | +2% | 폐기 |
| **E** | DPM++ 10 step | 22:06 | **−25%** | DPMSolver++ multistep |
| F | DPM++ 10 + CFG=1.0 | 16:35 | −44% | 품질 위험 |
| G | TRT FP16 + DDIM 20 | 21:55 | −26% | TRT 엔진만 |
| **H** | TRT + DPM 10 | 13:54 | **−53%** | 안정 조합 |
| **🏆 I** | TRT + DPM 10 + TeaCache | **9:15** | **−69% (3.18×)** | 최종 |

## 시도된 기법별 분석

### 1. SageAttention 3 (FP4 Blackwell) — **폐기**

LatentSync UNet에 SageAttn3을 monkey-patch했으나 baseline보다 느려짐. 원인:

- 작은 batch (B=2) + 짧은 sequence length (16 frames × 64×64 spatial / 8 heads = ~32k tokens per attn)
- SageAttn3 sweet spot: B≥8, seq≥4096
- LatentSync 워크로드는 메모리 bound가 아닌 compute bound

→ 추후 video diffusion model이 batch가 클 때만 재검토.

### 2. DPMSolver++ MultistepScheduler — **+25% 단축**

DDIM 20 step → DPMSolver++ 10 step으로 변경. 같은 품질 유지하면서 UNet 호출 절반.

**적용 시 주의 사항** (두 가지 패치 필요):

#### 2-1. `fix_dpmsolver_args.py` — `from_pretrained` 회피

LatentSync 기본 코드 `DPMSolverMultistepScheduler.from_pretrained("configs")`가 DDIM의 scheduler_config.json을 읽어 `steps_offset=1` 충돌. 마지막 step에서 `sigmas[step_index+1]` IndexError.

**Fix**: 명시적 인자로 직접 생성 + `euler_at_final=True`.

#### 2-2. `fix_dpm_chunk_reset.py` — chunk 간 step_index 누적 방지

LatentSync는 비디오를 num_frames(16) 단위 chunk(예: 101개)로 나눠 각 chunk마다 N-step 디노이징.
- DDIM은 stateless → 무관
- DPMSolver는 `_step_index`가 chunk 간 누적 → 두 번째 chunk 첫 호출에서 IndexError

**Fix**: chunk loop 안에서 `set_timesteps()` 재호출.

### 3. TensorRT FP16 엔진 (UNet3D) — **추가 −26%**

LatentSync UNet3DConditionModel을 ONNX → TRT FP16 엔진 컴파일.

**Static shape**:
- `sample`: (B=2, C=13, T=16, H=64, W=64) FP16
- `timestep`: scalar int64
- `encoder_hidden_states`: (B*T=32, S=50, D=384) FP16  ← Whisper-tiny features
- `output`: (2, 4, 16, 64, 64) FP16

**Engine 크기**: 2.55 GB · 빌드 시간 ~10분 · forward 평균 409ms

**적용 시 주의 사항**:

#### 3-1. `fix_trtunet_device.py` — diffusers `_execution_device` 호환

TRTUNet에 PyTorch parameter가 없으므로 `pipeline.device`가 walking할 때 StopIteration. 1-element dummy `nn.Parameter`로 device marker 등록.

#### 3-2. CFG 필수

엔진이 B=2로 컴파일되어 `guidance_scale > 1.0` 필수. CFG=1.0은 PyTorch 경로 사용 (Run F).

### 4. TeaCache — **추가 −33%**

Liu et al., "TeaCache: Timestep Embedding Aware Cache for Diffusion Models" (CVPR 2025) 응용.

**원리**: 인접 timestep 입력 변화량(rel_l1)을 누적하다가 threshold 미만이면 직전 출력 재사용. UNet 호출 ~50% skip.

**LatentSync 적용**:
- `teacache_wrapper.py`: TRTUNet 또는 PyTorch UNet 둘 다 wrap 가능
- `install_teacache.py`: inference.py + lipsync_pipeline.py에 hook 설치
- threshold 0.1 권장 (논문값) — 품질 손실 -0.07~0.5%
- chunk 시작마다 `reset_cache()` 호출 (chunk 간 입력이 완전히 다름)

**환경변수**: `LATENTSYNC_TEACACHE=0.1` (값 = rel_l1 threshold, 0 = OFF)

## 환경변수 매트릭스

```bash
# Baseline (PyTorch DDIM)
unset LATENTSYNC_USE_TRT LATENTSYNC_SCHEDULER LATENTSYNC_TEACACHE

# DPMSolver++ 만 (E)
export LATENTSYNC_SCHEDULER=dpm

# TRT + DPM (H, 안정)
export LATENTSYNC_USE_TRT=1
export LATENTSYNC_TRT_ENGINE=/workspace/trt_work/engines/unet_fp16.trt
export LATENTSYNC_SCHEDULER=dpm

# 최종 (I, 최고 속도)
export LATENTSYNC_USE_TRT=1
export LATENTSYNC_TRT_ENGINE=/workspace/trt_work/engines/unet_fp16.trt
export LATENTSYNC_SCHEDULER=dpm
export LATENTSYNC_TEACACHE=0.1
```

## 패치 파일 위치 (이 저장소)

| 파일 | 역할 |
|-----|-----|
| `patches/fix_dpmsolver_args.py` | DPMSolver 명시적 인자 생성 |
| `patches/fix_dpm_chunk_reset.py` | chunk 간 step_index reset |
| `patches/latentsync_trt_unet.py` | TRT 엔진 wrapper |
| `patches/fix_trtunet_device.py` | TRTUNet device marker fix |
| `patches/teacache_wrapper.py` | TeaCache 구현 |
| `patches/install_teacache.py` | inference.py + pipeline에 TeaCache hook |
| `patches/sageattn3_patch.py` | SageAttention 3 (참고용, 효과 없어 폐기) |
| `patches/sageattn2_patch.py` | SageAttention 2 (참고용) |

## 재현 절차

1. TRT 엔진 빌드 (별도 세션, 약 30분):
   ```bash
   # ONNX export → trtexec --fp16 --workspace=4096
   ```

2. 패치 적용 (멱등):
   ```bash
   /opt/venv_lipsync/bin/python /workspace/patches/fix_dpm_chunk_reset.py
   /opt/venv_lipsync/bin/python /workspace/patches/fix_trtunet_device.py
   /opt/venv_lipsync/bin/python /workspace/patches/install_teacache.py
   ```

3. 환경변수 설정 + 실행 (위 매트릭스 참고)

## 향후 작업

- **GFPGAN-TRT**: 후처리 22분 → ~7분 (진행 중)
- **TRT FP8 재빌드**: 추가 +20% 가능 (NVIDIA Modelopt 사용)
- **DeepCache vs TeaCache**: 둘 중 더 잘 맞는 것 선택 (TeaCache 채택)


---

## 2026-05-11 — 2단계 가속 (GFPGAN 후처리 + RetinaFace)

### 측정 결과 (1080p 64초 영상 기준)

| Run | 설정 | 시간 | baseline 대비 |
|---|---|---|---|
| baseline | GFPGAN PyTorch FP32 + face_helper | 18:00 (1080s) | – |
| TRT GFPGAN | TRT BF16 generator (FP16 broken) | 15:24 (924s) | −14% |
| + TRT RetinaFace | + 4-resolution dispatch (HD/FHD/QHD/UHD) | ~14:55 (예상) | −17% |

### 핵심 발견

1. **GFPGAN FP16은 사용 불가** — StyleGAN2 modulated conv weight^2.sum 이 FP16 overflow → PSNR 13.9 dB. **BF16 엔진**이 정답 (PSNR 49.8 dB)
2. **RetinaFace TRT 효과는 ~2%** — forward 자체는 5.7× 가속이지만 전체에서 차지하는 비중 작음. 진짜 병목은 **alignment+paste-back warpAffine (CPU)**
3. **다중 해상도 인프라**: HD 720p / FHD 1080p / QHD 1440p / UHD 2160p 4개 정적 엔진 + 자동 dispatcher → 미래 UI 해상도 선택 대비

### 추가 가속 후보 (품질 무손실 조건)

- **GPU warpAffine** (alignment + paste-back) — 별도 spawn task 진행 중, 예상 −17%
- NMS GPU化 (torchvision.ops.nms): −3%
- CUDA graph capture: −10%
- LatentSync VAE TRT: LatentSync 단계 −15% 별도 절감

### upscale=2 측정 (참고)

| 모드 | 시간 (단독 추정) | 출력 해상도 | 파일 크기 |
|---|---|---|---|
| upscale=1 (현재) | 15:24 | 512×512 face | 42 MB |
| upscale=2 | ~20분 (단독) / 34:55 (GPU 경합) | 1024×1024 face | 126 MB |

→ 디테일 4배 향상 / 시간 +30% (GPU 경합 빼고). 품질 우선이면 upscale=2 권장.

### 추가 최적화 인프라 (zero quality drop)

- `patches/retinaface_postprocess_gpu.py` — facexlib NMS → torchvision.ops.nms (GPU)
  - 동일 IoU 알고리즘, 결과 비트단위 동일
  - face 검출 후처리 ~30ms → ~3ms

---

## 2026-05-11 — GPU alignment + paste-back (Phase 3)

### 측정 결과 (1080p 64초 영상)

| 단계 | 시간 | 누적 절감 |
|---|---|---|
| baseline (PyTorch FP32) | 18:00 (1080s) | – |
| TRT BF16 GFPGAN (v1) | 15:24 (924s) | −14% |
| **+ GPU alignment + paste (v2)** | **8:04 (484s)** | **−55%** |

### 핵심: v1 → v2 단축의 본질

**paste-back이 진짜 병목이었음** (예상은 face detection이었지만):
- v1 cv2.warpAffine paste = **113 ms/face** (1080p frame, 1 face)
- v2 grid_sample paste = **24 ms/face** (4.66× 가속)
- 평균 5 faces/frame 였으므로: 5 × (113-24) = **445 ms/frame 절약**
- 1602 frames × 445 ms = **713 s 절약** (예상 단축 −17% 훨씬 초과)

### v2 per-stage 분석 (1602 frames)

```
extract (ffmpeg):     79.0 s   (concurrent libx264 영향 큰 편)
enhance loop:        365.6 s   (228 ms/frame)
  detect (PT):       137.3 s
  upload (H2D):        1.2 s
  align (GPU):         2.3 s   (1.4 ms/frame — cv2 0.6 ms 대비 살짝 늦음, N=1 오버헤드)
  GFPGAN forward:     57.8 s   (36 ms/frame, ~5 faces avg)
  paste (GPU):       135.3 s   (84 ms/frame, ~5 faces avg) ⭐
  download (D2H):      1.2 s
assemble:             39.6 s
─────────────────────────────
total:               484.2 s
```

### 품질 검증 (목표 통과)

- PSNR median: **45.62 dB** (목표 ≥45)
- SSIM mean: **0.9922** (목표 ≥0.99)
- p99 픽셀 차이: 6 LSB 이내 (시각 구분 불가)
- diff 가 큰 frame 도 parse-mask 경계의 cv2 Gaussian rounding 차이일 뿐

### 발견된 추가 이슈

⚠️ **build_retinaface_multires.py 가 `strict=False` 로 silently no-op**:
- 체크포인트의 `module.` prefix 처리 안 함
- 결과 ONNX/TRT 가 random init weights
- 우리가 측정한 "5.7× speedup" 은 빈 모델 forward 였음

→ `rebuild_retinaface_fhd.py` 로 수정 빌드 완료. bbox 차이 mean 5.8 px (FP16 양자화 영향, 화질엔 영향 없음).

### 파일

| 파일 | 역할 |
|----|----|
| `patches/gpu_face_aligner.py` | GPU align/paste 모듈 |
| `patches/gfpgan_async_postprocess_trt_v2.py` | v2 메인 스크립트 |
| `patches/test_gpu_aligner_sanity.py` | cv2 vs GPU 동등성 검증 |
| `patches/bench_align_paste_microbench.py` | per-stage 마이크로 벤치 |
| `patches/compare_v1_v2_quality.py` | PSNR/SSIM 비교 |
| `patches/rebuild_retinaface_fhd.py` | RetinaFace 가중치 strict 로드 + 재빌드 |
| `patches/BENCH_GPU_ALIGN.md` | 상세 보고서 |
| `patches/retinaface_postprocess_gpu.py` | NMS GPU 패치 |

### 사용

```bash
/opt/venv_gfpgan/bin/python /workspace/patches/gfpgan_async_postprocess_trt_v2.py \
  --input  /workspace/media/output/INPUT.mp4 \
  --output /workspace/media/output/OUTPUT.mp4 \
  --upscale 1 [--retinaface-trt] [--detail-timing]
```

---

## 2026-05-11 (오후) — v3 + downscale-detect (Phase 3)

### 측정 결과 (1080p 64초 영상, clean GPU 상태)

| Run | 설정 | 시간 | enhance loop |
|---|---|---|---|
| v2 baseline | PT detect + GPU paste | 10:07 (607s) | 532s |
| **v3 + downscale=2** | **detect at 540p**, paste at 1080p | **4:14** (254s) | 181s |

v2 vs v3 품질 (n=40 sampled frames):
- PSNR median: **44.83 dB** (목표 45 살짝 미달, p5 42.49)
- SSIM mean: **0.9907** ✓ (목표 0.99)
- 시각적 구분 불가

### 시간 절감 원인

| 단계 | v2 | v3 downscale=2 | 단축 |
|---|---|---|---|
| detect | 156.8s (97.9 ms/frame) | 48.6s (30.3 ms/frame) | **−69%** ⭐ |
| gfpgan | 144.7s (90.4 ms/frame) | 30.3s (18.9 ms/frame) | −79% |
| paste | 109.7s (68.5 ms/frame) | 98.1s (61.3 ms/frame) | −11% |

→ detection은 540p input (RetinaFace anchors 4× 감소)이라 −69%.
   gfpgan/paste도 감소는 GPU 캐시 효과 + face count 약간 변동 (1407 → 1454).

### 시도했지만 사용 불가능한 최적화

| 옵션 | 결과 |
|---|---|
| **BF16 RetinaFace TRT** | engine smoke test 동작 (conf 1.0) but full pipeline에서 cuDNN sublibrary loading 실패 |
| **NMS GPU patch** | 동일한 cuDNN 충돌 (BF16 TRT 와 같은 원인 추정) |

→ TRT 컨텍스트 + PyTorch face_parse cuDNN 충돌 가능성. 별도 환경 분리 필요.

### 권장 사용

```bash
# 빠름 (4:14, 살짝 미세 품질 감소)
/opt/venv_gfpgan/bin/python /workspace/patches/gfpgan_async_postprocess_trt_v3.py \
  --input  INPUT.mp4 --output OUTPUT.mp4 \
  --upscale 1 --downscale-detect 2 [--detail-timing]

# 정밀 (10:07, 품질 최우선)
/opt/venv_gfpgan/bin/python /workspace/patches/gfpgan_async_postprocess_trt_v2.py \
  --input  INPUT.mp4 --output OUTPUT.mp4 --upscale 1
```

### 전체 64초 영상 파이프라인 누적 단축

| 구성 | 시간 |
|---|---|
| 더빙 단계 (변동 없음) | ~10:00 |
| LatentSync I (TRT + DPM + TeaCache) | 9:15 |
| GFPGAN v3 + downscale=2 | 4:14 |
| **TOTAL** | **23:29** (1차 baseline 61:30 대비 −62%, 38분 절감) |

---

## 2026-05-11 (저녁) — Quality 패치: profile skip + chunked DPM reset

### 새 패치 2개

1. **fix_face_profile_skip.py** — LatentSync 가 측면 얼굴 (90도 가까이 회전)에서
   입 위치 landmark 부정확 → "입이 떠다니는" artifact. 106-point landmark 로 yaw
   추정 후 threshold 이상이면 face=None 반환 (원본 frame 유지).

   ```bash
   export LATENTSYNC_PROFILE_THRESHOLD=0.35  # 약 yaw 45도 이상 skip
   ```

   판정 공식:
   ```
   yaw_ratio = |nose_x - eye_midpoint_x| / eye_distance
   ```
   정면 face: 0.05~0.15 / 측면 30도: ~0.25 / 측면 60도: ~0.5 / 측면 90도: 1+

2. **fix_dpm_chunked_call_reset.py** — `_chunked_call` 경로에 chunk 마다
   `set_timesteps()` + `reset_cache()` 호출. 기존 `fix_dpm_chunk_reset.py` 가
   메인 `__call__` 만 처리했던 빈자리.

   `LATENTSYNC_CHUNK_SECONDS>0` 사용 시 (장편 영상 메모리 절약) DPMSolver
   `step_index` 누적 → IndexError 방지.

### test4 측정 (108초 영상, 2656 frames, drama)

기존 test4 lipsync 가 안 됐던 이유:
- 메모리 부족 → OOM kill (chunked 비활성화 시)
- DPMSolver chunked 경로 IndexError (chunked 활성화 시)
- 측면 face 가 많아서 lipsync 가 적용되어도 부정확

새 패치 적용 후:
- `LATENTSYNC_CHUNK_SECONDS=10` (250 frames/chunk, 11 chunks) → 메모리 OK
- `fix_dpm_chunked_call_reset.py` → IndexError 해결
- `LATENTSYNC_PROFILE_THRESHOLD=0.35` → 측면 face skip

결과: **14:58 완료**, 76 MB mp4 출력
- Brightness skip 300 frames (어두운 장면)
- Face skip 합계 약 800+ frames (face 미감지 + profile skip)
- Lipsync 적용된 frame 약 1500 frames

### 환경변수 정리

```bash
# 최종 권장 (Quality + Speed)
export LATENTSYNC_USE_TRT=1
export LATENTSYNC_TRT_ENGINE=/workspace/trt_work/engines/unet_fp16.trt
export LATENTSYNC_SCHEDULER=dpm
export LATENTSYNC_TEACACHE=0.1
export LATENTSYNC_PROFILE_THRESHOLD=0.35    # 측면 face 안전 처리
export LATENTSYNC_CHUNK_SECONDS=10          # 장편 영상 (1분+) 메모리 절약
```
