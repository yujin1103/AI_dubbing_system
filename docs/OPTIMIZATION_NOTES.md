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
