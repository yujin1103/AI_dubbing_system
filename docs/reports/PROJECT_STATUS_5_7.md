# 프로젝트 진행 상황 (2026-05-07)

**다국어 자동 더빙 시스템 — 2차 중간발표 직전**

---

## 📋 목차

1. [시스템 아키텍처](#-시스템-아키텍처)
2. [현재 파이프라인](#-현재-파이프라인)
3. [5/7 작업 내역](#-57-작업-내역)
4. [Production 설정](#-production-설정)
5. [알려진 한계](#-알려진-한계)
6. [코드 점검 결과](#-코드-점검-결과)
7. [향후 작업](#-향후-작업)

---

## 🏗 시스템 아키텍처

### 전체 컴포넌트

```
영상 입력 (mp4)
    ↓
[orchestrator.py 메인 파이프라인]
    ↓
┌─────────────────────────────────────┐
│ 1. 오디오 전처리                     │
│    - ffmpeg 추출                     │
│    - BS-Roformer (vocal/bgm 분리)    │
│    - Silero VAD (발화 구간)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. 화자 식별 + ASR                   │
│    - Qwen3-ASR-1.7B (텍스트 + word)  │
│    - DiariZen (화자 분리)            │
│    - ECAPA-TDNN (speaker embedding)  │
│    - LightASD (active speaker)       │
│    - emotion2vec+ (감정 감지)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 번역 + 합성                       │
│    - VectorEngine LLM 번역 (batch=7) │
│    - MOS 자동 평가 (reference 선택)  │
│    - CosyVoice3 (TTS, voice cloning) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. 영상 결합 + Lipsync               │
│    - ffmpeg mix (loudnorm 적용)      │
│    - LatentSync 1.6 (lipsync)        │
│    - GFPGAN async 2x (face quality)  │
└─────────────────────────────────────┘
    ↓
출력 영상 (mp4)
```

### 멀티 venv 구조 (Docker container)

```
/opt/venv_lipsync/    # 메인 (orchestrator, LatentSync, CosyVoice)
/opt/venv_asr/        # Qwen3-ASR
/opt/venv_diarizen/   # DiariZen
/opt/venv_gfpgan/     # GFPGAN, color_match
```

### Daemon 구조

```
port 8901: CosyVoice3 daemon
port 8902: Qwen3-ASR daemon
port 8903: DiariZen daemon
```

---

## 🔄 현재 파이프라인

### orchestrator.py 메인 함수: `run_pipeline()`

```python
파라미터:
  --input PATH             영상 입력
  --lang ko/ja/en/zh        타겟 언어
  --enable-lipsync          LatentSync 1.6 적용
  --use-lora                Korean LoRA 사용
  --enable-postprocess      GFPGAN 2x 적용
  --postprocess-upscale 2   업스케일 비율
  --smart-daemon            Daemon mode (속도 향상)
  --content-type drama      감정 정책 (auto/drama/movie/lecture/interview/news)
```

### 데이터 흐름

```
1. extract_audio (ffmpeg)
2. separate_audio (BS-Roformer) → vocals.wav + bgm.wav
3. apply_vad_filter (Silero VAD) → clean_vocals.wav
4. asr_qwen3 (daemon) → words[] with timestamps
5. diarize_pyannote (daemon) → speaker turns
6. fuse_av_diarization (LightASD + ECAPA) → spurious 제거
7. build_segments (words + diarization → Segment objects)
8. ★ refine_segments (segment_refiner.py, 5/7 NEW)
9. extract_emotion (emotion2vec+) → seg.emotion
10. build_speaker_profiles (MOS 평가) → reference WAV
11. translate_segments_llm (batch=7) → translated text
12. synthesize_chunk (CosyVoice3) → dubbed.wav
13. mix_audio (ffmpeg loudnorm) → chunk_final.mp4
14. apply_lipsync (LatentSync 1.6) → lipsync.mp4 [optional]
15. apply_gfpgan_postprocess (async I/O) → gfpgan.mp4 [optional]
16. concat_chunks → 최종 mp4
```

---

## 🛠 5/7 작업 내역

### ✅ 완료된 fixes

#### 1. CosyVoice deps 일괄 설치
원인: 다양한 의존성 missing → cascade crash
해결:
```bash
pip install hyperpyyaml modelscope openai-whisper inflect conformer \
            hydra-core gdown pyworld wget transformers sentencepiece \
            silero-vad x-transformers pyarrow funasr
```

#### 2. Daemon 자동 정리 (defense-in-depth)
원인: stale daemon이 GPU memory 점유 → CUDA driver crash
해결: orchestrator.py 두 곳에서 `_stop_daemons()` 호출
- `run_pipeline()` lipsync 진입 시 (line 3765)
- `apply_latent_sync()` 함수 시작 (line 3072)

#### 3. emotion2vec+ 정상 동작
원인: `funasr` 모듈 누락 → 모든 segment Neutral (0.00)
해결: `pip install funasr` → 다양한 감정 감지
```
Before: 모두 Neutral (0.00)
After:  Neutral (0.84), Sad (0.99), Angry (1.00), Happy (0.94)
```

#### 4. mix_audio loudnorm (perceptual loudness 매칭)
원인: 고정 비율 mixing → 더빙 음량 너무 큼
해결: `orchestrator.py:2966-3015`
```python
# Before: dubbed=0.7 + bgm=0.6 (고정)
# After: loudnorm I=-23 LUFS (EBU R128) + bgm=0.9
```

#### 5. Reference 음성 peak normalize
원인: BS-Roformer 분리 후 vocal quiet → CosyVoice 부자연
해결: `orchestrator.py:1862-1868`
```python
peak = float(np.abs(clip).max())
if peak > 0 and peak < 0.5:
    gain = min(0.7 / peak, 4.0)
    clip = clip * gain
    print(f"[Profile] reference boost ×{gain:.2f}")
```

#### 6. GFPGAN async I/O 파이프라인
원인: GPU idle while CPU read/write
해결: `patches/gfpgan_async_postprocess.py` (NEW)
```
prefetch=4 reader thread + write_pool=2
GPU never waits → -22% time
```

#### 7. orchestrator default 변경
- `lipsync_steps`: 20 (한국어 phoneme 정확도)
- `lipsync_vae_chunk`: 2 (v42 정확 매칭)
- `lipsync_deepcache`: False (sm_120 incompat)
- `LATENTSYNC_VAE_VARIANT`: ema (v42 검증)
- `postprocess_upscale`: 2 (v42 quality)
- `--lipsync-deepcache` CLI: store_true (default OFF)
- LLM `BATCH_SIZE`: 5 → 7 (-30% 시간)

#### 8. segment_refiner.py (NEW)
원인: DiariZen이 빠른 화자 교차 (drama)에서 segment merging
해결: `scripts/segment_refiner.py`
```
1. ASD per-frame face 변화 시점에서 segment 분할
2. 짧은 발화 over-extension 방지 (≤3 words + ≥5s → trim)
3. WordTiming dataclass / dict 둘 다 지원
```

### ⚠️ 폐기된 시도 (검증 후 부적합)

| 시도 | 결과 | 폐기 이유 |
|---|---|---|
| `Color Match v1` (rectangle) | 사각형 자국 | Face box paste-back |
| `Color Match v2` (ellipse) | 여전 자국 | Boundary visible |
| `Color Match v3` (diff mask) | VAE 노이즈로 rectangle | LatentSync 구조적 한계 |
| `lip_paste v1` (rectangle) | 머리 위 사각형 | Face detect 폴백 |
| `lip_paste v2` (fps sync) | Ghost effect | 공간 미스매치 |
| `Pink Correct` | GFPGAN이 덮어씀 | 효과 무효화 |
| `256 resolution` | 입 영역 artifact | 1.6은 512 전용 학습 |
| `nf=32` | motion module PE max 24 | 아키텍처 제약 |
| `DeepCache` | sm_120 crash | CUDA stream sync |
| `channels_last` | +65% slower | UNet3D 미적합 |
| `torch.compile` | +30% slower | Dynamic shape recompile |
| `SageAttention` | head_dim=512 incompat | LatentSync 미지원 |
| `vae_chunk=4` | 미세 quality drop | revert chunk=2 |
| `steps=15` | 한국어 phoneme ↓ | revert steps=20 |

**= 14개 가속/품질 개선 시도, 모두 sm_120 + LatentSync 1.6에서 부적합으로 검증.**

---

## 🏆 Production 설정 (5/7 final)

### 권장 명령어

```bash
# 강연/인터뷰 (단일 화자 정면 — 가장 적합)
python orchestrator.py \
  --input video.mp4 --lang ko \
  --enable-lipsync \
  --enable-postprocess --postprocess-upscale 2 \
  --smart-daemon

# 한국어 LoRA 사용 (품질 ↑)
python orchestrator.py \
  --input video.mp4 --lang ko \
  --enable-lipsync --use-lora \
  --lipsync-config stage2_512_nf16_smallmask.yaml \
  --enable-postprocess --postprocess-upscale 2 \
  --smart-daemon
```

### Default 값들

```python
# orchestrator.py
lipsync_steps:      20                       # v42 검증
lipsync_vae_chunk:  2                        # v42 정확 매칭
lipsync_deepcache:  False                    # sm_120 incompat
postprocess_upscale: 2                       # v42 quality
mix_audio.use_loudnorm: True                 # EBU R128
mix_audio.target_lufs: -23                   # 표준 dialogue
mix_audio.bgm_volume: 0.9                    # 원본 dynamics
LLM BATCH_SIZE:     7                        # batch=7

# 환경변수 (자동 설정)
LATENTSYNC_VAE_VARIANT: ema
LATENTSYNC_VAE_CHUNK:   2
LATENTSYNC_VAE_SLICING: 1 (강제)
GFPGAN_SCRIPT:    gfpgan_async_postprocess.py (우선)
GFPGAN_FALLBACK:  gfpgan_postprocess.py
```

---

## ⚠️ 알려진 한계

### 1. **드라마 빠른 화자 교차** (해결 시도 중)
- DiariZen이 < 1초 turn 못 잡음
- segment_refiner.py로 ASD 기반 split 시도 중
- 검증 진행 중

### 2. **LoRA cheek pink artifact** (구조적)
- 학습 nf=2, 추론 nf=16 mismatch
- 광대뼈 영역 미세 색감 시프트
- 해결: nf=4 재학습 (RTX 5080 가능) 또는 nf=16 재학습 (cloud A100)

### 3. **LatentSync 자체 가속 불가**
- sm_120에서 가능한 모든 기법 검증 → 모두 실패
- 후처리 (GFPGAN async) 만 가속 가능

### 4. **드라마 audio quality**
- BS-Roformer로 분리해도 reverb/processing 잔존
- Reference MOS 1.5-2.8 (Poor) → CosyVoice 기계음
- 적합: TED, 인터뷰, 강의 (clean recording)
- 부적합: 드라마, 영화 (theatrical processing)

### 5. **시간 비용**
- 100초 영상 → 35-80분 (variable, multi-speaker일수록 빠름)
- 실시간 처리 불가
- 사전 처리 후 시청 use case에 적합

---

## 🔍 코드 점검 결과

### ✅ 잘 된 부분

1. **모듈형 구조**: 각 단계가 독립 함수
2. **Daemon mode**: 모델 로딩 시간 절감
3. **Cache 전략**: ASD 결과 + AI Hub 데이터 캐싱
4. **방어 코드**: 각 단계 try/except + fallback
5. **Run ID 기반 격리**: 작업 공간 독립
6. **JSON 리포트**: 모든 결과 기록
7. **다국어 인프라**: `latentsync_<lang>.pt` 자동 인식

### ⚠️ 개선이 필요한 부분

#### 1. **orchestrator.py 비대화** (3,900+ lines)
- 한 파일에 너무 많은 책임
- 권장: 모듈 분리
  - `pipeline/audio.py` (vocal sep, VAD, ASR)
  - `pipeline/diarize.py` (DiariZen, ASD)
  - `pipeline/translate.py` (LLM 번역)
  - `pipeline/tts.py` (CosyVoice3)
  - `pipeline/video.py` (lipsync, postprocess)

#### 2. **CosyVoice deps ad-hoc 설치**
- 5/7 cascade crash로 14개 deps 수동 설치
- 권장: requirements_cosy.txt 작성 + 한 번에 설치
- 또는 별도 venv_cosy 분리

#### 3. **Patches 폴더 정리 필요**
폐기된 파일들이 그대로 있음:
```
patches/
├── color_match_postprocess.py    ❌ 폐기 (rectangle 자국)
├── lip_paste_postprocess.py      ❌ 폐기 (ghost)
├── pink_correct_postprocess.py   ❌ 폐기 (GFPGAN 덮어씀)
├── poisson_blend_postprocess.py  ❌ 폐기
├── codeformer_postprocess.py     ❌ 사용 안 됨
├── gfpgan_postprocess.py          ✅ Fallback
├── gfpgan_async_postprocess.py   ✅ Production (NEW)
├── latentsync_inference_v27.py   ✅ Production
├── latentsync_vae_chunk_patch.py ✅ Used
├── latentsync_train_patch.py     ✅ Used
├── cosyvoice_daemon.py            ✅ Used
├── asr_daemon.py                  ✅ Used
├── diarize_daemon.py              ✅ Used
└── ...
```

권장: 폐기된 5개 파일을 `patches/deprecated/`로 이동.

#### 4. **Volume normalization 하드코딩**
- `dubbed_volume=0.7`, `bgm_volume=0.9` 등 magic number
- 권장: config 파일로 분리 또는 dataclass로 그룹화

#### 5. **Error handling 비일관**
- 일부 함수: `try/except + 진행` (silent fail)
- 다른 함수: `raise RuntimeError` (강제 중단)
- 권장: 일관된 정책 (필수 단계는 raise, 선택 단계는 warn+continue)

#### 6. **테스트 부재**
- Unit tests 없음
- Integration tests 없음
- 새 영상에 대한 회귀 테스트 자동화 부족

#### 7. **로깅 구조**
- print() 일관 사용 (좋음)
- 그러나 색상/태그 일부만 적용 (`[Pipeline]`, `[Cosy]`, etc)
- 권장: logging 모듈 사용 + 통일된 prefix

#### 8. **Type hints 부분적**
- 일부 함수만 type annotation
- 권장: mypy 강제 + 모든 함수

#### 9. **문서화 부족**
- 큰 함수의 docstring은 있지만 args/returns 일부 누락
- 권장: numpy/google style 일관

#### 10. **하드코딩된 경로**
```python
"/opt/LatentSync/checkpoints/latentsync_unet.pt"
"/opt/gfpgan_models/GFPGANv1.4.pth"
"/workspace/media/lora/latentsync_ko.pt"
```
권장: config.yaml로 분리

### 🔴 Critical issues

#### 1. **segment_refiner 검증 미완**
- 5/7 NEW 모듈
- 첫 시도에서 WordTiming bug 발견 + fix
- 두 번째 시도 진행 중 (검증 대기)

#### 2. **MOS reference 품질 가이드 부재**
- MOS < 3.0 reference도 사용
- 권장: threshold 강제 (≥3.0) + 부족 시 alternate 검색

#### 3. **LightASD coverage 41% (drama)**
- 빠른 컷 + 다중 angle → face tracking 자주 끊김
- 권장: face detection threshold 완화 + 시간적 보간

---

## 📊 성능 측정 (15초 영상 기준)

```
단계                       시간      비율
────────────────────────────────────
Daemon load (cold)         1:30      —
Vocal separation           1:00      6%
Silero VAD                 0:30      3%
Qwen3-ASR (daemon)         1:00      6%
DiariZen + AV-Fusion       2:00      12%
Emotion + MOS              1:00      6%
LLM 번역 (batch=7)          0:30      3%
CosyVoice TTS               2:00      12%
ffmpeg mix                  0:15      1%
LatentSync 1.6              5:00      30%
GFPGAN async 2x             3:30      21%
────────────────────────────────────
Total                       17:15
```

100초 영상 (drama): 35-80분 (multi-speaker 비중에 따라 다름)

---

## 🚀 향후 작업 (우선순위)

### P0 (시연 직전 — 1-2일)
1. **segment_refiner 검증** (run v3 결과 평가)
2. **시연용 영상 사전 처리** (TED 등 깨끗한 영상 5-10개)
3. **2차 중간발표 슬라이드 작성** (대본 `중간발표_v2_대본.md` 활용)

### P1 (시연 후 — 1주 내)
1. **Korean LoRA nf=4 재학습** (3-4일 background)
2. **MOS threshold 강제** (≥3.0 reference만)
3. **patches/deprecated/ 정리**

### P2 (장기)
1. **A100 cloud에서 LoRA nf=16 재학습** ($200-400, 3-4일)
2. **Active speaker face_box 명시** (LatentSync에 직접 전달)
3. **orchestrator.py 모듈 분리**
4. **일본어 LoRA 학습** (데이터 수집 필요)

---

## 📁 핵심 파일 위치

```
E:\TTS_capstone\
├── orchestrator.py                    # 메인 (3,900 lines)
├── 중간발표_v2_대본.md                # 2차 발표 대본
├── PROJECT_STATUS_5_7.md              # 본 문서
│
├── patches/
│   ├── latentsync_inference_v27.py    # LatentSync 패치
│   ├── gfpgan_async_postprocess.py    # 5/7 NEW
│   ├── gfpgan_postprocess.py          # Fallback
│   ├── cosyvoice_daemon.py
│   ├── asr_daemon.py
│   ├── diarize_daemon.py
│   └── deprecated/                    # 정리 대상
│       ├── color_match_postprocess.py
│       ├── lip_paste_postprocess.py
│       ├── pink_correct_postprocess.py
│       └── ...
│
├── scripts/
│   ├── segment_refiner.py             # 5/7 NEW
│   ├── asd_runner.py
│   ├── av_fusion.py
│   ├── aihub_face_crop.py
│   └── ...
│
├── configs/
│   └── lora_full_train.yaml
│
└── media/
    ├── input/
    │   ├── test4.mp4                  # 드라마 클립 (108s)
    │   └── ...
    ├── output/
    │   ├── test4_ko_..._gfpgan.mp4    # 최종 결과
    │   └── archive/                   # 14단계 archive
    │       ├── 00_legacy_apr_may_dev/
    │       ├── 01_test_series_dev/
    │       ├── 02-11/
    │       └── ...
    ├── runs/
    │   └── 20260507_*/                # 작업 공간
    ├── reports/
    │   └── *.json                     # JSON 리포트
    ├── lora/
    │   └── latentsync_ko_50k.pt       # 4.4 GB
    └── training_outputs/
        └── lora_full_train/
            └── checkpoint-*.pt
```

---

## 🎓 학습한 것들

### sm_120 (Blackwell) GPU 환경의 한계

```
✗ DeepCache: CUDA stream sync issue
✗ channels_last: UNet3D fallback 비효율
✗ torch.compile: dynamic shape recompile
✗ SageAttention: head_dim=512 미지원
✗ vae_slicing OFF: cuDNN driver crash

→ LatentSync 자체 가속 불가능
→ 후처리만 가속 가능 (GFPGAN async I/O -22%)
```

### LatentSync 1.6 특성

```
- 512×512 전용 학습 (256은 1.5 시대)
- TREPA loss (1.6에서 도입)
- num_frames 최대 24 (motion module PE 한계)
- VAE: stabilityai/sd-vae-ft-ema (smooth)
- DDIM 20 steps (default), 15도 가능
```

### CosyVoice voice cloning 요구사항

```
- Reference 길이: 5-10초+ (3초는 부족)
- Reference 품질: MOS ≥ 3.0
- 단일 화자 reference (다중 화자 섞이면 robotic)
- Clean recording (reverb/effects 적은 게 좋음)
- 16kHz mono로 자동 변환
```

### 드라마 vs 강연 영상 적합성

| 항목 | 드라마 | TED 강연 |
|---|---|---|
| 화자 수 | 5-10명 | 1명 |
| 발화 속도 | 빠른 교차 | 느린 명확 |
| Audio quality | Theatrical processed | Clean lavalier |
| Reverb | 강함 | 약함 |
| Reference MOS | 1.5-3.0 | 4.0+ |
| TTS 품질 | Robotic | Natural |
| 우리 시스템 적합도 | ❌ | ✅ |

---

## 📝 결론

5월 7일 작업 요약:
- ✅ LatentSync 1.6 production 통합 완료
- ✅ Web UI 백엔드 (Docker microservices)
- ✅ 14개 가속 시도 검증 + 채택 (GFPGAN async)
- ✅ Volume normalization 시스템
- ✅ Emotion detection 정상화 (funasr)
- ✅ Reference quality 자동 보정
- ⚠️ segment_refiner 검증 진행 중

**다음 우선순위:**
1. segment_refiner 효과 검증 (drama 화자 교차 fix)
2. 시연용 영상 (TED 등) 사전 처리
3. 2차 중간발표 슬라이드 작성

이 문서는 `E:\TTS_capstone\PROJECT_STATUS_5_7.md`에 저장됨.
