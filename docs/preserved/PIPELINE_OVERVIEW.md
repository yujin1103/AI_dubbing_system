# 다국어 자동 더빙 시스템 — 파이프라인 가이드

> **한 줄 요약**: 영어 영상 → (오디오 추출 → 화자 분리 → 번역 → 한국어 voice cloning → 립싱크) → 한국어 더빙 영상
>
> **목표 사용자**: 처음 시스템을 보는 사람이 어떤 모델이 어디서 무슨 일을 하는지 한눈에 이해

---

## 1. 시스템 한눈에 보기

```
입력 영상 (영어 mp4)
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage 1. 오디오 분리 (vocals만 깨끗하게 추출)     │
│   ffmpeg → BS-Roformer → Silero VAD              │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage 2. 누가 / 뭐라고 / 어떤 감정으로 말했나     │
│   Qwen3-ASR + DiariZen + ECAPA + LightASD        │
│   + emotion2vec+                                  │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage 3. 화자별 깨끗한 reference + 한국어 번역    │
│   MOS 평가 → VectorEngine LLM 번역               │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage 4. 화자 목소리로 한국어 합성                │
│   CosyVoice3 (voice cloning + emotion)           │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Stage 5. 영상 결합 + 입모양 맞춤                  │
│   ffmpeg mix → LatentSync 1.6 → GFPGAN          │
└──────────────────────────────────────────────────┘
    │
    ▼
출력 영상 (한국어 더빙 mp4)
```

총 5개 스테이지, 약 12개 모델이 협력. 100초 영상 기준 약 17~35분 처리.

---

## 2. 사용 모델 한눈에 보기

| # | 모델 | 역할 | 한 마디로 |
|---|---|---|---|
| 1 | **BS-Roformer** | 음악/음성 분리 | "보컬만 떼어내기" |
| 2 | **Silero VAD v5** | 발화 구간 감지 | "말 안 하는 silence 제거" |
| 3 | **Qwen3-ASR-1.7B** | 음성 → 텍스트 + 단어 단위 타임스탬프 | "받아쓰기" |
| 4 | **DiariZen** | 화자 분리 (몇 명이 언제 말했나) | "누가 말했는지 추정" |
| 5 | **ECAPA-TDNN** (SpeechBrain) | 화자 임베딩 (192-dim 벡터) | "목소리 지문 추출" |
| 6 | **LightASD** | 영상 속 입 움직이는 사람 감지 | "어느 얼굴이 말하는 중?" |
| 7 | **emotion2vec+ Large** | 감정 분류 | "Happy/Sad/Angry/Neutral 등 감지" |
| 8 | **MOS Evaluator** (custom) | 음성 품질 평가 | "이 reference 깨끗한가?" |
| 9 | **VectorEngine LLM** (GPT-style) | 영어→한국어 번역 + 감정 묘사 | "립싱크에 맞는 syllable로 번역" |
| 10 | **CosyVoice3** (Fun-CosyVoice3-0.5B) | TTS (zero-shot voice cloning) | "5초 reference로 한국어 합성" |
| 11 | **LatentSync 1.6** | 입모양 동기화 | "한국어 발음에 입 맞추기" |
| 12 | **GFPGAN v1.4** | 얼굴 화질 복원 | "립싱크 후 디테일 살리기" |

---

## 3. 스테이지별 상세 (모델 / 역할 / 입출력)

### Stage 1. 오디오 분리

#### 1-1. ffmpeg
- **역할**: 영상에서 오디오 트랙 추출
- **입력**: `video.mp4`
- **출력**: `audio.wav` (44.1kHz stereo)

#### 1-2. BS-Roformer
- **역할**: 보컬과 배경음악(BGM) 분리. 더빙 시 BGM은 살리고 보컬만 한국어로 교체해야 함
- **입력**: `audio.wav`
- **출력**:
  - `vocals.wav` (대화)
  - `bgm.wav` (배경음악, 효과음)
- **한계**: vocal에 묻은 reverb는 분리 안 됨 → ECAPA 화자 임베딩 노이즈 원인

#### 1-3. Silero VAD v5
- **역할**: vocals.wav에서 실제 말이 있는 구간만 추출 (silence 제거)
- **입력**: `vocals.wav`
- **출력**: `clean_vocals.wav` (silence 제거된 보컬)
- **부가 출력**: 발화 구간 리스트 `[(start, end), ...]`

---

### Stage 2. 누가 / 뭐라고 / 어떤 감정

#### 2-1. Qwen3-ASR-1.7B (daemon :8902)
- **역할**: 한국어/영어 음성 → 텍스트 + 단어별 타임스탬프
- **입력**: `clean_vocals.wav`
- **출력**:
  - `words: [{word, start, end}, ...]` (단어 단위 타임스탬프)
  - `detected_lang: "en"` (자동 언어 감지)

#### 2-2. DiariZen (daemon :8903)
- **역할**: 화자 분리 — 음성 임베딩 cluster로 "누가 언제 말했나" 결정
- **입력**: `clean_vocals.wav`
- **출력**: `diarization: [(start, end, "SPEAKER_00"), ...]`
- **한계**:
  - 1초 미만 짧은 turn은 잘 못 잡음
  - reverb 환경 변화 큰 영상에서 같은 화자를 다른 SPEAKER로 over-detect

#### 2-3. ECAPA-TDNN (SpeechBrain)
- **역할**: DiariZen 결과 후처리 — 각 turn에서 192-dim 화자 임베딩 추출, centroid 거리로 over-detect 자동 병합 + outlier 짧은 발화 재할당
- **입력**: DiariZen turns + audio
- **출력**: 정제된 diarization + `speaker_centroids: {SPEAKER_xx: embedding}`
- **활용**: segment_refiner에서 ASD-split된 sub-segment의 화자 재할당, ECAPA sliding window로 화자 변화 감지

#### 2-4. LightASD (Active Speaker Detection)
- **역할**: 영상 frame별로 어느 face가 말하고 있는지 감지 (입 움직임 + 음성 동기 분석)
- **입력**: `video.mp4` + ASR 결과
- **출력**: face track별 frame당 score (양수 = 발화 중)
- **결과 캐싱**: video hash 기반 (재실행 시 1~2분 절감)

#### 2-5. AV-Fusion (자체 알고리즘)
- **역할**: audio diarization + visual ASD 결합. spurious 화자 제거, lipsync 대상 face track 결정
- **입력**: diarization + ASD 결과
- **출력**:
  - `spurious_speakers`: 발화 짧고 face match 없는 가짜 화자
  - `per_frame_target`: frame당 lipsync 적용할 face track
  - `speaker_face_map`: 화자 → face track 매핑

#### 2-6. emotion2vec+ Large
- **역할**: segment별 감정 분류 (audio 기반)
- **입력**: vocals_path + (start, end)
- **출력**: `(emotion, confidence)` — Neutral/Sad/Angry/Happy/Surprised/Scared 중 하나
- **활용**: TTS reference 선택 + LLM 번역 prompt에 hint 제공

---

### Stage 3. Reference 선택 + 번역

#### 3-1. MOS Evaluator (custom UTMOS-style)
- **역할**: 화자별 segment 음성 품질 평가 (1~5점)
- **입력**: 화자별 segment audio chunks
- **출력**: 화자별 감정별 best reference (가장 높은 MOS)
- **목적**: 깨끗한 reference 선택 → CosyVoice voice cloning 품질 향상

#### 3-2. VectorEngine LLM (Cloud API, batch=7)
- **역할**: 영어 → 한국어 번역 + 감정 묘사 생성
- **입력**: segment 텍스트 + duration + emotion hint + speaker
- **출력**:
  - `translated_text`: 한국어 번역 (syllable 가이드 따라 길이 조절)
  - `tts_emotion`: 감정 묘사 (TTS instruction)
- **특이점**: 발화 시간에 맞는 syllable 수 강제 (lip-sync 정확도)

---

### Stage 4. 한국어 합성

#### 4-1. CosyVoice3 (Fun-CosyVoice3-0.5B-2512, daemon :8901)
- **역할**: zero-shot voice cloning + 한국어 합성
- **입력**:
  - `tts_text`: prefix(`You are a helpful assistant.<|endofprompt|>`) + 한국어 텍스트
  - `prompt_wav`: 화자 reference (16kHz mono, 3~15초)
  - `speed`: 발화 속도 (default 1.0)
- **출력**: 24kHz wav (한국어 음성)
- **모드**: `inference_cross_lingual` (영어 reference로 한국어 합성)
- **후처리**: ffmpeg atempo로 시간 길이 lip-sync 맞춤 (0.85x~1.25x 한도)

---

### Stage 5. 영상 결합 + 립싱크

#### 5-1. ffmpeg mix
- **역할**: 한국어 더빙 + BGM 결합
- **입력**: `dubbed.wav` + `bgm.wav`
- **출력**: `chunk_final.mp4`
- **특이점**: loudnorm I=-23 LUFS (EBU R128 표준 dialogue level)

#### 5-2. LatentSync 1.6 (선택, `--enable-lipsync`)
- **역할**: 한국어 음성에 맞춰 입모양 재생성
- **입력**: `chunk_final.mp4` + 한국어 audio
- **출력**: `lipsync.mp4` (입모양 동기화)
- **기술**: VAE 기반 video diffusion, DDIM 20 steps
- **한국어 LoRA**: `media/lora/latentsync_ko.pt` (AIHub 588영상 50k step 학습, 4.10GB)

#### 5-3. GFPGAN v1.4 (선택, `--enable-postprocess`)
- **역할**: 립싱크 후 face 디테일 복원 (face restoration)
- **입력**: `lipsync.mp4`
- **출력**: `gfpgan.mp4` (2x upscale, async I/O)
- **성능**: prefetch=4 + write_pool=2로 GPU never idle (-22% time)

---

## 4. Daemon 구조 (모델 로딩 시간 절감)

```
port 8901 — CosyVoice3 daemon  (loading 60-90s, ~1GB GPU)
port 8902 — Qwen3-ASR daemon   (loading 30-60s)
port 8903 — DiariZen daemon    (loading 20-30s)
```

각 모델을 메모리 상주시켜 매 실행마다 재로딩 절약. `--smart-daemon` 옵션 사용 시 자동 활용.

---

## 5. 멀티 venv 구조 (Docker container)

```
/opt/venv_lipsync/    # 메인 (orchestrator, LatentSync, CosyVoice)
/opt/venv_asr/        # Qwen3-ASR (transformers 4.46+)
/opt/venv_diarizen/   # DiariZen (pyannote 3.3)
/opt/venv_gfpgan/     # GFPGAN, color_match
```

각 모델의 의존성 충돌 방지. orchestrator는 subprocess로 daemon 호출.

---

## 6. 데이터 흐름 (한 chunk 기준)

```
test4.mp4 (영상)
    │
    ├─ ffmpeg
    │   └─ test4_chunk_000.mp4 (영상 chunk)
    │
    ├─ extract_audio
    │   └─ test4_chunk_000.wav (44.1kHz)
    │
    ├─ BS-Roformer
    │   ├─ vocals/test4_chunk_000_vocals.wav
    │   └─ bgm/test4_chunk_000_bgm.wav
    │
    ├─ Silero VAD
    │   └─ vocals/test4_chunk_000_clean_vocals.wav
    │
    ├─ Qwen3-ASR (daemon)
    │   └─ words[] (단어 + timestamp)
    │
    ├─ DiariZen + ECAPA
    │   └─ diarization (화자 turns) + speaker_centroids
    │
    ├─ LightASD + AV-Fusion
    │   └─ asd_result + per_frame_target + spurious_speakers
    │
    ├─ build_segments + segment_refiner
    │   └─ segments[] (text + speaker + start/end)
    │
    ├─ emotion2vec+ → segment.emotion
    │
    ├─ MOS Evaluator
    │   └─ reference/SPEAKER_xx_Emotion.wav (화자별 reference)
    │
    ├─ LLM 번역
    │   └─ segment.translated_text (한국어)
    │
    ├─ CosyVoice3 (daemon)
    │   └─ dubbed/test4_chunk_000_dubbed.wav (한국어 더빙)
    │
    ├─ ffmpeg mix (loudnorm + BGM)
    │   └─ chunks/test4_chunk_000_final.mp4
    │
    ├─ LatentSync 1.6 (선택)
    │   └─ lipsync.mp4
    │
    └─ GFPGAN async (선택)
        └─ output/test4_ko_..._gfpgan.mp4
```

---

## 7. 성능 (15초 영상 기준)

| 단계 | 시간 | 비율 |
|---|---|---|
| Daemon load (cold) | 1:30 | — |
| Vocal separation (BS-Roformer) | 1:00 | 6% |
| Silero VAD | 0:30 | 3% |
| Qwen3-ASR (daemon) | 1:00 | 6% |
| DiariZen + AV-Fusion | 2:00 | 12% |
| Emotion + MOS | 1:00 | 6% |
| LLM 번역 (batch=7) | 0:30 | 3% |
| CosyVoice TTS | 2:00 | 12% |
| ffmpeg mix | 0:15 | 1% |
| LatentSync 1.6 | 5:00 | 30% |
| GFPGAN async 2x | 3:30 | 21% |
| **Total** | **17:15** | |

100초 영상 (drama): 35-80분 (multi-speaker일수록 빠름)

---

## 8. 사용 방법

### 권장 명령 (강연/인터뷰 — 가장 적합)
```bash
python orchestrator.py \
  --input video.mp4 --lang ko \
  --enable-lipsync \
  --enable-postprocess --postprocess-upscale 2 \
  --smart-daemon
```

### 한국어 LoRA 사용 (품질 ↑)
```bash
python orchestrator.py \
  --input video.mp4 --lang ko \
  --enable-lipsync --use-lora \
  --lipsync-config stage2_512_nf16_smallmask.yaml \
  --enable-postprocess --postprocess-upscale 2 \
  --smart-daemon
```

### 빠른 검증 (dubbing only, lipsync 생략)
```bash
python orchestrator.py \
  --input video.mp4 --lang ko \
  --content-type auto --smart-daemon
```

---

## 9. 알려진 한계

| 항목 | 영향 | 회피 방법 |
|---|---|---|
| 드라마 reverb 잔존 | ECAPA 화자 오인 + voice cloning robotic | dereverb 단계 추가 (CPU DeepFilterNet) |
| DiariZen <1초 turn 못 잡음 | 빠른 화자 교차 누락 | ECAPA sliding window split (5/8 구현) |
| 등돌이/off-screen 화자 | LightASD 실패 → AV-Fusion 무력 | ECAPA + temporal pattern 보조 |
| LatentSync 1.6 sm_120 가속 불가 | 100초 영상 ~35분 | 후처리만 가속 (GFPGAN async) |
| LoRA cheek pink artifact | 광대뼈 색감 시프트 | nf=4 재학습 또는 nf=16 cloud A100 학습 |

---

## 10. 출력 파일 구조

```
media/
├── input/
│   └── test4.mp4
├── runs/
│   └── 20260508_xxxx_test4_xxxxxx/
│       ├── vocals/   # BS-Roformer 결과
│       ├── bgm/
│       ├── reference/  # 화자별 MOS-best reference
│       ├── dubbed/   # CosyVoice 합성 결과
│       └── chunks/   # mix 결과
├── output/
│   └── test4_ko_xxxx.mp4   # 최종 결과
└── reports/
    └── test4_ko_xxxx.json  # 모든 segment 정보 (화자/감정/번역/MOS)
```

---

## 11. 시스템 사양

- **OS**: Windows 11 + Docker Desktop + WSL2
- **GPU**: RTX 5080 (16GB VRAM, sm_120 Blackwell)
- **RAM**: 63GB (Docker 50g 할당)
- **Storage**: E: 1.9TB (모델 캐시 ~60GB, AIHub 538 데이터 ~510GB)
- **Container**: `dubbing_pipeline` (이미지 56GB)

---

**문서 끝.** 처음 보는 사람도 어떤 모델이 어디서 무슨 입출력을 다루는지 한눈에 파악 가능.
