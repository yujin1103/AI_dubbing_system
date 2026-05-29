# AI Dubbing System — Automatic Movie/Video Dubbing Pipeline
# AI 더빙 시스템 — 영화/영상 자동 더빙 파이프라인

> **EN** — A modular dubbing pipeline that takes an English video and automatically performs speaker diarization → speech recognition → translation → speech synthesis → (optional) lip-sync.
>
> **KO** — 영어 영상을 입력하면 화자 분리 → 음성 인식 → 번역 → 음성 합성 → 립싱크까지 자동으로 수행하는 모듈러 더빙 파이프라인.

This repository is based on the `restructure-modular` branch. / 본 저장소는 `restructure-modular` 브랜치를 기준으로 모듈화된 파이프라인을 제공합니다.

---

## 파이프라인 개요 / Pipeline Overview

```
입력 영상 (영어)
   │
   ▼
[1] 오디오 추출 (extract_audio)
   │
   ▼
[2] 음원 분리 (separate_audio) ── BGM/음성 분리 (Demucs/BS-RoFormer)
   │
   ▼
[3] 화자 분리 (diarize) ── 4-way fusion (DiariZen + NeMo + pyannote×2)
   │
   ▼
[4] 화자-얼굴 매칭 (face_clustering) ── LightASD + InsightFace ArcFace
   │
   ▼
[5] 화자 정제 (repair patches) ── word/nemo/visual-ASD/face/gap-fill
   │
   ▼
[6] 음성 인식 (run_asr) ── Qwen3-ASR
   │
   ▼
[7] 번역 (translate) ── 문맥 인식 번역 + 길이 제어
   │
   ▼
[8] 감정 분석 (extract_emotion) ── emotion2vec
   │
   ▼
[9] 음성 합성 (run_tts) ── CosyVoice3 (inference_instruct2)
   │
   ▼
[10] 오디오 합성 (compose_audio) ── 타임라인 배치 + 페이드
   │
   ▼
[11] 립싱크 (lipsync, 선택) ── LatentSync + GFPGAN
   │
   ▼
출력 영상 (한국어 더빙)
```

---

## Quick Start / 빠른 시작

```bash
# 1. Start model daemons (TTS / ASR / Diarize / Fusion)
#    모델 데몬 시작
bash src/daemons/start_daemons.sh

# 2. Run the pipeline / 파이프라인 실행
python src/pipeline.py --config configs/default.json --input-video media/input/test4.mp4
```

**EN** — Models load once into long-lived daemons (first load: TTS 60–90 s, ASR 30–45 s, Diarize 20–30 s); afterwards each HTTP call responds in well under a second.
**KO** — 모델은 상시 데몬으로 한 번만 로드됩니다 (최초: TTS 60–90초, ASR 30–45초, Diarize 20–30초). 이후 HTTP 호출은 1초 미만으로 응답합니다.

> **GPU note / GPU 참고:** InsightFace face embedding runs on GPU under `venv_lipsync` (`onnxruntime-gpu`) with `LD_LIBRARY_PATH=""`. pyannote-3.1 loads under `venv_pyann` with `LD_LIBRARY_PATH=""`. / 얼굴 임베딩은 `venv_lipsync`에서 GPU로, pyannote-3.1은 `venv_pyann`에서 로드합니다.

---

## Directory Structure / 디렉토리 구조

```
Capstone_dub/
├── src/                      # Core pipeline code / 핵심 파이프라인 코드
│   ├── pipeline.py           # Pipeline orchestrator (11 steps) / 오케스트레이터
│   ├── diarize.py            # Speaker diarization / 화자 분리
│   ├── face_clustering.py    # Speaker–face matching (LightASD + ArcFace)
│   ├── run_asr.py            # Speech recognition (Qwen3-ASR)
│   ├── translate.py          # Translation / 번역
│   ├── run_tts.py            # Speech synthesis (CosyVoice3)
│   ├── compose_audio.py      # Audio composition / 오디오 합성
│   ├── repair_patches/       # Speaker-refinement patches / 화자 정제 패치
│   ├── daemons/              # Model daemons (TTS/ASR/Diarize/Fusion/pyannote)
│   └── preserved_orchestrator/  # Full end-to-end orchestrator / 전체 오케스트레이터
├── configs/                  # Config files (JSON) / 설정 파일
├── scripts/                  # Utility scripts / 유틸리티 스크립트
├── docs/                     # Documentation / 문서
├── requirements/             # Python dependencies / 의존성
├── docker/                   # Docker assets / 도커 자원
└── media/                    # I/O + ground-truth / 입출력·정답(GT)
    ├── input/                #   Input videos / 입력 영상
    └── gt/                   #   Ground-truth labels for evaluation / 평가용 정답
```

---

## Key Models & Techniques / 주요 기술

| Step / 단계 | Model / Technique · 모델/기법 |
|------|-----------|
| Source separation / 음원 분리 | Demucs / BS-RoFormer |
| Speaker diarization / 화자 분리 | DiariZen (WavLM-large) + NeMo TitaNet + pyannote-3.1 (fusion) |
| Selective local split / 선택적 국소 분할 | pyannote-3.1 split adopted locally (adds short-overlap speakers) |
| Speaker–face / 화자-얼굴 | LightASD (Active Speaker Detection) + InsightFace ArcFace (buffalo_l) |
| Voice merge / 목소리 병합 | ERes2NetV2 speaker-verification embeddings |
| Speech recognition / 음성 인식 | Qwen3-ASR-1.7B |
| Translation / 번역 | Context-aware LLM translation with duration control / 문맥 인식 + 길이 제어 |
| Emotion / 감정 분석 | emotion2vec-large |
| Speech synthesis / 음성 합성 | CosyVoice3-0.5B (`inference_instruct2`) |
| Lip-sync / 립싱크 | LatentSync + GFPGAN |

---

## Design Principles / 설계 원칙

- **Fully automatic — no per-video hardcoding.** Speaker count is auto-decided (`num_speakers=null`); thresholds are uniform across videos, not hand-tuned per clip.
  **완전 자동 — 영상별 하드코딩 없음.** 화자 수는 자동 결정, 임계값은 모든 영상에 동일하게 적용.
- **No video-specific rules.** Rules that fit one clip overfit and break on others, so they are avoided.
  **영상 전용 규칙 금지.** 특정 영상에만 맞춘 규칙은 다른 영상에서 깨지므로 추가하지 않음.
- **Multi-signal fusion.** Audio (diarization), voice embeddings, face identity, and on-screen active-speaker detection are combined to cover each model's blind spots.
  **다중 신호 결합.** 음성·목소리·얼굴·화면(ASD)을 함께 사용해 단일 모델 약점 보완.
- **Open-source only**, with source/maintainer verification before installing external models or packages.
  **오픈소스 기반**, 외부 모델·패키지는 출처 확인 후 사용.

---

## Diarization Results / 화자 분리 결과

Evaluated fully automatically (caches cleared, no per-video tuning) against ground truth. / 캐시 삭제·영상별 튜닝 없이 자동 실행한 결과를 정답(GT)과 비교.

| Video | Score | Speaker count (det / GT) | Notes / 비고 |
|---|---|---|---|
| **test4** | **1.1167** | **6 / 6 ✓** | man1·woman1·paramedic·dad·mom = 1.0; sean 0.5 (0.6 s utterance) |
| **test5** | **1.1143** | **4 / 4 ✓** (+BG) | dad·BG = 1.0; doctor 0.83, mom 0.67, sean 0.57 (short shouts) |

**EN** — `score = mean per-speaker consistency + speaker-count-match bonus`. The system outputs anonymous labels (`SPEAKER_00…`); the scorer maps each GT speaker's segments to the most time-overlapping detected cluster and measures consistency. Speaker **count** is matched exactly for both; remaining error is confined to sub-second utterances/shouts (an intrinsic signal limit).

**KO** — `점수 = 화자별 일관성 평균 + 화자 수 일치 보너스`. 시스템은 익명 라벨(`SPEAKER_00…`)을 출력하며, 채점기가 각 GT 화자를 시간이 가장 겹치는 검출 클러스터에 매칭해 일관성을 측정합니다. 두 영상 모두 화자 **수**는 정확히 일치하고, 남은 오차는 1초 미만 짧은 발화/외침에 한정된 본질적 한계입니다.

---

## License / 라이선스

For academic/research use. Each model follows its own license. / 학술·연구 목적. 각 모델의 라이선스를 따릅니다.
