# Full Multilingual Dubbing Pipeline

영상에서 화자 자동 분리 → 화자별 voice clone reference 추출 → LLM 기반 번역 (길이 정확히 맞춤) → CosyVoice3 음성 합성 → 원본 영상에 더빙 오디오 합성. **다음 단계로 립싱크 모델을 적용**할 수 있는 dubbed 영상을 출력.

## 핵심 특징

- **6+ 화자 자동 분리** — n_speakers 하드코딩 없음
- **자동 boundary 보정** — Silero VAD로 segment 시작점 자동 정확화 (사용자가 영상을 보고 알려줄 필요 없음)
- **4단계 길이 제어** (CosyVoice3 + LLM):
  1. Phoneme(자모) 기반 budget (14 jamo/sec)
  2. LABS multi-candidate (LLM이 short/normal/long 3개 한 번에 생성)
  3. Iterative correction (오차 >15%면 측정된 rate로 budget 재계산 후 재번역)
  4. CosyVoice speed (0.85–1.50) fine-tune + speed cap 도달 시 LLM ultra-compress
- **다국어 지원** — 영어 외에 ja/zh/등 (`--language`), 타겟 한국어 외에 다른 언어도 (`--target-lang`)

## 구조

```
full_dubbing_pipeline/
├── 1_diarize.py                 # Stage 1: 화자 분리 + ASR
├── 2_extract_speaker_refs.py    # Stage 2: 화자별 clean ref 추출
├── 3_dub_pipeline.py            # Stage 3: VAD 보정 + 번역 + 합성 + composite
├── example_speaker_config.json  # 화자별 emotion/tone hints (optional)
├── run_pipeline.sh              # End-to-end runner
├── requirements.txt
└── README.md
```

## 설치

```bash
# Python 3.10+ + CUDA 12+ + ~16GB GPU 권장
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# HuggingFace pyannote 토큰 (Stage 1)
# 1. https://huggingface.co/pyannote/speaker-diarization-3.1 약관 동의
# 2. https://huggingface.co/settings/tokens 에서 token 생성
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# LLM API (Stage 3 번역)
export VECTORENGINE_API_KEY=your_key
export VECTORENGINE_BASE_URL=https://api.vectorengine.ai
export VECTORENGINE_MODEL=gpt-5.4
# 또는 OpenAI-호환 다른 endpoint 사용 가능

# CosyVoice3 daemon (Stage 3 합성)
# 본 패키지는 HTTP 클라이언트만 포함. CosyVoice3 daemon은 별도 설치 필요.
# 본 메인 repo의 docker-compose 또는 patches/cosyvoice_daemon.py 참고.
export COSYVOICE_URL=http://127.0.0.1:8901
```

## End-to-End 사용

```bash
# 가장 간단한 방법
./run_pipeline.sh input.mp4 ./output_dir

# 또는 단계별 직접 실행
python 1_diarize.py --input input.mp4 --output out/diarize.json --language en

python 2_extract_speaker_refs.py \
  --video input.mp4 \
  --diarize-json out/diarize.json \
  --out-dir out/refs

python 3_dub_pipeline.py \
  --video input.mp4 \
  --diarize-json out/diarize.json \
  --refs-manifest out/refs/manifest.json \
  --out-dir out/dub \
  --target-lang Korean \
  --speaker-config example_speaker_config.json  # optional
```

**출력**: `out/dub/dubbed.mp4` — 원본 영상 + 한국어 더빙 오디오 합성된 영상.
→ 다음 단계로 LatentSync 등 립싱크 모델에 입력.

## 각 단계 상세

### Stage 1 — `1_diarize.py`
- pyannote 3.1 → segment + speaker
- WhisperX large-v3 → word-level timestamps
- ERes2NetV2 + multi-pass refiner → intra-segment split, contamination 제거
- 출력: `diarize.json` (segments[]: speaker/start/end/text/word_count)

### Stage 2 — `2_extract_speaker_refs.py`
- 각 화자의 최장 clean segment 선택 (5–12s 우선)
- Contamination 필터: 다른 화자 segment ±0.3s 침범 reject
- Peak normalize -3dB로 loudness 통일
- 짧은 ref(<3s)는 0.1s gap으로 loop-augment (CosyVoice 안정성)
- 출력: `{speaker}.wav` per speaker + `manifest.json`

### Stage 3 — `3_dub_pipeline.py`
세부 절차:
1. **VAD boundary refine**: 각 segment 시작점에 대해 Silero VAD onset 탐지. **prev_seg.end 이후만 search** → 옆 화자 발화 침범 차단. |shift|≥0.2s면 자동 보정.
2. **LLM 3-candidate 번역**: short/normal/long 후보를 한 번 호출에 생성. 자모 budget = `target_dur × 14 × {0.8, 1.0, 1.2}`.
3. **3 candidates 합성 + closest 선택**: CosyVoice3로 모두 합성 → target에 가장 가까운 것 채택.
4. **Iterative correction**: off > 15%면 측정된 jamo/sec로 budget 재계산, 두 번째 LLM 호출.
5. **Iterative speed fit**: 합성 결과가 available window 초과면 speed 점진 상향(최대 1.50). 비선형이라 반복 측정 후 재조정.
6. **LLM ultra-compress fallback**: speed cap 도달해도 안 맞으면 LLM에 더 짧은 번역 요청 (부수절 drop OK).
7. **Composite**: 각 clip을 corrected_start에 배치, 다음 non-skip segment 시작 직전에 fade out, mux로 영상에 합성.

추가 출력:
- `translations.json` — 각 segment의 영어 / 한국어 / target_dur / final_dur / corrected_start
- `baseline_vad_corrected.json` — VAD 보정된 segment 정보 (downstream에서 사용 가능)

## Speaker Config (선택)

`example_speaker_config.json`처럼 각 화자에 desc/emotion/tone 설정 가능. LLM 번역 시 어조 일관성, CosyVoice instruct에 감정 전달. **없어도 기본값 사용** (Neutral / natural).

```json
{
  "SPEAKER_03": {
    "desc": "young father — frustrated angry shouting",
    "emotion": "Frustrated",
    "tone": "frustrated shouting"
  }
}
```

## 한계 & Tip

- **다국어 한계**: CosyVoice3는 한/영/중/일 강한 편이지만 한국어 발음에 중국어 억양 잔존. 더 강한 다국어 TTS(Chatterbox Multilingual, IndexTTS-2 등) 교체 시 `CosyVoice` class만 swap하면 됨.
- **메모리**: 16GB GPU 권장. 8GB는 가능하지만 batch_size 조정 필요.
- **첫 실행**: pyannote(~200MB) + WhisperX large-v3(~3GB) + ERes2NetV2(~70MB) + Silero VAD(~2MB) 자동 다운로드, 5–10분 소요.
- **재실행 캐시**: 모델은 캐시 사용, 영상 10s당 처리 ~30–60s (Stage 1+2+3 합).
- **LLM 호출 횟수**: segment당 1–2회 + ultra-compress 0–1회. 26 segments 영상은 보통 30–50 호출.

## License + 모델 출처

| 단계 | 모델 | License |
|---|---|---|
| Diarization | pyannote/speaker-diarization-3.1 | MIT |
| ASR + alignment | whisperx (large-v3) | BSD-2-Clause |
| Voice embedding | ERes2NetV2 (Alibaba ModelScope) | Apache 2.0 |
| VAD | Silero VAD | MIT |
| TTS | CosyVoice3 (Alibaba) | Apache 2.0 |

## Troubleshooting

- **pyannote 401 Unauthorized** → HF token + 모델 약관 동의 재확인
- **ERes2NetV2 download fail** → ModelScope 접근성 (중국 미러) 확인
- **CosyVoice daemon down** → `curl $COSYVOICE_URL/health` 확인, daemon 재기동
- **LLM JSON parse fail** → temperature 낮추기 (현재 0.3), 모델 변경
- **CUDA OOM** → batch_size 줄이기 또는 CPU 모드

## 더 강력한 버전 (참고)

본 standalone은 v195 결과 + v13 길이 제어 + auto VAD 보정만 포함. 메인 repo는 추가로:
- **5-way fusion** (DiariZen + NeMo + pyannote_c1/3.1 + VBx)
- **Visual ASD** (LightASD + InsightFace) per-face speaking detection
- **Focused NeMo re-diarize** on suspicious windows
- 다양한 thresholds + face track continuity

본 패키지는 팀원이 자기 환경에 빠르게 적용 가능한 핵심 버전.

## Next Step (Lip-sync)

Stage 3 출력 `dubbed.mp4`에 LatentSync 또는 SadTalker로 립싱크 적용:
- 입력: `dubbed.mp4` (영상 + 더빙 오디오)
- 출력: 입 움직임이 더빙 한국어와 일치하는 영상
- 본 repo 내 `patches/lipsync_pipeline_*.py` 참고
