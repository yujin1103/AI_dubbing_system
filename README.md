# AI Dubbing System — Automatic Speaker Diarization + Multilingual Dubbing Pipeline
# AI 더빙 시스템 — 자동 화자 분리 + 다국어 더빙 파이프라인

> 🌐 **English** below · **한국어** 원문은 [여기로](#한국어-원문) 이동.

---

## English

A system that automatically dubs movie/drama videos into another language. From speaker diarization → translation → TTS → composition, **everything runs fully automatically with no per-video hardcoding.**

Pipeline: extract vocals from the source audio with BS-RoFormer → 4-way fusion diarization → Qwen3-ASR transcription → vectorengine GPT translation → CosyVoice3 speech synthesis → ffmpeg mux. Every stage is isolated into its own venv daemon for stability.

### Key Features

- **Video-agnostic automatic diarization**: 4-way fusion (DiariZen + NeMo + pyannote-3.1) + adaptive heuristic auto-decides per-video `OUTLIER_FAR_THRESH` / `gap_fill mm`. No sweep hardcoding.
- **Per-speaker face mapping**: LightASD + insightface ArcFace 512-dim cosine clustering auto-extracts a representative face thumbnail per speaker (43–89 clusters).
- **Multi-thr consensus algorithm**: runs e2e simultaneously with various `OUTLIER_FAR_THRESH` values, then auto-detects the main_count plateau to adopt the best thr.
- **8 repair patches**: gap_fill / word_level_split / focused_nemo_split / face_cluster_match / visual_asd_reassign / postprocess_reassign_text / boost_subchunk_asr / sweep_gt_match.
- **GT-based automatic validation**: auto-measures accuracy + DER (Diarization Error Rate) + segment-level mapping accuracy.

### Measured Accuracy (re-verified 2026-05-30)

Fully automatic — caches cleared, uniform params, scored against ground truth:

| Test | Length | Speaker count (det/GT) | Consistency | Score | GT mapping |
|---|---|---|---|---|---|
| test4.mp4 (Good Doctor) | 108s | **6 / 6 ✓** | **0.92** | **1.1167** | 24/26 = 92.3% |
| test5.mp4 | 84s | **4 / 4 ✓** (+BG) | 0.81 | **1.1143** | 16/19 = 84.2% |

Speaker **count** is auto-matched exactly for both, and main speakers are mostly perfectly consistent (1.0). Remaining error is confined to sub-second utterances/shouts (intrinsic signal limit: off-screen voice + same-gender speech).

**What each metric means** (the system never sees who the people are — it only outputs anonymous labels `SPEAKER_00, SPEAKER_01, …`; the scorer matches those to the ground-truth (GT) people by time overlap):

- **Speaker count (det / GT)** — how many distinct people the system found vs. how many actually speak. `6 / 6 ✓` means it auto-found exactly the right number, with no count hint given.
- **Consistency (0–1)** — for each real person, the fraction of their lines that landed in *one* label. `1.0` = all of that person's speech was grouped as a single speaker (perfect); `0.5` = their lines were split across two labels. The table shows the average over all people.
- **Score** — a single summary number: `mean consistency + 0.2 (if speaker count matches) + 0.1 (if a background speaker was detected)`. Above `1.0` means high consistency *and* the count matched.
- **GT mapping** — of all the labeled GT segments, how many were assigned to the correct person. `24/26 = 92.3%`.

<details><summary>Stricter frame/segment-level metrics (older preserved run)</summary>

| Test | DER (↓ better) | Segment acc. |
|---|---|---|
| test4 | 25.38% | 78.2% |
| test5 | – | 78.0% |

- **DER (Diarization Error Rate, lower is better)** — the field-standard metric. Over every moment of audio it sums the time wrongly labeled (wrong speaker + missed speech + speech detected during silence). Strict, because a single short boundary or overlap error counts against you. **25% DER is normal-to-good** for multi-speaker drama.
- **Segment accuracy** — the share of detected segments whose speaker label matches GT at the segment level. Lower than the consistency headline because it penalizes every short boundary mismatch.

These are stricter low-level metrics, not a contradiction of the high speaker-level accuracy above.
</details>

### System Architecture — 9-service microservices (teammate structure + our validated assets)

| Service | Role | venv |
|---|---|---|
| `controller` | CPU stages (ffmpeg, JSON conversion, translate API, build_timeline, mux) | system |
| `separator` | BS-RoFormer 4-stem + silero-vad | system |
| `diarizer` | 4-way fusion (DiariZen WavLM + NeMo TitaNet + pyannote-3.1) | venv_diarizen |
| `pyannote` | pyannote 4.0 community-1 (isolated) | system |
| `speaker` | ERes2NetV2 (voice 512-dim) + emotion2vec + Qwen3-ASR + ForcedAligner | venv_asr |
| `tts-cosyvoice` | CosyVoice3-0.5B inference_instruct2 (fade-in/out applied) | venv_lipsync |
| `face` (new) | LightASD speaking score + insightface ArcFace face cluster + thumbnail jpg | system |
| `webapp-backend` | FastAPI + docker socket (orchestration) | system |
| `webapp-frontend` | Vite + React + TS + Tailwind | Node.js |

### Pipeline — 17 steps (`src/pipeline.py`)

```
extract_audio → separate_audio → redirect_nonspeech → diarize → rttm_to_json
  → face_clustering (NEW) → apply_preserved_repair (NEW) → merge_chunks
  → cut_chunks → extract_emotion → run_asr → translate → build_timeline
  → generate_tts_instructions → run_tts → validate_tts → compose_audio → mux
```

`step_router.py` routes each step to the appropriate service; `webapp-backend` invokes `docker compose exec` via the docker socket.

### Automatic Algorithms (video-agnostic, no hardcoding)

| # | Algorithm | File | Role |
|---|---|---|---|
| 1 | adaptive thr/mm recommendation | `src/adaptive_thr.py` | recommends `OUTLIER_FAR_THRESH` + gap_fill mm from raw SPK stats |
| 2 | **multi-thr consensus** | `src/multi_thr_consensus.py` | runs multiple thr e2e, auto-adopts max main_count + BG priority |
| 3 | auto thr retry | `src/auto_thr_decision.py` | analyzes 1st result → auto-retries a different thr if insufficient |
| 4 | face_clustering + thumbnail | `src/face_clustering.py` | LightASD + ArcFace + intra-segment face split + SPK split + thumbnail jpg |
| 5 | apply_repair_patches | `src/apply_repair_patches.py` | batch-invokes 8 patches (word_split + focused_nemo + gap_fill + ...) |
| 6 | time_context_merge | `src/time_context_merge.py` | auto-reassigns sandwiched short outlier SPK |
| 7 | voice_safe_merge | `src/voice_safe_merge.py` | safe merge only when voice cosine sim ≥ 0.85 (+face cross-evidence) |
| 8 | auto_refine | `src/auto_refine.py` | adaptive threshold + voice + face cross-evidence |
| 9 | preserved_fusion | `src/preserved_fusion.py` | 4-way fusion daemon HTTP client |
| 10 | build_ui_metadata | `src/build_ui_metadata.py` | integrates per-speaker face thumbnail + segments + emotion + translation metadata |
| 11 | compute_der | `scripts/compute_der.py` | pyannote.metrics DER + segment-level accuracy |
| 12 | validate_against_gt | `scripts/validate_against_gt.py` | GT-based per-speaker consistency |

### Design Principles

- **Fully automatic — no per-video hardcoding.** Speaker count is auto-decided (`num_speakers=null`); thresholds are uniform across videos.
- **No video-specific rules.** Rules tuned to one clip overfit and break elsewhere, so they are avoided.
- **Multi-signal fusion.** Audio (diarization), voice embeddings, face identity, and on-screen ASD are combined to cover each model's blind spots.
- **Open-source only**, with source/maintainer verification before installing external models or packages.

### Quick Start

```bash
# 1) Start daemons (4-way fusion + cosy + asr)
bash src/daemons/start_daemons.sh

# 2) Run the pipeline
python src/pipeline.py --config configs/default.json --input-video media/input/test4.mp4
```

> **GPU note:** InsightFace face embedding runs on GPU under `venv_lipsync` (`onnxruntime-gpu`) with `LD_LIBRARY_PATH=""`; pyannote-3.1 loads under `venv_pyann` with `LD_LIBRARY_PATH=""`.

(Full build / model-download / e2e / multi-thr / GT-validation commands are in the **한국어 원문** below — they are language-neutral shell snippets.)

### Diarization Results

Evaluated fully automatically (caches cleared, no per-video tuning) against ground truth.

| Video | Score | Speaker count (det / GT) | Notes |
|---|---|---|---|
| **test4** | **1.1167** | **6 / 6 ✓** | man1·woman1·paramedic·dad·mom = 1.0; sean 0.5 (0.6 s utterance) |
| **test5** | **1.1143** | **4 / 4 ✓** (+BG) | dad·BG = 1.0; doctor 0.83, mom 0.67, sean 0.57 (short shouts) |

`score = mean per-speaker consistency + speaker-count-match bonus`. The system outputs anonymous labels (`SPEAKER_00…`); the scorer maps each GT speaker's segments to the most time-overlapping detected cluster and measures consistency. Speaker **count** matches exactly for both; remaining error is confined to sub-second utterances/shouts (an intrinsic signal limit).

### Intrinsic Limits (cannot be solved automatically)

| Limit case | Cause | Mitigation |
|---|---|---|
| off-screen voice + same-gender speech | speaker off-camera → no face + similar voice (male↔male, female↔female) | webapp UI inline SPK editor (Phase 3) — 1-click user correction |
| shout acoustic shift | mom shout F0 253Hz vs dad 237Hz — nearly identical | add emotion2vec + face_clustering |
| DiariZen non-determinism | slightly different result per run (GPU float) | `torch.manual_seed` + cuDNN deterministic |
| shared SPK | doctor/sean/mom partly share one SPK | over-merge gap_fill cannot separate perfectly |

### Models & Libraries

| Model | Role | License |
|---|---|---|
| DiariZen (WavLM-large-s80-md-v2) | speaker diarization | research |
| NeMo TitaNet-Large | speaker verification + clustering | Apache 2.0 |
| pyannote/speaker-diarization-3.1 | diarization | MIT |
| ERes2NetV2 | voice embedding | Apache 2.0 |
| LightASD (Junhua-Liao) | active speaker detection | research |
| insightface ArcFace (buffalo_l) | face embedding | research |
| Qwen3-ASR-1.7B + ForcedAligner-0.6B | ASR | Tongyi Open |
| CosyVoice3-0.5B (Fun-AudioLLM) | TTS inference_instruct2 | Tongyi Open |
| BS-RoFormer | vocals/instruments separation | MIT |
| silero-vad | voice activity detection | MIT |
| emotion2vec_plus_large | emotion classification | Apache 2.0 |

### Runtime Environment & Experiment Log (2026-06-18)

**Verified local runtime** — single self-contained root `Capstone_dub/`:

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5080, 16 GB VRAM |
| Host | Windows 11 + WSL2 + Docker Desktop |
| Driver / CUDA | driver CUDA 13.2; container base images `nvidia/cuda:12.1.1` / `12.4.1` |
| Python | 3.10 (CPU services) – 3.12 (GPU services), one venv per service |

Models load fully offline from the in-repo `media/model_cache` mount. `docker-compose.override.yml` repins `HF_HOME` / `MODELSCOPE_CACHE` / `TORCH_HOME` / `ERES2NETV2_CACHE_DIR` to `/workspace/project/...` so no re-download occurs. `src/daemon_lifecycle.py` juggles the diarize (8903) / ASR (8902) daemons one at a time to fit the single 16 GB GPU. Full verified model-path table: `PROJECT_CONTEXT.md`.

**Active docker-compose services**: `controller, separator, diarizer, pyannote, face, speaker, vibevoice-asr, tts-cosyvoice, webapp-backend, webapp-frontend`.

**Auxiliary services & experiments**: `pyannote`, `face` (visual ASD), and `vibevoice-asr` run as separate compose services — `pyannote` + NeMo feed the 4-way diarization fusion (active `diarization.engine = fusion_4way`), `face` does LightASD + insightface speaker re-assignment, `vibevoice-asr` is an alternate ASR backend. A trained **MOS evaluator** (`src/mos_evaluator.py`, wav2vec2 + MLP head, scores dubbed speech 1–5) exists for TTS naturalness but is not yet wired into the active `validate_tts` (`tts.validation.enabled=false` by default). A separate real-time speech-to-speech interpreter prototype lives in `realtime_interpreter/` (not part of the batch pipeline).

**Recent ASR experiments**: full-audio ASR + Viterbi word-to-chunk assignment to cut short-chunk word-drop / hallucination — now consolidated into the modular `src/run_asr.py` and `src/repair_patches/boost_subchunk_asr.py` (the earlier standalone `scripts/full_boost_asr.py` / `window_asr.py` prototypes were folded in and removed).

---
---

## 한국어 (원문)

영화/드라마 영상을 자동으로 다국어 더빙하는 시스템. 화자 분리 → 번역 → TTS → 합성까지 **영상별 hardcoding 없이 완전 자동**으로 수행.

원본 음성에서 BS-RoFormer로 boca 추출 → 4-way fusion diarization → Qwen3-ASR 전사 → vectorengine GPT 번역 → CosyVoice3 음성 합성 → ffmpeg mux. 모든 단계가 venv 격리 daemon으로 분리되어 안정성 보장.

## 주요 특징

- **영상 무관 자동 화자 분리**: 4-way fusion (DiariZen + NeMo + pyannote-3.1) + adaptive heuristic으로 영상별 `OUTLIER_FAR_THRESH` / `gap_fill mm` 자동 결정. sweep hardcoding 없음.
- **각 화자 얼굴 매핑**: LightASD + insightface ArcFace ArcFace 512-dim cosine clustering으로 SPK → 대표 face thumbnail 자동 추출 (43-89 cluster).
- **multi-thr 합의 알고리즘**: 다양한 `OUTLIER_FAR_THRESH` 값으로 동시 e2e 실행 후 main_count plateau 자동 검출하여 best thr 채택.
- **8 repair patches**: gap_fill / word_level_split / focused_nemo_split / face_cluster_match / visual_asd_reassign / postprocess_reassign_text / boost_subchunk_asr / sweep_gt_match.
- **GT 기반 자동 검증**: 정확도 + DER (Diarization Error Rate) + segment-level 매핑 정확도 자동 측정.

## 실측 정확도 (2026-05-30 재검증)

완전 자동 — 캐시 삭제·통일 파라미터·GT 대비:

| Test | 영상 길이 | 화자 수 (검출/GT) | 일관성 | score | GT 매핑 |
|---|---|---|---|---|---|
| test4.mp4 (Good Doctor) | 108s | **6 / 6 ✓** | **0.92** | **1.1167** | 24/26 = 92.3% |
| test5.mp4 | 84s | **4 / 4 ✓** (+BG) | 0.81 | **1.1143** | 16/19 = 84.2% |

화자 **수**는 둘 다 자동으로 정확히 일치, 주요 화자는 대부분 일관성 1.0. 남은 오차는 1초 미만 짧은 발화/외침에 한정된 본질적 한계(off-screen voice + 동성 발화).

**각 지표 설명** (시스템은 사람이 누구인지 모릅니다 — 익명 라벨 `SPEAKER_00, SPEAKER_01, …`만 출력하고, 채점기가 시간 겹침으로 정답(GT) 인물과 매칭합니다):

- **화자 수 (검출/GT)** — 시스템이 찾은 사람 수 vs 실제 말한 사람 수. `6 / 6 ✓` = 화자 수 힌트 없이 정확한 인원을 자동으로 찾음.
- **일관성 (0–1)** — 각 실제 인물의 발화 중 *하나의* 라벨로 묶인 비율. `1.0` = 그 사람의 모든 발화가 한 화자로 묶임(완벽), `0.5` = 두 라벨로 쪼개짐. 표의 값은 전체 인물 평균.
- **score** — 한 줄 요약 숫자: `일관성 평균 + 0.2(화자 수 일치 시) + 0.1(배경 화자 검출 시)`. `1.0` 초과 = 일관성이 높으면서 화자 수도 맞음.
- **GT 매핑** — 라벨된 GT 구간 중 올바른 인물에 배정된 비율. `24/26 = 92.3%`.

<details><summary>더 엄격한 frame/segment 단위 지표 (옛 preserved run)</summary>

| Test | DER (↓ 좋음) | segment 정확도 |
|---|---|---|
| test4 | 25.38% | 78.2% |
| test5 | – | 78.0% |

- **DER (Diarization Error Rate, 낮을수록 좋음)** — 분야 표준 지표. 오디오의 매 순간에 대해 잘못 라벨된 시간(틀린 화자 + 놓친 발화 + 묵음 구간을 발화로 검출)을 모두 합산. 짧은 경계·겹침 오차 하나도 감점되어 엄격함. 다화자 드라마에서 **25% DER은 정상~양호**.
- **segment 정확도** — 검출된 구간 중 화자 라벨이 GT와 segment 단위로 맞는 비율. 짧은 경계 불일치를 모두 감점하므로 일관성 헤드라인보다 낮음.

모순이 아니라 더 엄격한 저수준 지표입니다.
</details>

## 시스템 구성

### 9 services 마이크로서비스 (팀원 구조 + 우리 검증 자산)

| Service | 역할 | venv |
|---|---|---|
| `controller` | CPU 단계 (ffmpeg, JSON 변환, translate API, build_timeline, mux) | system |
| `separator` | BS-RoFormer 4-stem + silero-vad | system |
| `diarizer` | 4-way fusion (DiariZen WavLM + NeMo TitaNet + pyannote-3.1) | venv_diarizen |
| `pyannote` | pyannote 4.0 community-1 (격리) | system |
| `speaker` | ERes2NetV2 (voice 512-dim) + emotion2vec + Qwen3-ASR + ForcedAligner | venv_asr |
| `tts-cosyvoice` | CosyVoice3-0.5B inference_instruct2 (fade-in/out 적용) | venv_lipsync |
| `face` (신규) | LightASD speaking score + insightface ArcFace face cluster + thumbnail jpg | system |
| `webapp-backend` | FastAPI + docker socket (오케스트레이션) | system |
| `webapp-frontend` | Vite + React + TS + Tailwind | Node.js |

### Pipeline 17 steps (`src/pipeline.py`)

```
extract_audio → separate_audio → redirect_nonspeech → diarize → rttm_to_json
  → face_clustering (NEW) → apply_preserved_repair (NEW) → merge_chunks
  → cut_chunks → extract_emotion → run_asr → translate → build_timeline
  → generate_tts_instructions → run_tts → validate_tts → compose_audio → mux
```

`step_router.py`가 각 step을 적절한 service로 라우팅. `webapp-backend`가 docker socket을 통해 `docker compose exec` 호출.

## 자동 알고리즘 (영상 무관, hardcoding 없음)

| # | 알고리즘 | 파일 | 역할 |
|---|---|---|---|
| 1 | adaptive thr/mm 추천 | `src/adaptive_thr.py` | raw SPK stats 기반 `OUTLIER_FAR_THRESH` + gap_fill mm 자동 추천 |
| 2 | **multi-thr 합의** | `src/multi_thr_consensus.py` | thr 여러 값 동시 e2e 후 main_count 최대 + BG 우선 자동 채택 |
| 3 | 자동 thr retry | `src/auto_thr_decision.py` | 1차 결과 분석 → 부족 시 다른 thr 자동 재시도 |
| 4 | face_clustering + thumbnail | `src/face_clustering.py` | LightASD + ArcFace + intra-segment face split + SPK split + thumbnail jpg |
| 5 | apply_repair_patches | `src/apply_repair_patches.py` | 8 patches 일괄 호출 (word_split + focused_nemo + gap_fill + ...) |
| 6 | time_context_merge | `src/time_context_merge.py` | sandwich된 짧은 outlier SPK 자동 reassign |
| 7 | voice_safe_merge | `src/voice_safe_merge.py` | voice cosine sim ≥ 0.85 (+face cross-evidence) 만 안전 merge |
| 8 | auto_refine | `src/auto_refine.py` | adaptive threshold + voice + face cross-evidence |
| 9 | preserved_fusion | `src/preserved_fusion.py` | 4-way fusion daemon HTTP 클라이언트 |
| 10 | build_ui_metadata | `src/build_ui_metadata.py` | 화자별 face thumbnail + segments + 감정 + 번역 통합 metadata |
| 11 | compute_der | `scripts/compute_der.py` | pyannote.metrics DER + segment-level 정확도 |
| 12 | validate_against_gt | `scripts/validate_against_gt.py` | GT 기반 per-speaker consistency 측정 |

## 실측 시간

### 단일 영상 처리 (영상 무관 default)

| 단계 | 시간 |
|---|---|
| daemons 기동 (4-way fusion + cosy + asr) | 60-75s |
| e2e (extract+separate+diarize+ASR+translate+TTS+mux) | 12-20분 |
| face_clustering (LightASD + ArcFace + thumbnail) | 14분 |
| adaptive mm + gap_fill | 1-2분 |
| GT validation | 1초 |
| **합 (단일 thr)** | **약 30분** |

### Multi-thr 합의 (진짜 자동)

| 단계 | 시간 |
|---|---|
| daemons 기동 1회 | 75s |
| e2e thr=0.40 | 12-18분 |
| e2e thr=0.50 | 12-20분 |
| face_clustering 1회 | 14분 |
| multi_thr_consensus + adaptive | 1분 |
| **합 (multi-thr 2개)** | **약 40-50분** |

GPU: 16GB VRAM (RTX 5080) 동시 사용 가능 한도. cosy + asr + 4 diarize daemons + face = 약 14-15 GiB.

## 빠른 시작

### 1. 환경 준비

```bash
# Docker Desktop 또는 Docker Engine
docker --version

# .env 작성 (vectorengine API 키 등)
cp .env.example .env
# .env 편집: VECTORENGINE_API_KEY, HF_TOKEN
```

### 2. 빌드

```bash
# 단일 dubbing_pipeline 컨테이너 (모든 venv 통합, 보존된 환경)
docker build -f docker/Dockerfile.preserved-base -t tts_base:latest .
docker build -f docker/Dockerfile.preserved-pipeline -t dubbing_pipeline:latest .

# face service (LightASD + ArcFace)
docker build -f docker/Dockerfile.face -t movie-dubbing/face:local .

docker compose -f docker-compose.preserved.yml up -d
```

또는 팀원 형식 (9 services 분리):

```bash
docker compose up -d  # docker-compose.yml (9 services 분리)
```

### 3. 모델 다운로드

```bash
hf auth login  # HuggingFace 로그인 (gated 모델)

# pyannote/speaker-diarization-3.1 cache 받기
docker exec dubbing_pipeline /opt/venv_diarizen/bin/huggingface-cli download pyannote/speaker-diarization-3.1
```

LightASD weight + S3FD face detector weight은 face 컨테이너 빌드 시 자동 clone.

### 4. e2e 실행 (단일 영상)

```bash
# daemons 기동
docker exec dubbing_pipeline bash /workspace/patches/start_daemons.sh

# 4-way fusion sub-daemons 추가
docker exec -d dubbing_pipeline bash -c "nohup /opt/venv_diarizen/bin/python /workspace/patches/nemo_diarize_daemon.py --port 8923 &"
docker exec -d dubbing_pipeline bash -c "PYANNOTE_MODEL=pyannote/speaker-diarization-3.1 nohup /opt/venv_diarizen/bin/python /workspace/patches/pyannote_diarize_daemon.py --port 8943 &"
docker exec -d dubbing_pipeline bash -c "FUSION_DIARIZEN_URL=http://127.0.0.1:8903 FUSION_NEMO_URL=http://127.0.0.1:8923 FUSION_PYANNOTE2_URL=http://127.0.0.1:8943 nohup /opt/venv_diarizen/bin/python /workspace/patches/fusion_diarize_daemon.py --port 8918 &"

# e2e (영상 무관 default thr=0.40)
docker exec -e LATENTSYNC_OUTLIER_OFF=0 -e LATENTSYNC_OUTLIER_FAR_THRESH=0.40 -e DIARIZE_DAEMON_URL=http://127.0.0.1:8918 \
    dubbing_pipeline python /workspace/orchestrator.py \
    --input /workspace/media/input/test.mp4 \
    --name test --lang ko --content-type drama --smart-daemon
```

### 5. face_clustering + 자동 후처리

```bash
# face_clustering (영상 1편당 14분)
docker exec movie-dubbing-project-face-1 bash -c \
  "cd /workspace/project && /usr/bin/python src/face_clustering.py \
    media/runs/<RUN_ID>/chunks \
    media/runs/<RUN_ID>/meta/test_chunk_000_segments.json \
    --out-face-clusters media/runs/<RUN_ID>/meta/face_clusters.json \
    --out-remapped media/runs/<RUN_ID>/meta/diarization_face_matched.json"

# adaptive mm 추천 + apply_repair_patches 자동
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python \
  /workspace/Capstone_dub_src/adaptive_thr.py \
  /workspace/media/runs/<RUN_ID>/meta/test_chunk_000_segments.json

# 추천된 mm 적용
docker exec dubbing_pipeline /opt/venv_diarizen/bin/python \
  src/apply_repair_patches.py /workspace/media/runs/<RUN_ID> \
  --main-merge 0.50 --bg-merge 0.30 --sim-match 0.45 --pad 0.5
```

### 6. multi-thr 합의 (진짜 자동, 시간 큼)

```bash
# 동일 영상에 thr 다른 값으로 N번 e2e
for THR in 0.30 0.40 0.50; do
  docker exec -e LATENTSYNC_OUTLIER_OFF=0 -e LATENTSYNC_OUTLIER_FAR_THRESH=$THR \
    -e DIARIZE_DAEMON_URL=http://127.0.0.1:8918 \
    dubbing_pipeline python /workspace/orchestrator.py \
    --input /workspace/media/input/test.mp4 \
    --name test_thr${THR/./} --lang ko --content-type drama --smart-daemon
done

# 합의로 best thr 자동 채택
docker exec dubbing_pipeline python /workspace/Capstone_dub_src/multi_thr_consensus.py --thr-run \
  0.30:/workspace/media/runs/test_thr030 \
  0.40:/workspace/media/runs/test_thr040 \
  0.50:/workspace/media/runs/test_thr050
```

### 7. GT 검증

```bash
# GT 파일 작성 (예: media/gt/test_gt.json)
# {"main_count": 4, "bg_count": 1, "main_speakers": [...], "segments": [...]}

docker exec dubbing_pipeline /opt/venv_diarizen/bin/python \
  /workspace/full_dubbing_pipeline/validate_against_gt.py \
  /workspace/media/runs/<RUN_ID>/meta/test_chunk_000_segments_gapfilled.json \
  /workspace/media/gt/test_gt.json out.json

# DER + segment 정확도
docker exec dubbing_pipeline python \
  /workspace/full_dubbing_pipeline/compute_der.py \
  /workspace/media/runs/<RUN_ID>/meta/test_chunk_000_segments_gapfilled.json \
  /workspace/media/gt/test_gt.json out_der.json
```

## 자동 알고리즘 흐름 (영상 1편 처리 시)

```
[1] e2e (orchestrator.py)
    └ extract_audio → separate (BS-RoFormer)
    └ diarize (4-way fusion: DZ + NeMo + pyannote-3.1)
    └ ASR (Qwen3-ASR) + translate (vectorengine GPT) + TTS (CosyVoice3)
    └ raw segments.json 생성

[2] adaptive_thr.py 자동 분석
    └ raw SPK 분포 (main / outlier)
    └ heuristic mm 추천:
       - outlier > main, main 1-2 → mm=0.40
       - outlier > main, main ≥3 → mm=0.50 (test5 case)
       - outlier ≥ main/2 → mm=0.50 (test4 case)
       - outlier 1-2 → mm=0.55
       - outlier 0 → mm=0.99 (보존 default)

[3] face_clustering.py (영상 무관 default)
    └ LightASD subprocess → tracks + speaking scores
    └ insightface ArcFace 512-dim → cosine greedy (sim ≥ 0.4)
    └ SPK split (한 SPK가 여러 face cluster → 새 SPK 라벨)
    └ intra-segment face split (한 segment 내 face cluster 변경 시 자동 split)
    └ face thumbnail jpg 자동 추출 (cluster_NNN.jpg)
    └ speaker_face_map (SPK ↔ face cluster + thumbnail) 생성

[4] apply_repair_patches.py (영상 무관 default)
    └ word_level_split (F0 jump + LR cos)
    └ focused_nemo_split (NeMo 재 diarize)
    └ visual_asd_reassign
    └ face_cluster_match (보존 logic)
    └ gap_fill (adaptive mm/bm/sm)
    └ postprocess_reassign_text

[5] voice_safe_merge.py (선택)
    └ SPK 쌍 voice cosine sim 계산
    └ face cross-evidence + sim ≥ 0.85 만 안전 merge

[6] GT validation (있을 때)
    └ per-GT-speaker consistency
    └ DER (pyannote.metrics)
    └ segment-level 정확도
```

## 자동 알고리즘 한계 (본질적, 자동 풀 수 없음)

| 한계 케이스 | 원인 | 해결 방법 |
|---|---|---|
| **off-screen voice + 동성 발화** | 카메라가 다른 사람 보는 동안 발화 → face 안 보임 + voice 비슷 (남성 끼리, 여성 끼리) | webapp UI 인라인 SPK 에디터 (Phase 3) — 사용자가 1-click 보정 |
| **외침 voice acoustic shift** | mom 외침 F0 253Hz vs dad 237Hz — F0 거의 동일 | emotion2vec 추가 + face_clustering 필요 |
| **DiariZen non-determinism** | 같은 영상에 약간 다른 결과 (GPU 부동소수점) | `torch.manual_seed` + cuDNN deterministic (코드 수정 필요) |
| **공유 SPK** | 의사/션/엄마 발화 일부가 한 SPK 공유 | over-merge gap_fill 단계가 정확한 분리 못 함 |

본질적 한계는 **webapp UI 인라인 SPK 에디터** (사용자 1-click 보정 + 부분 재합성)로 해결.

## 검증 자산 (`references/preserved/`)

```
references/preserved/
├── BEST_BASELINE_v194.json + .md     # 검증된 baseline (test4 ≈ 90.3%)
├── test4_gt.json + test5_gt.json     # 사용자 작성 GT 라벨
├── sweep_results_test4.json          # 40 config grid sweep 결과
├── runs/                             # 검증된 run dirs (segments_*.json)
│   ├── test4_v305_full/              # 8 stage segments
│   ├── test5_v305_full/
│   ├── test5_v195env_outlier_on/
│   └── t4_verify_th070/
├── validation/                       # GT 비교 결과
│   ├── val_test4_face_arcface.json
│   ├── val_test4_fusion_4way.json
│   ├── val_test4_FINAL_FINAL.json    # ★ test4 best (0.9897, main=6 ✓)
│   ├── val_test5_FRESH_FINAL.json    # ★ test5 best (1.1381, main=4 ✓)
│   ├── der_test4_FINAL.json
│   ├── e2e_full_pipeline/            # e2e 전체 + multi-thr + auto ceiling
│   └── e2e_integration/              # Phase 1 통합 후 1차 검증
└── face_thumbnails_sample/           # face cluster jpg 10개 sample
```

## 문서 (`docs/preserved/`)

| 문서 | 내용 |
|---|---|
| `STATUS.md` | canonical project state |
| `EXPERIMENT_LOG.md` | v17~v305 실험 로그 |
| `DIARIZATION_SWEEP_LOG.md` | sweep 결과 |
| `VALIDATION_RESULTS.md` | GT 검증 요약 |
| `DER_AND_ACCURACY.md` | DER + segment 정확도 분석 |
| `FINAL_PIPELINE_BEST.md` | 최종 best 결과 |
| `AUTO_LIMIT_ANALYSIS.md` | 자동 알고리즘 한계 분석 |
| `AUTO_ALGORITHMS_FINAL.md` | 5개 자동 알고리즘 차례 적용 결과 |
| `INTEGRATION_STATUS.md` | Phase 1 통합 현황 |

## 환경 변수 (영상 무관 default)

```bash
# diarize
LATENTSYNC_OUTLIER_OFF=0           # outlier 검출 활성
LATENTSYNC_OUTLIER_FAR_THRESH=0.40 # default (multi-thr 합의로 자동 채택 가능)

# v178 baseline
TIME_GAP_SPLIT=1
ASD_GATED_FACE_TRACK=1
SINGLETON_SKIP=1

# v190
VISUAL_ASD_TRIGGER=1
FOCUSED_NEMO_RE_DIARIZE=1

# v194 word_level_split
WORD_LEVEL_SPLIT=1
WORD_F0_JUMP_HZ=100  # 영상 무관
WORD_LR_COS_MAX=0.35
WORD_SIDE_SIM_MIN=0.40
```

## 모델 + 라이브러리

| 모델 | 역할 | 라이센스 |
|---|---|---|
| DiariZen (WavLM-large-s80-md-v2) | speaker diarization | research |
| NeMo TitaNet-Large | speaker verification + clustering | Apache 2.0 |
| pyannote/speaker-diarization-3.1 | diarization | MIT |
| ERes2NetV2 (iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common) | voice embedding | Apache 2.0 |
| LightASD (Junhua-Liao) | active speaker detection | research |
| insightface ArcFace (buffalo_l) | face embedding | research |
| Qwen3-ASR-1.7B + ForcedAligner-0.6B | ASR | Tongyi Open |
| CosyVoice3-0.5B (Fun-AudioLLM) | TTS inference_instruct2 | Tongyi Open |
| BS-RoFormer | vocals/instruments separation | MIT |
| silero-vad | voice activity detection | MIT |
| emotion2vec_plus_large | emotion classification | Apache 2.0 |

## 런타임 환경 + 실험 기록 (2026-06-18)

**검증된 로컬 런타임** — 단일 자족 루트 `Capstone_dub/`:

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 5080, 16 GB VRAM |
| 호스트 | Windows 11 + WSL2 + Docker Desktop |
| 드라이버 / CUDA | 드라이버 CUDA 13.2 / 컨테이너 base 이미지 `nvidia/cuda:12.1.1`·`12.4.1` |
| Python | 3.10(CPU 서비스) ~ 3.12(GPU 서비스), 서비스별 venv |

모델은 레포 내 `media/model_cache` 마운트에서 **완전 오프라인 로드**된다. `docker-compose.override.yml`이 `HF_HOME`·`MODELSCOPE_CACHE`·`TORCH_HOME`·`ERES2NETV2_CACHE_DIR`을 `/workspace/project/...`로 교정해 재다운로드가 발생하지 않는다. `src/daemon_lifecycle.py`가 단일 16 GB GPU에 맞춰 diarize(8903)/ASR(8902) 데몬을 한 번에 하나씩 juggling한다. 전수 검증된 모델 경로표는 `PROJECT_CONTEXT.md` 참조.

**활성 docker-compose 서비스**: `controller, separator, diarizer, pyannote, face, speaker, vibevoice-asr, tts-cosyvoice, webapp-backend, webapp-frontend`.

**보조 서비스 & 실험**: `pyannote`·`face`(시각 ASD)·`vibevoice-asr`가 별도 compose 서비스로 동작 — `pyannote`+NeMo는 4-way 화자분리 융합(활성 `diarization.engine = fusion_4way`), `face`는 LightASD+insightface 화자 재배정, `vibevoice-asr`는 대체 ASR 백엔드. 학습된 **MOS 평가기**(`src/mos_evaluator.py`, wav2vec2 + MLP head, 더빙 음성 1~5점)는 TTS 자연스러움 채점용으로 구현돼 있으나 활성 `validate_tts`엔 아직 미연결(`tts.validation.enabled=false` 기본). 실시간 음성↔음성 통역 프로토타입은 별도 `realtime_interpreter/`에 있다(배치 파이프라인과 분리).

**최근 ASR 실험**: 통짜 오디오 ASR + Viterbi 단어-청크 배정으로 짧은 청크 단어 유실/환각 완화 — 현재는 모듈화된 `src/run_asr.py`와 `src/repair_patches/boost_subchunk_asr.py`로 통합(초기 standalone `scripts/full_boost_asr.py`·`window_asr.py` 프로토타입은 흡수·삭제됨).

## License

본 저장소는 학술/연구 목적. 모델 라이센스는 각 모델 저장소 참조.

## Acknowledgments

- 팀원 [Stanl2y/Capstone_dub](https://github.com/Stanl2y/Capstone_dub) 구조 base
- 보존 자산은 [E:\TTS_capstone](https://github.com/yujin1103/AI_dubbing_system/tree/main) 의 v17~v305 실험 결과
- 자동 알고리즘 + DER 검증 + face thumbnail 매핑은 본 저장소 작업
