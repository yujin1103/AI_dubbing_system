# Preserved 자산 — yujin1103/AI_dubbing_system

E:\TTS_capstone 에서 이식된 검증 자산. 팀원 구조 (Stanl2y/Capstone_dub) base 위에 우리 패치/baseline/GT 추가.

## 보존 영역

### `src/daemons/` — 4-way fusion + ASR + TTS daemon (14개)
| 파일 | 역할 | 포트 |
|---|---|---|
| `fusion_diarize_daemon.py` | 4-way fusion endpoint | 8903 |
| `diarize_daemon.py` | DiariZen WavLM-large-s80-md-v2 | 8913 |
| `nemo_diarize_daemon.py` | NeMo TitaNet-Large | 8923 |
| `pyannote_diarize_daemon.py` | pyannote community-1 / 3.1 | 8933 / 8943 |
| `asr_daemon.py` | Qwen3-ASR-1.7B + ForcedAligner-0.6B | 8902 |
| `cosyvoice_daemon.py` | CosyVoice3-0.5B inference_instruct2 (fade-in/out) | 8901 |
| `eres2netv2_helper.py` | ERes2NetV2 192-dim voice embedding (FBank input) | - |
| `audio_f0_gender.py` | F0 기반 gender 필터 | - |
| `asd_runner.py` / `asd_filter.py` | LightASD speaking score | - |
| `start_daemons.sh` / `stop_daemons.sh` | 8 daemons 일괄 기동/종료 | - |
| `campplus_*.py` | CAM++ 보조 diarization (실험) | - |

### `src/repair_patches/` — 8 검증 patches
| 파일 | 역할 |
|---|---|
| `gap_fill.py` | over-merge + BG cluster + sim-match (영상별 sweep best) |
| `word_level_split.py` | v194 word-level intra-segment split (F0 jump + LR cos) |
| `focused_nemo_split.py` | v190 long segment 재diarize + gap NeMo |
| `face_cluster_match.py` | face cluster dominant SPK 매칭 |
| `visual_asd_reassign.py` | ASD speaking score + gap-add |
| `postprocess_reassign_text.py` | word-level text reassign |
| `boost_subchunk_asr.py` | 3.0x volume boost + 4s window sub-chunk ASR (test5 +15 words, Adam x2 detect) |
| `sweep_gt_match.py` | GT 기반 config grid sweep (40 configs) |

### `src/preserved_orchestrator/orchestrator.py`
원본 모놀리식 orchestrator (5500+ lines). Phase 1 src/pipeline.py 통합 진행 중. 검증된 통합 로직 (4-way fusion 호출, F0 gender, BG speaker no-translate, word-level reassign 자동 호출) 보존.

### `configs/preserved/`
- `best_config_test4.json` — test4.mp4 GT sweep best (score 0.998, main=6 ✓)
- `best_config_test5.json` — test5.mp4 GT sweep best (score 1.167, main=4 ✓, BG ✓)
- `daemons_ports.yaml` — 7 daemon 포트/venv/모델 매핑
- `diarize_env.env` — v178/v190/v194 환경 변수 (TimeGapSplit, focused NeMo, word-level split 등)

### `references/preserved/`
- `test4_gt.json` / `test5_gt.json` — GT segments (mom, dad_phone, dialogue, frustrated_dad, Brian, Sean / 엄마, 아빠, 의사, 션, BG)
- `BEST_BASELINE_v194.json` + `.md` — test4 v194 baseline (24 segs, 6 SPK)
- `sweep_results_test4.json` — 40 config grid 결과

### `docs/preserved/`
- `STATUS.md` — canonical project state
- `EXPERIMENT_LOG.md` — v17~v305 실험 로그
- `DIARIZATION_SWEEP_LOG.md` — diarize sweep 결과
- `PROGRESS.md` — 진행 상황
- `PIPELINE_OVERVIEW.md` — 파이프라인 overview

## 검증된 핵심 수치

### test4.mp4 (GT: 메인 6명)
- best config: `main_merge=0.99 bg_merge=0.30 sim_match=0.10 pad=0.5` → score **0.998**
- consistency: mom 1.00 / frustrated_dad 1.00 / Sean 1.00 / dialogue 0.71 / dad_phone 0.57 / Brian 0.50
- 진행: v92 (4-way fusion) → v121 (90.3%) → v178 (95.7-97.8%) → v190 (6 unique SPK) → **v194 (word-level split)**

### test5.mp4 (GT: 메인 4명 + BG 1)
- best config: `main_merge=0.40 bg_merge=0.30 sim_match=0.45 pad=0.5` → score **1.167** (avg 0.87)
- consistency: 엄마 0.67 / 아빠 1.00 / 의사 0.67 / 션 1.00 / BG 1.00
- boost subchunk: 141 → 156 words (+15 fresh, Adam x2 detect)

### 환경
- GPU 16GB total, RAM 50GB (WSL2)
- LatentSync per-video: test4 `OUTLIER_OFF=1`, test5 `OUTLIER_FAR_THRESH=0.70`
- mom 88s drift = acoustic ceiling (F0 mom 253Hz vs dad 237Hz, 구분 어려움)

## 다음 단계 (Phase 1 src/ 통합)
1. `src/repair_patches/` 8개 → `src/repair_diarization.py` `REPAIR_MODULES` 등록
2. `src/daemons/` 7개 → 팀원 docker-compose 9 services 와 연결 (Dockerfile.diarizer/asr/tts 안에 venv_* 통합)
3. `src/preserved_orchestrator/orchestrator.py` 5500 lines → `src/pipeline.py` slim version 통합
4. webapp UI 인라인 에디터 + 부분 재합성 API
5. push to yujin1103/AI_dubbing_system main (merge PR)
