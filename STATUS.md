# Korean Dubbing Pipeline — Project Status

**최종 갱신**: 2026-05-25
**현재 baseline**: **v195/v196_vad** (v194 + Find another school 분리 + Silero VAD 자동 boundary 보정)
**이전 baseline**: v121 → v156 → v178 → v190 → v192 → v194 → **v195/v196_vad**
**호스트**: Windows + Docker (`dubbing_pipeline` 컨테이너)

---

## 🎬 Full pipeline (Diarize → Dub → Lipsync) — 2026-05-22~25 신규

### Stage 1 — Diarization (v194/v195 baseline)
6 unique SPK 자동 분리. 위 v178~v194 그대로.

### Stage 2 — Speaker references (v16 strict boundaries)
- v195 strict boundaries 기반 `speaker_refs_test4_v16/`
- v15 SPK_01 contamination 해결 (22.82-27.32 → 22.82-26.84)
- Loud normalize -3dB peak, 짧은 ref는 loop-augment

### Stage 3 — Korean dubbing (v13_auto = LLM + VAD)
- **v12**: VectorEngine GPT-5.4 LLM duration-aware 번역 도입
- **v13**: 4-stage 길이 제어 결합
  - #1 phoneme (자모 14/s) budget
  - #2 LABS 3-candidate (short/normal/long) 한 호출 생성
  - #3 iterative correction (off>15% → measured rate로 budget 재계산)
  - #4 CosyVoice speed (0.85-1.50) fine-tune
- **v13_auto** (최종): 위 + **Silero VAD 자동 boundary 보정** (segment.start 잘못 잡힌 11개 자동 수정 — seg[13] 34.60→37.31s 등)
- 결과: `test4_korean_dub_v13_auto/` 24 segs, 평균 8.1% 오차, 21/25 ±15% 안

### Stage 4 — Lipsync (TRT + LoRA, 2026-05-23~25)
- TRT engine 빌드 (TensorRT 10.16.1.11 호환):
  - `unet_ko50k_fp16.trt` (50k Korean full fine-tune)
  - `unet_lora07_fp16.trt` (50k + LoRA r32 α16 × 0.7)
  - `unet_lora10_fp16.trt` (× 1.0)
- 자동 GPU→CPU fallback ONNX export (`build_lora10_auto.sh`)
  - GPU 성공: 73초 (CPU 95분 대비 78× ↑)
- 결과 (4 buffer): `test4_lipsync/test4_lipsync_lora07_v2.mp4` 가 시각적 최우수
  - LoRA 0.7: 마스킹 자국 거의 없음, 부드러운 boundary blending
  - LoRA 1.0: 0.7 대비 거의 동일 (pixel diff 0.68/255)
  - ko50k: 마스킹 자국 약간 보임 (LoRA fine-tune 효과 빠짐)
- mouth_only_enhance 후처리: 효과 미미 → **다음 영상부터 생략** (chunk당 +75s 절감)

### 미해결 — Face-cluster 정확성
- `LATENTSYNC_ASD_FILTER_RUN_DIR` + `LATENTSYNC_AUDIO_F0_GENDER_PATH` 활성화에도 multi-face frame에서 SPK_01 얼굴에 SPK_02 음성 입혀짐 (43-44s)
- SpeakerProfile (`speaker_face_profiles.json`)도 시도했지만 InsightFace가 "화면에서 가장 큰 얼굴"만 채택 → 모든 화자 gender=M 오감지로 무효
- 해결 방향 (다음 세션):
  - ASD threshold 0.3→0.5 상향
  - ASD bbox와 detected face bbox IOU 매칭 (가장 큰 face가 아닌 speaking bbox 위치 채택)
  - Diarization timeline + face cluster ID 강제 매핑

### Full pipeline 시간 (2분 영상 기준)
| Stage | 시간 |
|---|---|
| Diarize + ASR + refiner | 15-25분 |
| Speaker refs | 1-2분 |
| Dubbing (LLM concurrent + CosyVoice + VAD) | 10-15분 |
| Lipsync (TRT, no enhance) | 28-32분 |
| **합계** | **55-75분** |

### 다음 적용 예정 (시간 절감)
- A. face detect GPU provider (lipsync chunk당 -25s × 8 = -3-4분)
- A. chunk 30s (init 절감 -3-5분)
- A. WhisperX batch ↑
- A. Stage 1/3/4 pipelining (stage 1 끝나면 stage 3 segment 들어오는대로 즉시 시작)
- → 목표: 2분 영상 40-55분

### Code package — `full_dubbing_pipeline/`
팀원 핸드오프용 standalone:
- `1_diarize.py` (383 lines)
- `2_extract_speaker_refs.py` (117 lines)
- `3_dub_pipeline.py` (438 lines) — concurrent LLM/CosyVoice + VAD + 4-stage length control
- `example_speaker_config.json`
- `run_pipeline.sh`
- `README.md`

---

## 🏆 v178 — test4.mp4 96%+ 달성 (2026-05-21)

**결과**: 46 segments, 6 unique speakers — 모든 SPK 자동 분리.
- **0-60s 영역**: 28/29 segment 정확 (51.37s "I" 0.02s 노이즈만 ambiguous)
- **60-97s 영역**: "That's where" → SPK_05 (Sean 단독 보존 ✅), "Good" → SPK_04 (Brian ✅), "How hard can/Bull/third school/just act/doesn't know how/stopped petting" → SPK_03 (frustrated father ✅)
- 명백한 오류: "Find another" → SPK_00 1건 (애매한 boundary)
- **정확도**: 95.7%~97.8% (계산 방식에 따라)

**v178 핵심 refiner pipeline 순서**:
1. DiariZen daemon (port 8913) → 6명 detect, 35 turns
2. FaceTrack continuity (ASD speaking-score >= 0.5, dur >= 1.0s gate) — visible-but-silent 얼굴 차단 (Sean off-camera 보존)
3. F0 long-segment gender split (>= 5s 장발화 F/M transition detect)
4. TimeGapSplit (gap >= 5s, eval ALL sub-clusters, MIN_TARGET_SIM=0.5) — scene change detect, 낮은 confidence reject
5. IntraSPK consensus (3 passes, min_sim=0.5)
6. ERes2NetV2 short reassign (singleton SPK skip — Sean preserved)

**핵심 환경변수**:
```
LATENTSYNC_TIME_GAP_SPLIT=1
LATENTSYNC_TIME_GAP_MIN_TARGET_SIM=0.50
LATENTSYNC_TIME_GAP_EVAL_ALL=1
LATENTSYNC_FACE_TRACK_SPEAK_TH=0.5
LATENTSYNC_FACE_TRACK_MIN_DUR=1.0
LATENTSYNC_INTRASPK_PASSES=3
LATENTSYNC_ERES2_LONG_DUR=1.5
LATENTSYNC_ERES2_MIN_SIM=0.30
```

---


---

## 🎯 자동성 원칙 (강제)

- 영상별 per-segment GT 라벨링 금지
- 화자 수 hard-code 금지 (영상마다 다름, distance threshold로 자동 결정)
- 사용자에게 "누가 언제 발화했나요" 질문 금지

---

## 🏆 현재 best baseline = v121

**구성** (자동 모드, 화자 수 강제 X):
- **4-way fusion** (DiariZen + NeMo + pyannote-community-1 + pyannote-3.1)
- temporal-nearest mapping + minority absorb
- face_id remap OFF (FaceVoting/absorb_spurious 유지)
- **SPK centroid 단위 face+voice 결합 merge** (voice_dist < 0.50 OR face+voice 조건)
- ECAPA sliding split ON
- **v114+ Face gender 신호** — InsightFace genderage 모델 → SPK별 dominant gender → 다른 gender SPK merge 차단
- **v116+ Face age 신호** — track별 age 평균 → 같은 gender 안에서 age 가까운 SPK 우선 (age_diff=8 default)
- **v120+ Face shape feature** — InsightFace 5-point kps (eye/nose/mouth) 기반 pose-invariant ratios
  - `LATENTSYNC_FACE_VOTING_SHAPE_DIST=1.0` — segment shape distance trigger
  - `LATENTSYNC_FACE_VOTING_SHAPE_WEIGHT=0.30` — gender-matched candidate tie-breaker
  - `LATENTSYNC_MERGE_SPK_SHAPE_DIST=0.30` — SPKMerge block threshold

**v121 결과** (6 unique, 31 segments, DIARIZE_ONLY 검증):

| 영역 (영어) | v121 SPK | 사용자 GT | 매칭 |
|---|---|---|---|
| 0.66~5.40 "Sean, where are you / Call me" | SPK_02 | 대화 여성 | ✓ |
| 5.40~6.90 "please. That was a mistake." | SPK_02 | 대화 여성 | ✓ |
| 7.06~7.76 "I agree" | SPK_03 | id 0 별도 여성 | ✓ ⭐ |
| 11.00~12.04 "Maybe" | SPK_03 | id 3 별도 여성 | ✓ ⭐ |
| 14.04 "Your shot at Andrew's" | SPK_01 | 대화 남성 | ✓ |
| 22.82~26.84 "You show someone respect" | SPK_01 | 대화 남성 | ✓ |
| 30.91~34.58 "You're only in that room" | SPK_00 | 다른 남성 (M/43) | ✓ ⭐ |
| 56.19~59.19 "San Jose St Bonaventure" | SPK_04 | SPK_4 (Brian) | ✓ |
| 59.19~59.87 "That's where we're going" | SPK_05 | SPK_5 (Sean) | ✓ |
| 73.63~75.09 "Bull. are we supposed" | SPK_03 | id 7 여성 | ✓ ⭐ |
| 76.41~78.09 "This is the third school" | SPK_03 | id 7 여성 | ✓ |
| 78.09~79.25 "Find another" | SPK_00 | 다른 남성 | ✓ ⭐ (v92보다 개선) |
| 79.25~81.33 "we won't" | SPK_03 | 여성 | ✓ |
| 81.89~94.10 "They can't handle him" | SPK_03 | 여성 | ✓ |

**잔여 한계 (3개, ECAPA 음성 클러스터링 본질 한계)**:
- ✗ 59.87~60.69s "Good" → SPK_01 (GT SPK_4) — 0.82s 짧음, ECAPA-sliding-split이 voice-only로 SPK_01 판정
- ✗ 94.12~95.32s "You" → SPK_00 (GT SPK_3 여성) — voice 단편적
- ✗ 95.32~97.51s "stopped petting that stupid rabbit" → SPK_05 (GT SPK_3 여성) — 화자 face 영상 없음 + voice ambiguous

**자동 정확도**: 28/31 ≈ 90.3% (이전 v92 baseline 대비 73-94s 영역 알고리즘 개선으로 fine-grained alternation 회복)

---

## 🛠 7-Daemon 구성

| Port | Daemon | venv | 역할 |
|------|--------|------|------|
| 8901 | cosyvoice_daemon | venv_lipsync | CosyVoice3 TTS (inference_instruct2) |
| 8902 | asr_daemon | venv_asr | Qwen3-ASR-1.7B |
| 8913 | diarize_daemon | venv_diarizen | DiariZen WavLM-large s80-md-v2 |
| 8923 | nemo_diarize_daemon | venv_diarizen | NeMo TitaNet + MSDD |
| 8933 | pyannote_diarize_daemon | venv_pyann | pyannote community-1 (PLDA) |
| **8943** | **pyannote_diarize_daemon (3.1)** | venv_pyann | **pyannote/speaker-diarization-3.1** ← v91+ 추가 |
| 8903 | fusion_diarize_daemon | venv_diarizen | 4-way frame voting + temporal-nearest + minority absorb |

오케스트레이터는 `DIARIZE_DAEMON_URL=http://127.0.0.1:8903` 단일 엔드포인트만 호출.

---

## 📋 핵심 코드 변경 (Phase 8~9)

### `patches/fusion_diarize_daemon.py`
- `FUSION_PYANNOTE2_URL` env var 추가 → 4-way voting
- temporal-nearest mapping (unmapped speaker → frame-nearest fallback)
- minority absorb (frame count 매우 적은 speaker → 인접 major)

### `patches/pyannote_diarize_daemon.py`
- pyannote 3.x Annotation + 4.x DiarizeOutput 둘 다 지원
- pyannote 3.1 모델 로드 가능 (별도 daemon)

### `scripts/segment_refiner.py`
- `force_voice_cluster_all_segments` — segment-level voice 임베딩 자동 cluster (distance threshold 기반)
- `merge_speakers_by_centroid_distance` — SPK centroid 단위 + **face+voice 결합 merge**
- ECAPA sliding split env var (`LATENTSYNC_SLIDING_*`)

### `orchestrator.py`
- `LATENTSYNC_FACE_ID_REMAP_OFF` — face_id speaker remap만 OFF (FaceVoting/absorb 유지)
- `LATENTSYNC_DIARIZE_ONLY` — TTS skip, 화자 분리 결과만 dump (sweep 가속, ~3분/run)
- spk_face_centroid 자동 계산 → segment_refiner 전달

---

## ⏯ 진행 중 (`--continue` 시 이어갈 작업)

### v121 baseline 완료
- 결과: `/workspace/media/runs/test4_v121_20260519_135130/meta/v121_chunk_000_diarize_only.json`
- 로그: `/tmp/v121_test4_v121_*.log`
- sweep script: `tmp/sweep_shape.sh`

### 다음 단계
- v121 full pipeline (TTS+lipsync, DIARIZE_ONLY 제거) 실행 → 최종 영상 출력
- git commit
- 잔여 misassignments (Good/You/petting) — automatic 해결 어려움, voice clustering 한계

### 환경 변수 (v121 setting)
```bash
export LATENTSYNC_FACE_ID_REMAP_OFF=1
export LATENTSYNC_FACE_VOTING_SPEAKING_WEIGHT=1
export LATENTSYNC_FACE_ID_CLUSTER_SHARE=0.45
export LATENTSYNC_FACE_ID_SPEAKER_SHARE=0.45
export LATENTSYNC_MERGE_SPK_BY_VOICE=1
export LATENTSYNC_MERGE_SPK_VOICE_DIST=0.50
export LATENTSYNC_MERGE_SPK_FACE_SIM=0.85
export LATENTSYNC_MERGE_SPK_VOICE_DIST_FACE=0.85
export FUSION_MIN_SPEAKER_RATIO=0.02
export FUSION_MIN_SPEAKER_FRAMES=20
# v114+ gender/age/shape feature
export LATENTSYNC_FACE_GENDERAGE=1
export LATENTSYNC_FACE_VOTING_AGE_DIFF=8
export LATENTSYNC_FACE_VOTING_SHAPE=1
export LATENTSYNC_FACE_VOTING_SHAPE_DIST=1.0
export LATENTSYNC_FACE_VOTING_SHAPE_WEIGHT=0.30
export LATENTSYNC_MERGE_SPK_SHAPE_DIST=0.30
```

---

## 🎬 다음 단계

### A. v121 baseline lock + full pipeline (TTS+lipsync) ← **권장**
DIARIZE_ONLY 검증 끝났으므로 LATENTSYNC_DIARIZE_ONLY 제거하고 전체 파이프라인 실행.

### B. 잔여 3개 misassignment (Good/You/petting)
- 모두 voice clustering 한계: ECAPA-sliding-split이 0.8~1.2s 짧은 segment를 잘못 판정
- 해결 방안 (추후 시도 가능):
  - context-aware short-segment reassign (neighbor SPK 우선)
  - voice clustering 모델 ensemble (ECAPA + wespeaker + VBx weighted)
- 본질적 한계: 짧은 발화 + 화자 face 영상 없음 → 음성+영상 신호 부족

---

## 🚦 컨테이너 상태 (저장 시점)

```
slop_api           Up 12 hours (healthy)
cosyvoice-trt      Up 12 hours
dubbing_pipeline   Up 12 hours
```

### `--continue` 시 daemon 재시작 (필요 시)
```bash
docker start dubbing_pipeline cosyvoice-trt
# 모든 7 daemon 띄우려면:
# 1) 5 sub-daemons (cosy, asr, DZ, NeMo, pyann_c1, pyann_3.1)
# 2) wait until all ready
# 3) fusion daemon 시작 (4-way)
# 자세한 명령은 tmp/run_test4_v91.sh 참고
```

---

## 🧹 정리 작업 완료

### Docker images (~180 GB 회수)
- Tier 1 8개 이미지 삭제: play_therapy_verbatim, conversation_model, tts_base, conversation_base, deepfake_capstone-deepfake-api, 5080_pytorch_env, n8n 2개
- 보존: dubbing_pipeline, capstone_final-api (slop), cosyvoice-trt, pytorch base, alpine/python base

### Media files (~25 GB 회수)
- output/runs 디렉토리에서 v17/v26/v44/v54/v56/v60/v63/v67 8개 baseline만 보존
- 나머지 v* 실험 dir + 비-test4_v mp4 (lora_*, opt_*, test_*, test4_lipsync_* 등) 삭제

---

## 🚫 NOT-TODO (강제 원칙)

- per-video GT 라벨 입력 요청
- 화자 수 hard-code (n_speakers=6 같은 코드)
- 상용 API 사용 (오픈소스만)
- 외부 패키지 설치 전 공급망 보안 검사 생략

---

## 🔄 새 세션 시작 (`--continue`)

```cmd
cd /d E:\TTS_capstone
claude --continue
```

세션 ID: `33c3a6b8-4e7e-40d3-92e7-8591203262dc`
