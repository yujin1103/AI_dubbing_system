# Korean Dubbing Pipeline — Project Status

**최종 갱신**: 2026-05-19
**현재 baseline**: **v121** (5-way fusion + face gender + age + shape feature)
**이전 baseline**: v92 → v116 → **v121** (face shape tie-breaker 추가로 73-94s 영역 개선)
**호스트**: Windows + Docker (`dubbing_pipeline` 컨테이너)

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
