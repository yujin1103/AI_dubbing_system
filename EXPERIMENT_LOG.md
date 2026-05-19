# test4.mp4 화자 분리 실험 기록 (v2~v67)

**목적**: 영어 드라마 → 한국어 더빙 자동 파이프라인. 화자 분리 정확도 개선이 주 과제.

**테스트 영상**: `test4.mp4` (~98초, 6명 화자, 영어, 드라마 한 장면)

**평가 기준 (자동)**: `tmp/eval_diarization_auto.py`
- n_groups: 총 발화 그룹 수
- n_speakers: detect 화자 수 (GT 6명)
- selfref %: profile 못 만든 화자가 self-reference 사용 비율 (TTS 품질 위험)
- fragmented sent: 같은 sentence가 여러 화자로 쪼개진 건수
- rapid transitions <1s: 1초 이내 화자 전환 (over-fragment 신호)
- SPK_06: GT에 없는 가짜 화자 over-detect 그룹 수

**User GT (디버깅 검증용, 자동 시스템에 입력 안 됨)**:
- 총 화자 6명
- SPK_5 = "thats where we're going" 1발화만
- SPK_4 = id 12 전체 + id 14 "good"
- SPK_6 = 존재하지 않음 (탐지 시 over-detect)
- id 0 = SPK_1 "mistake" + SPK_2 "I agree" 혼재
- id 3 "maybe" = SPK_01
- id 7 = SPK_3
- id 9 = SPK_0 + SPK_3 혼재

---

## 단계별 핵심 마일스톤

### Phase 1 — 초기 baseline (v2~v17)
DiariZen 단독 + post-processing 튜닝. 안정성 확보 단계.

| 버전 | 주요 변경 | 결과 | 평가 |
|------|----------|------|------|
| v2~v9 | Qwen3-ASR + CosyVoice3 + LatentSync 첫 통합 | 작동 시작 | 초기 |
| v17 | C 작업: soft syllable range + CAPEL | 6명, runtime 9m 50s | **첫 안정** |

### Phase 2 — Face Recognition 시도 (v18~v25)
Audio + ArcFace 결합으로 화자 분리 정확도 향상 시도. 우리 영상에서 효과 미미.

| 버전 | 변경 | 결과 |
|------|------|------|
| v18 | SPEAKER_02 회귀 (남성화) | ❌ 회귀 |
| v19 | v18 fix | OK |
| v20 | OUTLIER 활성화 → 가짜 SPEAKER_99/98/97 양산 | ❌ |
| v21 | OUTLIER OFF | 복원 |
| v22~v23 | Face Recognition 재설계 | over-merge |
| v24~v25 | Face threshold 튜닝 (0.45~0.60) | 효과 미미 |

### Phase 3 — 안정 baseline (v26~v32)
Face Recognition OFF + audio 튜닝 + CAPEL 버그 fix. **사용자 만족 baseline**.

| 버전 | 변경 | 결과 |
|------|------|------|
| **v26** | **Face OFF + CAPEL fix** | **15m 47s, 6명, SPK_05=5, trim 5** ← 안정 |
| v27 | 같은 화자 인접 발화 안전 병합 | SPK_05=4, label 일부 회귀 |
| v28~v32 | env var, AV reassign tuning | marginal |

### Phase 4 — ArcFace FaceID 재도입 (v33~v38)
사용자 영상의 face 신호 약하지만 cluster 기반으로 활용 시도.

| 버전 | 변경 | 결과 |
|------|------|------|
| v33 | FACE_ID 활성, ArcFace cluster | 일부 회복 |
| v37 | AV-Reassign OFF + FaceVoting only | balance |
| v38 | DiariZen ahc_threshold 튜닝 | marginal |

### Phase 5 — NeMo + 다른 모델 시도 (v40~v50)
DiariZen 한계 인식, 다른 화자 분리 모델 시도.

| 버전 | 모델 | 결과 |
|------|------|------|
| v40~v42 | NeMo TitaNet + MSDD 단독 | 4명, over-merge |
| v43 | DiariZen raw post_process 약화 | 22 groups |
| v44 | DiariZen + LightASD speaking-weighted FaceVoting | **5명, 24 groups, selfref 12.5%** ← Phase 5 best |
| v45~v50 | 2-model fusion (DiariZen + NeMo) | 7명, SPK_06 over |

### Phase 6 — pyannote community-1 + 3-model fusion (v51~v56)
pyannote 4.x + community-1 추가, 3-model voting.

| 버전 | 변경 | 결과 |
|------|------|------|
| v51~v53 | pyannote community-1 단독 + VAD 강화 | 4명, "maybe" 정확 |
| v54 | **3-model fusion (DZ+NeMo+pyannote)** | **6명, SPK_06 over 4, selfref 41.7%** |
| v55 | face_id 0.30 강함 | 4명 collapse, SPK_5 손실 |
| v56 | face_id 0.45/0.45 보수 | **6명, SPK_06 over 3, GT 일부 정답** |

### Phase 8 — ECAPA voice cluster (segment-level voice 임베딩 기반 자동 SPK 통합) (v71~)

**핵심 발견**: v68 best baseline도 사용자 정밀 GT 미달. SPK_03이 GT SPK_2 + GT SPK_3 두 사람 공유, SPK_05도 GT SPK_5 + GT SPK_3 공유. 즉 라벨링 일관성 부족. → ECAPA voice 임베딩으로 segment 단위 자동 re-cluster.

**자동성 원칙 (사용자 명시)**:
- 화자 수 hard-code 금지 (영상마다 다름)
- AgglomerativeClustering with `distance_threshold` (자동 cluster 수 결정)
- 환경변수 `LATENTSYNC_VOICE_CLUSTER_DIST` (cosine distance threshold, 영상 무관 보편 값)

**구현**: `scripts/segment_refiner.py` 의 `force_voice_cluster_all_segments` 함수. orchestrator의 `LATENTSYNC_DIARIZE_ONLY=1` 모드 추가 (TTS skip — fast iteration, 각 run ~4분).

**Sweep round 1 (v75~v78)**:
| 버전 | dist threshold | n_clusters (자동) | 평가 |
|------|--------|-----|---|
| v75 | 0.55 | 27 | 너무 over-split |
| v76 | 0.65 | 21 | over-split |
| v77 | 0.75 | 15 | over-split |
| v78 | 0.85 | **9** | GT 6명에 가까움. SPK_3 일관성 매우 우수 (id 7+id 9+id 11 모두 SPK_03) |

**Sweep round 2 (v79~v82)** — segment voice cluster:
| 버전 | dist | n_clusters | GT 매핑 |
|------|--------|------|----|
| v79 | 0.90 | 5 | SPK_3 일관성 ✓ but "산호세"/"좋아" SPK_4 손실 |
| v80 | 0.95 | 4 | under-merge |
| v81 | 1.00 | 2 | massive collapse |
| v82 | 1.05 | 1 | all collapse |

**Sweep round 3 (v83~v86)** — SPK centroid 단위 voice merge (segment cluster OFF):
| 버전 | voice_dist | unique | 평가 |
|------|--------|------|----|
| v83 | 0.45 | 6 | merge 발동 안 함, baseline 유지 |
| v84/v85 | 0.55/0.65 | 6 | merge 안 함, segment count 일부 변화 |
| v86 | 0.75 | 3 | over-merge (3개 pair 합쳐짐) |

**SPK pairwise distance 측정 결과**:
- SPK_03 ↔ SPK_05: voice 0.666 (가장 가까움)
- SPK_01 ↔ SPK_02: voice 0.701, face 0.785
- SPK_02 ↔ SPK_04: voice 0.865 (다른 사람)
- SPK_00 ↔ SPK_03: voice 0.811, **face 0.76**
- SPK_04 ↔ SPK_05: voice 0.939, **face 0.81**

**Sweep round 4 (v87~v90)** — face + voice 결합 SPK merge:
조건: `voice_dist < base_threshold` OR (`face_sim ≥ face_threshold` AND `voice_dist < face_voice_dist`)

| 버전 | face_sim | voice_dist_face | merges | unique | GT 매칭 |
|------|---|---|---|---|---|
| **v87** | 0.70 | 0.85 | 0 (face 임계 미통과) | **6** | **best**. id 0 분리 ✓, id 6 SPK_3 ✓, id 7/9/11 모두 SPK_3 ✓, "산호세"/"좋아"/"거기로 가" 모두 정확 |
| v88 | 0.75 | 0.85 | SPK_00+03 (face 0.76) | 5 | "걔 다치잖아" SPK_03에 흡수 |
| v89 | 0.65 | 0.90 | SPK_00+03, 01+02 | 4 | over-merge |
| v90 | 0.70 | 0.95 | SPK_00+03, 04+05 | 4 | 산호세+좋아+거기로 모두 SPK_05 |

**v87** (3-way fusion, face_centroid 미계산 — 실제 face 보조 비활성):
- 6 unique, 21 groups
- SPK_3 일관성 매우 우수
- 다만 face+voice 결합 발동 안 함

---

### Phase 9 — 4-way fusion (pyannote 3.1 추가) + face+voice merge 활성 (v91~v92)

**구현**:
- `patches/fusion_diarize_daemon.py`: `FUSION_PYANNOTE2_URL` env var 추가, 4-way voting
- 4번째 sub-daemon: port 8943, pyannote/speaker-diarization-3.1
- `orchestrator.py`: spk_face_centroid 정상 계산 + segment_refiner 전달
- 총 7개 daemon (cosy 8901, asr 8902, DZ 8913, NeMo 8923, pyann_c1 8933, pyann_3.1 8943, fusion 8903)

**v91** (face_sim 0.70):
- 4-way fusion 정상 작동, SPK_01+02 (voice 0.683 + face 0.784) auto merge → **5 unique**
- 영어 원본 text dump 가능 (GT 검증 매우 유용)
- 다만 사용자 GT는 6명 → over-merge

**v92** (face_sim 0.85, SPK_01+02 보존) — **final best (확정)**:
- **6 unique, 39 segments**
- ✓ "please That was a" SPK_01 (GT mistake=SPK_1)
- ✓ **"Maybe" SPK_01 (GT id 3 SPK_01)** ⭐
- ✓ "It was also disrespectful" SPK_01 (id 1 시작)
- ✓ "San Jose"/"Good" SPK_04 (GT SPK_4)
- ✓ "That's where we're going" SPK_05 (GT SPK_5)
- ✓ id 7 "we supposed"/"third school" SPK_03
- ✓ id 9 "can't handle him I don't blame them" SPK_03
- ✓ id 9 "What happened" SPK_03
- ✓ id 11 "What did you do Sean" SPK_03
- ✗ "on get"/"act normal" 영역 — voice+face가 SPK_1/SPK_5와 유사해 자동 분리 한계
- ✗ id 11 "You"+"stop being stupid rabbit" SPK_01/05 (GT SPK_3) — face voting 미흡

---

### Phase 10 — wespeaker 5th signal 시도 (v98~v99) + SPKMerge 버그 fix

**시도**: ECAPA-TDNN voice embedding 외에 wespeaker (voxceleb_resnet34_LM, ResNet152_LM ONNX) 추가 신호로 SPK centroid merge 보강.

**구현**:
- `wespeakerruntime` (PyPI) 설치 (venv_diarizen + venv_lipsync 모두)
- voxceleb_resnet34_LM.onnx (25 MB), voxceleb_resnet152_LM.onnx (79 MB) 다운로드 (`~/.wespeaker/`)
- `scripts/segment_refiner.py`:
  - `merge_speakers_by_centroid_distance`에 `wespeaker_model` 인자 추가
  - SPK별 wespeaker embedding centroid 계산
  - combined distance = (ECAPA + wespeaker) / 2 → 임계 비교
  - SPKMerge mapping 적용 디버그 print 추가 (`segments relabeled` 카운트)
- `orchestrator.py`:
  - `LATENTSYNC_WESPEAKER=1` env var
  - `LATENTSYNC_WESPEAKER_ONNX` 명시적 onnx 경로 (R34 default / R152 명시)

**버그 fix**: v98 (R34)에서 `[SPKMerge] 6 → 4 SPKs` log 출력되었지만 segments dump 6 unique 그대로 → wespeaker가 venv_lipsync에 없어 load 실패 → SPKMerge가 ECAPA만으로 0 merges. wespeaker venv_lipsync 설치 후 v99에서 정상 작동 (`19/39 segments relabeled`).

**v98 (ResNet34)**: wespeaker dist 모든 화자 쌍에서 작게 출력 (SPK_01↔02=0.337, SPK_01↔03=0.343). over-merge (4 unique 의도 but venv 문제로 적용 안 됨).

**v99 (ResNet152)**:
- wespeaker dist 더 작아짐 (SPK_01↔02=0.139, SPK_01↔03=0.159)
- combined 0.410/0.431 — threshold 0.55 통과 → 2 merges
- 결과: **4 unique** (SPK_02, SPK_03 → SPK_01로 흡수). 사용자 GT SPK_3 영역 모두 SPK_01에 들어감 → **회귀**.

**결론**: wespeaker voxceleb 모델 (R34/R152) — drama 영상의 다양한 voice에서 distance를 일률적으로 작게 출력 (다른 화자도 0.13~0.4). 우리 영상엔 부정확. **ECAPA-TDNN 단독이 더 정확**.

**최종 baseline: v92 (확정)**. 자동 모드에서 더 이상 화자 분리 향상 어려움. 추가 voice 모델 / 추가 신호는 over-merge 위험.

---

### Phase 11 — VBx (Brno Bayesian) 5-way fusion + face encoder + PLDA scoring (v111~v113)

**시도**: VBx ResNet101 ONNX + (옵션) LDA→PLDA scoring을 fusion에 5th model로 통합.

**구현**:
- `/tmp/VBx` clone (BUT-FIT)
- `patches/vbx_diarize_daemon.py` (port 8953):
  - features.py (mel fbank 64-dim, kaldi-style)
  - VBx ResNet101 ONNX (256-dim x-vector)
  - 옵션: LDA (256→128 via transform.h5) + PLDA (mu/tr/psi) + cosine AHC
  - 옵션 env: `VBX_USE_PLDA=1`, `VBX_AHC_THRESHOLD=0.85`
- `patches/fusion_diarize_daemon.py`: `FUSION_VBX_URL` 추가 → **5-way fusion**

**VBx 단독 (cosine, PLDA OFF)**:
- threshold sweep: 0.65 (39 unique) ~ 0.85 (6 unique) ~ 0.90 (4) ~ 1.0 (1)
- **0.88 sweet spot = 6 unique (GT 일치)**, 27 segments

**VBx + PLDA (LDA reduce 256→128 + PLDA transform)**:
- threshold sweep: 0.3 (116 unique) ~ 0.85 (8) ~ 0.9 (4)
- PLDA 효과 marginal — cosine과 유사

**v111 (5-way fusion + VBx cosine)**: 28 segments, 6 unique. v92와 GT 매핑 동일.

**v112 (dominant face cluster centroid)**:
- 핵심 발견: SPK_04 dominant cluster = SPK_05 dominant cluster = **cluster_14** (face encoder가 두 GT 화자를 같은 cluster로 잡음)
- mixed centroid 회피 적용했지만 결과는 v111과 동일

**v113 (5-way + VBx PLDA + dominant cluster)**: 31 segments, 6 unique. 동일 미해결 영역.

**Phase 11 결론**:
- VBx PLDA, dominant cluster, 5-way fusion 모두 marginal — 같은 trade-off 패턴
- **face encoder 한계**: 같은 시간대 다른 화자 face가 InsightFace 기준 같은 cluster (cluster_14 = SPK_4 + SPK_5 dominant 공유)
- **자동 모드 GT 매칭률 75~80% 천장 확정**
- 추가 시도 = trade-off만 발생 (다른 영역에서 정답/회귀 교체)

---

### Phase 12 — face gender/age feature 추가 (v114~v116)

**핵심 아이디어**: voice + face cluster + speaking score 외 **gender + age** 신호 활용. 같은 시간대 화자 face가 비슷해도 gender/age는 다를 수 있음.

**구현**:
- `scripts/face_id_embedder.py`:
  - `extract_face_id_embedding_with_gender` 함수 → (embedding, gender 'M'/'F', age int) tuple
  - `compute_track_face_embeddings`: track별 dominant gender + mean age 집계
  - `_last_track_genders`, `_last_track_ages` module-level state
  - `reassign_segments_by_face_voting`에 gender mismatch + age diff 강제 reassign 추가
- `orchestrator.py`: SPK별 dominant gender 계산 + segment_refiner 전달
- ENV:
  - `LATENTSYNC_FACE_GENDERAGE=1` (default ON)
  - `LATENTSYNC_FACE_VOTING_AGE_DIFF=15` (default, 영상 무관 보편 임계)

**InsightFace buffalo_l의 genderage 모델 활성**:
- ResNet50 ArcFace + genderage (96x96 ONNX) 동시 추론
- gender accuracy ~95% in clean face
- age 추정 평균 오차 ~5세 (face age estimation 일반 표준)

**v114 (gender 추가)**: face encoder 작동 (gender=48 tracks). 첫 시도 dict bug → fix 후 v115.

**v115 (segment-level gender reassign in face_voting)**:
- `[FaceVoting/Gender]` log 작동
- SPK_01 = M (대화 남성), SPK_02 = F (대화 여성), 나머지 4명 모두 M
- 같은 성별 SPK 간 효과 미미 (남성 4명 사이 분리 X)

**v116 (gender + age 결합 reassign)** — **자동 정확도 큰 향상**:
- track ages range: ~28~67세 다양
- `[FaceVoting/G+A]` log 다수 출력
- **여자 2명 + 남자 4명 GT 일치 자동 detect**:
  - SPK_01 = M/39 (대화 남성)
  - SPK_02 = F/33 (대화 여성)
  - **SPK_03 = F/33** (Bull/third school/can't handle 화자 — 다른 여성)
  - SPK_00 = M/43
  - SPK_04 = M (산호세)
  - SPK_05 = M (거기로 가)

**v116 GT 매핑 진전**:
- ✓ id 0 "I agree" → SPK_03 (다른 여성 — F/33 별도)
- ✓ id 0 "Maybe / them" → SPK_03
- ✓ id 7 "Bull / supposed to do" → SPK_03 (id 7 GT SPK_3 = 여성)
- ✓ id 7 "third school" → SPK_03
- ✓ id 9 "can't handle / Cause obviously" → SPK_03
- ✓ id 9 "They can't handle him" → SPK_03
- ✓ "You're only in that room" → SPK_00 (M/43 다른 남성)
- ✓ "Find another" → SPK_00
- ✗ "Good" → SPK_01 (GT SPK_4 산호세 남성과 다름)
- ✗ "stopped petting rabbit" → SPK_05 (GT SPK_3 여성과 다름) — 마지막 영역 face cluster 한계

**리포트 파일 위치**:
- segments JSON: `E:/TTS_capstone/media/runs/test4_v116_20260519_122011/meta/test4_v116_chunk_000_diarize_only.json`
- log: `/tmp/test4_v116_*.log` (Docker container 내부)

**Phase 12 결론**: gender + age 자동 feature로 **여성 2명 분리 + SPK_3 (여성) 일관성 큰 진전**. 사용자 GT "여자 2 + 남자 4" 정확 매칭. 남성 4명 사이 분리는 face cluster 한계로 일부 잔존.

---

### Phase 13 — age_diff sweep (v117/118/119) + face shape feature (v120)

**v117 (age_diff=5)** — 5 unique speakers (over-merge). SPK_02 → SPK_03 흡수. age_diff 너무 낮으면 같은 gender F/F 강제 합침.

**v118 (age_diff=8)** — v116과 동일 (6 unique). SPK_02 (대화 여성) 보존 + SPK_03 (별도 여성) 분리. GT 매칭 유지.

**v119 (age_diff=12) + face shape feature mid-sweep**:
- 5-point kps 기반 face shape feature 신규 추가 (mouth width, face length, nose position ratios)
- shape distance trigger threshold 0.08 (초기 시도)
- 결과: 18개 G+A or G+A+S reassign 발동 — 일부 양호 (56s SPK_05→04 ✓), 일부 회귀 (Bull → SPK_02 ✗ )
- **shape distance 분포 매우 wide (0.126 ~ 4.148)** — head pose가 dominant signal → noisy

**v120/v121/v122 shape sweep 결과**:
- v120 (shape disabled as trigger, weight=0.20): n=28, == v116 baseline
- **v121 (shape trigger 1.0, weight=0.30, merge_block 0.30)**: n=31 ⭐ — 73-94s 영역 fine-grained alternation 회복 (SPK_03 여성 ↔ SPK_00 "Find another" 다른 남성)
- v122 (shape trigger 0.5, more aggressive): n=31, 일부 over-split

**v121 선정 이유 (new baseline)**:
- 73-94s 영역 SPK_03/SPK_00 alternation 제대로 detect: "Bull/are we supposed/third school/we won't/can't handle" (여성) ↔ "Find another" (다른 남성) 자동 분리
- 6 unique SPKs 유지 (over-detection 없음)
- 잔여 misassignments 3개만 (Good/You/petting) — 모두 voice clustering 본질 한계
- shape feature 효과 검증됨: 같은 gender 후보 중 mouth/face geometry 가까운 SPK 우선 → fine-grained 분리

**자동 정확도**: 28/31 ≈ 90.3% (v92 대비 73-94s region 개선)

**잔여 한계 (자동 모드 본질 한계)**:
- "Good" (59.87s, 0.82s) — ECAPA-sliding-split이 voice-only 판단으로 SPK_01 (dialogue character)에 부여
- "You" (94.12s, 1.20s), "stopped petting" (95.32s, 2.19s) — 화자(여성)가 영상에 나오지 않음 + voice ambiguous

---

### Phase 7+ — face_id remap 분리 + sliding split 환경변수화 (v68~v70)

**핵심 발견 (v60→v68)**: v60의 face_id remap이 GT SPK_0 ("방법을 몰라", "애 다치잖아" 영역)를 GT SPK_3 ("어쩌라고", "숀 무슨짓" 영역)에 합치는 원인. 두 화자의 face가 카메라 각도 유사해서 같은 face cluster에 속함. remap 끄면 6명 보존 + FaceVoting/absorb로 정확도 유지.

**코드 변경 (v68+)**:
- `orchestrator.py`: `LATENTSYNC_FACE_ID_REMAP_OFF=1` env var 추가. face_id remap만 끄고 FaceVoting/absorb_spurious는 유지.
- `scripts/segment_refiner.py`: sliding split 파라미터 env var화 (`LATENTSYNC_SLIDING_MIN_CONSEC`, `LATENTSYNC_SLIDING_MIN_SUB_DUR`, `LATENTSYNC_SLIDING_MIN_SEG_DUR`)

| 버전 | 변경 | 결과 |
|------|------|------|
| **v68** | **face_id REMAP_OFF + FaceVoting ON + sliding default** | **6명, 30 groups, GT 완벽 (id 0 분리 + SPK_5 "거기로" + SPK_4 "좋아" + SPK_0 "방법/애 다치잖아"), selfref 26.7%** |
| v69 | v68 + sliding 매우 보수 (consec 3, sub 1.0, seg 3.0) | 6명, 22 groups, GT 손실 (id 0 분리 X, "거기로 가" X) |
| **v70** | **v68 + sliding 약간만 보수 (consec 3, sub 0.5, seg 2.0)** | **6명, 27 groups, GT 거의 (id 0 ✓, SPK_0 ✓), "거기로 가" 손실 (SPK_4와 합쳐짐)** |

---

### Phase 7 — Fusion daemon 알고리즘 개선 (v57~v67)
근본 분석: 73~85s 영역 본질적 2명이지만 fusion이 7명 화자 만듦 → fusion daemon 자체 개선.

**주요 코드 변경 (v60+)**:
- `patches/fusion_diarize_daemon.py`:
  - **Temporal-nearest mapping**: NM/pyannote speaker가 DZ와 frame overlap 0인 경우 시간상 가장 가까운 DZ speaker로 fallback 매핑 (unmapped → 새 화자 방지)
  - **Minority absorb**: frame count 적은 speaker (< min_ratio 또는 < min_abs frames)를 인접 major speaker로 reassign
  - 환경변수: `FUSION_MIN_SPEAKER_RATIO`, `FUSION_MIN_SPEAKER_FRAMES`
- `scripts/segment_refiner.py`:
  - **ECAPA sliding split OFF env var**: `LATENTSYNC_SLIDING_SPLIT_OFF=1`

| 버전 | 변경 | 결과 |
|------|------|------|
| v57~v59 | spurious 임계 / DiariZen ahc 변경 | marginal |
| **v60** | **fusion temporal-nearest + minority absorb (0.02/20)** | **5명, 24 groups, selfref 20.8%, SPK_06 0, "좋아"=SPK_04 ✓** |
| v61 | minority 0.015/15 (relaxed) | fusion 6명 but face_id remap이 5명으로 |
| v62 | face_id 0.55 보수 | 5명, no change |
| **v63** | **face_id OFF** | **6명, 22 groups, selfref 18.2%, fragmented 2** ← 화자 수 ✓, "좋아" 손실 |
| v64 | face 0.55/0.70 | 6명 but 30 groups (over-frag) |
| v65 | face 0.99/0.99 (remap no-op) + FaceVoting ON | 30 groups (sliding split 영향) |
| v66 | + sliding split OFF | 10 groups (under-frag), SPK_05 손실 |

---

## 최종 비교표 (주요 baseline)

| 버전 | 화자 | 그룹 | selfref | fragmented | "좋아" SPK_4 | id 0 mistake/agree | "거기로 가" SPK_5 | "방법을 몰라" SPK_0 | SPK_06 | 평가 |
|---|---|---|---|---|---|---|---|---|---|---|
| v44 | 5 | 24 | 12.5% | 5 | - | - | - | - | 0 | DiariZen+FaceVoting |
| v54 | 6 | 24 | 41.7% | 5 | ✓ | ✗ | ✗ | ✗ | 4 over | 3-model fusion 첫 |
| v56 | 6 | 21 | 42.9% | 4 | ✓ | ✗ | ✓ | ✗ | 3 over | face_id 보수 |
| v60 | 5 | 24 | 20.8% | 6 | ✓ | ✓ | ✓ | ✗ | 0 | minority absorb |
| v63 | 6 | 22 | **18.2%** | **2** | ✗ | ✓ | ✓ | ✓ | 0 | face_id OFF |
| **v68** | **6** | 30 | 26.7% | 7 | **✓** | **✓** | **✓** | **✓** | **0** | **GT 완벽** ← top |
| v69 | 6 | **22** | 31.8% | 5 | ✓ | ✗ | ✗ | ✓ | 0 | sliding 매우 보수 |
| **v70** | **6** | 27 | **25.9%** | 6 | **✓** | **✓** | ✗ | **✓** | **0** | **균형** |

**top 후보**:
- **v68** (GT 완벽): 30 groups, 모든 GT 발화 정답 라벨. 사용자 의도 "더 빡세게 분리"에 가장 부합.
- **v70** (균형): 27 groups, GT 대부분 ✓ ("거기로 가"만 손실). selfref 25.9% (v68 26.7% 비슷).

---

## 기술 스택 (현재)

### 6-Daemon 구성
| Port | Daemon | 역할 |
|------|--------|------|
| 8901 | cosyvoice_daemon | TTS (CosyVoice3 inference_instruct2, fade fix) |
| 8902 | asr_daemon | Qwen3-ASR-1.7B |
| 8913 | diarize_daemon | DiariZen WavLM-large s80-md-v2 |
| 8923 | nemo_diarize_daemon | NeMo TitaNet + MSDD |
| 8933 | pyannote_diarize_daemon | pyannote community-1 (PLDA) |
| 8903 | fusion_diarize_daemon | 3-way frame voting + temporal-nearest + minority absorb |

### 평가 도구
- `tmp/eval_diarization_auto.py` — GT-free metric
- `media/gt/test4_gt.json` — 디버깅용 (학습 입력 X)

---

## 잔존 한계

1. **SPK_5 over-split (자동 해결 불가)**: GT는 1발화이지만 모든 baseline에서 multiple segments. SPK_5 face track 4개 모두 ArcFace 0.55+ 유사도 → 자동 face cluster 분리 불가. ECAPA voice 측면에서도 다른 시점이라 묶이지 않음.

2. **GT "좋아" vs 6명 detect trade-off**: v60은 GT 라벨 정확하지만 5명. v63은 6명 detect하지만 "좋아"가 SPK_01에 흡수. face_id 활성/비활성 trade-off.

3. **selfref 의존 (TTS 품질 위험)**: 짧은 화자가 6s+ profile 못 만들 시 segment 자체를 reference로. v60 20.8%, v63 18.2%로 개선됐지만 0% 목표.
