# face_clustering ArcFace + 2-way fusion 검증 결과

작성: 2026-05-27

## Phase A — face_clustering ArcFace 구현 + 검증

### 환경
- `face` 컨테이너 신규 (Dockerfile.face: LightASD repo + insightface 0.7.3 + onnxruntime-gpu)
- `_cluster_face_tracks`: insightface FaceAnalysis (buffalo_l) + cosine greedy (sim ≥ 0.4)
- LightASD subprocess (Columbia_test.py) → tracks/scores → face crop → ArcFace embedding 512-dim
- 단점: onnxruntime CUDA libs 누락 (libcublasLt.so.12) — CPU 사용 (느림)

### 시간
- test4 (1 chunk, 62 tracks): **7분 38초** (LightASD 6분 + face crop+embed 1분 38초)
- test5 (2 chunks, 222 tracks): **14분 1초**

### 클러스터링 결과
| | tracks | ArcFace clusters | 보존 v305f clusters |
|---|---|---|---|
| test4 | 62 | **15** | 16 (동등 ★) |
| test5 | 222 | **20** | (미측정) |

### GT 비교 (raw vs face-remapped)
| Test | raw score | face score | mom/엄마 | 비고 |
|---|---|---|---|---|
| test4 | 0.7302 | **0.5873** | 0.33 | 감소 — raw 가 over-merge (main 5) 상태라 face split 효과 X |
| test5 | 0.2429 | 0.2476 | 0.33 | 미미 — raw over-split (main 10), face cluster 가 audio SPK 다수 매핑 |

**결론**: ArcFace clustering 자체는 보존 v305f 동등 수준 (15/20 clusters). 단 GT 점수는 단독 적용으로 향상 X. 진짜 효과는 **face_cluster + gap_fill + 4-way fusion 결합** 필요.

## Phase B — fusion (2-way + 4-way 비교)

### 4-way fusion (DiariZen + NeMo + pyannote-c1 + pyannote-3.1)
**4 sub-daemon 띄움 후 fusion endpoint 호출.**
- pyannote daemons (8933 community-1, 8943 3.1) lazy load — 첫 호출 시 ~3분 대기
- fusion daemon log: `[Fusion] 3-model: dz=38, nm=36, pyann=0 → fused=40` — pyannote-c1 응답 segments=0 (community-1 HF cache 없어 fallback). pyannote-3.1만 cascade fuse.
- **실효 3-way** (DiariZen + NeMo + pyannote-3.1)

| Test4 | score | main | mom | dad_phone | dialogue | frustrated_dad | Sean | Brian |
|---|---|---|---|---|---|---|---|---|
| raw | 0.7302 | 5 | 0.33 | 0.57 | 0.71 | 1.00 | 1.00 | 0.50 |
| fusion 2-way | 0.6825 | 5 | 0.67 ★ | 0.86 ★ | 0.57 | 0.50 | 1.00 | 0.50 |
| **fusion 4-way (실효 3-way)** | **0.7302** | 5 | **0.67** | **1.00** ★★ | 0.71 | 0.50 | 1.00 | 0.50 |
| 보존 v305f | 0.9976 | 6 | 1.00 | 0.57 | 0.71 | 1.00 | 1.00 | 0.50 |

→ **4-way (실효 3-way) 추가 향상**:
- dad_phone 0.86 → **1.00** (pyannote-3.1 효과)
- dialogue 0.57 → 0.71 (2-way 손실 복구)
- mom 0.67 유지

| Test5 | score | main | bg | 엄마 | 아빠 | 의사 | 션 | BG |
|---|---|---|---|---|---|---|---|---|
| raw | 0.2429 | 10 | ✗ | 0.33 | 0.00 | 0.17 | 0.71 | 0.00 |
| fusion 2-way | 0.2429 | 12 | ✗ | 0.33 | 0.00 | 0.17 | 0.71 | 0.00 |
| **fusion 4-way** | **0.2429** | 12 | ✗ | 0.33 | 0.00 | 0.17 | 0.71 | 0.00 |
| 보존 v305f | 1.1667 | 4 | ✓ | 0.67 | 1.00 | 0.67 | 1.00 | 1.00 |

→ test5 fusion 4-way 효과 없음 (raw 동일). **over-merge gap_fill 후처리** 가 test5 핵심.

### 시간
| Test | 단일 DiariZen e2e | 2-way fusion | 4-way fusion |
|---|---|---|---|
| test4 | 17분 54초 | 12분 29초 | **11분 40초** ✓ |
| test5 | 7분 24초 | 7분 39초 | 7분 33초 |

### Daemon 자원 (4-way 동시)
- GPU 12.5 / 16.3 GiB (DiariZen + NeMo + cosy + asr, pyannote x2는 lazy)
- pyannote 첫 호출 시 추가 ~4GB (총 ~16GB 한계 직전)

## Phase B — 2-way fusion (DiariZen + NeMo)

### 환경
- 추가 daemon: `nemo_diarize_daemon` (port 8923)
- `fusion_diarize_daemon` (port 8918, DIARIZEN_URL=8903 + NEMO_URL=8923 2-way)
- orchestrator: `DIARIZE_DAEMON_URL=http://127.0.0.1:8918` 으로 호출
- GPU 사용: 12775/16303 MiB (cosy + asr + DiariZen + NeMo)

### 시간
- test4: **12분 29초** (단일 DiariZen 17:54 보다 빠름)
- test5: **7분 39초**

### GT 비교
| Test4 | score | main | mom | dad_phone | dialogue | frustrated_dad | Sean | Brian |
|---|---|---|---|---|---|---|---|---|
| raw (단일 DiariZen) | 0.7302 | 5 | 0.33 | 0.57 | 0.71 | 1.00 | 1.00 | 0.50 |
| face ArcFace | 0.5873 | 5 | 0.33 | 0.43 | 0.43 | 0.83 | 1.00 | 0.50 |
| **fusion 2-way** | **0.6825** | 5 | **0.67** ★ | **0.86** ★ | 0.57 | 0.50 | 1.00 | 0.50 |
| 보존 v305f gapfilled | 0.9976 | 6 | 1.00 | 0.57 | 0.71 | 1.00 | 1.00 | 0.50 |

→ **fusion 2-way 효과: mom 0.33 → 0.67 (2배), dad_phone 0.57 → 0.86**. 단 frustrated_dad/dialogue 감소. main=5 (6명 못 도달, SPEAKER_02 누락).

| Test5 | score | main | bg | 엄마 | 아빠 | 의사 | 션 | BG |
|---|---|---|---|---|---|---|---|---|
| raw | 0.2429 | 10 | ✗ | 0.33 | 0.00 | 0.17 | 0.71 | 0.00 |
| face ArcFace | 0.2476 | 9 | ✗ | 0.33 | 0.00 | 0.33 | 0.57 | 0.00 |
| **fusion 2-way** | **0.2429** | 12 | ✗ | 0.33 | 0.00 | 0.17 | 0.71 | 0.00 |
| 보존 v305f gapfilled | 1.1667 | 4 | ✓ | 0.67 | 1.00 | 0.67 | 1.00 | 1.00 |

→ test5 fusion 효과 없음. NeMo + DiariZen 모두 over-split 경향. test5 는 **over-merge gap_fill (main_merge=0.40) 후처리** 가 핵심.

## 종합 결론

### 효과 있는 조합
1. **mom 88s drift (test4)**: fusion 2-way (DiariZen + NeMo) — 0.33 → 0.67 (2배)
2. **dad_phone 분리 (test4)**: fusion 2-way — 0.57 → 0.86
3. **BG detect (test5)**: gap_fill bg_merge — fusion만으로는 0/1.00

### 진짜 보존 수준 (0.9976 / 1.1667) 도달에 필요한 것
1. **4-way fusion** (DiariZen + NeMo + pyannote-c1 + pyannote-3.1) — 현재는 2-way만
2. **face_cluster ↔ gap_fill 연결** — face_clusters.json 을 apply_repair_patches.py 의
   face_cluster_match 단계 입력으로 (현재는 단독 적용으로만)
3. **GT sweep 재실행** — e2e raw 가 보존과 다르므로 새 best config 필요 (main_merge / sim_match)

### 다음 단계
- pyannote_diarize_daemon (8933 community-1 + 8943 3.1) 추가 → 4-way fusion 완성
- src/face_clustering.py 결과를 src/apply_repair_patches.py 의 face_cluster_match.py
  로 전달 (run_dir 안 face_clusters.json 자동 인식)
- 새 e2e raw 에 sweep_gt_match.py 다시 돌려 best config 갱신

## 보존 결과 파일
- `references/preserved/validation/e2e_full_pipeline/val_test{4,5}_face_arcface.json`
- `references/preserved/validation/e2e_full_pipeline/val_test{4,5}_fusion_2way.json`
- final mp4: `E:\TTS_capstone\media\output\test{4,5}_fusion_ko_*.mp4`
- run dir: `E:\TTS_capstone\media\runs\20260527_*test{4,5}fusio_*\`
