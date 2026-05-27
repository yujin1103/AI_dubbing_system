# Best Pipeline Results — face SPK split + 4-way fusion + sweep best config

작성: 2026-05-27  
파이프라인: orchestrator (4-way fusion) → face_clustering (SPK split) → apply_repair_patches (gap_fill mm sweep)

## 파이프라인 흐름

```
test.mp4 →
  extract_audio → separate (BS-RoFormer) → diarize (4-way fusion: DiariZen+NeMo+pyannote-3.1)
  → cut_chunks → translate → TTS (CosyVoice3) → compose → mux
  ↓
새 단계 (linsync 제외):
  face_clustering (LightASD + ArcFace + SPK split via face cluster)
  apply_repair_patches (gap_fill mm/bm/sm sweep + word_split + focused_nemo + postprocess)
  validate_against_gt (per-GT-speaker consistency + main_count + bg_detected)
```

## test4 최종 결과 (GT: main=6, BG=0)

### 단계별 score
| 단계 | score | main | mom | dad_phone | dialogue | Sean | frustrated_dad | Brian |
|---|---|---|---|---|---|---|---|---|
| raw 단일 DiariZen | 0.7302 | 5 | 0.33 | 0.57 | 0.71 | 1.00 | 1.00 | 0.50 |
| 2-way fusion (DZ+NeMo) | 0.6825 | 5 | 0.67 | 0.86 | 0.57 | 1.00 | 0.50 | 0.50 |
| 4-way fusion (DZ+NeMo+pyannote-3.1) | 0.7302 | 5 | 0.67 | 1.00 | 0.71 | 1.00 | 0.50 | 0.50 |
| + face SPK split | 0.6349 | 8 ✗ over-split | 0.67 | 0.71 | 0.43 | 1.00 | 0.50 | 0.50 |
| **+ face split + gap_fill mm=0.50** | **0.7897 ★** | **3** | **1.00** ★ | 0.86 | 0.71 | 1.00 | 0.67 | 0.50 |
| (참고) 보존 v305f gapfilled | 0.9976 | 6 | 1.00 | 0.57 | 0.71 | 1.00 | 1.00 | 0.50 |

→ **best: face split + mm=0.50** — mom 0.33 → **1.00 ★**, dad_phone 0.57 → 0.86, frustrated_dad 0.67 회복. main=3 (GT 6 미달, gap_fill over-merge 부작용).

### mm sweep 결과
| mm | score | main | mom | dad_phone |
|---|---|---|---|---|
| 0.40 | 0.9167 (trivial) | 1 | 1.00 | 1.00 |
| **0.50** | **0.7897 ★** | 3 | **1.00** | 0.86 |
| 0.60 | 0.7897 | 4 | 1.00 | 0.86 |
| 0.70 | 0.7897 | 4 | 1.00 | 0.86 |
| 0.80 | 0.7063 | 5 | 0.67 | 0.86 |
| 0.99 | 0.7063 | 5 | 0.67 | 0.86 |

→ 보존 v305f 의 mm=0.99 는 보존 raw 6 SPK 기준. 우리 raw 5 SPK + face split 추가 시 best = mm=0.50.

## test5 최종 결과 (GT: main=4, BG=1)

### 단계별 score
| 단계 | score | main | bg | 아빠 | 의사 | 션 |
|---|---|---|---|---|---|---|
| raw 단일 | 0.2429 | 10 | ✗ | 0.00 | 0.17 | 0.71 |
| 2-way fusion | 0.2429 | 12 | ✗ | 0.00 | 0.17 | 0.71 |
| 4-way fusion | 0.2429 | 12 | ✗ | 0.00 | 0.17 | 0.71 |
| + face (split 0개, over-split) | 0.2190 | 11 | ✗ | 0.00 | 0.33 | 0.43 |
| **+ face + gap_fill mm=0.40** | **0.4048 ★** | **2** | ✗ | 0.00 | **0.83** ★ | **0.86** ★ |
| (참고) 보존 v305f gapfilled | 1.1667 | 4 | ✓ | 1.00 | 0.67 | 1.00 |

→ **best: face + mm=0.40** — 의사 0.17 → 0.83, 션 0.71 → 0.86. 아빠/BG 여전히 0. main=2 (GT 4 미달).

### sm sweep (test5)
- sm=0.10 ~ 0.85 모두 동일 (0.4048) — raw 에 BG cluster 자체 없어 gap_fill BG 단계 효과 X.
- BG 살리려면 e2e 재실행 `LATENTSYNC_OUTLIER_OFF=0` 또는 다른 `OUTLIER_FAR_THRESH` 필요.

## 핵심 발견

### 효과 있던 단계 (순서)
1. **2-way fusion (DZ+NeMo)**: test4 mom 0.33 → 0.67 (큰 효과)
2. **4-way fusion (+pyannote-3.1)**: test4 dad_phone 0.57 → 1.00 (실효 3-way, pyannote-c1 응답 0)
3. **face SPK split**: 단독으로는 over-split 야기 — gap_fill 필요
4. **gap_fill mm sweep**: test4 mm=0.50 (0.7063 → 0.7897), test5 mm=0.40 (0.2429 → 0.4048)

### 보존 v305f 와의 갭
| Test | 우리 best | 보존 | 차이 |
|---|---|---|---|
| test4 | 0.7897 | 0.9976 | **-0.21** |
| test5 | 0.4048 | 1.1667 | **-0.76** |

### 남은 한계
1. **main_count 미달** (test4 3 vs GT 6, test5 2 vs GT 4) — face split 과 gap_fill 의 balance 부족
2. **test5 BG 미검출** — raw segments 에 BG cluster (`SPEAKER_BG_*` 라벨) 없음
3. pyannote-c1 응답 0 — HF cache 다운로드 완료했지만 daemon 재기동 안 함

## 시간 측정 (최종 best pipeline)

| 단계 | test4 | test5 |
|---|---|---|
| extract_audio + separate + ASR | ~50s | ~50s |
| 4-way fusion (DZ+NeMo+pyannote-3.1) | ~30s | ~30s |
| translate + emotion + refs | ~7-11분 | ~5-7분 |
| TTS (CosyVoice3) | ~5-7분 | ~3-5분 |
| concat + mux | 5s | 5s |
| **자동 e2e (lipsync 제외)** | **11분 40초** | **7분 33초** |
| face_clustering (별도) | 14분 7초 | 13분 48초 |
| apply_repair_patches | 2분 30초 | 2분 |
| GT 비교 | 1초 | 1초 |
| **전체 (face + repair 포함)** | **약 28분** | **약 24분** |

## Push 자산
- `references/preserved/validation/e2e_full_pipeline/val_test4_full_BEST.json` — test4 mm=0.50 결과
- `references/preserved/validation/e2e_full_pipeline/val_test5_full_BEST.json` — test5 mm=0.40 결과
- `references/preserved/validation/e2e_full_pipeline/val_test4_face_split_only.json` — face split 단독
- `src/face_clustering.py` — SPK split 추가 버전 (보존 face_cluster_match 동등)
- run dir (E:\TTS_capstone): `20260527_011926_test4fusio_039322` / `20260527_013156_test5fusio_93ca37`
- final mp4 (lipsync 제외, 4-way fusion 기반): `test4_fusion4way_ko_*.mp4` / `test5_fusion4way_ko_*.mp4`

## 추가 검증/개선 후보
1. **pyannote-c1 daemon 재기동** (HF cache 받았으니 진짜 4-way 가능)
2. **e2e 재실행 (LATENTSYNC_OUTLIER_OFF=0)** — test5 BG 살릴 가능성
3. **face cluster sim_threshold tuning** (현재 0.4) — split 임계값 조정
4. **min_split_chunks 조정** — test4 SPEAKER_01 4-way split 너무 공격적 → 3 또는 4 로 올림
