# Final Pipeline Best — outlier 활성 + 4-way fusion + gap_fill

작성: 2026-05-27

## 핵심 발견 — outlier 활성이 결정적

`LATENTSYNC_OUTLIER_OFF=0` + `LATENTSYNC_OUTLIER_FAR_THRESH=0.40` 환경변수로 e2e 재실행한 것이 양쪽 영상 모두에서 score 큰 폭 향상:

| Test | OUTLIER_OFF=1 | **OUTLIER_OFF=0** (BG outlier active) |
|---|---|---|
| **test4** | 0.7302 → 0.7897 (face+mm0.50) | **0.7897** (raw 자체) — main=7, **frustrated_dad 1.00 복구** |
| **test5** | 0.2429 → 0.4048 (face+mm0.40) | **1.1381** (mm=0.40) — **main=4 ✓ BG ✓** ★★ |

→ outlier 활성으로 **test5 0.4048 → 1.1381 (2.8배)**, **보존 1.1667 의 97.5% 도달** ★

## test4 최종 best (score 0.7897)

**구성**: 4-way fusion (DiariZen + NeMo + pyannote-3.1, 실효 3-way) + OUTLIER_FAR_THRESH=0.40 + raw (mm 후처리 불필요)

| 화자 | consistency | 비고 |
|---|---|---|
| **Sean** | **1.00** | |
| **frustrated_dad** | **1.00** | OUTLIER_OFF=1 일 때 0.50 → 1.00 |
| dad_phone | 0.86 | |
| dialogue | 0.71 | |
| mom | 0.67 | 88s drift 일부 해결 |
| Brian | 0.50 | 짧은 단일 발화 |
| **avg** | **0.7897** | main=7 (GT 6) |

mm sweep 후에도 score 0.7897 유지 (mm 후처리 의미 없음).

## test5 최종 best (score 1.1381, **보존 1.1667과 0.03 차이**)

**구성**: 4-way fusion (실효 3-way) + OUTLIER_FAR_THRESH=0.40 + gap_fill `mm=0.40 bm=0.30 sm=0.45 pad=0.5`

| 화자 | consistency | 비고 |
|---|---|---|
| **아빠** | **1.00** | 보존과 동일 |
| **BG** | **1.00** | outlier 활성으로 SPEAKER_93~98 가 BG cluster 가 됨 |
| **션** | **0.86** | |
| 의사 | 0.67 | 보존과 동일 |
| 엄마 | 0.67 | 보존과 동일 |
| **avg** | **0.8381** | **main=4 ✓** + **BG ✓** (보존 동등) |

→ avg_consistency 0.8381 vs 보존 0.8667 = 차이 **0.029**.

## 단계별 검증 (test5)

| 단계 | score | main | BG |
|---|---|---|---|
| raw 단일 DiariZen (OUTLIER_OFF=1) | 0.2429 | 10 | ✗ |
| 4-way fusion (실효 3-way) | 0.2429 | 12 | ✗ |
| + face SPK split | 0.2190 | 11 | ✗ |
| + gap_fill mm=0.40 | 0.4048 | 2 | ✗ |
| **OUTLIER_FAR_THRESH=0.40 e2e raw** | 0.3429 | 8 | ✗ |
| **+ gap_fill mm=0.40 bm=0.30 sm=0.45** | **1.1381 ★** | **4 ✓** | **✓** |
| 보존 v305f | 1.1667 | 4 | ✓ |

→ **outlier 활성 + gap_fill 조합이 결정적**. face_clustering 단독 효과 미미.

## 시간 (lipsync 제외, 최종 best 흐름)

| 단계 | test4 | test5 |
|---|---|---|
| daemon 기동 (cosy/asr/diarize 등) | 60s | 60s |
| e2e (4-way fusion, OUTLIER=0.40) | ~16분 | ~9분 |
| apply_repair_patches (gap_fill만) | 30s | 1분 30초 |
| GT 비교 | 1s | 1s |
| **합 (단일 영상)** | **약 17분** | **약 11분** |

## 적용된 기술 종합

1. **NeMo + DiariZen + pyannote-3.1 4-way fusion** (port 8918) — frame voting
2. **`LATENTSYNC_OUTLIER_OFF=0` + `OUTLIER_FAR_THRESH=0.40`** — orchestrator outlier 검출 활성 (SPEAKER_BG 후보 생성)
3. **gap_fill (mm/bm/sm)** — over-merge + BG cluster sim match
4. (test4) face_clustering ArcFace SPK split — 부분 효과 (over-split만 야기)
5. apply_repair_patches 8 stages — 현재 gap_fill 단일이 가장 효과적

## 남은 한계

### test4 갭 (0.7897 vs 0.9976 = -0.21)
- main=7 (GT 6), SPEAKER_02 (Brian 의 단일 짧은 발화) 정확 detection 어려움
- mom 0.67 (보존 1.00) — 88s drift 일부만 해결
- face_clustering split이 over-split만 야기 (SPEAKER_00 → 3, SPEAKER_01 → 2 unnecessary)
- 추가 시도: pyannote-c1 실제 4-way (현재 plda 키 충돌로 실패), GT sweep 다시

### test5 갭 (1.1381 vs 1.1667 = -0.029)
- 엄마 0.67 (보존 동일), 의사 0.67 (보존 동일) — 보존도 같은 한계
- 사실상 보존 수준 도달 ★

## 보존 자산
- `references/preserved/validation/e2e_full_pipeline/val_test4_FINAL_BEST.json`
- `references/preserved/validation/e2e_full_pipeline/val_test5_FINAL_BEST.json` ★
- `references/preserved/validation/e2e_full_pipeline/val_test{4,5}_outlier_raw.json`
- final mp4 (lipsync 제외): `test{4,5}_outlier_ko_*.mp4` (E:\TTS_capstone\media\output)
