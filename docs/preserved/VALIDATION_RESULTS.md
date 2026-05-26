# Validation Results — GT 매칭 검증

검증 일자: 2026-05-26  
검증 환경: dubbing_pipeline 컨테이너 (venv_diarizen, 4-way fusion + gap_fill)  
검증 script: `scripts/validate_against_gt.py`  
GT 파일: `references/preserved/test4_gt.json`, `test5_gt.json`

## Score 산출 식 (sweep_gt_match.py와 동일)
```
score = avg(per_gt_speaker_consistency)
      + 0.2 (main_count_match 시)
      + 0.1 (bg_detected 시)
```
per-GT-speaker consistency = 같은 GT 화자 발화들이 같은 detect SPK 로 매핑된 비율 (1.0 = 완벽).

## 검증 결과 요약

| Run | Stage | Score | Avg cons. | Main | BG | 비고 |
|---|---|---|---|---|---|---|
| **test4 v305f** | gapfilled | **0.9976** | 0.7976 | 6 ✓ | - | ★ BEST — mom/Sean/frustrated_dad 1.00 |
| test4 v305f | raw | 0.9421 | 0.7421 | 6 ✓ | - | mom 0.67 (gap_fill 전) |
| t4 verify th070 | raw | 0.5992 | 0.5992 | **12 ✗** | - | repair 미적용 (over-split) |
| **test5 v305f** | gapfilled | **1.1667** | 0.8667 | 4 ✓ | ✓ | ★ BEST — 아빠/션/BG 1.00 |
| test5 v305f | raw | 0.4000 | 0.4000 | 6 ✗ | ✗ | gap_fill 전 (BG 검출 실패) |
| test5 v195env | gapfilled | 1.1381 | 0.8381 | 4 ✓ | ✓ | 차선 (션 0.86) |

## gap_fill 효과 입증

**test4 v305f** (mm=0.99 bm=0.30 sm=0.10)
- raw → gapfilled: 0.9421 → **0.9976** (+0.055)
- mom consistency: 0.67 → **1.00** (over-split된 mom 발화를 단일 SPK로 병합)

**test5 v305f** (mm=0.40 bm=0.30 sm=0.45)
- raw → gapfilled: 0.4000 → **1.1667** (+0.77, **2.9x**)
- main_count: 6 → **4** (over-split된 4개를 정확히 4명으로 병합)
- BG detect: ✗ → **✓** (3rd-party 외침을 SPEAKER_BG_* 클러스터로 자동 분리)
- 아빠: 0.0 → **1.0**, BG: 0.0 → **1.0**

## per-GT-speaker consistency 상세

### test4 v305f gapfilled (score 0.9976)
| GT speaker | consistency | 비고 |
|---|---|---|
| mom | **1.00** | gap_fill로 88s drift 부분 통합 |
| frustrated_dad | **1.00** | |
| Sean | **1.00** | |
| dialogue | 0.71 | 짧은 발화 일부 다른 SPK로 분류 |
| dad_phone | 0.57 | 전화 노이즈로 acoustic 다름 |
| Brian | 0.50 | 단일 짧은 발화 (small sample) |

### test5 v305f gapfilled (score 1.1667)
| GT speaker | consistency | 비고 |
|---|---|---|
| 아빠 | **1.00** | |
| 션 | **1.00** | |
| BG | **1.00** | "Adam!" 외침 등 3rd-party 정확 분리 |
| 엄마 | 0.67 | 외침 부분 dad와 acoustic 가까움 (F0 253Hz vs 237Hz) |
| 의사 | 0.67 | |

## Best config (sweep 결과)

### test4.mp4
```json
{ "main_merge": 0.99, "bg_merge": 0.30, "sim_match": 0.10, "pad": 0.5 }
```
- LATENTSYNC env: `OUTLIER_OFF=1`

### test5.mp4
```json
{ "main_merge": 0.40, "bg_merge": 0.30, "sim_match": 0.45, "pad": 0.5 }
```
- LATENTSYNC env: `OUTLIER_FAR_THRESH=0.70`
- boost subchunk: 141 → 156 words (+15 fresh, "Adam!" x2 detect)

## 보존 결과 파일

| 파일 | 위치 |
|---|---|
| 검증 결과 6개 JSON | `references/preserved/validation/val_*.json` |
| segments_*.json (per stage) | `references/preserved/runs/<run>/` |
| words_*.json (ASR 결과) | `references/preserved/runs/<run>/` |
| GT 정답 | `references/preserved/test4_gt.json`, `test5_gt.json` |
| sweep 40 configs 결과 | `references/preserved/sweep_results_test4.json` |
| BEST_BASELINE_v194 (이전) | `references/preserved/BEST_BASELINE_v194.json` + `.md` |

## 재현 방법
```bash
# 검증 실행 (gapfilled vs GT)
python scripts/validate_against_gt.py \
    references/preserved/runs/test4_v305_full/test4_v305_full_chunk_000_segments_gapfilled.json \
    references/preserved/test4_gt.json \
    out.json
```
