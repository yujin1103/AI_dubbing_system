# DER + 시간대 정확도 측정 결과

작성: 2026-05-28  
script: `scripts/compute_der.py` (pyannote.metrics 4.0 + segment-level overlap)

## 측정 지표 차이

| 지표 | 측정 방식 | 한계 |
|---|---|---|
| **consistency (이전)** | 각 GT 화자 별로 "가장 자주 매핑된 detect SPK" 비율 | over-merge에 좋게 나옴 (1 SPK 가 여러 GT 화자 다 흡수해도 높음) |
| **DER (Diarization Error Rate)** | Missed + False Alarm + Speaker Error (시간 비율) | 표준, 시간대 단위 |
| **segment-level accuracy** | GT segment 별 mapping된 detect SPK overlap 시간 비율 | 시간대 + 화자 식별 동시 |

## test4 결과 (4-way fusion + thr=0.50 + mm=0.50)

### DER
- **DER: 28.63%**
- missed detection: 4.1% (음성 놓침)
- false alarm: 5.2% (가짜 음성)
- **speaker error: 19.3%** (잘못된 화자 라벨)
- 음성 감지는 양호 (9.3%), 화자 식별이 주 오차

### Segment-level accuracy (시간대 정확도)
- **전체 78.8% 정확** (61.16s / 77.62s)
- 화자별:
  - **Sean: 95.4% ★** (대부분 정확)
  - **frustrated_dad: 89.3% ★**
  - dad_phone: 83.4%
  - Brian: 81.6%
  - dialogue: 68.9%
  - **mom: 61.2%** (88s drift 일부 SPEAKER_03 으로 잘못)

### 화자 매핑 혼동
- **Brian + Sean 둘 다 SPEAKER_04 매핑** — 두 다른 사람을 같은 detect SPK 가 흡수
- **frustrated_dad + mom 둘 다 SPEAKER_03 매핑** — mom 의 88s drift 가 SPEAKER_03 (frustrated_dad) 로 잘못 흡수

→ main_count=6 ✓ 이지만 실제로는 6개 detect SPK 가 6 GT 화자에 1-to-1 매핑 안 됨 (몇 개는 2 GT 화자 흡수).

## test5 결과 (4-way fusion + thr=0.40 + mm=0.40)

### DER
- **DER: 103.02%** (!! 100% 넘음)
- missed detection: 20.4%
- **false alarm: 53.1%** (매우 높음 — detect duration 55.3s vs GT 41.25s)
- speaker error: 29.6%

### Segment-level accuracy
- **전체 78.0% 정확** (36.28s / 46.50s) — test4 와 비슷
- 화자별:
  - **아빠: 93.3% ★**
  - **션: 86.4% ★**
  - BG: 69.0%
  - 의사: 67.2%
  - **엄마: 50.0%** (절반만 정확)

### 핵심 문제 — over-merge
detect SPK 분포 (segments_gapfilled.json):
- **SPEAKER_01: 20 segments** (main 다 흡수)
- SPEAKER_93/94/96: 각 1-3 segments (outlier)
- SPEAKER_BG_00: 2 segments

→ main=4 (01/93/94/96) ✓ 이지만, 실제로는 **SPEAKER_01 하나가 GT 4 화자 (션/아빠/엄마/의사) 발화 대부분 흡수**. consistency 점수는 이런 over-merge에 좋게 나옴.

## 종합: consistency vs DER 갭

| | consistency score | main_match | DER | segment 정확도 |
|---|---|---|---|---|
| **test4** | 0.9897 (99.2%) | ✓ main=6 | **28.63%** | **78.8%** |
| **test5** | 1.1381 (97.5%) | ✓ main=4 | **103.0%** | **78.0%** |

**해석**:
- test4 는 consistency + DER 모두 양호 — 진짜 보존 수준 (DER 약 28%)
- test5 는 consistency 매우 높지만 DER 매우 나쁨 — over-merge 함정 (SPEAKER_01에 4 화자 흡수)
- segment-level 정확도는 양쪽 비슷 (78~79%) — 시간대 단위로 보면 절대 정확도는 비슷

## 결론

1. **consistency 점수는 main_count 만족하면 일관성만 본다** → over-merge 시 trivial 1.0 발생 가능
2. **DER + segment-level 가 진짜 정확도**:
   - test4 78.8% (양호)
   - test5 78.0% (consistency 점수 1.1381이 보이는 것만큼 좋지는 않음)
3. test5 의 진짜 개선은 over-merge 풀고 main 4 화자 별도 detect → 그러려면 OUTLIER_FAR_THRESH 다른 값 + mm 더 높게 + face_cluster_match 적용

## 보존 파일
- `references/preserved/validation/e2e_full_pipeline/der_test4.json` (DER 28.63%, segment 78.8%)
- `references/preserved/validation/e2e_full_pipeline/der_test5.json` (DER 103%, segment 78.0%)
- `scripts/compute_der.py` (DER 측정 스크립트)
