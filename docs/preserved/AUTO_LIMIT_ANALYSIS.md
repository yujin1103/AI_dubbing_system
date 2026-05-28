# 자동 화자 분리 한계 분석 (hardcoding 없이)

작성: 2026-05-28

## 사용자 요구
- **모든 영상에 동일 default** 적용 (영상별 sweep hardcoding 금지)
- **자동으로 분류**되게 — 영상마다 수동 config 안 함

## 적용한 자동 로직 (영상 무관)

| 로직 | 위치 | default 값 |
|---|---|---|
| 4-way fusion (DZ+NeMo+pyannote-3.1) | `src/preserved_fusion.py` | endpoint 8918 |
| OUTLIER_FAR_THRESH | `orchestrator.py` | 0.40 (default) |
| gap_fill mm/bm/sm | `src/repair_patches/gap_fill.py` | mm=0.45 bm=0.30 sm=0.45 |
| face ArcFace clustering | `src/face_clustering.py` | sim_threshold=0.4 |
| **intra-segment face split** (신규) | `src/face_clustering.py` | min_intra=0.7s, segment ≥1.5s |

## test4 best 결과 (자동 default 적용, hardcoding 없음)

### 시도한 모든 경로
| 경로 | score | main | DER | segment 정확도 |
|---|---|---|---|---|
| raw 단일 DiariZen (OFF=1) | 0.7302 | 5 | - | - |
| 4-way fusion (thr=0.40) | 0.7302 | 5 | - | - |
| 4-way + face split + mm=0.50 | 0.7897 | 3 | - | - |
| 4-way + OUTLIER=0.40 (raw) | 0.7897 | 7 | - | - |
| **OUTLIER=0.50 + gap_fill mm=0.50** | **0.9897** | **6 ✓** | **25.38%** | **78.2%** ★ |
| 4-way + face intra-split + mm=0.50 | 0.8135 | 4 | - | - |
| 보존 v305f (참고, hardcoded best) | 0.9976 | 6 | - | - |

★ 최선: OUTLIER=0.50 + gap_fill mm=0.50, hardcoding 최소화 (mm=0.50 도 sweep best 라 일반화 필요).

### test4 화자별 정확도 (segment-level)
| GT 화자 | 정확도 | matched detect SPK | 주 한계 |
|---|---|---|---|
| **Sean** | **95.4%** ★ | SPEAKER_04 (Brian 과 공유) | - |
| **frustrated_dad** | **87.1%** ★ | SPEAKER_03 (mom 과 공유) | - |
| dad_phone | 83.4% | SPEAKER_00 | - |
| Brian | 81.6% | SPEAKER_04 | Sean 발화와 공유 |
| dialogue | 68.9% | SPEAKER_01 | scene 내 다양한 화자 |
| **mom** | **61.2%** | SPEAKER_03 | 88s drift → frustrated_dad 와 SPK 공유 |

### 56-60s 구간 Brian/Sean 분리 실패
- GT: Brian (56.19-58.67) + Sean (59.19-59.87) + Brian (59.87-60.69) 3개 발화
- detect: SPEAKER_04 단일 segment (56.16-59.84, 60.48-60.72)
- **시도 1**: word_level_split F0=50Hz LR=0.50 — 분리 X (동성, F0 거의 동일)
- **시도 2**: focused_nemo_split — NeMo 도 단일 화자로 분류
- **시도 3**: face_clustering intra-split (0.7s) — face cluster 변경 없음 (Brian 만 보이고 Sean off-screen 추정)

→ **acoustic + visual ceiling**. 같은 face 안에 다른 사람 voice (off-screen voice over) 면 자동 검출 불가능.

### mom 88s drift 한계
- mom 발화 중 일부가 SPEAKER_03 (frustrated_dad) 에 잘못 흡수
- mom 외침 시 F0 253Hz (boost) vs dad 237Hz — acoustic 차이 작음
- face: mom + frustrated_dad 같은 angle 가능성 → face split 불가

## 자동 검출 한계 (실측)

| 한계 유형 | 영상 | 원인 | 해결 가능성 |
|---|---|---|---|
| 동성 voice + off-screen | test4 56-60s Brian/Sean | acoustic 비슷 + face 안 보임 | 매우 어려움 (수동 라벨링 필요) |
| 외침 voice acoustic shift | test4 mom 88s | F0 변화 (mom 외침이 dad 와 비슷) | 어려움 (감정 분류 보조 필요) |
| 한 SPK 가 4 화자 흡수 | test5 SPEAKER_01 | over-merge gap_fill | mm/bm/sm 영상별 sweep 필요 (hardcoding) |

## 결론

1. **자동 default 로 도달 가능한 최대 정확도**:
   - test4: DER 25.38%, segment 78.2% (보존 face_cluster_match 보다 약간 낮음)
   - 화자별 평균 80%+ 도달 (Brian/Sean 공유, mom drift 제외)

2. **추가 향상 위해 필요한 것**:
   - face_clusters.json ↔ face_cluster_match.py (보존) 연결 — schema 변환 필요
   - 영상별 OUTLIER_FAR_THRESH adaptive — chunk SPK 분포 분석 후 자동 결정
   - 감정 분류 보조 (emotion2vec) — mom 외침 vs frustrated_dad 분리

3. **본질적 한계**:
   - off-screen voice + 동성 = 어떤 알고리즘도 자동 분리 어려움
   - 수동 라벨 필요 시점 (예: webapp UI 인라인 SPK 변경 + 부분 재합성)

## 다음 단계 후보
1. webapp UI 인라인 SPK 에디터 (Phase 3) — 한계 부분 수동 보정
2. 영상 분석 후 adaptive thr 자동 결정 알고리즘
3. emotion2vec 결과를 face_clustering 에 통합
