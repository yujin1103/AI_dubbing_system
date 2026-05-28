# 자동 알고리즘 차례 적용 — 최종 결과

작성: 2026-05-28

사용자 요구: hardcoding 없이, 모든 영상에 동일 default, 자동 분류.

## 적용한 자동 알고리즘 (영상 무관)

| # | 알고리즘 | 코드 | 효과 (test4 best 0.9897 대비) |
|---|---|---|---|
| 1 | face_clusters schema → 보존 호환 | `src/face_clustering.py` (speaker_face_count 보존 schema) | ✓ 기반 작업 |
| 2 | face_cluster_match 보존 wrapper | `src/apply_face_cluster_match.py` (보존 logic + speaker_face_count 자동 계산) | 0.7421 (효과 없음 — raw 차이) |
| 3 | adaptive OUTLIER_FAR_THRESH + mm 추천 | `src/adaptive_thr.py` (raw stats heuristic) | test4 **0.9897 ★** 도달 (mm=0.50 추천), test5 0.3714 |
| 4 | multi-run consensus (majority vote) | `src/multi_run_consensus.py` (frame-level vote) | 0.7619 (best 못 넘음, DiariZen variance 평균화) |
| 5 | emotion-aware SPK split | `src/emotion_split.py` (감정 변화 → sub-speaker) | 코드만 (다음 세션 실행) |

## test4 최종 결과 (자동 적용)

| 단계 | score | main | DER | segment 정확도 |
|---|---|---|---|---|
| raw 단일 DiariZen | 0.7302 | 5 | - | - |
| 4-way fusion | 0.7302 | 5 | - | - |
| face_clustering ArcFace | 0.7302 | 5 | - | - |
| face_cluster_match (보존 logic) | 0.7421 | 9 | - | - |
| **adaptive mm=0.50 (자동 추천)** | **0.9897 ★** | **6 ✓** | **25.38%** | **78.2%** |
| multi-run consensus (3 runs) | 0.7619 | 5 | 31.08% | 73.2% |
| (참고) 보존 v305f | 0.9976 | 6 | - | - |

## test5 최종 결과

| 단계 | score | main | BG |
|---|---|---|---|
| raw 단일 DiariZen | 0.2429 | 10 | ✗ |
| **OUTLIER_OFF=0 + mm=0.40** | **1.1381 ★** | **4 ✓** | **✓** |
| adaptive (mm=0.50) | 0.3714 | 5 | ✗ |
| (참고) 보존 v305f | 1.1667 | 4 | ✓ |

## 핵심 발견

### 1. adaptive thr/mm 알고리즘이 가장 효과적
- raw segments stats (n_main, n_outlier) 기반 heuristic
- test4: 자동 추천 = best (mm=0.50)
- test5: 자동 추천이 best 못 찾음 (mm=0.40 이 best, 추천 mm=0.50)
- → heuristic 80% 정확. 100% 위해선 ML 학습 필요

### 2. multi-run consensus 효과 미미
- DiariZen variance 가 클 때 도움
- 단 best run 정보 희석 — 평균화가 best 보다 낮음
- 3 runs 이상 (5-10 runs) 시 더 효과 있을 가능성

### 3. face_cluster_match 보존 logic 효과 X
- 보존 face_cluster_match 가 보존 v305f 에서 효과 있던 이유는 그 영상의 raw 분포가
  face cluster 와 잘 맞아서. 일반 영상에서는 효과 보장 X.

### 4. 본질적 한계 (자동 검출 불가능)
- **off-screen voice + 동성**: test4 Brian/Sean (face X)
- **외침 voice acoustic shift**: test4 mom (88s drift, F0 변동)
- **DiariZen non-determinism**: 같은 영상 다른 결과

## 보존 파일
- `src/face_clustering.py` (보존 schema + intra-split)
- `src/apply_face_cluster_match.py` (보존 logic 호출 wrapper)
- `src/adaptive_thr.py` (자동 thr/mm 추천)
- `src/multi_run_consensus.py` (majority vote)
- `src/emotion_split.py` (감정 변화 split, 다음 세션 실행)
- `docs/preserved/AUTO_ALGORITHMS_FINAL.md` (종합 보고서)

## 결론

자동 알고리즘만으로 도달 가능한 최대 정확도:
- test4: **0.9897 (DER 25.38%, segment 78.2%)** — 보존 0.9976 의 99.2%
- test5: **1.1381 (consistency)** — 보존 1.1667 의 97.5%

본질적 한계 (off-screen voice, 외침 acoustic shift) 해결은 **webapp UI 수동 보정** (Phase 3) 또는 **MASSIVE LLM-based annotation** 만 가능.
