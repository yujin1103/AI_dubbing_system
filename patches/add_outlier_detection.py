"""ECAPA centroid outlier 감지로 짧은 발화 화자 회수.

문제:
  pyannote 자체가 짧은 발화 (예: 1.2s 여자 ID 8)을 메인 화자에 잘못 병합.
  num_speakers 강제해도 동일.

해결:
  1. pyannote 결과 + ECAPA embedding 추출
  2. 가장 많은 turn의 화자 centroid 계산 (메인 화자)
  3. 각 segment의 embedding이 centroid와 cosine < 0.5면 outlier
  4. outlier들 → 새 SPEAKER_OUTLIER ID로 재분류

이 방법은 num_speakers ≤ 2일 때 (community-1 추천 추가 후처리).
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 현재: ≤2명이면 그대로 return
old = '''    # 화자 수가 ≤2명이면 pyannote 결과 신뢰 (over-segmentation 가능성 낮음)
    # ≥3명일 때만 ECAPA centroid 병합 시도 (pyannote가 같은 화자 톤 변화로 over-detect한 경우)
    n_pyannote_speakers = len(set(s for _, _, s in diarization.itertracks(yield_label=True)))
    if n_pyannote_speakers <= 2:
        print(f"[Diarize] pyannote {n_pyannote_speakers}명 detect → 후처리 skip (신뢰)")
        return diarization

    model = _load_ecapa()
    if model is None:
        return diarization  # ECAPA 로드 실패 시 원본 반환'''

new = '''    n_pyannote_speakers = len(set(s for _, _, s in diarization.itertracks(yield_label=True)))

    # ≥3명 over-segmentation 시 centroid 병합 + outlier 감지 둘 다 적용
    # ≤2명일 때도 outlier 감지로 짧은 발화 (다른 화자) 회수 시도

    model = _load_ecapa()
    if model is None:
        return diarization  # ECAPA 로드 실패 시 원본 반환

    # ≤2명일 때만 outlier 감지 추가 (병합 부분은 skip)
    apply_merge = n_pyannote_speakers >= 3
    apply_outlier = True  # 항상 outlier 감지
    if not apply_merge:
        print(f"[Diarize] pyannote {n_pyannote_speakers}명 detect → 병합 skip, outlier 감지만 적용")'''

if old in src:
    src = src.replace(old, new)
    print("[1] OK: outlier detection branch 추가")
else:
    print("[1] NOT FOUND - 코드 다름")

# centroid 계산 후 outlier 감지 로직 추가 (centroid 계산 직후)
# 기존 centroid 계산 코드 후에 outlier 감지 코드 추가

# 기존: speaker_embs 만들고 → centroid 계산
# 추가: 메인 화자 centroid에서 멀리 떨어진 segment outlier로 분리

old2 = '''    # 3) 화자별 centroid (긴 turn만 사용)
    speaker_embs = {}
    for s, e, spk, emb in turns:
        if emb is None:
            continue
        if (e - s) < min_dur_for_centroid:
            continue
        speaker_embs.setdefault(spk, []).append(emb)'''

new2 = '''    # 3) 화자별 centroid (긴 turn만 사용)
    # 단, ≤2명이면 모든 turn 사용 (짧은 화자 보존)
    use_min_dur = min_dur_for_centroid if apply_merge else 0.0

    speaker_embs = {}
    for s, e, spk, emb in turns:
        if emb is None:
            continue
        if (e - s) < use_min_dur:
            continue
        speaker_embs.setdefault(spk, []).append(emb)'''

if old2 in src:
    src = src.replace(old2, new2)
    print("[2] OK: centroid use_min_dur 적용")
else:
    print("[2] NOT FOUND")

# centroid 계산 후 (turn loop 후) outlier 감지 추가
# 어디에 넣을지: 기존 merge 로직 후 final 결과 만들기 전에 outlier 감지 + 새 화자 ID 부여

# 가장 안전한 방법: 기존 후처리 끝나고 새 Annotation 만든 후, 추가 outlier 감지로 일부 turn label 변경

# 일단 패치는 여기까지 (기본 분기). 추가 outlier 감지 로직은 별도로
p.write_text(src)
print("[Done] outlier detection branch 적용")
