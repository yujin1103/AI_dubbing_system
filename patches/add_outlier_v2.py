"""ECAPA outlier detection — 짧은 다른 화자 발화 회수.

post_process_diarization 끝부분에 outlier 감지 단계 추가.

알고리즘:
  1. 모든 segment의 ECAPA embedding 추출
  2. 각 화자의 centroid 계산 (긴 turn만)
  3. 각 segment에 대해:
     - 자기 화자 centroid와 거리 = self_dist
     - 다른 화자 centroid와 거리 = other_dist (가장 가까운)
     - self_dist > 0.5 + other_dist (자기보다 다른 화자에 더 가까우면) → 잘못 분류
     - cosine 0.4 이하면 모든 centroid에서 멀음 → 새 화자
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 일단 post_process 함수의 ≤2명 skip 로직 원복
old1 = '''    n_pyannote_speakers = len(set(s for _, _, s in diarization.itertracks(yield_label=True)))

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

new1 = '''    # 화자 수가 ≤2명이면 pyannote 결과 신뢰 (병합 skip).
    # 단, outlier 감지는 항상 적용 (짧은 다른 화자 발화 회수).
    n_pyannote_speakers = len(set(s for _, _, s in diarization.itertracks(yield_label=True)))
    apply_merge = n_pyannote_speakers >= 3

    model = _load_ecapa()
    if model is None:
        return diarization'''

if old1 in src:
    src = src.replace(old1, new1)
    print("[1] OK: branch 정리")
else:
    print("[1] NOT FOUND")

# 기존 centroid use_min_dur 부분
old2 = '''    # 3) 화자별 centroid (긴 turn만 사용)
    # 단, ≤2명이면 모든 turn 사용 (짧은 화자 보존)
    use_min_dur = min_dur_for_centroid if apply_merge else 0.0

    speaker_embs = {}
    for s, e, spk, emb in turns:
        if emb is None:
            continue
        if (e - s) < use_min_dur:
            continue
        speaker_embs.setdefault(spk, []).append(emb)'''

new2 = '''    # 3) 화자별 centroid (긴 turn만 사용)
    # ≤2명이면 모든 turn 사용 (짧은 화자 보존)
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
    print("[2] OK: 동일 (확인용)")
else:
    print("[2] NOT FOUND")

# 함수 끝부분에 outlier 감지 추가
# 기존 함수 마지막 부분 찾아서 outlier detection 코드 삽입

# 함수 마지막 return 직전 search
old3 = '''    print(f"[Diarize] turn 수: {len(turns)} → {sum(1 for t in collapsed if (t[1]-t[0])>=min_segment_duration)} (병합/정제)")

    # 5) 새 Annotation 만들기 (collapsed 결과)
    from pyannote.core import Annotation, Segment
    final_diar = Annotation()
    for s, e, spk in collapsed:
        if (e - s) >= min_segment_duration:
            final_diar[Segment(s, e)] = spk
    return final_diar'''

new3 = '''    print(f"[Diarize] turn 수: {len(turns)} → {sum(1 for t in collapsed if (t[1]-t[0])>=min_segment_duration)} (병합/정제)")

    # ====== OUTLIER DETECTION ======
    # 각 segment의 embedding이 자기 화자 centroid와 too far → 잘못 분류
    # 그 segment를 새 화자(SPEAKER_OUT) 또는 다른 기존 화자에 재할당
    if model is not None and len(centroids) >= 1:
        OUTLIER_SELF_THRESH = 0.55  # 이하 = 자기 centroid와 너무 멈
        OUTLIER_REASSIGN_THRESH = 0.65  # 이상 = 다른 centroid와 충분히 가까움

        outlier_changes = []
        max_speaker_idx = max(
            (int(s.split("_")[-1]) for s, _, _ in [(spk, 0, 0) for spk in centroids.keys()] if "_" in s),
            default=0
        )

        # collapsed turn 다시 처리하여 각 turn embedding 추출
        for ci, (cs, ce, cspk) in enumerate(collapsed):
            # 짧은 turn은 embedding 안정성 부족 — skip
            if (ce - cs) < 0.5:
                continue
            s_idx, e_idx = int(cs * sr), int(ce * sr)
            emb = _ecapa_embedding(audio[s_idx:e_idx], sr, model)
            if emb is None:
                continue

            # 자기 centroid와 거리
            if cspk not in centroids:
                continue
            self_sim = float(np.dot(emb, centroids[cspk]))

            # 다른 centroid 중 가장 가까운
            best_other_spk = None
            best_other_sim = -1.0
            for ospk, oc in centroids.items():
                if ospk == cspk:
                    continue
                osim = float(np.dot(emb, oc))
                if osim > best_other_sim:
                    best_other_sim = osim
                    best_other_spk = ospk

            # 자기보다 다른 화자에 더 가까우면 재할당
            if best_other_spk and best_other_sim > self_sim and best_other_sim >= OUTLIER_REASSIGN_THRESH:
                outlier_changes.append((ci, cspk, best_other_spk, self_sim, best_other_sim))
                collapsed[ci] = (cs, ce, best_other_spk)

            # 자기 centroid와 너무 멀고 + 다른 centroid에서도 멀면 새 화자
            elif self_sim < OUTLIER_SELF_THRESH and best_other_sim < OUTLIER_SELF_THRESH:
                # 새 화자 ID 만들기
                new_spk = f"SPEAKER_{99 - len(outlier_changes):02d}"
                outlier_changes.append((ci, cspk, new_spk, self_sim, best_other_sim))
                collapsed[ci] = (cs, ce, new_spk)

        if outlier_changes:
            print(f"[Diarize] Outlier {len(outlier_changes)}개 재할당:")
            for ci, old_spk, new_spk, ss, os in outlier_changes:
                cs, ce = collapsed[ci][0], collapsed[ci][1]
                print(f"  turn{ci} [{cs:.2f}~{ce:.2f}] {old_spk}→{new_spk} (self={ss:.2f}, other={os:.2f})")

    # 5) 새 Annotation 만들기 (collapsed + outlier 결과)
    from pyannote.core import Annotation, Segment
    final_diar = Annotation()
    for s, e, spk in collapsed:
        if (e - s) >= min_segment_duration:
            final_diar[Segment(s, e)] = spk
    return final_diar'''

if old3 in src:
    src = src.replace(old3, new3)
    print("[3] OK: outlier detection 추가")
else:
    print("[3] NOT FOUND")

p.write_text(src)
print("[Done] outlier detection 적용")
