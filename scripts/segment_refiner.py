"""ASD-guided segment refinement for multi-speaker drama.

목적:
  DiariZen이 빠른 화자 교차 (drama)를 못 잡는 한계 보완.
  LightASD per-frame active face score로 segment 안의 화자 변화 감지 → split.

3가지 기능:
  1. ASD-guided split: face_id 변화 시점에서 segment 분할
  2. Tail attribution: 끝 1.5초가 다른 화자면 별도 segment
  3. Short utterance preservation: ASR 단어 적은데 duration 길면 trim

사용:
  from segment_refiner import refine_segments
  segments = refine_segments(segments, asd_result, words, vocals_path)
"""
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np

# 파라미터
MIN_TURN_DURATION = 0.5      # 0.5초 미만 turn은 split 안 함 (noise 방지)
TAIL_LOOKBACK = 1.5          # 끝 1.5초만 별도 검사 (tail attribution)
SHORT_UTTERANCE_WORDS = 3    # 단어 ≤3개 + duration ≥5s = 강제 trim
TRIM_MIN_DURATION = 1.0      # trim 후 최소 duration


def _build_per_frame_face(asd_result: Dict) -> List[Optional[int]]:
    """Per-frame active face index 빌드.

    각 frame에서 ASD score > 0 인 face 중 가장 높은 점수의 face index 반환.
    아무도 active 아니면 None.
    """
    if not asd_result or not asd_result.get("tracks"):
        return []
    n_frames = asd_result["n_frames"]
    tracks = asd_result["tracks"]
    per_frame = [None] * n_frames

    # frame별 best face 계산
    for face_idx, track in enumerate(tracks):
        track_start = track.get("frame_start", track.get("start", 0))
        scores = track.get("scores", [])
        for local_i, score in enumerate(scores):
            frame_i = track_start + local_i
            if frame_i >= n_frames:
                continue
            if score > 0:
                cur = per_frame[frame_i]
                if cur is None:
                    per_frame[frame_i] = (face_idx, score)
                else:
                    if score > cur[1]:
                        per_frame[frame_i] = (face_idx, score)

    # face_idx만 남기기
    return [p[0] if p is not None else None for p in per_frame]


def _detect_face_changes(
    seg_faces: List[Optional[int]],
    fps: float,
    seg_start_frame: int,
    min_turn_frames: int,
) -> List[int]:
    """Segment 안에서 face 변화 frame 감지.

    Returns: change_points (절대 frame 인덱스 리스트)
    """
    change_points = []
    if not seg_faces:
        return change_points

    last_face = None
    last_change_frame = seg_start_frame
    stable_count = 0
    candidate_face = None
    candidate_start = None

    for i, face in enumerate(seg_faces):
        cur_frame = seg_start_frame + i
        if face is None:
            # gap (no detection) — 유지
            continue
        if face != last_face:
            # 새 face 후보
            if face != candidate_face:
                candidate_face = face
                candidate_start = cur_frame
                stable_count = 1
            else:
                stable_count += 1
            # stable enough?
            if stable_count >= min_turn_frames:
                # 이전 face와 다르면 change point
                if last_face is not None and last_face != face:
                    if candidate_start - last_change_frame >= min_turn_frames:
                        change_points.append(candidate_start)
                        last_change_frame = candidate_start
                last_face = face
                candidate_face = None
                stable_count = 0
        else:
            stable_count = 0

    return change_points


def _snap_to_word_boundary(
    frame_time: float,
    words: List[Dict],
) -> float:
    """가장 가까운 word boundary로 snap. words=[{start, end, word}]"""
    if not words:
        return frame_time
    best_diff = float("inf")
    best_t = frame_time
    for w in words:
        for boundary in (w.get("start", 0), w.get("end", 0)):
            d = abs(boundary - frame_time)
            if d < best_diff:
                best_diff = d
                best_t = boundary
    return best_t


def _split_words(words: List[Dict], split_t: float) -> Tuple[List[Dict], List[Dict]]:
    """words를 split_t 기준 앞/뒤로 분할."""
    before, after = [], []
    for w in words:
        w_end = w.get("end", w.get("start", 0))
        if w_end <= split_t:
            before.append(w)
        else:
            after.append(w)
    return before, after


def refine_segment_with_asd(
    seg: Any,
    asd_per_frame_face: List[Optional[int]],
    fps: float,
    seg_words: List[Dict],
) -> List[Any]:
    """단일 segment를 ASD 결과로 분할.

    Returns: refined segments (1개 또는 N개)
    """
    seg_start_frame = int(seg.start * fps)
    seg_end_frame = int(seg.end * fps)
    seg_end_frame = min(seg_end_frame, len(asd_per_frame_face))

    if seg_end_frame <= seg_start_frame:
        return [seg]

    seg_faces = asd_per_frame_face[seg_start_frame:seg_end_frame]
    min_turn_frames = max(2, int(MIN_TURN_DURATION * fps))

    # face 변화 감지
    change_points = _detect_face_changes(
        seg_faces, fps, seg_start_frame, min_turn_frames
    )

    if not change_points:
        return [seg]

    # change_points를 시간으로 변환 + word boundary로 snap
    refined = []
    prev_t = seg.start
    for cp_frame in change_points:
        cp_t = cp_frame / fps
        # snap
        cp_t = _snap_to_word_boundary(cp_t, seg_words)
        # 너무 짧은 sub-segment 방지
        if cp_t - prev_t < MIN_TURN_DURATION:
            continue

        sub = deepcopy(seg)
        sub.start = prev_t
        sub.end = cp_t
        before_words, _ = _split_words(seg_words, cp_t)
        sub_text = " ".join(w.get("word", "") for w in before_words).strip()
        # Segment.text 필드 갱신 (JSON dump + LLM 번역 input)
        sub.text = sub_text
        sub.original_text = sub_text  # 호환성용 동적 attr
        # sub.words도 시간 범위 기준 필터 (TTS 등 후속 단계 정확도)
        _filter_words_by_time(sub, prev_t, cp_t)
        sub._was_split = True  # ECAPA 재할당 대상 표시
        refined.append(sub)
        prev_t = cp_t
        seg_words = [w for w in seg_words if w.get("end", w.get("start", 0)) > cp_t]

    # 마지막 sub-segment
    if seg.end - prev_t >= MIN_TURN_DURATION:
        last = deepcopy(seg)
        last.start = prev_t
        last.end = seg.end
        last_text = " ".join(w.get("word", "") for w in seg_words).strip()
        last.text = last_text
        last.original_text = last_text
        _filter_words_by_time(last, prev_t, seg.end)
        last._was_split = True  # ECAPA 재할당 대상 표시
        refined.append(last)

    return refined if len(refined) > 1 else [seg]


def _filter_words_by_time(seg: Any, start: float, end: float) -> None:
    """seg.words를 [start, end] 범위 안 단어만 남기게 inplace 필터.
    WordTiming dataclass 또는 dict 둘 다 지원."""
    if not hasattr(seg, "words") or not seg.words:
        return
    def _ws(w):
        return getattr(w, "start", None) if not isinstance(w, dict) else w.get("start")
    def _we(w):
        return getattr(w, "end", None) if not isinstance(w, dict) else w.get("end")
    seg.words = [
        w for w in seg.words
        if _ws(w) is not None and _we(w) is not None
        and _ws(w) >= start - 0.001 and _we(w) <= end + 0.001
    ]


def detect_short_utterance_extension(seg: Any, words: List[Dict]) -> Optional[Tuple[float, float]]:
    """짧은 발화 over-extension 감지.

    Returns: (new_start, new_end) if needs trim, else None.
    """
    word_count = len(words)
    duration = seg.end - seg.start

    if word_count <= SHORT_UTTERANCE_WORDS and duration >= 5.0:
        # 실제 발화 시점만 사용
        if words:
            actual_start = words[0].get("start", seg.start)
            actual_end = words[-1].get("end", seg.end)
            actual_duration = actual_end - actual_start
            if actual_duration >= TRIM_MIN_DURATION:
                # 약간 padding 추가
                pad = 0.2
                return (max(seg.start, actual_start - pad),
                        min(seg.end, actual_end + pad))
    return None


def _compute_ecapa_emb(
    audio: np.ndarray,
    sr: int,
    ecapa_model,
) -> Optional[np.ndarray]:
    """ECAPA 임베딩 추출 (192-dim, L2-normalized). 외부 모델 inject 받음."""
    try:
        import torch
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(audio) < int(sr * 0.4):
            return None
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = ecapa_model.encode_batch(wav).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return None
        return emb / norm
    except Exception as e:
        print(f"[Refine] ECAPA 임베딩 실패: {e}")
        return None


def reassign_speakers_by_ecapa(
    refined_segments: List[Any],
    vocals_path: str,
    centroids: Dict[str, np.ndarray],
    ecapa_model,
    min_sim: float = 0.5,
) -> List[Any]:
    """ASD split된 sub-segment를 ECAPA centroid bank와 비교해 speaker 재할당.

    - `_was_split` marker가 있는 segment만 처리 (단일 segment는 보존)
    - centroid 거리(cosine sim)가 min_sim 이상인 best speaker로 재할당
    - min_sim 이하면 원래 라벨 유지 (outlier 보호)
    """
    if not centroids or ecapa_model is None or not vocals_path:
        return refined_segments
    try:
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
    except Exception as e:
        print(f"[Refine] vocals 로드 실패 (재할당 스킵): {e}")
        return refined_segments
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    reassigned = 0
    skipped_outlier = 0
    for seg in refined_segments:
        if not getattr(seg, "_was_split", False):
            continue  # split 안 된 원본은 건드리지 않음
        s_idx = int(seg.start * sr)
        e_idx = min(int(seg.end * sr), len(audio))
        if e_idx - s_idx < int(0.4 * sr):
            continue  # 0.4초 미만은 임베딩 부족
        chunk = audio[s_idx:e_idx]
        emb = _compute_ecapa_emb(chunk, sr, ecapa_model)
        if emb is None:
            continue
        sims = {spk: float(np.dot(emb, cv)) for spk, cv in centroids.items()}
        best_spk = max(sims, key=sims.get)
        best_sim = sims[best_spk]
        if best_sim < min_sim:
            skipped_outlier += 1
            continue  # 모든 centroid에서 멀음 → 원래 라벨 유지
        if best_spk != seg.speaker:
            print(f"[Refine] speaker reassign [{seg.start:.2f}~{seg.end:.2f}] "
                  f"{seg.speaker} → {best_spk} (sim={best_sim:.2f})")
            seg.speaker = best_spk
            reassigned += 1
    if reassigned:
        print(f"[Refine] {reassigned}개 sub-segment speaker 재할당")
    if skipped_outlier:
        print(f"[Refine] {skipped_outlier}개 sub-segment outlier 보호 (원래 라벨 유지)")
    return refined_segments


def merge_speakers_by_centroid_distance(
    segments: List[Any],
    vocals_path: str,
    ecapa_model,
    min_seg_dur: float = 0.5,
    spk_face_centroid: Optional[Dict[str, "np.ndarray"]] = None,
    wespeaker_model=None,
    spk_gender: Optional[Dict[str, str]] = None,
    spk_shape_feats: Optional[Dict[str, "np.ndarray"]] = None,
) -> List[Any]:
    """SPK centroid 단위 voice cluster — fusion이 자동 detect한 화자들의 voice centroid를
    cosine distance 기반으로 자동 통합 (similar voice = 같은 사람으로 합침).

    v87+: 옵션으로 face centroid (spk_face_centroid)도 함께 사용. face cos sim 높고
    voice distance 적절히 낮으면 merge — voice만으론 구분 어려운 케이스 해결.

    Env var:
      LATENTSYNC_MERGE_SPK_BY_VOICE=1 → 활성
      LATENTSYNC_MERGE_SPK_VOICE_DIST (default 0.55) — voice cosine distance 임계
      LATENTSYNC_MERGE_SPK_FACE_SIM (default 0.0) — face cos sim 보조 임계
        (0.0 = face 정보 무시, 양수면 voice dist < 별도 임계 + face sim ≥ 이 값)
      LATENTSYNC_MERGE_SPK_VOICE_DIST_FACE (default 0.85) — face 통과 시 사용할 voice dist
    """
    if ecapa_model is None or not vocals_path or not segments:
        return segments
    import os as _os
    if _os.environ.get("LATENTSYNC_MERGE_SPK_BY_VOICE", "0") != "1":
        return segments
    dist_threshold = float(_os.environ.get("LATENTSYNC_MERGE_SPK_VOICE_DIST", "0.55"))
    face_sim_threshold = float(_os.environ.get("LATENTSYNC_MERGE_SPK_FACE_SIM", "0.0"))
    dist_threshold_face = float(_os.environ.get("LATENTSYNC_MERGE_SPK_VOICE_DIST_FACE", "0.85"))

    try:
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
    except Exception as e:
        print(f"[SPKMerge] vocals 로드 실패: {e}")
        return segments
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # SPK별 segment 임베딩 모음 (ECAPA + wespeaker)
    from collections import defaultdict
    spk_embs = defaultdict(list)
    spk_wespeaker_embs = defaultdict(list)
    # wespeaker는 wav 파일로 직접 호출하므로 임시 chunk 저장
    import tempfile, soundfile as _sf, os as _os2
    for seg in segments:
        dur = seg.end - seg.start
        if dur < min_seg_dur:
            continue
        chunk = audio[int(seg.start * sr):min(int(seg.end * sr), len(audio))]
        emb = _compute_ecapa_emb(chunk, sr, ecapa_model)
        if emb is not None:
            spk_embs[seg.speaker].append(emb)
        # wespeaker embedding (옵션)
        if wespeaker_model is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    _sf.write(tmp.name, chunk.astype(np.float32), sr)
                    tmp_path = tmp.name
                ws_emb = wespeaker_model.extract_embedding(tmp_path)
                _os2.unlink(tmp_path)
                if ws_emb is not None:
                    ws_arr = np.asarray(ws_emb, dtype=np.float32).flatten()
                    spk_wespeaker_embs[seg.speaker].append(ws_arr)
            except Exception as _we:
                pass

    if len(spk_embs) < 2:
        print(f"[SPKMerge] only {len(spk_embs)} SPKs, no merge")
        return segments

    # 각 SPK centroid (L2 normalized mean)
    spk_centroids = {}
    for spk, embs_list in spk_embs.items():
        c = np.mean(np.stack(embs_list), axis=0)
        c = c / max(np.linalg.norm(c), 1e-9)
        spk_centroids[spk] = c

    # wespeaker centroid (옵션)
    spk_ws_centroids = {}
    for spk, embs_list in spk_wespeaker_embs.items():
        if not embs_list:
            continue
        c = np.mean(np.stack(embs_list), axis=0)
        c = c / max(np.linalg.norm(c), 1e-9)
        spk_ws_centroids[spk] = c
    if spk_ws_centroids:
        print(f"[SPKMerge] wespeaker centroid 계산: {list(spk_ws_centroids.keys())}")

    # pairwise distance matrix
    spks = sorted(spk_centroids.keys())
    n = len(spks)
    print(f"[SPKMerge] {n} SPKs detected, computing pairwise distance...")
    dist_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            d = 1.0 - float(np.dot(spk_centroids[spks[i]], spk_centroids[spks[j]]))
            dist_pairs.append((d, spks[i], spks[j]))
    dist_pairs.sort()
    for d, a, b in dist_pairs[:10]:
        print(f"  {a} ↔ {b}: dist={d:.3f}")

    # union-find로 distance < threshold인 SPK 합침
    parent = {spk: spk for spk in spks}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # 더 많은 segment 가진 SPK가 canonical (라벨 보존)
            n_a = len(spk_embs[a])
            n_b = len(spk_embs[b])
            if n_a >= n_b:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # wespeaker dist 계산 (있으면)
    ws_dist_map = {}
    if spk_ws_centroids:
        for d, a, b in dist_pairs:
            ca = spk_ws_centroids.get(a)
            cb = spk_ws_centroids.get(b)
            if ca is not None and cb is not None:
                ws_dist_map[(a, b)] = 1.0 - float(np.dot(ca, cb))
                ws_dist_map[(b, a)] = ws_dist_map[(a, b)]
        if ws_dist_map:
            print("[SPKMerge] wespeaker pairwise:")
            for d, a, b in dist_pairs[:10]:
                wd = ws_dist_map.get((a, b))
                if wd is not None:
                    print(f"  {a} ↔ {b}: ecapa_dist={d:.3f}, wespeaker_dist={wd:.3f}, avg={(d+wd)/2:.3f}")

    # combined distance threshold (env)
    ws_combine = float(_os.environ.get("LATENTSYNC_MERGE_SPK_COMBINED_DIST", "0.0"))

    n_merged = 0
    n_blocked_gender = 0
    for d, a, b in dist_pairs:
        merge = False
        face_sim = None
        ws_dist = ws_dist_map.get((a, b)) if ws_dist_map else None
        combined = (d + ws_dist) / 2 if ws_dist is not None else d
        if d < dist_threshold:
            merge = True
        elif ws_combine > 0 and ws_dist is not None and combined < ws_combine:
            merge = True
        elif face_sim_threshold > 0 and spk_face_centroid is not None and d < dist_threshold_face:
            ca = spk_face_centroid.get(a)
            cb = spk_face_centroid.get(b)
            if ca is not None and cb is not None:
                face_sim = float(np.dot(ca, cb))
                if face_sim >= face_sim_threshold:
                    merge = True
        # v114+: gender 다른 SPK는 merge 금지 (강한 신호)
        if merge and spk_gender:
            ga = spk_gender.get(a)
            gb = spk_gender.get(b)
            if ga and gb and ga != gb:
                merge = False
                n_blocked_gender += 1
                print(f"[SPKMerge] {a}({ga}) ↔ {b}({gb}): merge blocked by gender mismatch", flush=True)
        # v120+: face shape feature 차이가 큰 SPK는 merge 금지
        if merge and spk_shape_feats:
            sa = spk_shape_feats.get(a)
            sb = spk_shape_feats.get(b)
            if sa is not None and sb is not None:
                shape_dist = float(np.linalg.norm(sa - sb))
                shape_block_th = float(_os.environ.get("LATENTSYNC_MERGE_SPK_SHAPE_DIST", "0.10"))
                if shape_dist >= shape_block_th:
                    merge = False
                    print(f"[SPKMerge] {a} ↔ {b}: merge blocked by shape mismatch (dist={shape_dist:.3f} ≥ {shape_block_th})", flush=True)
        if merge and find(a) != find(b):
            union(a, b)
            tag = f"voice_dist={d:.3f}"
            if ws_dist is not None:
                tag += f", ws_dist={ws_dist:.3f}, combined={combined:.3f}"
            if face_sim is not None:
                tag += f", face_sim={face_sim:.3f}"
            print(f"[SPKMerge] {a} + {b} ({tag}) → merge")
            n_merged += 1

    # 각 SPK → canonical (root) 매핑
    spk_to_canon = {spk: find(spk) for spk in spks}
    n_unique_before = len(spks)
    n_unique_after = len(set(spk_to_canon.values()))
    print(f"[SPKMerge] {n_unique_before} → {n_unique_after} SPKs (threshold={dist_threshold}, {n_merged} merges)")
    print(f"[SPKMerge debug] mapping: {spk_to_canon}")

    # 적용 — segments 안의 모든 speaker 값 변경 (없는 경우는 그대로)
    changed = 0
    for seg in segments:
        sp = getattr(seg, "speaker", None)
        if sp is None:
            continue
        canon = spk_to_canon.get(sp, sp)
        if canon != sp:
            seg.speaker = canon
            changed += 1
    print(f"[SPKMerge debug] {changed}/{len(segments)} segments relabeled")

    return segments


def force_voice_cluster_all_segments(
    segments: List[Any],
    vocals_path: str,
    ecapa_model,
    min_seg_dur: float = 0.5,
) -> List[Any]:
    """모든 segment의 ECAPA 임베딩 → AgglomerativeClustering (distance threshold 기반) → 자동 라벨 통합.

    Purpose: fusion + face_id가 같은 사람을 다른 시간대에 다른 SPK로 잡는 문제 해결.
    voice 신호만으로 자동 재라벨링. 화자 수 강제 안 함 (영상마다 다른 화자 수 자동 처리).

    Env var:
      LATENTSYNC_FORCE_VOICE_CLUSTER=1 → 활성
      LATENTSYNC_VOICE_CLUSTER_DIST (default 0.35) — cosine distance threshold
        값이 작을수록 cluster 더 많이 (보수), 클수록 적게 (적극 합침).
    """
    if ecapa_model is None or not vocals_path:
        return segments
    if not segments:
        return segments

    import os as _os
    if _os.environ.get("LATENTSYNC_FORCE_VOICE_CLUSTER", "0") != "1":
        return segments
    dist_threshold = float(_os.environ.get("LATENTSYNC_VOICE_CLUSTER_DIST", "0.35"))

    try:
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
    except Exception as e:
        print(f"[VoiceCluster] vocals 로드 실패: {e}")
        return segments
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 각 segment의 ECAPA 임베딩 추출
    embs = []
    valid_idx = []
    for i, seg in enumerate(segments):
        dur = seg.end - seg.start
        if dur < min_seg_dur:
            embs.append(None)
            continue
        s_idx = int(seg.start * sr)
        e_idx = min(int(seg.end * sr), len(audio))
        chunk = audio[s_idx:e_idx]
        emb = _compute_ecapa_emb(chunk, sr, ecapa_model)
        embs.append(emb)
        if emb is not None:
            valid_idx.append(i)

    if len(valid_idx) < 2:
        print(f"[VoiceCluster] valid embeddings {len(valid_idx)} too few, skip")
        return segments

    valid_embs = np.array([embs[i] for i in valid_idx])
    # L2 normalize (cosine similarity = dot product)
    norms = np.linalg.norm(valid_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    valid_embs = valid_embs / norms

    try:
        from sklearn.cluster import AgglomerativeClustering
        # distance threshold 기반 자동 cluster 수 결정
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(valid_embs)
        n_clusters = len(set(labels))
        print(f"[VoiceCluster] distance_threshold={dist_threshold} → {n_clusters} clusters (자동 결정)")
    except Exception as e:
        print(f"[VoiceCluster] clustering 실패: {e}")
        return segments

    # 각 cluster의 dominant 기존 SPK 라벨을 canonical로
    from collections import Counter, defaultdict
    cluster_old_spks = defaultdict(Counter)
    for i_valid, label in enumerate(labels):
        seg_idx = valid_idx[i_valid]
        old_spk = segments[seg_idx].speaker
        cluster_old_spks[label][old_spk] += 1

    # cluster → canonical SPEAKER_XX 라벨 매핑 (dominant 라벨)
    # 가장 큰 cluster부터 우선권 → smaller cluster가 같은 SPK 못 가져가도록
    sorted_clusters = sorted(cluster_old_spks.items(), key=lambda x: -sum(x[1].values()))
    cluster_to_spk = {}
    used_spks = set()
    next_idx = 0
    for cluster_id, spk_counts in sorted_clusters:
        assigned = False
        for spk, _ in spk_counts.most_common():
            if spk not in used_spks:
                cluster_to_spk[cluster_id] = spk
                used_spks.add(spk)
                assigned = True
                break
        if not assigned:
            # 새 라벨 부여 (충돌 회피)
            while f"SPEAKER_{next_idx:02d}" in used_spks:
                next_idx += 1
            new_label = f"SPEAKER_{next_idx:02d}"
            cluster_to_spk[cluster_id] = new_label
            used_spks.add(new_label)
            next_idx += 1

    # 각 valid segment에 새 라벨 부여
    n_changed = 0
    for i_valid, label in enumerate(labels):
        seg_idx = valid_idx[i_valid]
        new_spk = cluster_to_spk[label]
        old_spk = segments[seg_idx].speaker
        if new_spk != old_spk:
            segments[seg_idx].speaker = new_spk
            n_changed += 1
            print(f"[VoiceCluster] [{segments[seg_idx].start:.2f}~{segments[seg_idx].end:.2f}] {old_spk} → {new_spk}")

    # invalid (짧은 segment) → 인접 segment 라벨 따라감
    for i, emb in enumerate(embs):
        if emb is not None:
            continue
        # 가장 가까운 valid segment 라벨
        best_d = float("inf")
        best_spk = segments[i].speaker
        mid = (segments[i].start + segments[i].end) / 2
        for vi in valid_idx:
            v_mid = (segments[vi].start + segments[vi].end) / 2
            d = abs(v_mid - mid)
            if d < best_d:
                best_d = d
                best_spk = segments[vi].speaker
        if best_spk != segments[i].speaker:
            segments[i].speaker = best_spk

    print(f"[VoiceCluster] {n_changed}/{len(valid_idx)} segments 재라벨 (auto n={n_clusters}, dist_th={dist_threshold})")
    return segments


def refine_segments(
    segments: List[Any],
    asd_result: Optional[Dict] = None,
    words_by_seg: Optional[List[List[Dict]]] = None,
    speaker_centroids: Optional[Dict[str, np.ndarray]] = None,
    vocals_path: Optional[str] = None,
    ecapa_model=None,
    spk_face_centroid: Optional[Dict[str, np.ndarray]] = None,
    wespeaker_model=None,
    spk_gender: Optional[Dict[str, str]] = None,
    spk_shape_feats: Optional[Dict[str, np.ndarray]] = None,
) -> List[Any]:
    """전체 segment list refinement.

    INPUT:
      segments: List[Segment]
      asd_result: dict from run_asd() (optional)
      words_by_seg: List[List[Dict]] - segment별 word timestamps (optional)
      speaker_centroids: ECAPA centroid bank {speaker: embedding} (optional)
      vocals_path: vocals.wav 경로 (재할당 시 필요)
      ecapa_model: ECAPA-TDNN 모델 (재할당 시 필요)

    OUTPUT:
      refined segments (split + trim + speaker reassign 적용)
    """
    if not segments:
        return segments

    # ASD-guided split (asd_result + words 둘 다 있으면)
    refined = []
    if asd_result and words_by_seg:
        per_frame_face = _build_per_frame_face(asd_result)
        fps = asd_result["fps"]

        for seg, seg_words in zip(segments, words_by_seg):
            sub_segs = refine_segment_with_asd(seg, per_frame_face, fps, seg_words)
            refined.extend(sub_segs)

        if len(refined) > len(segments):
            print(f"[Refine] ASD split: {len(segments)} → {len(refined)} segments")
    else:
        refined = list(segments)

    # 짧은 발화 trim
    final = []
    trim_count = 0
    for i, seg in enumerate(refined):
        seg_words = words_by_seg[i] if words_by_seg and i < len(words_by_seg) else []
        if not seg_words:
            # words 없으면 trim 못함 → 그대로
            final.append(seg)
            continue
        trim_result = detect_short_utterance_extension(seg, seg_words)
        if trim_result:
            new_start, new_end = trim_result
            old_dur = seg.end - seg.start
            new_dur = new_end - new_start
            seg.start = new_start
            seg.end = new_end
            print(f"[Refine] short utterance trim: {old_dur:.1f}s → {new_dur:.1f}s "
                  f"({len(seg_words)} words)")
            trim_count += 1
        final.append(seg)

    if trim_count > 0:
        print(f"[Refine] {trim_count} short utterances trimmed")

    # ASD-split된 sub-segment를 ECAPA centroid로 speaker 재할당
    if speaker_centroids and ecapa_model is not None and vocals_path:
        final = reassign_speakers_by_ecapa(
            final, vocals_path, speaker_centroids, ecapa_model
        )

    # ECAPA sliding window로 audio-blind 화자 변화 감지 + split
    # (DiariZen이 turn을 못 만든 빠른 화자 교차 대응)
    # LATENTSYNC_SLIDING_SPLIT_OFF=1 → 비활성 (v66+: over-fragmented 회피)
    import os as _os
    _slide_off = _os.environ.get("LATENTSYNC_SLIDING_SPLIT_OFF", "0") == "1"
    if not _slide_off and speaker_centroids and ecapa_model is not None and vocals_path:
        final = sliding_split_all(final, vocals_path, speaker_centroids, ecapa_model)
    elif _slide_off:
        print("[Refine] ECAPA sliding split 비활성화 (LATENTSYNC_SLIDING_SPLIT_OFF=1)", flush=True)

    # 너무 짧은 outlier segment 흡수 (ECAPA 신뢰도 낮음)
    final = _absorb_short_outliers(final)

    # v71+: ECAPA voice-based 자동 재라벨링 (distance threshold 기반, 화자 수 자동 결정)
    if ecapa_model is not None and vocals_path:
        final = force_voice_cluster_all_segments(
            final, vocals_path, ecapa_model,
        )

    # v83+: SPK centroid 단위 자동 통합 (segment cluster보다 robust)
    # v87+: spk_face_centroid 전달 시 face cos sim 보조 조건 활성화
    # v98+: wespeaker_model 전달 시 ECAPA+wespeaker combined distance 활성화
    # v114+: spk_gender 전달 시 다른 gender SPK는 merge 금지
    # v120+: spk_shape_feats 전달 시 shape 차이 큰 SPK는 merge 금지
    if ecapa_model is not None and vocals_path:
        final = merge_speakers_by_centroid_distance(
            final, vocals_path, ecapa_model,
            spk_face_centroid=spk_face_centroid,
            wespeaker_model=wespeaker_model,
            spk_gender=spk_gender,
            spk_shape_feats=spk_shape_feats,
        )

    # v123+: short segment context-aware reassignment
    # 짧은 segment (<2.5s) + 오래 침묵한 SPK 라벨 → 최근 active SPK로 reassign
    # voice clustering 본질 한계 보완 (e.g. "Good" 0.82s, "stopped petting rabbit" 2.19s)
    if ecapa_model is not None and vocals_path:
        final = _reassign_short_with_voice_continuity(
            final, vocals_path, ecapa_model,
        )

    # v166+: face-track-cluster based reassignment (multi-modal)
    # 각 segment의 dominant face track → cluster_id → 일관성 강제
    final = _reassign_by_face_track_cluster(final)

    # v173+: face-TRACK continuity (cluster보다 fine — 같은 track id = same person)
    final = _reassign_by_face_track_continuity(final)

    # v160+: split long segments by F0 gender transitions (yelling 끼어듦 detect)
    # 긴 segment 내에서 female F0 spike → 다른 화자 끼어듦 (e.g. 13s SPK_3 monologue + SPK_0 yelling)
    if vocals_path:
        final = _split_long_by_f0_transitions(final, vocals_path)

    # v174+: time-gap based SPK split (scene change detect via temporal discontinuity)
    # 같은 SPK 라벨에 큰 시간 gap (>=5s) 있으면 sub-cluster들 voice 검증 후 다른 SPK로 reassign.
    # Test4 SPK_05 케이스: Sean (59s) + 다른 SPK (68-97s, 8.8s+ gap) 자동 분리.
    if vocals_path:
        final = _split_spk_by_time_gap(final, vocals_path)

    # v153+: Intra-SPK voice consistency split (contaminated cluster 자동 분리)
    # v158+: iterate until convergence
    if vocals_path:
        import os as _os_intra
        n_iter = int(_os_intra.environ.get("LATENTSYNC_INTRASPK_PASSES", "1"))
        for pi in range(n_iter):
            before = [s.speaker for s in final]
            final = _split_contaminated_spk_eres2(final, vocals_path)
            after = [s.speaker for s in final]
            n_ch = sum(1 for a, b in zip(before, after) if a != b)
            print(f"[IntraSPK] iteration {pi+1}/{n_iter}: {n_ch} changes", flush=True)
            if n_ch == 0:
                break

    # v133+: ERes2NetV2 short-clip embedding reassignment
    # 짧은 segment에서 ECAPA가 잘못 cluster한 케이스를 ERes2NetV2로 fix.
    # 실험 확인: "Good" 0.82s → Brian, "stopped petting" → other_female 자동 detect.
    # v151+: multi-pass — centroid 재계산 후 추가 reassign으로 cascading fix
    if vocals_path:
        import os as _os_mp
        n_passes = int(_os_mp.environ.get("LATENTSYNC_ERES2_PASSES", "1"))
        for pass_i in range(n_passes):
            before = [s.speaker for s in final]
            final = _reassign_short_with_eres2netv2(final, vocals_path, _pass_num=pass_i + 1)
            after = [s.speaker for s in final]
            n_changed = sum(1 for a, b in zip(before, after) if a != b)
            print(f"[ERes2Short] pass {pass_i+1}/{n_passes}: {n_changed} changes", flush=True)
            if n_changed == 0:
                break  # converged

    # v139+: F0-based gender enforcement (직교 신호 — voice embedding과 다른 차원)
    # 각 segment의 dominant F0 gender 계산.
    # SPK centroid gender (face_id 기반) 와 다르면 reassign to closest matching-gender SPK.
    if vocals_path:
        final = _enforce_f0_gender(final, vocals_path)

    # v181+: 최종 singleton sandwich-override pass.
    # 모든 reassign 끝난 상태에서 SPK가 1 segment만 남았고, 양옆 neighbor가
    # 같은 다른 SPK + gap <= 0.3s 직접 접촉이면 그 SPK로 reassign.
    # "Find another" 78.09 SPK_00 → SPK_03 같은 case 자동 fix.
    final = _final_singleton_sandwich_pass(final)

    return final


def _final_singleton_sandwich_pass(segments: List[Any]) -> List[Any]:
    """v181+: All-pass 종료 후 singleton SPK sandwich override.
    1개 segment SPK + 양옆 직접 접촉(gap<=th) + 양옆 같은 다른 SPK → reassign.
    Sean 같은 unique 0.52s+ gap 보존 (sandwich 아님).
    """
    import os as _os_sw
    if _os_sw.environ.get("LATENTSYNC_FINAL_SANDWICH", "1") != "1":
        return segments
    if len(segments) < 3:
        return segments
    th = float(_os_sw.environ.get("LATENTSYNC_SANDWICH_GAP", "0.3"))
    from collections import Counter
    counts = Counter(s.speaker for s in segments)
    singletons = {spk for spk, n in counts.items() if n == 1}
    if not singletons:
        return segments
    n_changed = 0
    for i in range(1, len(segments) - 1):
        seg = segments[i]
        if seg.speaker not in singletons:
            continue
        prev_s = segments[i - 1]
        next_s = segments[i + 1]
        prev_gap = seg.start - prev_s.end
        next_gap = next_s.start - seg.end
        if (prev_s.speaker == next_s.speaker
                and prev_s.speaker != seg.speaker
                and 0 <= prev_gap <= th
                and 0 <= next_gap <= th):
            old = seg.speaker
            seg.speaker = prev_s.speaker
            n_changed += 1
            print(f"[FinalSandwich] [{seg.start:.2f}~{seg.end:.2f}] {old} → "
                  f"{seg.speaker} (neighbors={prev_s.speaker}, "
                  f"gaps={prev_gap:.2f}/{next_gap:.2f}s)", flush=True)
    if n_changed:
        print(f"[FinalSandwich] {n_changed} singleton(s) sandwich-reassigned")
    return segments


def _join_words_text(words) -> str:
    """WordTiming 또는 dict 리스트에서 text 추출."""
    if not words:
        return ""
    parts = []
    for w in words:
        if isinstance(w, dict):
            parts.append(w.get("word", ""))
        else:
            parts.append(getattr(w, "word", ""))
    return " ".join(p for p in parts if p).strip()


def split_by_ecapa_sliding(
    seg: Any,
    vocals_path: str,
    centroids: Dict[str, np.ndarray],
    ecapa_model,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    window: float = 0.6,
    hop: float = 0.2,
    min_consecutive: int = 2,
    min_sub_dur: float = 0.5,
    min_seg_dur: float = 1.5,
) -> List[Any]:
    import os as _os
    min_consecutive = int(_os.environ.get("LATENTSYNC_SLIDING_MIN_CONSEC", str(min_consecutive)))
    min_sub_dur = float(_os.environ.get("LATENTSYNC_SLIDING_MIN_SUB_DUR", str(min_sub_dur)))
    min_seg_dur = float(_os.environ.get("LATENTSYNC_SLIDING_MIN_SEG_DUR", str(min_seg_dur)))
    """segment 안에서 ECAPA sliding window로 화자 변화 감지 → split.

    DiariZen이 turn을 못 만든 빠른 화자 교차에 대응. ASD-blind 케이스 보완.

    PARAMETERS:
      window: ECAPA window 크기 (초). 0.6은 임베딩 안정성 + 시간 해상도 균형.
      hop: 윈도우 이동 간격
      min_consecutive: 화자 변화로 인정할 연속 윈도우 수 (1-window 깜빡임 무시)
      min_sub_dur: split된 sub-segment 최소 길이
      min_seg_dur: 이 길이 미만 segment는 sliding 분석 안 함
    """
    if not centroids or ecapa_model is None or not vocals_path:
        return [seg]
    if seg.end - seg.start < min_seg_dur:
        return [seg]

    # audio 로드 (외부에서 inject 받으면 재사용)
    if audio is None or sr is None:
        try:
            import soundfile as sf
            audio, sr = sf.read(vocals_path)
        except Exception:
            return [seg]
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

    # window별 best speaker
    estimates = []  # list of (start_t, best_spk, best_sim)
    t = seg.start
    while t + window <= seg.end:
        chunk = audio[int(t * sr):int((t + window) * sr)]
        emb = _compute_ecapa_emb(chunk, sr, ecapa_model)
        if emb is None:
            t += hop
            continue
        sims = {spk: float(np.dot(emb, cv)) for spk, cv in centroids.items()}
        best = max(sims, key=sims.get)
        estimates.append((t, best, sims[best]))
        t += hop

    if len(estimates) < min_consecutive * 2:
        return [seg]

    # smoothing: min_consecutive 미만 연속은 인접으로 흡수
    spk_seq = [e[1] for e in estimates]
    smoothed = list(spk_seq)
    i = 0
    while i < len(smoothed):
        j = i
        while j < len(smoothed) and smoothed[j] == smoothed[i]:
            j += 1
        run_len = j - i
        if run_len < min_consecutive:
            # 이전 화자로 흡수 (없으면 다음)
            if i > 0:
                replace = smoothed[i - 1]
            elif j < len(smoothed):
                replace = smoothed[j]
            else:
                replace = smoothed[i]
            for k in range(i, j):
                smoothed[k] = replace
        i = j

    # 변화 지점 (smoothed sequence에서)
    change_indices = [i for i in range(1, len(smoothed)) if smoothed[i] != smoothed[i - 1]]
    if not change_indices:
        return [seg]

    # 변화 시간 (해당 윈도우 시작점)
    change_times = [estimates[i][0] for i in change_indices]

    # word boundary로 snap
    if hasattr(seg, "words") and seg.words:
        def _ws(w):
            return getattr(w, "start", None) if not isinstance(w, dict) else w.get("start")
        def _we(w):
            return getattr(w, "end", None) if not isinstance(w, dict) else w.get("end")
        for i, ct in enumerate(change_times):
            best_diff = float("inf")
            best_t = ct
            for w in seg.words:
                ws, we = _ws(w), _we(w)
                if ws is None or we is None:
                    continue
                for boundary in (ws, we):
                    d = abs(boundary - ct)
                    if d < best_diff:
                        best_diff = d
                        best_t = boundary
            change_times[i] = best_t

    # split
    refined = []
    prev_t = seg.start
    prev_idx = 0
    for ct, change_idx in zip(change_times, change_indices):
        sub_dur = ct - prev_t
        if sub_dur < min_sub_dur:
            continue
        sub_spk = smoothed[prev_idx]
        sub = deepcopy(seg)
        sub.start = prev_t
        sub.end = ct
        sub.speaker = sub_spk
        _filter_words_by_time(sub, prev_t, ct)
        if hasattr(sub, "words"):
            sub.text = _join_words_text(sub.words)
            sub.original_text = sub.text
        sub._was_split = True
        sub._split_method = "ecapa_sliding"
        refined.append(sub)
        prev_t = ct
        prev_idx = change_idx

    # 마지막 sub-segment
    if seg.end - prev_t >= min_sub_dur:
        sub_spk = smoothed[prev_idx]
        last = deepcopy(seg)
        last.start = prev_t
        last.end = seg.end
        last.speaker = sub_spk
        _filter_words_by_time(last, prev_t, seg.end)
        if hasattr(last, "words"):
            last.text = _join_words_text(last.words)
            last.original_text = last.text
        last._was_split = True
        last._split_method = "ecapa_sliding"
        refined.append(last)

    # 빈 word sub-segment가 있으면 split 자체를 취소 (silence/noise false positive 방지)
    has_empty = any(not getattr(s, "words", []) for s in refined)
    if has_empty:
        empty_ranges = [(s.start, s.end) for s in refined if not getattr(s, "words", [])]
        print(f"[Refine] ECAPA-sliding split 취소 [{seg.start:.2f}~{seg.end:.2f}]: "
              f"빈 sub {empty_ranges} (silence/noise false positive)")
        return [seg]

    if len(refined) > 1:
        summary = " | ".join(f"{s.speaker}[{s.start:.2f}~{s.end:.2f}]" for s in refined)
        print(f"[Refine] ECAPA-sliding split [{seg.start:.2f}~{seg.end:.2f}] "
              f"{seg.speaker} → {summary}")
        return refined
    return [seg]


def sliding_split_all(
    segments: List[Any],
    vocals_path: str,
    centroids: Dict[str, np.ndarray],
    ecapa_model,
) -> List[Any]:
    """전체 segment list에 ECAPA sliding split 일괄 적용. audio 1회 로드."""
    if not centroids or ecapa_model is None or not vocals_path:
        return segments
    try:
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
    except Exception as e:
        print(f"[Refine] sliding split 위해 vocals 로드 실패: {e}")
        return segments
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    new_segs = []
    n_split = 0
    for seg in segments:
        subs = split_by_ecapa_sliding(
            seg, vocals_path, centroids, ecapa_model,
            audio=audio, sr=sr,
        )
        if len(subs) > 1:
            n_split += 1
        new_segs.extend(subs)
    if n_split:
        print(f"[Refine] ECAPA-sliding 추가 split: {n_split}개 segment → 총 {len(new_segs)}개")
    return new_segs


def _enforce_f0_gender(
    segments: List[Any],
    vocals_path: str,
) -> List[Any]:
    """v139+: F0-based gender enforcement.

    Voice embedding (ECAPA/ERes2NetV2) 는 timbre 위주라 동일 성별 화자 분리 어려움.
    F0 pitch는 male/female 명확히 구분 (165/175Hz 기준).

    Algorithm:
      1. librosa.pyin 으로 per-frame F0 추출
      2. 각 segment의 dominant gender (male/female/unknown) 결정
      3. SPK centroid gender 계산 (segment dominant gender의 mode)
      4. Segment gender ≠ SPK gender → matching-gender SPK 중 voice sim 최고로 reassign

    환경변수:
      LATENTSYNC_F0_GENDER (default 1) — 활성
      LATENTSYNC_F0_MALE_MAX (default 165) — male F0 상한
      LATENTSYNC_F0_FEMALE_MIN (default 175) — female F0 하한
      LATENTSYNC_F0_MIN_SHARE (default 0.6) — segment gender 결정 최소 share
    """
    import os as _os_f0
    if _os_f0.environ.get("LATENTSYNC_F0_GENDER", "1") != "1":
        return segments
    if not vocals_path or not segments:
        return segments

    male_max = float(_os_f0.environ.get("LATENTSYNC_F0_MALE_MAX", "165"))
    female_min = float(_os_f0.environ.get("LATENTSYNC_F0_FEMALE_MIN", "175"))
    min_share = float(_os_f0.environ.get("LATENTSYNC_F0_MIN_SHARE", "0.6"))

    try:
        import librosa
        y, sr = librosa.load(vocals_path, sr=None, mono=True)
        print(f"[F0Gender] computing pyin for {len(y)/sr:.1f}s audio ...", flush=True)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048,
        )
        pyin_times = librosa.times_like(f0, sr=sr)
    except Exception as _e:
        print(f"[F0Gender] pyin failed: {_e}")
        return segments

    # Per-segment dominant gender
    # v140+: yelling/shouting 시 F0 spike (200-320Hz) → median 잘못. 25th percentile 사용.
    # 차분한 음성의 진정한 F0가 화자 정체성에 더 가까움.
    def _seg_gender(start, end):
        mask = (pyin_times >= start) & (pyin_times < end)
        f0_chunk = f0[mask]
        valid = f0_chunk[~np.isnan(f0_chunk)] if hasattr(f0_chunk, '__len__') else []
        if len(valid) < 3:
            return None, 0.0
        # v140: 25th percentile (lower) — yelling 시 F0 spike 무시, 안정 음성만 사용
        f0_robust = float(np.percentile(valid, 25))
        n_male = int(np.sum(valid < male_max))
        n_female = int(np.sum(valid > female_min))
        n_total = len(valid)
        male_share = n_male / n_total
        female_share = n_female / n_total
        # robust median (25th percentile) 기준 추가 — yelling 시 보호
        if f0_robust < male_max:
            return "M", f0_robust
        if f0_robust > female_min:
            # share도 high 일 때만
            if female_share >= min_share:
                return "F", f0_robust
            return None, f0_robust
        # boundary zone: share 기준
        if male_share >= min_share and male_share > female_share:
            return "M", f0_robust
        if female_share >= min_share and female_share > male_share:
            return "F", f0_robust
        return None, f0_robust

    # Compute SPK gender (mode of segment genders)
    from collections import defaultdict
    spk_gender_votes = defaultdict(lambda: {"M": 0.0, "F": 0.0})
    seg_genders = []
    for seg in segments:
        g, f0_med = _seg_gender(seg.start, seg.end)
        seg_genders.append((g, f0_med))
        if g:
            spk_gender_votes[seg.speaker][g] += (seg.end - seg.start)

    spk_gender = {}
    for spk, votes in spk_gender_votes.items():
        if votes["M"] > votes["F"] * 1.5:
            spk_gender[spk] = "M"
        elif votes["F"] > votes["M"] * 1.5:
            spk_gender[spk] = "F"
        else:
            spk_gender[spk] = None  # ambiguous

    print(f"[F0Gender] SPK genders: {spk_gender}")

    # Build per-SPK ECAPA centroid for reassign (use existing segment_refiner global if available)
    # Simpler: compute voice sim via ERes2NetV2 centroids (already exists logic)
    try:
        import sys as _sys
        if "/workspace" not in _sys.path:
            _sys.path.insert(0, "/workspace")
        from patches.eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model
        get_eres2netv2_model()
        import soundfile as sf
        audio, sr_v = sf.read(vocals_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Per-SPK centroid from segments matching SPK's dominant gender
        spk_long_embs = defaultdict(list)
        for seg in segments:
            dur = seg.end - seg.start
            if dur < 2.0:
                continue
            chunk = audio[int(seg.start * sr_v):int(seg.end * sr_v)]
            emb = extract_eres2netv2_emb(chunk, sr=sr_v)
            if emb is not None:
                spk_long_embs[seg.speaker].append(emb)

        spk_centroids = {}
        for spk, embs in spk_long_embs.items():
            c = np.mean(np.stack(embs), axis=0)
            c = c / max(np.linalg.norm(c), 1e-9)
            spk_centroids[spk] = c.astype(np.float32)
    except Exception as _e:
        print(f"[F0Gender] ERes2NetV2 setup failed: {_e}")
        return segments

    n_changed = 0
    for i, seg in enumerate(segments):
        seg_g, f0_med = seg_genders[i]
        cur_spk = seg.speaker
        cur_spk_g = spk_gender.get(cur_spk)
        # Only reassign when segment gender is CLEAR and disagrees with cur SPK
        if not seg_g or not cur_spk_g or seg_g == cur_spk_g:
            continue

        # Find best matching-gender SPK by voice sim
        chunk = audio[int(seg.start * sr_v):int(seg.end * sr_v)]
        emb = extract_eres2netv2_emb(chunk, sr=sr_v)
        if emb is None:
            continue
        candidates = [(spk, c) for spk, c in spk_centroids.items() if spk_gender.get(spk) == seg_g]
        if not candidates:
            continue
        best_spk, best_sim = None, -1.0
        for spk, c in candidates:
            sim = float(np.dot(emb, c))
            if sim > best_sim:
                best_sim = sim
                best_spk = spk
        if best_spk and best_spk != cur_spk and best_sim >= 0.30:
            old_spk = cur_spk
            seg.speaker = best_spk
            n_changed += 1
            dur = seg.end - seg.start
            print(f"[F0Gender] [{seg.start:.2f}~{seg.end:.2f}] {old_spk}({cur_spk_g}) "
                  f"→ {best_spk}({seg_g}, sim={best_sim:.3f}) F0={f0_med:.0f}Hz dur={dur:.2f}s",
                  flush=True)
    if n_changed:
        print(f"[F0Gender] {n_changed} segments reassigned by F0 gender mismatch", flush=True)
    return segments


def _reassign_by_face_track_cluster(segments: List[Any]) -> List[Any]:
    """v166+: face-track-cluster 기반 segment 일관성 reassignment.

    각 segment 시간대에 dominant face track → 그 track의 face cluster_id 결정.
    같은 face cluster_id를 가진 segments는 같은 SPK 라벨이어야 함 (face 정체성).

    Algorithm:
      1. asd_result에서 face track frames 가져옴
      2. 각 segment 시간대(start~end) 안에 어떤 track이 가장 많은 frame 가지는지
      3. 그 track의 face cluster_id 조회
      4. cluster_id → SPK 매핑 vote (각 cluster의 majority SPK)
      5. segment의 face cluster_id 매핑 SPK가 현재 라벨과 다르고 vote 강하면 reassign

    환경변수:
      LATENTSYNC_FACE_CLUSTER_REASSIGN (default 1)
      LATENTSYNC_FACE_CLUSTER_MIN_VOTE_RATIO (default 0.7) — cluster내 majority SPK 비율
    """
    import os as _os_fc
    if _os_fc.environ.get("LATENTSYNC_FACE_CLUSTER_REASSIGN", "1") != "1":
        return segments
    try:
        import sys as _sys
        # Match orchestrator's import path — face_id_embedder (NOT scripts.face_id_embedder)
        # to share global state correctly
        for _p in ("/workspace/scripts", "/workspace"):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        import face_id_embedder as _fie
        track_to_cluster = _fie.get_last_track_to_cluster()
        asd_result = _fie.get_last_asd_result()
        print(f"[FaceCluster] starting: track_to_cluster={len(track_to_cluster)} entries, "
              f"asd_result={'set' if asd_result else 'None'}", flush=True)
        if not track_to_cluster or not asd_result:
            print(f"[FaceCluster] missing data — skip", flush=True)
            return segments
    except Exception as _e:
        print(f"[FaceCluster] exception: {_e}", flush=True)
        return segments

    fps = float(asd_result.get("fps", 25.0))
    n_frames_total = int(asd_result.get("n_frames", 0))
    tracks = asd_result.get("tracks", [])
    if not tracks:
        return segments

    # Build frame → list of (track_idx, score) mapping
    frame_to_tracks = [[] for _ in range(max(n_frames_total, 1))]
    for tidx, t in enumerate(tracks):
        t_frames = t.get("frames", [])
        t_scores = t.get("scores", [1.0] * len(t_frames))
        for fi, f in enumerate(t_frames):
            if 0 <= f < len(frame_to_tracks):
                score = float(t_scores[fi]) if fi < len(t_scores) else 0.0
                frame_to_tracks[f].append((tidx, score))

    # For each segment, find dominant face cluster
    from collections import defaultdict, Counter
    seg_face_cluster = []
    for seg in segments:
        f1 = max(0, int(seg.start * fps))
        f2 = min(len(frame_to_tracks), int(seg.end * fps + 1))
        track_score = defaultdict(float)
        for fi in range(f1, f2):
            for tidx, score in frame_to_tracks[fi]:
                # weight by speaking score (LightASD)
                track_score[tidx] += max(0.0, score)
        if not track_score:
            seg_face_cluster.append(None)
            continue
        best_track = max(track_score.items(), key=lambda x: x[1])[0]
        cluster_id = track_to_cluster.get(best_track)
        seg_face_cluster.append(cluster_id)

    # cluster_id → SPK label voting
    min_vote_ratio = float(_os_fc.environ.get("LATENTSYNC_FACE_CLUSTER_MIN_VOTE_RATIO", "0.7"))
    cluster_spk_votes = defaultdict(Counter)
    for i, seg in enumerate(segments):
        cid = seg_face_cluster[i]
        if cid is not None:
            cluster_spk_votes[cid][seg.speaker] += (seg.end - seg.start)

    # Determine dominant SPK for each cluster
    cluster_to_spk = {}
    for cid, votes in cluster_spk_votes.items():
        total = sum(votes.values())
        if total < 1.0:
            continue
        top_spk, top_n = votes.most_common(1)[0]
        ratio = top_n / total
        if ratio >= min_vote_ratio:
            cluster_to_spk[cid] = top_spk

    if not cluster_to_spk:
        print("[FaceCluster] no strong cluster → SPK mapping")
        return segments

    print(f"[FaceCluster] cluster→SPK: {cluster_to_spk}")

    # Reassign segments whose face cluster maps to different SPK
    n_reassigned = 0
    for i, seg in enumerate(segments):
        cid = seg_face_cluster[i]
        if cid is None:
            continue
        target_spk = cluster_to_spk.get(cid)
        if target_spk and target_spk != seg.speaker:
            print(f"[FaceCluster] [{seg.start:.2f}~{seg.end:.2f}] "
                  f"{seg.speaker} → {target_spk} (face_cluster={cid})", flush=True)
            seg.speaker = target_spk
            n_reassigned += 1
    if n_reassigned:
        print(f"[FaceCluster] {n_reassigned} segments reassigned by face cluster consistency")
    return segments


def _split_spk_by_time_gap(segments: List[Any], vocals_path: Optional[str] = None) -> List[Any]:
    """v174+: 같은 SPK 라벨이 큰 시간 gap (>=gap_th)로 떨어져 있으면 scene change 가능성.
    Sub-cluster들 voice 다르면 (ERes2 cos < threshold) 다른 SPK로 split.

    Test4 SPK_05 케이스:
      - 59.19~59.87 (Sean, taxi 씬)
      - 68.70 (8.83s gap, 다른 씬)
      - 95.32 (23s gap)
    같은 SPK_05이지만 시간상 분리 → voice도 분리되면 다른 SPK로.
    """
    import os as _os_tg
    if _os_tg.environ.get("LATENTSYNC_TIME_GAP_SPLIT", "1") != "1":
        return segments
    if not vocals_path or not segments:
        return segments

    gap_th = float(_os_tg.environ.get("LATENTSYNC_TIME_GAP_TH", "5.0"))
    voice_diff_th = float(_os_tg.environ.get("LATENTSYNC_TIME_GAP_VOICE_DIFF", "0.20"))
    min_target_sim = float(_os_tg.environ.get("LATENTSYNC_TIME_GAP_MIN_TARGET_SIM", "0.50"))
    eval_all_subs = _os_tg.environ.get("LATENTSYNC_TIME_GAP_EVAL_ALL", "1") == "1"

    try:
        import sys as _sys
        if "/workspace" not in _sys.path:
            _sys.path.insert(0, "/workspace")
        from patches.eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model
        get_eres2netv2_model()
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    except Exception as _e:
        print(f"[TimeGapSplit] init 실패: {_e}")
        return segments

    # Group segments by SPK label
    from collections import defaultdict
    spk_groups = defaultdict(list)  # spk → list of (seg_idx, seg)
    for i, seg in enumerate(segments):
        spk_groups[seg.speaker].append((i, seg))

    # Compute ERes2 embeddings per segment
    seg_embs = {}
    for i, seg in enumerate(segments):
        dur = seg.end - seg.start
        if dur < 0.4:
            continue
        chunk = audio[int(seg.start * sr):int(seg.end * sr)]
        emb = extract_eres2netv2_emb(chunk, sr=sr)
        if emb is not None:
            seg_embs[i] = emb

    n_changed = 0
    print(f"[TimeGapSplit] checking SPKs with gap >= {gap_th}s, voice_diff >= {voice_diff_th}", flush=True)

    for spk, group in spk_groups.items():
        if len(group) < 2:
            continue
        # Sort by start
        group_sorted = sorted(group, key=lambda x: x[1].start)
        # Find sub-clusters by time gap
        sub_clusters = [[group_sorted[0]]]
        for prev_idx in range(len(group_sorted) - 1):
            cur = group_sorted[prev_idx + 1]
            prev = group_sorted[prev_idx]
            gap = cur[1].start - prev[1].end
            if gap >= gap_th:
                sub_clusters.append([cur])
            else:
                sub_clusters[-1].append(cur)
        if len(sub_clusters) < 2:
            continue
        # Compute centroid per sub-cluster
        sub_centroids = []
        for sc in sub_clusters:
            embs = [seg_embs[i] for i, _ in sc if i in seg_embs]
            if not embs:
                sub_centroids.append(None)
                continue
            c = np.mean(np.stack(embs), axis=0)
            c = c / max(np.linalg.norm(c), 1e-9)
            sub_centroids.append(c.astype(np.float32))
        # Check if sub-clusters are voice-different
        if all(c is None for c in sub_centroids):
            continue
        # Find largest sub-cluster (anchor)
        largest_idx = max(range(len(sub_clusters)), key=lambda x: sum(s[1].end - s[1].start for s in sub_clusters[x]))
        anchor_centroid = sub_centroids[largest_idx]
        if anchor_centroid is None:
            continue
        # Build OTHER SPK centroids for cross-comparison
        other_spk_centroids = {}
        for other_spk, other_group in spk_groups.items():
            if other_spk == spk:
                continue
            embs = [seg_embs[i] for i, _ in other_group if i in seg_embs]
            if not embs:
                continue
            c = np.mean(np.stack(embs), axis=0)
            c = c / max(np.linalg.norm(c), 1e-9)
            other_spk_centroids[other_spk] = c.astype(np.float32)
        # v175+: evaluate ALL sub-clusters (not just non-anchor).
        # 한 sub-cluster이 다른 SPK centroid에 더 잘 맞으면 reassign — anchor 자체가 contamination인 경우도 catch.
        # MIN_TARGET_SIM: 낮은 confidence 재할당 (e.g. Sean 0.351) reject — singleton의 unique speaker 보호.
        sub_iter = range(len(sub_clusters)) if eval_all_subs else [i for i in range(len(sub_clusters)) if i != largest_idx]
        for sci in sub_iter:
            sc = sub_clusters[sci]
            if sub_centroids[sci] is None:
                continue
            sim_to_anchor = float(np.dot(sub_centroids[sci], anchor_centroid)) if sci != largest_idx else 1.0
            # Compute own avg (excluding this sub-cluster) for self-fit reference
            own_other_embs = []
            for j, sc_j in enumerate(sub_clusters):
                if j == sci:
                    continue
                own_other_embs.extend([seg_embs[i] for i, _ in sc_j if i in seg_embs])
            if own_other_embs:
                own_other_centroid = np.mean(np.stack(own_other_embs), axis=0)
                own_other_centroid = own_other_centroid / max(np.linalg.norm(own_other_centroid), 1e-9)
                sim_to_own_rest = float(np.dot(sub_centroids[sci], own_other_centroid))
            else:
                sim_to_own_rest = 1.0  # only sub — can't compare
            # If voice diverges from rest of own SPK and a target SPK has higher sim → reassign
            if (1.0 - sim_to_own_rest) >= voice_diff_th and other_spk_centroids:
                best_sim = -1.0
                best_target = None
                for other_spk, oc in other_spk_centroids.items():
                    s = float(np.dot(sub_centroids[sci], oc))
                    if s > best_sim:
                        best_sim = s
                        best_target = other_spk
                if (best_target and best_sim >= min_target_sim
                        and best_sim > sim_to_own_rest + 0.05):
                    sub_dur = sum(s[1].end - s[1].start for s in sc)
                    print(f"[TimeGapSplit] SPK={spk} sub-cluster (t={sc[0][1].start:.1f}s, "
                          f"n={len(sc)}, dur={sub_dur:.1f}s, sim_own={sim_to_own_rest:.3f}) "
                          f"→ {best_target} (sim={best_sim:.3f})", flush=True)
                    for idx, seg in sc:
                        segments[idx].speaker = best_target
                        n_changed += 1
                else:
                    if best_target:
                        sub_dur = sum(s[1].end - s[1].start for s in sc)
                        print(f"[TimeGapSplit] SPK={spk} sub-cluster (t={sc[0][1].start:.1f}s, "
                              f"n={len(sc)}, dur={sub_dur:.1f}s, sim_own={sim_to_own_rest:.3f}) "
                              f"→ {best_target} REJECTED (sim={best_sim:.3f} < {min_target_sim} "
                              f"or not enough margin)", flush=True)
    if n_changed:
        print(f"[TimeGapSplit] {n_changed} segments reassigned by time-gap split")
    return segments


def _reassign_by_face_track_continuity(segments: List[Any]) -> List[Any]:
    """v173+: face TRACK (cluster 아님) 연속성으로 segment SPK 강제 통일.

    한 face track은 연속된 frame에서 한 사람의 얼굴 — 같은 SPK여야 함.
    cluster (여러 track 묶음)보다 더 강한 신호 — track 한 개 = 한 person 확정.
    """
    import os as _os_ft
    if _os_ft.environ.get("LATENTSYNC_FACE_TRACK_REASSIGN", "1") != "1":
        return segments
    try:
        import sys as _sys
        for _p in ("/workspace/scripts", "/workspace"):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        import face_id_embedder as _fie
        asd_result = _fie.get_last_asd_result()
        if not asd_result:
            return segments
    except Exception as _e:
        print(f"[FaceTrack] no asd: {_e}")
        return segments

    fps = float(asd_result.get("fps", 25.0))
    n_frames_total = int(asd_result.get("n_frames", 0))
    tracks = asd_result.get("tracks", [])
    if not tracks:
        return segments

    # frame → list of (track_idx, score)
    frame_to_tracks = [[] for _ in range(max(n_frames_total, 1))]
    for tidx, t in enumerate(tracks):
        t_frames = t.get("frames", [])
        t_scores = t.get("scores", [1.0] * len(t_frames))
        for fi, f in enumerate(t_frames):
            if 0 <= f < len(frame_to_tracks):
                score = float(t_scores[fi]) if fi < len(t_scores) else 0.0
                frame_to_tracks[f].append((tidx, score))

    # segment → dominant track  (v176+: ASD speaking-score gate)
    # 가시 face가 있더라도 LightASD score가 낮으면 (입 안 움직이면) 화자가 아님.
    # off-camera 화자 (e.g. Sean 같은 EMT) 케이스 → face track 신호 무시.
    speak_th = float(_os_ft.environ.get("LATENTSYNC_FACE_TRACK_SPEAK_TH", "0.5"))
    min_seg_dur = float(_os_ft.environ.get("LATENTSYNC_FACE_TRACK_MIN_DUR", "1.0"))
    from collections import defaultdict, Counter
    seg_dom_track = []
    seg_dom_score = []
    for seg in segments:
        dur = seg.end - seg.start
        if dur < min_seg_dur:
            # 짧은 segment는 face track 신호 신뢰도 낮음 → skip
            seg_dom_track.append(None)
            seg_dom_score.append(0.0)
            continue
        f1 = max(0, int(seg.start * fps))
        f2 = min(len(frame_to_tracks), int(seg.end * fps + 1))
        track_score_sum = defaultdict(float)
        track_score_cnt = defaultdict(int)
        for fi in range(f1, f2):
            for tidx, score in frame_to_tracks[fi]:
                # Only count POSITIVE speaking-score frames
                if score > 0:
                    track_score_sum[tidx] += score
                    track_score_cnt[tidx] += 1
        if not track_score_sum:
            seg_dom_track.append(None)
            seg_dom_score.append(0.0)
            continue
        # AVG score per track (not sum) — long but silent track shouldn't win
        track_avg = {t: track_score_sum[t] / max(track_score_cnt[t], 1) for t in track_score_sum}
        best_track, best_avg = max(track_avg.items(), key=lambda x: x[1])
        # Gate: only trust if best track's avg speaking score >= threshold
        if best_avg < speak_th:
            seg_dom_track.append(None)
            seg_dom_score.append(best_avg)
            continue
        seg_dom_track.append(best_track)
        seg_dom_score.append(best_avg)

    # track → SPK vote (longest duration wins)
    track_spk_votes = defaultdict(Counter)
    for i, seg in enumerate(segments):
        t = seg_dom_track[i]
        if t is not None:
            track_spk_votes[t][seg.speaker] += (seg.end - seg.start)

    min_vote_ratio = float(_os_ft.environ.get("LATENTSYNC_FACE_TRACK_VOTE_RATIO", "0.7"))
    track_to_spk = {}
    for t, votes in track_spk_votes.items():
        total = sum(votes.values())
        if total < 0.5:
            continue
        top, top_n = votes.most_common(1)[0]
        if top_n / total >= min_vote_ratio:
            track_to_spk[t] = top

    if not track_to_spk:
        return segments

    n_changed = 0
    for i, seg in enumerate(segments):
        t = seg_dom_track[i]
        if t is None:
            continue
        target = track_to_spk.get(t)
        if target and target != seg.speaker:
            asd_avg = seg_dom_score[i] if i < len(seg_dom_score) else 0.0
            print(f"[FaceTrack] [{seg.start:.2f}~{seg.end:.2f}] "
                  f"{seg.speaker} → {target} (track={t}, asd={asd_avg:.2f})", flush=True)
            seg.speaker = target
            n_changed += 1
    if n_changed:
        print(f"[FaceTrack] {n_changed} segments reassigned by face track continuity")
    return segments


def _split_long_by_f0_transitions(
    segments: List[Any],
    vocals_path: str,
) -> List[Any]:
    """v160+: 긴 segment를 F0 gender transitions으로 sub-split.

    Use case: 13s SPK_3 male monologue 안에 SPK_0 female yelling ("You're hurting him! No!") 끼어듦.

    Algorithm:
      1. 긴 segment (>= min_long_dur, default 5s)
      2. ~0.5s 단위로 F0 계산
      3. M ↔ F transition 검출
      4. transition 시점에서 segment split
      5. 새 sub-segments는 일단 원본 SPK 유지 (이후 ERes2/IntraSPK가 reassign)

    환경변수:
      LATENTSYNC_F0_SPLIT (default 1)
      LATENTSYNC_F0_SPLIT_MIN_DUR (default 5.0s)
      LATENTSYNC_F0_SPLIT_WIN (default 0.5s)
      LATENTSYNC_F0_SPLIT_MIN_SUB (default 0.5s) — sub-segment 최소 길이
    """
    import os as _os_s
    if _os_s.environ.get("LATENTSYNC_F0_SPLIT", "1") != "1":
        return segments
    if not vocals_path or not segments:
        return segments

    min_long_dur = float(_os_s.environ.get("LATENTSYNC_F0_SPLIT_MIN_DUR", "5.0"))
    win = float(_os_s.environ.get("LATENTSYNC_F0_SPLIT_WIN", "0.5"))
    min_sub = float(_os_s.environ.get("LATENTSYNC_F0_SPLIT_MIN_SUB", "0.5"))
    male_max = float(_os_s.environ.get("LATENTSYNC_F0_MALE_MAX", "165"))
    female_min = float(_os_s.environ.get("LATENTSYNC_F0_FEMALE_MIN", "175"))

    # Check if any segment is long enough
    needs_split = [i for i, s in enumerate(segments) if (s.end - s.start) >= min_long_dur]
    if not needs_split:
        return segments

    try:
        import librosa
        y, sr = librosa.load(vocals_path, sr=None, mono=True)
        f0_arr, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=2048,
        )
        pyin_times = librosa.times_like(f0_arr, sr=sr)
    except Exception as _e:
        print(f"[F0Split] pyin failed: {_e}")
        return segments

    def _window_gender(t0, t1):
        mask = (pyin_times >= t0) & (pyin_times < t1)
        chunk = f0_arr[mask]
        valid = chunk[~np.isnan(chunk)] if hasattr(chunk, '__len__') else []
        # v163: stricter — require ENOUGH voiced frames in window
        min_voiced = int(_os_s.environ.get("LATENTSYNC_F0_SPLIT_MIN_VOICED", "5"))
        if len(valid) < min_voiced:
            return "U"  # unknown / silence
        f0_med = float(np.median(valid))
        # v163: strict cutoff — median 기반 (not percentile)
        # voice modulation 시 25th은 낮을 수 있지만 median은 화자 본질에 가까움
        female_strict = float(_os_s.environ.get("LATENTSYNC_F0_SPLIT_FEMALE_STRICT", "200"))
        male_strict = float(_os_s.environ.get("LATENTSYNC_F0_SPLIT_MALE_STRICT", "150"))
        if f0_med < male_strict:
            return "M"
        if f0_med > female_strict:
            return "F"
        return "U"

    print(f"[F0Split] checking {len(needs_split)} long segments (>= {min_long_dur}s)", flush=True)
    out_segments = []
    n_splits = 0
    for i, seg in enumerate(segments):
        dur = seg.end - seg.start
        if dur < min_long_dur:
            out_segments.append(seg)
            continue
        # Compute per-window gender
        n_wins = int(dur / win)
        if n_wins < 3:
            out_segments.append(seg)
            continue
        win_genders = []
        for wi in range(n_wins):
            t0 = seg.start + wi * win
            t1 = min(seg.end, t0 + win)
            g = _window_gender(t0, t1)
            win_genders.append((t0, t1, g))
        # Find transitions (consecutive same gender for min_sub period)
        # Smoothing: require at least ceil(min_sub/win) consecutive windows of same gender
        min_consec = max(1, int(np.ceil(min_sub / win)))
        smoothed = [g for _, _, g in win_genders]
        # Replace short runs of 'U' or different gender with neighboring dominant
        n = len(smoothed)
        i_w = 0
        while i_w < n:
            j_w = i_w
            while j_w < n and smoothed[j_w] == smoothed[i_w]:
                j_w += 1
            run = j_w - i_w
            if run < min_consec and i_w > 0:
                for k in range(i_w, j_w):
                    smoothed[k] = smoothed[i_w - 1]
            i_w = j_w
        # Identify split points (gender transitions)
        split_idxs = [k for k in range(1, n) if smoothed[k] != smoothed[k - 1]]
        if not split_idxs:
            out_segments.append(seg)
            continue
        # Build sub-segments
        prev_idx = 0
        sub_segments_made = []
        for sp in split_idxs:
            sub_start = win_genders[prev_idx][0]
            sub_end = win_genders[sp][0]
            if sub_end - sub_start >= min_sub:
                # Create new segment
                new_seg = type(seg).__new__(type(seg))
                new_seg.__dict__ = dict(seg.__dict__)
                new_seg.start = sub_start
                new_seg.end = sub_end
                sub_segments_made.append((new_seg, smoothed[prev_idx]))
            prev_idx = sp
        # final tail
        sub_start = win_genders[prev_idx][0]
        sub_end = seg.end
        if sub_end - sub_start >= min_sub:
            new_seg = type(seg).__new__(type(seg))
            new_seg.__dict__ = dict(seg.__dict__)
            new_seg.start = sub_start
            new_seg.end = sub_end
            sub_segments_made.append((new_seg, smoothed[prev_idx]))
        if len(sub_segments_made) > 1:
            n_splits += 1
            print(f"[F0Split] [{seg.start:.2f}~{seg.end:.2f}] {seg.speaker} → "
                  f"{len(sub_segments_made)} sub-segs ({[g for _, g in sub_segments_made]})", flush=True)
            # v162+: F sub-segments에 unique label 부여 시도 (SPK_06 등 새 SPK 생성)
            # 부모 segment의 SPK는 male 기반인데 F sub-segment이면 다른 화자 가능성
            # → 다른 SPK label 부여 (기존 SPK가 male이면 새 high index)
            use_new_spk = _os_s.environ.get("LATENTSYNC_F0_SPLIT_NEW_SPK", "0") == "1"
            parent_spk = seg.speaker
            for ns, _g in sub_segments_made:
                if use_new_spk and _g == "F" and parent_spk and parent_spk.startswith("SPEAKER_"):
                    # parent SPK가 male-dominant이면 F sub-segment를 새 SPK label로
                    # parent ID + offset to create unique female label
                    try:
                        parent_num = int(parent_spk.replace("SPEAKER_", ""))
                        # Use SPEAKER_06 ~ SPEAKER_09 range for F sub-segments
                        new_num = 6 + (parent_num % 4)  # deterministic mapping
                        ns.speaker = f"SPEAKER_{new_num:02d}"
                    except (ValueError, AttributeError):
                        pass
                out_segments.append(ns)
        else:
            out_segments.append(seg)

    if n_splits:
        print(f"[F0Split] {n_splits} long segments split by F0 gender transitions", flush=True)
    return out_segments


def _split_contaminated_spk_eres2(
    segments: List[Any],
    vocals_path: str,
) -> List[Any]:
    """v153+: Intra-SPK voice consistency split — contaminated cluster 자동 분리.

    한 DZ_SPK label 안에 voice 다른 segments가 섞인 경우 검출 + 분리:
      1. 각 SPK label의 segments 수집 (>= 2)
      2. 모든 segment pair의 ERes2NetV2 cos sim 계산
      3. 만약 min(pairwise sim) < threshold (e.g. 0.35), bimodal voice → contaminated 의심
      4. 2-cluster split (greedy: outlier vs majority)
      5. Minority cluster의 segments를 다른 SPK 중 voice sim 최고로 reassign
         (자기 자신 cluster average보다 다른 SPK centroid가 더 가까우면 이동)

    환경변수:
      LATENTSYNC_INTRASPK_SPLIT (default 1) — 활성
      LATENTSYNC_INTRASPK_MIN_SIM (default 0.40) — minority가 다른 SPK 매칭에 필요한 최소 cos
      LATENTSYNC_INTRASPK_GAP (default 0.20) — pairwise min < majority avg - gap이면 split

    Test4 SPK_05 케이스:
      - "That's where" (Sean) vs "How hard can/just act/stopped petting" (SPK_3)
      - 4 segments 안에 2 voice cluster → 3 male을 SPK_3로 자동 이동
    """
    import os as _os_i
    if _os_i.environ.get("LATENTSYNC_INTRASPK_SPLIT", "1") != "1":
        return segments
    if not vocals_path or not segments:
        return segments

    min_sim = float(_os_i.environ.get("LATENTSYNC_INTRASPK_MIN_SIM", "0.40"))
    gap = float(_os_i.environ.get("LATENTSYNC_INTRASPK_GAP", "0.20"))

    try:
        import sys as _sys
        if "/workspace" not in _sys.path:
            _sys.path.insert(0, "/workspace")
        from patches.eres2netv2_helper import get_eres2netv2_model, extract_eres2netv2_emb
        get_eres2netv2_model()
    except Exception as _e:
        print(f"[IntraSPK] ERes2NetV2 load 실패: {_e}")
        return segments

    try:
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
    except Exception:
        return segments
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Group segments by speaker
    from collections import defaultdict
    spk_groups = defaultdict(list)
    for i, seg in enumerate(segments):
        spk_groups[seg.speaker].append(i)

    # Step 1: compute embeddings for all segments
    seg_embs = {}
    for i, seg in enumerate(segments):
        dur = seg.end - seg.start
        if dur < 0.4:
            continue
        chunk = audio[int(seg.start * sr):int(seg.end * sr)]
        emb = extract_eres2netv2_emb(chunk, sr=sr)
        if emb is not None:
            seg_embs[i] = emb

    # Step 2: build per-SPK centroid (from current labels) — for cross-SPK matching
    spk_centroids = {}
    for spk, idxs in spk_groups.items():
        embs = [seg_embs[i] for i in idxs if i in seg_embs]
        if not embs:
            continue
        c = np.mean(np.stack(embs), axis=0)
        c = c / max(np.linalg.norm(c), 1e-9)
        spk_centroids[spk] = c.astype(np.float32)

    if len(spk_centroids) < 2:
        return segments

    n_reassigned = 0
    print(f"[IntraSPK] checking {len(spk_groups)} SPKs for voice contamination (min_sim={min_sim}, gap={gap})")

    # Step 3: per-SPK consistency check
    for spk, idxs in spk_groups.items():
        valid_idxs = [i for i in idxs if i in seg_embs]
        if len(valid_idxs) < 2:
            continue

        # Pairwise cos matrix
        n = len(valid_idxs)
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for ii in range(n):
            for jj in range(n):
                sim_matrix[ii, jj] = float(np.dot(seg_embs[valid_idxs[ii]], seg_embs[valid_idxs[jj]]))

        # Average sim per segment to ALL others in SPK
        # For each segment, avg sim to other SPK members
        avg_sim_self = []
        for ii in range(n):
            others = [sim_matrix[ii, jj] for jj in range(n) if jj != ii]
            avg_sim_self.append(float(np.mean(others)) if others else 1.0)
        avg_sim_self = np.array(avg_sim_self)
        spk_med = float(np.median(avg_sim_self))

        # v155+: ALL segments check (not just outliers)
        # 각 segment의 own-cluster (excluding self) sim vs best other-SPK centroid sim
        # If best other-SPK > own + margin → reassign (majority contamination 해결)
        # v157+: skip very short segments (보호 — 단발 발화 화자 유지)
        # v164+: multi-model consensus voting (ERes2NetV2 + CAM++)
        gap_margin = float(_os_i.environ.get("LATENTSYNC_INTRASPK_REASSIGN_GAP", "0.05"))
        min_dur_reassign = float(_os_i.environ.get("LATENTSYNC_INTRASPK_MIN_DUR", "0.8"))
        use_consensus = _os_i.environ.get("LATENTSYNC_INTRASPK_CONSENSUS", "0") == "1"

        # Build CAM++ centroids if consensus enabled
        cam_centroids = {}
        seg_cam_embs = {}
        if use_consensus:
            try:
                from patches.campplus_helper import extract_campplus_emb, get_campplus_session
                get_campplus_session()
                # extract CAM++ embeddings
                for i, seg in enumerate(segments):
                    dur = seg.end - seg.start
                    if dur < 0.4:
                        continue
                    chunk = audio[int(seg.start * sr):int(seg.end * sr)]
                    cam = extract_campplus_emb(chunk, sr=sr)
                    if cam is not None:
                        seg_cam_embs[i] = cam
                # build CAM++ centroids per SPK
                from collections import defaultdict as _dd_cam
                cam_groups = _dd_cam(list)
                for i, seg in enumerate(segments):
                    if i in seg_cam_embs:
                        cam_groups[seg.speaker].append(seg_cam_embs[i])
                for s, embs in cam_groups.items():
                    if not embs:
                        continue
                    c = np.mean(np.stack(embs), axis=0)
                    c = c / max(np.linalg.norm(c), 1e-9)
                    cam_centroids[s] = c.astype(np.float32)
                print(f"[IntraSPK] CAM++ consensus enabled, {len(cam_centroids)} centroids", flush=True)
            except Exception as _ce:
                print(f"[IntraSPK] CAM++ unavailable: {_ce}")
                use_consensus = False

        for ll, seg_i in enumerate(valid_idxs):
            seg_dur_check = segments[seg_i].end - segments[seg_i].start
            if seg_dur_check < min_dur_reassign:
                continue  # protect short utterances (one-line characters like Sean)
            seg_emb = seg_embs[seg_i]
            own_avg = avg_sim_self[ll]
            sims_to_other = []
            for other_spk, c in spk_centroids.items():
                if other_spk == spk:
                    continue
                sims_to_other.append((float(np.dot(seg_emb, c)), other_spk))
            if not sims_to_other:
                continue
            sims_to_other.sort(reverse=True)
            best_sim, best_spk = sims_to_other[0]
            # Reassign if best_sim > own_avg + gap_margin AND best_sim >= min_sim
            if best_sim > own_avg + gap_margin and best_sim >= min_sim:
                # v164+: consensus check with CAM++
                consensus_ok = True
                cam_best_spk = None
                cam_best_sim = -1.0
                if use_consensus and seg_i in seg_cam_embs and cam_centroids:
                    cam_emb = seg_cam_embs[seg_i]
                    cam_sims = sorted(
                        ((float(np.dot(cam_emb, cv)), s) for s, cv in cam_centroids.items()),
                        reverse=True
                    )
                    if cam_sims:
                        cam_best_sim, cam_best_spk = cam_sims[0]
                        # require CAM++ also points to same SPK (or close to it)
                        if cam_best_spk != best_spk:
                            # 2nd-best CAM++ might agree
                            cam_top2_spks = [s for _, s in cam_sims[:2]]
                            if best_spk not in cam_top2_spks:
                                consensus_ok = False
                if not consensus_ok:
                    print(f"[IntraSPK-Consensus] [{segments[seg_i].start:.2f}~{segments[seg_i].end:.2f}] "
                          f"SKIP — ERes2 says {best_spk} but CAM++ says {cam_best_spk}", flush=True)
                    continue
                old_spk = segments[seg_i].speaker
                segments[seg_i].speaker = best_spk
                dur = segments[seg_i].end - segments[seg_i].start
                cam_note = f" [CAM++={cam_best_spk}({cam_best_sim:.3f})]" if use_consensus else ""
                print(f"[IntraSPK] [{segments[seg_i].start:.2f}~{segments[seg_i].end:.2f}] "
                      f"{old_spk}(own_avg={own_avg:.3f}) → {best_spk}(cross={best_sim:.3f}) "
                      f"dur={dur:.2f}s{cam_note}", flush=True)
                n_reassigned += 1

    if n_reassigned:
        print(f"[IntraSPK] {n_reassigned} segments reassigned by intra-SPK voice inconsistency", flush=True)
    return segments


def _reassign_short_with_eres2netv2(
    segments: List[Any],
    vocals_path: str,
    _pass_num: int = 1,
) -> List[Any]:
    """v133+: ERes2NetV2 short-clip embedding reassignment.

    실험 확인 (test4):
      - "Good" 0.82s: ECAPA→SPK_01(잘못), ERes2NetV2→Brian(cos=0.496)
      - "stopped petting" 2.19s: ECAPA→SPK_05(잘못), ERes2NetV2→other_female(cos=0.605)

    Algorithm:
      1. ERes2NetV2 load (lazy, 1회만)
      2. SPK centroid 계산 (long segments >= min_long_dur 기준)
      3. 각 short segment (<short_dur) 에 대해:
          - ERes2NetV2 embedding
          - 모든 SPK centroid와 cos similarity 계산
          - best_sim - 2nd > margin AND best_sim > min_threshold → reassign

    환경변수:
      LATENTSYNC_ERES2_SHORT (default 1) — 활성
      LATENTSYNC_ERES2_SHORT_DUR (default 2.5) — 짧은 segment 임계
      LATENTSYNC_ERES2_LONG_DUR (default 3.0) — centroid 계산용 long segment 임계
      LATENTSYNC_ERES2_MIN_SIM (default 0.35) — best match 최소 cos
      LATENTSYNC_ERES2_MARGIN (default 0.10) — best vs 2nd margin
    """
    import os as _os_e
    if _os_e.environ.get("LATENTSYNC_ERES2_SHORT", "1") != "1":
        return segments
    if not vocals_path or not segments:
        return segments

    short_dur = float(_os_e.environ.get("LATENTSYNC_ERES2_SHORT_DUR", "2.5"))
    long_dur = float(_os_e.environ.get("LATENTSYNC_ERES2_LONG_DUR", "3.0"))
    min_sim = float(_os_e.environ.get("LATENTSYNC_ERES2_MIN_SIM", "0.35"))
    margin = float(_os_e.environ.get("LATENTSYNC_ERES2_MARGIN", "0.10"))

    # Quick check: any short segments?
    short_indices = [i for i, s in enumerate(segments) if (s.end - s.start) < short_dur]
    if not short_indices:
        return segments

    try:
        import sys as _sys
        if "/workspace" not in _sys.path:
            _sys.path.insert(0, "/workspace")
        from patches.eres2netv2_helper import get_eres2netv2_model, extract_eres2netv2_emb
        get_eres2netv2_model()
    except Exception as _le:
        print(f"[ERes2Short] load 실패: {_le} — skip")
        return segments

    try:
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
    except Exception as _ae:
        print(f"[ERes2Short] vocals load 실패: {_ae}")
        return segments
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Build per-SPK centroid from long segments only (more reliable embedding)
    from collections import defaultdict, Counter
    spk_long_embs = defaultdict(list)
    spk_seg_count = Counter()
    for seg in segments:
        spk_seg_count[seg.speaker] += 1
        dur = seg.end - seg.start
        if dur < long_dur:
            continue
        chunk = audio[int(seg.start * sr):int(seg.end * sr)]
        emb = extract_eres2netv2_emb(chunk, sr=sr)
        if emb is not None:
            spk_long_embs[seg.speaker].append(emb)

    spk_centroids = {}
    for spk, embs in spk_long_embs.items():
        c = np.mean(np.stack(embs), axis=0)
        c = c / max(np.linalg.norm(c), 1e-9)
        spk_centroids[spk] = c.astype(np.float32)

    # v177+: identify singleton SPKs (1 segment, no long-ref centroid) — skip reassignment.
    # Sean's unique "That's where" 0.68s preserved without making SPK_05 a sticky centroid for others.
    singleton_spks = {spk for spk, n in spk_seg_count.items() if n == 1 and spk not in spk_centroids}

    if len(spk_centroids) < 2:
        print(f"[ERes2Short] only {len(spk_centroids)} SPKs with long ref — skip")
        return segments

    print(f"[ERes2Short] enabled (short<{short_dur}s, long>={long_dur}s, min_sim={min_sim}, margin={margin})")
    print(f"[ERes2Short] SPK centroids: {sorted(spk_centroids.keys())}")

    # v179+: pre-compute neighbor info for sandwich-override
    # singleton SPK preserved by default, BUT if seg is between two contiguous (gap<=0.3s)
    # same-other-SPK neighbors, it's a misclustered short utterance — allow reassign.
    sandwich_gap_th = float(_os_e.environ.get("LATENTSYNC_SANDWICH_GAP", "0.3"))

    n_changed = 0
    for i in short_indices:
        seg = segments[i]
        cur_spk = seg.speaker
        # v177+: preserve singleton SPKs (e.g. Sean's only line) — don't move.
        # v179+: EXCEPT if sandwich-override (both neighbors same other-SPK, gap <= 0.3s)
        sandwich_override = False
        if cur_spk in singleton_spks and 0 < i < len(segments) - 1:
            prev_seg = segments[i - 1]
            next_seg = segments[i + 1]
            prev_gap = seg.start - prev_seg.end
            next_gap = next_seg.start - seg.end
            if (prev_seg.speaker == next_seg.speaker
                    and prev_seg.speaker != cur_spk
                    and 0 <= prev_gap <= sandwich_gap_th
                    and 0 <= next_gap <= sandwich_gap_th):
                sandwich_override = True
                print(f"[ERes2Short] [{seg.start:.2f}~{seg.end:.2f}] singleton {cur_spk} "
                      f"sandwich-override (neighbors={prev_seg.speaker}, "
                      f"gaps={prev_gap:.2f}/{next_gap:.2f}s)", flush=True)
        if cur_spk in singleton_spks and not sandwich_override:
            continue
        dur = seg.end - seg.start
        chunk = audio[int(seg.start * sr):int(seg.end * sr)]
        emb = extract_eres2netv2_emb(chunk, sr=sr)
        if emb is None:
            continue
        # Score against all centroids
        sims = sorted(((float(np.dot(emb, c)), s) for s, c in spk_centroids.items()), reverse=True)
        best_sim, best_spk = sims[0]
        second_sim = sims[1][0] if len(sims) > 1 else 0.0
        cur_sim = next((cs for cs, s in sims if s == cur_spk), 0.0)
        # Apply reassignment criteria
        if (best_spk != cur_spk
                and best_sim >= min_sim
                and (best_sim - second_sim) >= margin
                and (best_sim - cur_sim) >= margin):
            old_spk = cur_spk
            seg.speaker = best_spk
            n_changed += 1
            print(f"[ERes2Short] [{seg.start:.2f}~{seg.end:.2f}] {old_spk}({cur_sim:.3f}) "
                  f"→ {best_spk}({best_sim:.3f}, 2nd={second_sim:.3f}) (dur={dur:.2f}s)", flush=True)

    if n_changed:
        print(f"[ERes2Short] {n_changed}/{len(short_indices)} short-segments reassigned", flush=True)
    return segments


def _reassign_short_with_voice_continuity(
    segments: List[Any],
    vocals_path: str,
    ecapa_model,
) -> List[Any]:
    """v123+: Short segment context-aware reassignment using local dominance.

    각 짧은 segment (<min_dur)에 대해 *past N초 window* 내 각 SPK의 총 발화 시간 계산.
    cur_spk보다 훨씬 더 active한 다른 SPK가 있고 voice sim도 within margin이면 reassign.

    환경변수:
      LATENTSYNC_SHORT_SEG_CONTINUITY (default 1)
      LATENTSYNC_SHORT_SEG_DUR (default 2.5) — 짧은 segment 임계
      LATENTSYNC_SHORT_SEG_WINDOW (default 30.0) — past window (초)
      LATENTSYNC_SHORT_SEG_DOMINANCE (default 3.0) — best SPK가 cur SPK보다 이 배수 이상 active면 trigger
      LATENTSYNC_SHORT_SEG_MARGIN (default 0.20) — voice sim margin (best가 cur - margin 이상이면 switch)

    "Good" 예시 (59.87s, SPK_01):
      - past 30s window (29.87~59.87):
        - SPK_01: 51.37~56.19 = 4.82s, others sums...
        - SPK_04: 56.19~58.67 = 2.48s
        - SPK_05: 58.67~59.87 = 1.20s
      - SPK_01 dominance: 4.82s vs SPK_04 2.48s vs SPK_05 1.20s
      - SPK_01 가장 많이 active → trigger 안 됨 (heuristic 보수적)

    "stopped petting" 예시 (95.32s, SPK_05):
      - past 30s window (65.32~95.32):
        - SPK_05: 68.70~71.98(2.38) + 94.12~95.32(1.20) = 3.58s
        - SPK_03: 73.63~75.09(1.46)+76.41~78.09(1.68)+79.63~81.33(1.70)+81.89~94.10(12.21) = 17.05s
        - SPK_00: 72.27~73.15(0.88)+78.09~79.63(1.54) = 2.42s
      - SPK_03 dominance: 17.05s vs cur SPK_05 3.58s → ratio 4.76× → trigger
      - voice sim 비교 후 within margin이면 switch → SPK_03 ✓
    """
    import os as _os
    if _os.environ.get("LATENTSYNC_SHORT_SEG_CONTINUITY", "1") != "1":
        print("[ShortSeg] disabled (LATENTSYNC_SHORT_SEG_CONTINUITY=0)", flush=True)
        return segments
    if ecapa_model is None or not vocals_path or not segments:
        return segments

    min_dur = float(_os.environ.get("LATENTSYNC_SHORT_SEG_DUR", "2.5"))
    window = float(_os.environ.get("LATENTSYNC_SHORT_SEG_WINDOW", "30.0"))
    dominance = float(_os.environ.get("LATENTSYNC_SHORT_SEG_DOMINANCE", "3.0"))
    margin = float(_os.environ.get("LATENTSYNC_SHORT_SEG_MARGIN", "0.20"))

    try:
        import soundfile as sf
        audio, sr = sf.read(vocals_path)
    except Exception as e:
        print(f"[ShortSeg] vocals 로드 실패: {e}")
        return segments
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Compute final SPK centroids from all segments
    from collections import defaultdict
    spk_embs = defaultdict(list)
    for seg in segments:
        dur = seg.end - seg.start
        if dur < 0.4:
            continue
        chunk = audio[int(seg.start * sr):min(int(seg.end * sr), len(audio))]
        emb = _compute_ecapa_emb(chunk, sr, ecapa_model)
        if emb is not None:
            spk_embs[seg.speaker].append(emb)

    speaker_centroids = {}
    for spk, embs_list in spk_embs.items():
        c = np.mean(np.stack(embs_list), axis=0)
        c = c / max(np.linalg.norm(c), 1e-9)
        speaker_centroids[spk] = c

    if len(speaker_centroids) < 2:
        return segments

    print(f"[ShortSeg] enabled (min_dur={min_dur}, window={window}s, dominance={dominance}×, margin={margin})", flush=True)

    n_changed = 0
    n_candidates = 0
    for i, seg in enumerate(segments):
        dur = seg.end - seg.start
        if dur >= min_dur:
            continue
        cur_spk = seg.speaker
        if cur_spk not in speaker_centroids:
            continue
        n_candidates += 1

        # Compute past-window speaking duration per speaker (excluding current seg)
        window_start = seg.start - window
        spk_dur_window = defaultdict(float)
        for j in range(len(segments)):
            if j == i:
                continue
            other = segments[j]
            # Only past segments
            if other.end > seg.start:
                continue
            # Within window
            if other.end < window_start:
                continue
            spk_dur_window[other.speaker] += (other.end - other.start)

        cur_dur = spk_dur_window.get(cur_spk, 0.0)

        # v123+: safeguard — 첫 등장 화자 보호 (cur SPK도 미래 segments에서 발화함 가능성 있음)
        # cur_dur가 너무 작으면 (예: 처음 등장) reassign 안 함
        min_cur_dur = float(_os.environ.get("LATENTSYNC_SHORT_SEG_MIN_CUR_DUR", "1.0"))
        if cur_dur < min_cur_dur:
            print(f"[ShortSeg] skip [{seg.start:.2f}~{seg.end:.2f}] {cur_spk} — cur_dur "
                  f"{cur_dur:.2f}s < min {min_cur_dur}s (첫 등장 보호)", flush=True)
            continue

        # Find dominant SPK in window
        if not spk_dur_window:
            continue
        best_spk_window, best_spk_dur = max(spk_dur_window.items(), key=lambda x: x[1])

        # Check dominance ratio
        if best_spk_window == cur_spk:
            continue  # current is already the dominant
        if best_spk_dur < dominance * cur_dur:
            # not dominant enough
            print(f"[ShortSeg] skip [{seg.start:.2f}~{seg.end:.2f}] {cur_spk}({cur_dur:.1f}s) "
                  f"vs {best_spk_window}({best_spk_dur:.1f}s) — ratio {best_spk_dur/max(cur_dur,1e-6):.2f}×<{dominance}×", flush=True)
            continue

        # Compute seg voice embedding
        chunk = audio[int(seg.start * sr):min(int(seg.end * sr), len(audio))]
        emb = _compute_ecapa_emb(chunk, sr, ecapa_model)
        if emb is None:
            continue
        emb = emb / max(np.linalg.norm(emb), 1e-9)

        cur_sim = float(np.dot(emb, speaker_centroids[cur_spk]))
        best_sim = float(np.dot(emb, speaker_centroids[best_spk_window]))

        # Require best_sim within margin of cur_sim (or better)
        if best_sim < cur_sim - margin:
            print(f"[ShortSeg] skip [{seg.start:.2f}~{seg.end:.2f}] {cur_spk}({cur_sim:.3f}) "
                  f"vs {best_spk_window}({best_sim:.3f}) — voice gap > margin {margin}", flush=True)
            continue

        old_spk = cur_spk
        seg.speaker = best_spk_window
        n_changed += 1
        print(f"[ShortSeg] [{seg.start:.2f}~{seg.end:.2f}] "
              f"{old_spk}(window={cur_dur:.1f}s,sim={cur_sim:.3f}) → "
              f"{best_spk_window}(window={best_spk_dur:.1f}s,sim={best_sim:.3f}) "
              f"(dur={dur:.2f}s, ratio={best_spk_dur/max(cur_dur,1e-6):.1f}×)", flush=True)

    if n_candidates:
        print(f"[ShortSeg] {n_changed}/{n_candidates} short-segments reassigned (window dominance)", flush=True)
    return segments


def _absorb_short_outliers(
    segments: List[Any],
    min_dur: float = 0.4,
    outlier_threshold: int = 90,
) -> List[Any]:
    """outlier 화자(SPEAKER_9X)의 짧은 segment를 인접 main speaker로 흡수.

    근거:
      - 0.4초 미만은 ECAPA 임베딩 신뢰도 낮음 (실제로 _compute_ecapa_emb가 None 반환)
      - outlier 라벨이 ASD-blind한 false positive일 가능성 큼
      - 인접 main speaker로 흡수하면 자연스러움
    """
    if len(segments) < 2:
        return segments

    def is_outlier(spk):
        if not spk or not isinstance(spk, str) or not spk.startswith("SPEAKER_"):
            return False
        try:
            return int(spk.replace("SPEAKER_", "")) > outlier_threshold
        except ValueError:
            return False

    absorbed = 0
    for i, seg in enumerate(segments):
        dur = seg.end - seg.start
        if dur >= min_dur or not is_outlier(seg.speaker):
            continue
        # 시간상 가까운 main speaker 찾기 (좌/우)
        prev_main = None
        next_main = None
        for j in range(i - 1, -1, -1):
            if not is_outlier(segments[j].speaker):
                prev_main = segments[j]
                break
        for j in range(i + 1, len(segments)):
            if not is_outlier(segments[j].speaker):
                next_main = segments[j]
                break
        candidates = []
        if prev_main is not None:
            candidates.append(("prev", seg.start - prev_main.end, prev_main.speaker))
        if next_main is not None:
            candidates.append(("next", next_main.start - seg.end, next_main.speaker))
        if not candidates:
            continue
        # 시간 거리 가장 가까운 main으로
        best = min(candidates, key=lambda x: abs(x[1]))
        old_spk = seg.speaker
        seg.speaker = best[2]
        print(f"[Refine] short outlier 흡수 [{seg.start:.2f}~{seg.end:.2f}] "
              f"{old_spk} → {best[2]} (dur={dur:.2f}s, side={best[0]})")
        absorbed += 1
    if absorbed:
        print(f"[Refine] {absorbed}개 짧은 outlier 흡수 완료")
    return segments
