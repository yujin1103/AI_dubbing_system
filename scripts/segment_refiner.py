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


def refine_segments(
    segments: List[Any],
    asd_result: Optional[Dict] = None,
    words_by_seg: Optional[List[List[Dict]]] = None,
    speaker_centroids: Optional[Dict[str, np.ndarray]] = None,
    vocals_path: Optional[str] = None,
    ecapa_model=None,
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
    if speaker_centroids and ecapa_model is not None and vocals_path:
        final = sliding_split_all(final, vocals_path, speaker_centroids, ecapa_model)

    # 너무 짧은 outlier segment 흡수 (ECAPA 신뢰도 낮음)
    final = _absorb_short_outliers(final)

    return final


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
