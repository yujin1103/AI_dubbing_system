"""Audio-Visual Diarization Fusion v2.

핵심 개선:
  - per-frame ASD 기반 lipsync target 결정 (단순 + 정확)
  - speaker_face_map은 부가 정보 (스피커 추적, off-screen 감지용)

알고리즘 (per_frame_target):
  for each frame f:
    1. audio diarization으로 active speaker 확인 (없으면 lipsync skip)
    2. frame f에 존재하는 face tracks 모음
    3. 그 중 ASD score가 가장 높고 양수인 face = lipsync target
    4. 없으면 None (off-screen)

speaker_face_map (부가):
  audio speaker → 가장 자주 매칭된 face track 리스트
  → spurious speaker 감지에 사용
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# 임계값
THRESH_SPEAKING = 0.0       # ASD score 이 이상 = speaking
THRESH_LIPSYNC = -0.5       # 이 미만이면 lipsync 적용 안 함 (확신 없음)
MIN_SPEAKER_DUR = 0.6       # 이 이하 audio speaker = spurious 후보
MIN_OVERLAP_FRAMES = 3      # face/audio overlap 최소 frame 수


def fuse_av_diarization(
    audio_segments: List[Tuple[float, float, str]],
    asd_result: Dict,
    verbose: bool = True,
) -> Dict:
    """Audio diarization과 Visual ASD를 융합."""
    if not asd_result or not asd_result.get("tracks"):
        return {
            "speaker_face_map": {},
            "spurious_speakers": [],
            "per_frame_target": [None] * (asd_result.get("n_frames", 0) if asd_result else 0),
            "report": "ASD result empty",
        }

    fps = asd_result["fps"]
    n_frames = asd_result["n_frames"]
    tracks = asd_result["tracks"]
    n_tracks = len(tracks)

    # === 1. frame-level lookup 구조 ===
    # face_at_frame[f] = [(track_idx, score), ...]
    face_at_frame: List[List[Tuple[int, float]]] = [[] for _ in range(n_frames)]
    for tidx, t in enumerate(tracks):
        for i, f in enumerate(t["frames"]):
            if 0 <= f < n_frames:
                face_at_frame[f].append((tidx, t["scores"][i]))

    # frame → active audio speaker (가장 마지막에 언급된 발화)
    frame_active_speaker: List[Optional[str]] = [None] * n_frames
    for s, e, spk in audio_segments:
        f1 = max(0, int(s * fps))
        f2 = min(n_frames, int(e * fps + 1))
        for f in range(f1, f2):
            frame_active_speaker[f] = spk

    # === 2. per-frame lipsync target ===
    per_frame_target: List[Optional[int]] = [None] * n_frames
    for f in range(n_frames):
        if frame_active_speaker[f] is None:
            continue  # silence → no lipsync
        if not face_at_frame[f]:
            continue  # no face on screen → no lipsync
        # ASD score 가장 높은 face
        candidates = face_at_frame[f]
        best_track, best_score = max(candidates, key=lambda x: x[1])
        if best_score > THRESH_LIPSYNC:
            per_frame_target[f] = best_track

    # === 3. speaker_face_map 산출 (부가 정보) ===
    # 각 audio speaker가 가장 많이 매칭된 face tracks
    speaker_face_count: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for f in range(n_frames):
        spk = frame_active_speaker[f]
        tgt = per_frame_target[f]
        if spk is not None and tgt is not None:
            speaker_face_count[spk][tgt] += 1

    speaker_face_map: Dict[str, List[int]] = {}
    for spk, counts in speaker_face_count.items():
        # count 내림차순으로 정렬, 5+ frames 매칭된 것만
        sorted_tracks = sorted(
            [(tidx, c) for tidx, c in counts.items() if c >= 5],
            key=lambda x: -x[1]
        )
        speaker_face_map[spk] = [tidx for tidx, _ in sorted_tracks]

    # === 4. speaker별 통계 + spurious 후보 ===
    speaker_total_dur: Dict[str, float] = defaultdict(float)
    for s, e, spk in audio_segments:
        speaker_total_dur[spk] += e - s

    spurious_speakers = []
    for spk, dur in speaker_total_dur.items():
        face_match = speaker_face_map.get(spk, [])
        # spurious = (a) 발화 시간 너무 짧음, AND (b) face matching 없음
        if dur < MIN_SPEAKER_DUR and not face_match:
            spurious_speakers.append(spk)

    # === 5. report ===
    report_lines = ["=== AV Fusion Report v2 ==="]
    report_lines.append(f"Audio speakers: {len(speaker_total_dur)} / Face tracks: {n_tracks}")
    report_lines.append(f"Frames: {n_frames} @ {fps}fps")

    # Track 별 speaking 통계
    report_lines.append(f"\n[Face Tracks summary]")
    for tidx, t in enumerate(tracks):
        if not t["frames"]:
            continue
        sc = np.array(t["scores"])
        f0, f1 = t["frames"][0], t["frames"][-1]
        speaking_pct = (sc > THRESH_SPEAKING).mean() * 100
        max_s = sc.max()
        report_lines.append(f"  Track {tidx}: frames {f0}~{f1} ({len(t['frames'])}), "
                           f"speaking {speaking_pct:.1f}%, max={max_s:.2f}")

    # Speaker → Face mapping
    report_lines.append(f"\n[Speaker → Face Mapping]")
    for spk in sorted(speaker_total_dur):
        dur = speaker_total_dur[spk]
        faces = speaker_face_map.get(spk, [])
        if faces:
            face_str = ", ".join(f"track{i}({speaker_face_count[spk][i]}f)" for i in faces)
            report_lines.append(f"  {spk} ({dur:.1f}s) → {face_str}")
        else:
            tag = "SPURIOUS" if spk in spurious_speakers else "OFF-SCREEN/NO-MATCH"
            report_lines.append(f"  {spk} ({dur:.1f}s) → {tag}")

    # 통계
    n_lipsync = sum(1 for t in per_frame_target if t is not None)
    n_audio_active = sum(1 for s in frame_active_speaker if s is not None)
    n_face = sum(1 for f_list in face_at_frame if f_list)
    report_lines.append(f"\n[Per-frame Stats]")
    report_lines.append(f"  Audio active frames: {n_audio_active}/{n_frames} "
                       f"({n_audio_active/max(1,n_frames)*100:.1f}%)")
    report_lines.append(f"  Face visible frames: {n_face}/{n_frames} "
                       f"({n_face/max(1,n_frames)*100:.1f}%)")
    report_lines.append(f"  Lipsync target frames: {n_lipsync}/{n_frames} "
                       f"({n_lipsync/max(1,n_frames)*100:.1f}%)")
    if spurious_speakers:
        report_lines.append(f"\nSpurious speakers (제거 권장): {spurious_speakers}")

    report = "\n".join(report_lines)
    if verbose:
        print(report)

    return {
        "speaker_face_map": speaker_face_map,
        "speaker_face_count": dict(speaker_face_count),  # v14: 외부 노출 (자동 화자 병합용)
        "spurious_speakers": spurious_speakers,
        "per_frame_target": per_frame_target,
        "frame_active_speaker": frame_active_speaker,
        "fps": fps,
        "n_frames": n_frames,
        "report": report,
    }


def detect_face_based_merges(
    speaker_face_count: Dict[str, Dict[int, int]],
    min_shared_frames: int = 10,
    min_share_ratio: float = 0.30,
) -> List[Tuple[str, str, int]]:
    """같은 face track에 매핑된 speaker pair 검출 → 자동 병합 후보.

    근거: 한 face = 한 사람. 두 audio speaker가 같은 face와 N frame 이상 매칭되면
    DiariZen이 같은 사람을 over-detect한 것일 가능성 매우 높음.

    PARAMETERS:
      min_shared_frames: 공유 face track의 최소 frame 수 (노이즈 방지)
      min_share_ratio: 작은 쪽 speaker의 face matching frame 중 공유 비율
                       예: A가 face1에 100frame, face2에 5frame 매칭이고
                            B도 face1에 50frame 매칭이면
                            A의 face1은 105 / 110 ≈ 95% (강한 신호)
                            B의 face1은 50 / 50 = 100% (확정)

    OUTPUT:
      [(speaker_a, speaker_b, shared_frames), ...] 병합 후보 list
    """
    pairs = []
    speakers = list(speaker_face_count.keys())
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            s1, s2 = speakers[i], speakers[j]
            faces1 = speaker_face_count[s1]
            faces2 = speaker_face_count[s2]
            # 공유 face track + 합산 frame
            shared_tracks = set(faces1.keys()) & set(faces2.keys())
            if not shared_tracks:
                continue
            shared_frames = sum(min(faces1[t], faces2[t]) for t in shared_tracks)
            if shared_frames < min_shared_frames:
                continue
            # 작은 쪽 기준 share_ratio 검증 (false merge 방지)
            total1 = sum(faces1.values())
            total2 = sum(faces2.values())
            if total1 == 0 or total2 == 0:
                continue
            share1 = shared_frames / total1
            share2 = shared_frames / total2
            min_share = min(share1, share2)
            if min_share < min_share_ratio:
                continue
            pairs.append((s1, s2, shared_frames))
    # shared_frames 큰 순서로
    pairs.sort(key=lambda x: -x[2])
    return pairs


def get_lipsync_bbox(asd_result: Dict, fusion: Dict, frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
    """frame_idx에 lipsync 적용할 face의 bbox 가져오기.

    Returns:
        (x1, y1, x2, y2) 또는 None
    """
    track_idx = fusion["per_frame_target"][frame_idx]
    if track_idx is None:
        return None
    track = asd_result["tracks"][track_idx]
    if frame_idx in track["frames"]:
        i = track["frames"].index(frame_idx)
        x1, y1, x2, y2 = track["bboxes"][i]
        return (int(x1), int(y1), int(x2), int(y2))
    return None


# ─── v17 NEW: per-segment AV reassignment ────────────────────────────────
def reassign_segments_by_face_owner(
    audio_segments: List[Tuple[float, float, str]],
    fusion: Dict,
    asd_result: Dict,
    min_dominant_ratio: float = 0.60,
    min_seg_face_share: float = 0.50,
    max_seg_duration: Optional[float] = None,
    verbose: bool = True,
) -> List[Tuple[float, float, str, str]]:
    """face track 기반 audio segment 재할당.

    근거: 한 face = 한 사람. face_track i 의 owner는 그 face와 가장 많이 매칭된
    speaker. 어떤 audio segment의 dominant face track이 다른 speaker로 라벨링되어
    있으면 → face owner로 재할당.

    이는 (a) spurious 제거나 (b) 화자 병합과 *별개*인 *재할당* 작업.
    SPEAKER_3 segments가 SPEAKER_5/0/1로 잘못 라벨링된 케이스 등을 잡는다.

    Args:
        audio_segments: [(start, end, speaker), ...] DiariZen 결과
        fusion: fuse_av_diarization 반환 dict (speaker_face_count 포함)
        asd_result: run_asd 반환 dict
        min_dominant_ratio: face owner 결정 시 dominant speaker의 face 점유율 최소값
                            (예: face1이 SPEAKER_5와 80%, SPEAKER_3와 20% 매칭이면
                             owner=SPEAKER_5, share=80% — 0.60 이상이면 채택)
        min_seg_face_share: segment 구간 frames 중 dominant face의 최소 점유율
                            (face track이 잠깐만 보이면 신뢰 안 함)
        max_seg_duration: None이면 무제한. 값이 있으면 그보다 긴 segment는 재할당 skip
                          (긴 발화의 화자 라벨은 audio diarization을 신뢰한다는 의미.
                          v18 회귀 fix: 짧은 segment만 보정해서 false positive 줄임)

    Returns:
        [(start, end, old_speaker, new_speaker), ...] 재할당된 segments.
        diarization Annotation 자체는 호출자가 수정.
    """
    speaker_face_count = fusion.get("speaker_face_count", {})
    if not speaker_face_count:
        return []

    n_frames = asd_result.get("n_frames", 0)
    fps = asd_result.get("fps", 25.0)
    tracks = asd_result.get("tracks", [])
    if n_frames <= 0 or not tracks:
        return []

    # 1. face_track i 의 owner speaker 결정
    # face_owner[face_idx] = (speaker, share_of_this_face_frames)
    all_face_indices = set()
    for s, faces in speaker_face_count.items():
        all_face_indices.update(faces.keys())
    face_owner: Dict[int, Tuple[str, float]] = {}
    for face_idx in all_face_indices:
        spk_counts = {s: speaker_face_count[s].get(face_idx, 0)
                      for s in speaker_face_count}
        total = sum(spk_counts.values())
        if total == 0:
            continue
        best_spk, best_count = max(spk_counts.items(), key=lambda x: x[1])
        share = best_count / total
        if share >= min_dominant_ratio:
            face_owner[face_idx] = (best_spk, share)

    if not face_owner:
        if verbose:
            print(f"[AV-Reassign] face owner 결정 못함 (dominant<{min_dominant_ratio:.0%})")
        return []

    if verbose:
        owner_str = ", ".join(f"f{i}→{spk}({sh:.0%})"
                              for i, (spk, sh) in sorted(face_owner.items()))
        print(f"[AV-Reassign] face owners: {owner_str}")

    # 2. frame → face track 리스트 lookup
    face_at_frame: List[List[int]] = [[] for _ in range(n_frames)]
    for tidx, t in enumerate(tracks):
        for f in t["frames"]:
            if 0 <= f < n_frames:
                face_at_frame[f].append(tidx)

    # 3. per-segment dominant face → 재할당
    reassignments: List[Tuple[float, float, str, str]] = []
    for start, end, spk in audio_segments:
        # v19: max_seg_duration보다 긴 segment는 신뢰 (재할당 skip)
        if max_seg_duration is not None and (end - start) > max_seg_duration:
            continue
        f1 = max(0, int(start * fps))
        f2 = min(n_frames, int(end * fps + 1))
        seg_frames = max(1, f2 - f1)
        if f2 <= f1:
            continue

        # 이 segment 구간에서 face track 별 frame 수
        face_freq: Dict[int, int] = {}
        for f in range(f1, f2):
            for tidx in face_at_frame[f]:
                face_freq[tidx] = face_freq.get(tidx, 0) + 1
        if not face_freq:
            continue
        dom_face, dom_count = max(face_freq.items(), key=lambda x: x[1])
        dom_share = dom_count / seg_frames
        if dom_share < min_seg_face_share:
            continue  # segment에서 face가 dominant 안 함 (얼굴 자주 가려짐 등)

        owner = face_owner.get(dom_face)
        if owner is None:
            continue  # 이 face track은 owner 결정 안 됨
        new_spk, owner_share = owner
        if new_spk == spk:
            continue  # 이미 맞게 라벨링됨

        reassignments.append((start, end, spk, new_spk))
        if verbose:
            print(f"[AV-Reassign] {start:.2f}~{end:.2f}s "
                  f"{spk} → {new_spk} "
                  f"(face{dom_face} segshare={dom_share:.0%}, owner={owner_share:.0%})")

    return reassignments


def apply_reassignment_to_diarization(
    diarization,
    reassignments: List[Tuple[float, float, str, str]],
    return_intervals: bool = True,
):
    """diarization Annotation에 재할당 적용 (v19 재구조).

    Args:
        diarization: pyannote.core.Annotation
        reassignments: reassign_segments_by_face_owner 결과
                       [(start, end, old_speaker, new_speaker), ...]
        return_intervals: True면 변경된 (start, end, old_speaker) 리스트 같이 반환

    Returns:
        return_intervals=True: (new_diarization, [(start, end, old_spk), ...])
        return_intervals=False: new_diarization
    """
    from pyannote.core import Annotation

    if not reassignments:
        empty: List[Tuple[float, float, str]] = []
        return (diarization, empty) if return_intervals else diarization

    # (start, end) → (new_spk, old_spk) map
    reassign_map: Dict[Tuple[float, float], Tuple[str, str]] = {
        (round(s, 3), round(e, 3)): (new, old)
        for s, e, old, new in reassignments
    }
    new_diar = Annotation()
    reassigned_intervals: List[Tuple[float, float, str]] = []
    for turn, track, spk in diarization.itertracks(yield_label=True):
        k = (round(turn.start, 3), round(turn.end, 3))
        new_info = reassign_map.get(k)
        if new_info is not None:
            new_spk, old_spk = new_info
            new_diar[turn, track] = new_spk
            reassigned_intervals.append((turn.start, turn.end, old_spk))
        else:
            new_diar[turn, track] = spk

    return (new_diar, reassigned_intervals) if return_intervals else new_diar


def overlaps_excluded_intervals(
    start: float,
    end: float,
    excluded: List[Tuple[float, float, str]],
    min_overlap_ratio: float = 0.5,
) -> bool:
    """segment(start, end)가 excluded intervals 중 하나와 min_overlap_ratio 이상
    겹치면 True. v19 ref bank 오염 방지용.

    excluded: apply_reassignment_to_diarization이 반환하는 (start, end, old_spk).
              old_spk는 여기서 무시 (구조 일관성용).
    """
    seg_dur = max(1e-3, end - start)
    for ex_start, ex_end, _old in excluded:
        overlap = max(0.0, min(end, ex_end) - max(start, ex_start))
        if overlap / seg_dur >= min_overlap_ratio:
            return True
    return False


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/scripts")
    from asd_runner import run_asd

    if len(sys.argv) < 2:
        print("Usage: av_fusion.py <video.mp4>")
        sys.exit(1)

    asd = run_asd(sys.argv[1])
    if not asd:
        sys.exit("ASD 실패")

    # 더미 audio segments (test3 가정)
    dummy_audio = [
        (0.5, 4.5, "SPEAKER_00"),
        (15.0, 18.5, "SPEAKER_01"),
        (20.0, 22.0, "SPEAKER_00"),
        (25.0, 35.0, "SPEAKER_01"),
        (40.0, 60.0, "SPEAKER_01"),
        (65.0, 70.0, "SPEAKER_01"),
        (71.0, 71.4, "SPEAKER_02"),  # spurious 후보
    ]

    fusion = fuse_av_diarization(dummy_audio, asd)
