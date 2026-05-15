"""Face Recognition cross-check for speaker diarization.

핵심 아이디어:
  - audio diarization은 *화자 정체성* 결정에 부정확 (같은 사람 다른 톤, 짧은 발화)
  - 동영상에는 얼굴이 있고 얼굴은 *결정적* (얼굴 같으면 같은 사람)
  - LightASD가 face track을 이미 만들었음 (위치/움직임 정보)
  - 우리는 거기에 *얼굴 정체성* (face ID embedding) 추가

작동 흐름:
  1. asd_result의 각 face track에서 N개 frame sampling
  2. InsightFace로 face embedding 추출 (512-dim, L2-normalized)
  3. cosine similarity로 track 간 동일인 판정 (cluster)
  4. cluster ID 기반 audio diarization 재구조:
     - cluster 같은데 audio speaker 다름 → 통합 (audio over-split fix)
     - cluster 다른데 audio speaker 같음 → 분리 (audio over-merge fix)

사용자 SPEAKER_05 케이스:
  face track f36/f41/f56/f57 모두 audio에선 SPEAKER_05로 잡힘.
  → 이 4개 face가 *진짜 같은 얼굴*인지 확인 (face_id cosine)
  → 같으면: SPEAKER_05 1명 OK (audio over-split이 아니라 진짜 같은 사람)
  → 다르면: 4명을 분리 (audio over-merge fix)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# Global InsightFace app (lazy init)
_face_app = None


def _get_face_app(ctx_id: int = 0):
    """InsightFace FaceAnalysis lazy load.

    v24: det_size + det_thresh 환경변수화 (작은 얼굴/측면 얼굴 검출률 향상):
      LATENTSYNC_FACE_DET_SIZE (default 320, was 640 — 작은 얼굴 더 잘 잡음)
      LATENTSYNC_FACE_DET_THRESH (default 0.30, was 0.50 — 낮은 confidence 수용)
    """
    global _face_app
    if _face_app is not None:
        return _face_app
    from insightface.app import FaceAnalysis
    det_size = int(os.environ.get("LATENTSYNC_FACE_DET_SIZE", "320"))
    det_thresh = float(os.environ.get("LATENTSYNC_FACE_DET_THRESH", "0.30"))
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _face_app = FaceAnalysis(
        name='buffalo_l',
        providers=providers,
    )
    _face_app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
    return _face_app


def extract_face_id_embedding(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """단일 이미지(BGR)에서 가장 큰 얼굴의 ArcFace 임베딩 추출.

    Returns:
        np.ndarray (512-dim, L2-normalized) 또는 None
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    try:
        app = _get_face_app()
        faces = app.get(image_bgr)
        if not faces:
            return None
        # bbox 면적 기준 가장 큰 얼굴
        def area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0, (x2 - x1) * (y2 - y1))
        biggest = max(faces, key=area)
        emb = biggest.normed_embedding  # 이미 L2-normalized
        return emb.astype(np.float32)
    except Exception as e:
        print(f"[FaceID] embedding 추출 실패: {e}")
        return None


def compute_track_face_embeddings(
    asd_result: Dict,
    video_path: str,
    n_samples_per_track: int = 7,
    bbox_padding_ratio: float = 0.40,
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    """asd_result의 각 face track에서 frame sampling → 평균 임베딩.

    Args:
        asd_result: asd_runner.run_asd 결과 (tracks 리스트 포함)
        video_path: 원본 video chunk path
        n_samples_per_track: 트랙당 sampling할 frame 수
        bbox_padding_ratio: face bbox 패딩 비율 (얼굴 잘림 방지)

    Returns:
        Dict[track_idx, embedding] — 임베딩 추출 성공한 track만
    """
    import cv2

    # v24: 환경변수로 sampling/padding 조정 가능 (default 인자 override)
    n_samples_per_track = int(os.environ.get(
        "LATENTSYNC_FACE_SAMPLES", str(n_samples_per_track)
    ))
    bbox_padding_ratio = float(os.environ.get(
        "LATENTSYNC_FACE_PADDING", str(bbox_padding_ratio)
    ))

    tracks = asd_result.get("tracks", [])
    if not tracks:
        return {}

    if not os.path.exists(video_path):
        print(f"[FaceID] video not found: {video_path}")
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FaceID] cv2 open 실패: {video_path}")
        return {}

    h, w = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    track_embeddings: Dict[int, np.ndarray] = {}
    n_total_extracts = 0
    n_failed_extracts = 0

    for tidx, t in enumerate(tracks):
        frames = t.get("frames", [])
        bboxes = t.get("bboxes", [])
        if not frames or not bboxes:
            continue

        n = len(frames)
        # 균등 sampling (시작/중간/끝 포함)
        if n <= n_samples_per_track:
            sample_indices = list(range(n))
        else:
            step = (n - 1) / (n_samples_per_track - 1)
            sample_indices = [int(round(i * step)) for i in range(n_samples_per_track)]

        embs: List[np.ndarray] = []
        for si in sample_indices:
            frame_idx = frames[si]
            bbox = bboxes[si]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            pw = bw * bbox_padding_ratio
            ph = bh * bbox_padding_ratio
            x1i = max(0, int(x1 - pw))
            y1i = max(0, int(y1 - ph))
            x2i = min(w, int(x2 + pw))
            y2i = min(h, int(y2 + ph))
            if x2i <= x1i or y2i <= y1i:
                continue
            crop = frame[y1i:y2i, x1i:x2i]
            n_total_extracts += 1
            emb = extract_face_id_embedding(crop)
            if emb is None:
                n_failed_extracts += 1
                continue
            embs.append(emb)

        if embs:
            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            norm = float(np.linalg.norm(mean_emb))
            if norm > 1e-8:
                mean_emb = mean_emb / norm
            track_embeddings[tidx] = mean_emb.astype(np.float32)

    cap.release()
    if verbose:
        n_tracks_with_emb = len(track_embeddings)
        print(f"[FaceID] {n_tracks_with_emb}/{len(tracks)} face tracks 임베딩 추출 "
              f"(samples {n_total_extracts}, fail {n_failed_extracts})")
    return track_embeddings


def cluster_tracks_by_face(
    track_embeddings: Dict[int, np.ndarray],
    similarity_threshold: float = 0.50,
    verbose: bool = True,
) -> Dict[int, int]:
    """face track들을 동일 인물끼리 cluster (Union-Find).

    Args:
        track_embeddings: compute_track_face_embeddings 반환
        similarity_threshold: cosine sim >= 이 값이면 동일인 (ArcFace 기준 0.5는 안전)

    Returns:
        Dict[track_idx, cluster_id] — cluster_id는 0부터 시작 정수
    """
    track_indices = sorted(track_embeddings.keys())
    if not track_indices:
        return {}

    parent = {ti: ti for ti in track_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sim_pairs: List[Tuple[int, int, float]] = []
    for i, ti in enumerate(track_indices):
        for tj in track_indices[i + 1:]:
            sim = float(np.dot(track_embeddings[ti], track_embeddings[tj]))
            if sim >= similarity_threshold:
                union(ti, tj)
                sim_pairs.append((ti, tj, sim))

    # cluster ID assign (root 기준)
    roots = sorted(set(find(ti) for ti in track_indices))
    root_to_cid = {r: i for i, r in enumerate(roots)}
    track_to_cluster = {ti: root_to_cid[find(ti)] for ti in track_indices}

    if verbose:
        from collections import Counter
        cnt = Counter(track_to_cluster.values())
        print(f"[FaceID] {len(track_indices)} tracks → {len(cnt)} face clusters "
              f"(threshold={similarity_threshold:.2f})")
        for cid in sorted(cnt):
            members = [ti for ti, c in track_to_cluster.items() if c == cid]
            print(f"  cluster_{cid}: {len(members)} tracks {members}")
        if sim_pairs[:5]:
            print(f"[FaceID] top similar pairs:")
            for ti, tj, s in sorted(sim_pairs, key=lambda x: -x[2])[:5]:
                print(f"  f{ti}↔f{tj}: cos={s:.3f}")
    return track_to_cluster


def derive_speaker_remap_from_face_clusters(
    fusion: Dict,
    track_to_cluster: Dict[int, int],
    min_cluster_share: float = 0.50,
    min_cluster_tracks: int = 2,
    min_speaker_to_cluster_share: float = 0.60,
    verbose: bool = True,
) -> Dict[str, str]:
    """face cluster 기반 audio speaker → canonical speaker remap.

    각 face cluster마다 *대표 audio speaker*를 결정하고
    cluster에 속한 다른 audio speaker는 대표로 통합.

    Args:
        fusion: fuse_av_diarization 반환 (speaker_face_count 포함)
        track_to_cluster: cluster_tracks_by_face 반환
        min_cluster_share: cluster의 대표 audio speaker 점유율 최소값

    Returns:
        Dict[old_speaker, canonical_speaker]
        — audio over-merge면 같은 cluster의 여러 speaker가 같은 canonical로
        — audio over-split이어도 이 함수론 해결 안 됨 (cluster 정의가 audio speaker 기준이라)

    Note: 이 함수는 face가 같은 사람인데 audio가 둘로 잘못 분리한 케이스를 통합.
    audio가 두 사람을 한 speaker로 통합한 케이스는 face가 다른 cluster여도 audio가 같으므로 검출 불가 (그 경우엔 face-based segment-level reassign 필요).
    """
    # v24: env var 오버라이드 (default 인자보다 환경변수 우선)
    min_cluster_share = float(os.environ.get(
        "LATENTSYNC_FACE_ID_CLUSTER_SHARE", str(min_cluster_share)
    ))
    min_cluster_tracks = int(os.environ.get(
        "LATENTSYNC_FACE_ID_MIN_TRACKS", str(min_cluster_tracks)
    ))
    min_speaker_to_cluster_share = float(os.environ.get(
        "LATENTSYNC_FACE_ID_SPEAKER_SHARE", str(min_speaker_to_cluster_share)
    ))

    speaker_face_count: Dict[str, Dict[int, int]] = fusion.get("speaker_face_count", {})
    if not speaker_face_count or not track_to_cluster:
        return {}

    # cluster별 track 수
    from collections import Counter
    cluster_track_count = Counter(track_to_cluster.values())

    # cluster_id → audio speaker별 매핑 frame 수
    cluster_speaker_frames: Dict[int, Dict[str, int]] = {}
    for spk, face_map in speaker_face_count.items():
        for face_idx, n_frames in face_map.items():
            cid = track_to_cluster.get(face_idx)
            if cid is None:
                continue
            cluster_speaker_frames.setdefault(cid, {})
            cluster_speaker_frames[cid][spk] = cluster_speaker_frames[cid].get(spk, 0) + n_frames

    # cluster마다 대표 audio speaker 결정 (안전장치):
    #   1) cluster track 수 ≥ min_cluster_tracks (단일 track cluster는 noisy)
    #   2) dominant speaker 점유 ≥ min_cluster_share
    cluster_canonical: Dict[int, str] = {}
    for cid, spk_frames in cluster_speaker_frames.items():
        # safety 1: cluster size
        if cluster_track_count.get(cid, 0) < min_cluster_tracks:
            if verbose:
                print(f"[FaceID] cluster_{cid} skip — tracks {cluster_track_count.get(cid, 0)} < {min_cluster_tracks}")
            continue
        total = sum(spk_frames.values())
        if total == 0:
            continue
        top_spk, top_count = max(spk_frames.items(), key=lambda x: x[1])
        share = top_count / total
        # safety 2: dominant 점유율
        if share >= min_cluster_share:
            cluster_canonical[cid] = top_spk
        elif verbose:
            print(f"[FaceID] cluster_{cid} skip — top {top_spk}({share:.0%}) < {min_cluster_share:.0%}")

    # speaker → canonical 매핑
    # safety 3: 이 speaker의 face frame 중 dominant cluster 점유율 ≥ min_speaker_to_cluster_share
    speaker_to_clusters: Dict[str, Dict[int, int]] = {}
    for spk, face_map in speaker_face_count.items():
        for face_idx, n_frames in face_map.items():
            cid = track_to_cluster.get(face_idx)
            if cid is None:
                continue
            speaker_to_clusters.setdefault(spk, {})
            speaker_to_clusters[spk][cid] = speaker_to_clusters[spk].get(cid, 0) + n_frames

    speaker_remap: Dict[str, str] = {}
    for spk, cluster_frames in speaker_to_clusters.items():
        if not cluster_frames:
            continue
        # 이 speaker의 dominant cluster + 점유율
        spk_total = sum(cluster_frames.values())
        top_cid, top_n = max(cluster_frames.items(), key=lambda x: x[1])
        spk_cluster_share = top_n / spk_total if spk_total > 0 else 0
        # safety 3: speaker의 face가 한 cluster에 충분히 집중되어 있을 때만 remap
        if spk_cluster_share < min_speaker_to_cluster_share:
            if verbose:
                print(f"[FaceID] {spk} remap skip — 분산 (top cluster {top_cid} share {spk_cluster_share:.0%} < {min_speaker_to_cluster_share:.0%})")
            continue
        canonical = cluster_canonical.get(top_cid)
        if canonical and canonical != spk:
            speaker_remap[spk] = canonical

    if verbose and speaker_remap:
        print(f"[FaceID] speaker remap (face cluster 기반):")
        for old, new in speaker_remap.items():
            print(f"  {old} → {new}")
    elif verbose:
        print(f"[FaceID] speaker remap 없음 (audio diarization이 face cluster와 일치)")
    return speaker_remap


def apply_speaker_remap_to_diarization(diarization, speaker_remap: Dict[str, str]):
    """Annotation에 speaker remap 적용 → 새 Annotation 반환."""
    if not speaker_remap:
        return diarization
    from pyannote.core import Annotation
    new_anno = Annotation(uri=getattr(diarization, "uri", None))
    for turn, track, spk in diarization.itertracks(yield_label=True):
        canon = speaker_remap.get(spk, spk)
        new_anno[turn, track] = canon
    return new_anno


def reassign_segments_by_face_voting(
    audio_segments,
    asd_result,
    track_embeddings,
    speaker_face_count,
    similarity_threshold: float = 0.55,
    margin_threshold: float = 0.05,
    min_seg_face_share: float = 0.50,
    max_seg_duration=None,
    verbose: bool = True,
):
    """Per-segment ArcFace voting based speaker reassignment (가설 C).

    cluster_canonical safety (single-track cluster, dominant share, speaker share)
    를 우회. 각 audio segment의 실제 face embedding을 각 speaker의 대표 embedding과
    직접 비교 → 가장 가까운 speaker로 reassign.

    환경변수 override:
        LATENTSYNC_FACE_VOTING_THRESHOLD (default 0.55)
        LATENTSYNC_FACE_VOTING_MARGIN (default 0.05)
        LATENTSYNC_FACE_VOTING_MIN_SHARE (default 0.50)
    """
    threshold = float(os.environ.get(
        "LATENTSYNC_FACE_VOTING_THRESHOLD", str(similarity_threshold)
    ))
    margin = float(os.environ.get(
        "LATENTSYNC_FACE_VOTING_MARGIN", str(margin_threshold)
    ))
    min_share = float(os.environ.get(
        "LATENTSYNC_FACE_VOTING_MIN_SHARE", str(min_seg_face_share)
    ))

    if not track_embeddings or not speaker_face_count:
        return []
    n_frames = asd_result.get("n_frames", 0)
    fps = asd_result.get("fps", 25.0)
    tracks = asd_result.get("tracks", [])
    if n_frames <= 0 or not tracks:
        return []

    speaker_ref_emb = {}
    for spk, face_map in speaker_face_count.items():
        weighted = []
        weights = []
        for face_idx, n_frames_spk in face_map.items():
            emb = track_embeddings.get(face_idx)
            if emb is None:
                continue
            weighted.append(emb * n_frames_spk)
            weights.append(n_frames_spk)
        if not weighted:
            continue
        ref = np.sum(weighted, axis=0) / max(1, sum(weights))
        nrm = float(np.linalg.norm(ref))
        if nrm > 1e-8:
            ref = ref / nrm
        speaker_ref_emb[spk] = ref.astype(np.float32)

    if len(speaker_ref_emb) < 2:
        if verbose:
            print(f"[FaceVoting] speaker ref embeddings <2 → skip")
        return []

    if verbose:
        print(f"[FaceVoting] speaker ref embeddings: {sorted(speaker_ref_emb.keys())}")
        spk_list = sorted(speaker_ref_emb.keys())
        for i, a in enumerate(spk_list):
            for b in spk_list[i + 1:]:
                s = float(np.dot(speaker_ref_emb[a], speaker_ref_emb[b]))
                if s >= 0.50:
                    print(f"  {a} ↔ {b}: cos={s:.3f}")

    face_at_frame = [[] for _ in range(n_frames)]
    for tidx, t in enumerate(tracks):
        if tidx not in track_embeddings:
            continue
        for f in t["frames"]:
            if 0 <= f < n_frames:
                face_at_frame[f].append(tidx)

    reassignments = []
    for start, end, spk in audio_segments:
        if max_seg_duration is not None and (end - start) > max_seg_duration:
            continue
        if spk not in speaker_ref_emb:
            continue
        f1 = max(0, int(start * fps))
        f2 = min(n_frames, int(end * fps + 1))
        seg_frames = max(1, f2 - f1)
        if f2 <= f1:
            continue

        face_freq = {}
        for f in range(f1, f2):
            for tidx in face_at_frame[f]:
                face_freq[tidx] = face_freq.get(tidx, 0) + 1
        if not face_freq:
            continue
        total_face = sum(face_freq.values())
        if total_face < seg_frames * min_share:
            continue

        seg_weighted = []
        seg_weights = []
        for tidx, n in face_freq.items():
            emb = track_embeddings.get(tidx)
            if emb is None:
                continue
            seg_weighted.append(emb * n)
            seg_weights.append(n)
        if not seg_weighted:
            continue
        seg_emb = np.sum(seg_weighted, axis=0) / max(1, sum(seg_weights))
        nrm = float(np.linalg.norm(seg_emb))
        if nrm > 1e-8:
            seg_emb = seg_emb / nrm

        sims = {s: float(np.dot(seg_emb, ref)) for s, ref in speaker_ref_emb.items()}
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        best_spk, best_sim = sorted_sims[0]
        second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
        cur_sim = sims.get(spk, 0.0)

        if (best_spk != spk
                and best_sim >= threshold
                and (best_sim - second_sim) >= margin
                and best_sim > cur_sim + margin):
            reassignments.append((start, end, spk, best_spk))
            if verbose:
                print(f"[FaceVoting] {start:.2f}~{end:.2f}s {spk}({cur_sim:.3f}) → "
                      f"{best_spk}({best_sim:.3f}, 2nd {second_sim:.3f})")

    if verbose:
        print(f"[FaceVoting] {len(reassignments)}/{len(audio_segments)} segments reassigned "
              f"(thr={threshold}, margin={margin}, min_share={min_share})")
    return reassignments
