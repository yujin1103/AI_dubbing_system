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
    # v113+: face encoder upgrade (default buffalo_l, antelopev2=R100 더 정밀)
    model_name = os.environ.get("LATENTSYNC_FACE_MODEL", "buffalo_l")
    # v114+: gender/age 모델 추가 (SPK merge 강화 신호)
    use_genderage = os.environ.get("LATENTSYNC_FACE_GENDERAGE", "1") == "1"
    allowed_modules = ["detection", "recognition"]
    if use_genderage:
        allowed_modules.append("genderage")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _face_app = FaceAnalysis(
        name=model_name,
        providers=providers,
        allowed_modules=allowed_modules,
    )
    _face_app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
    print(f"[FaceID] InsightFace model: {model_name} (modules: {allowed_modules})", flush=True)
    return _face_app


def extract_face_id_embedding(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """단일 이미지(BGR)에서 가장 큰 얼굴의 ArcFace 임베딩 추출.

    Returns:
        np.ndarray (512-dim, L2-normalized) 또는 None
    """
    res = extract_face_id_embedding_with_gender(image_bgr)
    return res[0] if res else None


def _compute_shape_features(kps) -> Optional[np.ndarray]:
    """InsightFace 5-point keypoints에서 pose-invariant face shape ratios.

    kps: (5, 2) array — [eye_L, eye_R, nose, mouth_L, mouth_R]
    Returns: 5-dim feature vector or None if invalid.

    v120+: 동일 gender + age 화자 분리용 face geometry signal.
    All ratios normalized by inter-eye distance → 2D similarity invariant.
    """
    if kps is None:
        return None
    try:
        kps = np.asarray(kps, dtype=np.float32)
        if kps.shape != (5, 2):
            return None
        eye_l, eye_r, nose, mouth_l, mouth_r = kps[0], kps[1], kps[2], kps[3], kps[4]
        eye_dist = float(np.linalg.norm(eye_r - eye_l))
        if eye_dist < 1e-3:
            return None
        eye_center = (eye_l + eye_r) / 2.0
        mouth_center = (mouth_l + mouth_r) / 2.0
        mouth_width = float(np.linalg.norm(mouth_r - mouth_l))
        eye_to_mouth = float(np.linalg.norm(mouth_center - eye_center))
        eye_to_nose = float(np.linalg.norm(nose - eye_center))
        nose_to_mouth = float(np.linalg.norm(mouth_center - nose))
        # nose lateral offset from eye-mouth midline (face symmetry signal)
        mid_em = (eye_center + mouth_center) / 2.0
        face_axis = mouth_center - eye_center
        axis_len = float(np.linalg.norm(face_axis)) + 1e-6
        face_axis_n = face_axis / axis_len
        # perpendicular distance nose to face axis
        rel_nose = nose - eye_center
        proj = float(np.dot(rel_nose, face_axis_n))
        perp = float(np.linalg.norm(rel_nose - proj * face_axis_n))
        feats = np.array([
            mouth_width / eye_dist,           # mouth width ratio
            eye_to_mouth / eye_dist,          # face length ratio
            eye_to_nose / eye_dist,           # nose top-position
            nose_to_mouth / eye_dist,         # nose-to-mouth length
            perp / eye_dist,                  # nose lateral offset (asymmetry)
        ], dtype=np.float32)
        # filter NaN/inf
        if not np.all(np.isfinite(feats)):
            return None
        return feats
    except Exception:
        return None


def extract_face_id_embedding_with_gender(image_bgr: np.ndarray):
    """단일 이미지에서 가장 큰 얼굴의 임베딩 + gender + age + shape feats.

    Returns: (embedding, gender_str, age, shape_feats) or None
        gender_str: 'M' or 'F' (genderage 모델 활성 시), None (없으면)
        age: estimated age (int) or None
        shape_feats: 5-dim np.ndarray (mouth/face geometry ratios) or None
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    try:
        app = _get_face_app()
        faces = app.get(image_bgr)
        if not faces:
            return None
        def area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0, (x2 - x1) * (y2 - y1))
        biggest = max(faces, key=area)
        emb = biggest.normed_embedding.astype(np.float32)
        gender = None
        if hasattr(biggest, "sex"):
            gender = biggest.sex
        elif hasattr(biggest, "gender") and biggest.gender is not None:
            gender = "F" if biggest.gender == 0 else "M"
        age = None
        if hasattr(biggest, "age") and biggest.age is not None:
            age = int(biggest.age)
        # v120+: face shape ratios from kps (5-point)
        shape_feats = None
        if hasattr(biggest, "kps") and biggest.kps is not None:
            shape_feats = _compute_shape_features(biggest.kps)
        return emb, gender, age, shape_feats
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
    track_genders: Dict[int, str] = {}
    track_ages: Dict[int, int] = {}
    track_shape_feats: Dict[int, np.ndarray] = {}
    n_total_extracts = 0
    n_failed_extracts = 0

    for tidx, t in enumerate(tracks):
        frames = t.get("frames", [])
        bboxes = t.get("bboxes", [])
        if not frames or not bboxes:
            continue

        n = len(frames)
        if n <= n_samples_per_track:
            sample_indices = list(range(n))
        else:
            step = (n - 1) / (n_samples_per_track - 1)
            sample_indices = [int(round(i * step)) for i in range(n_samples_per_track)]

        embs: List[np.ndarray] = []
        gender_votes = []
        age_votes = []
        shape_votes: List[np.ndarray] = []
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
            result = extract_face_id_embedding_with_gender(crop)
            if result is None:
                n_failed_extracts += 1
                continue
            emb, gender, age, shape_feats = result
            embs.append(emb)
            if gender:
                gender_votes.append(gender)
            if age is not None:
                age_votes.append(age)
            if shape_feats is not None:
                shape_votes.append(shape_feats)

        if embs:
            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            norm = float(np.linalg.norm(mean_emb))
            if norm > 1e-8:
                mean_emb = mean_emb / norm
            track_embeddings[tidx] = mean_emb.astype(np.float32)
            if gender_votes:
                from collections import Counter as _C
                track_genders[tidx] = _C(gender_votes).most_common(1)[0][0]
            if age_votes:
                track_ages[tidx] = int(round(sum(age_votes) / len(age_votes)))
            if shape_votes:
                track_shape_feats[tidx] = np.mean(np.stack(shape_votes, axis=0), axis=0).astype(np.float32)

    cap.release()
    if verbose:
        n_tracks_with_emb = len(track_embeddings)
        print(f"[FaceID] {n_tracks_with_emb}/{len(tracks)} face tracks 임베딩 추출 "
              f"(samples {n_total_extracts}, fail {n_failed_extracts}, gender={len(track_genders)}, shape={len(track_shape_feats)})")
    # 사이드 효과: gender/age/shape info를 module-level global에 저장
    global _last_track_genders, _last_track_ages, _last_track_shape_feats, _last_asd_result
    _last_track_genders = track_genders
    _last_track_ages = track_ages
    _last_track_shape_feats = track_shape_feats
    # v166+: asd_result 저장 (face track frame lookup용)
    _last_asd_result = asd_result
    if verbose and track_ages:
        ages_summary = sorted(track_ages.values())
        print(f"[FaceID] track ages range: {min(ages_summary)} ~ {max(ages_summary)} (median={ages_summary[len(ages_summary)//2]})", flush=True)
    return track_embeddings


_last_track_genders: Dict[int, str] = {}
_last_track_ages: Dict[int, int] = {}
_last_track_shape_feats: Dict[int, np.ndarray] = {}
_last_track_to_cluster: Dict[int, int] = {}
_last_asd_result: Optional[Dict] = None  # asd_runner result for face frame lookup


def get_last_track_genders() -> Dict[int, str]:
    """compute_track_face_embeddings 호출 후 track별 gender 가져오기."""
    return dict(_last_track_genders)


def get_last_track_shape_feats() -> Dict[int, np.ndarray]:
    """compute_track_face_embeddings 호출 후 track별 shape feature 가져오기."""
    return dict(_last_track_shape_feats)


def get_last_track_to_cluster() -> Dict[int, int]:
    """cluster_tracks_by_face 호출 후 track → cluster_id 매핑 가져오기."""
    return dict(_last_track_to_cluster)


def get_last_asd_result() -> Optional[Dict]:
    """face_id 처리에 사용된 asd_result (face frames + bboxes) 가져오기."""
    return _last_asd_result


def get_last_track_ages() -> Dict[int, int]:
    """compute_track_face_embeddings 호출 후 track별 age 가져오기."""
    return dict(_last_track_ages)


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

    # v166+: 글로벌 cache for segment_refiner 통합
    global _last_track_to_cluster
    _last_track_to_cluster = dict(track_to_cluster)

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


def absorb_spurious_speakers(
    audio_segments,
    asd_result,
    track_embeddings,
    speaker_face_count,
    max_n_segments: int = 1,
    max_total_dur: float = 3.0,
    similarity_threshold: float = 0.30,
    verbose: bool = True,
):
    """1-turn spurious speaker (n_segs ≤ max_n_segments + total_dur < max_total_dur)
    → 그 segment 시간대의 face embedding과 가장 가까운 *다른* speaker로 reassign.

    환경변수:
        LATENTSYNC_SPURIOUS_MAX_SEGS (default 1)
        LATENTSYNC_SPURIOUS_MAX_DUR  (default 3.0)
        LATENTSYNC_SPURIOUS_SIM      (default 0.30) 최소 face cosine sim
    """
    max_n_segments = int(os.environ.get(
        "LATENTSYNC_SPURIOUS_MAX_SEGS", str(max_n_segments)
    ))
    max_total_dur = float(os.environ.get(
        "LATENTSYNC_SPURIOUS_MAX_DUR", str(max_total_dur)
    ))
    sim_thr = float(os.environ.get(
        "LATENTSYNC_SPURIOUS_SIM", str(similarity_threshold)
    ))

    if not track_embeddings or not speaker_face_count:
        return []
    n_frames = asd_result.get("n_frames", 0)
    fps = asd_result.get("fps", 25.0)
    tracks = asd_result.get("tracks", [])
    if n_frames <= 0 or not tracks:
        return []

    from collections import defaultdict
    spk_segs = defaultdict(list)
    for s in audio_segments:
        spk_segs[s[2]].append((s[0], s[1]))

    spurious_spks = []
    for spk, segs in spk_segs.items():
        if len(segs) > max_n_segments:
            continue
        total_dur = sum(e - s for s, e in segs)
        if total_dur < max_total_dur:
            spurious_spks.append(spk)
    if not spurious_spks:
        if verbose:
            print(f"[Spurious] no spurious speakers (max_segs={max_n_segments}, max_dur={max_total_dur}s)")
        return []
    if verbose:
        print(f"[Spurious] candidates: {spurious_spks}")

    # speaker별 face embedding mean (spurious 제외)
    speaker_ref_emb = {}
    for spk, face_map in speaker_face_count.items():
        if spk in spurious_spks:
            continue
        weighted, weights = [], []
        for face_idx, n_f in face_map.items():
            emb = track_embeddings.get(face_idx)
            if emb is None:
                continue
            weighted.append(emb * n_f)
            weights.append(n_f)
        if not weighted:
            continue
        ref = np.sum(weighted, axis=0) / max(1, sum(weights))
        nrm = float(np.linalg.norm(ref))
        if nrm > 1e-8:
            ref = ref / nrm
        speaker_ref_emb[spk] = ref.astype(np.float32)

    if not speaker_ref_emb:
        if verbose:
            print(f"[Spurious] no non-spurious speaker references → skip")
        return []

    face_at_frame = [[] for _ in range(n_frames)]
    for tidx, t in enumerate(tracks):
        if tidx not in track_embeddings:
            continue
        for f in t.get("frames", []):
            if 0 <= f < n_frames:
                face_at_frame[f].append(tidx)

    reassignments = []
    for spk in spurious_spks:
        for start, end in spk_segs[spk]:
            f1 = max(0, int(start * fps))
            f2 = min(n_frames, int(end * fps + 1))
            if f2 <= f1:
                continue
            face_freq = {}
            for f in range(f1, f2):
                for tidx in face_at_frame[f]:
                    face_freq[tidx] = face_freq.get(tidx, 0) + 1
            if not face_freq:
                if verbose:
                    print(f"[Spurious] {spk} {start:.2f}~{end:.2f}s — no face in segment → skip")
                continue

            # segment 대표 embedding
            seg_w, seg_wts = [], []
            for tidx, n_f in face_freq.items():
                emb = track_embeddings.get(tidx)
                if emb is None:
                    continue
                seg_w.append(emb * n_f)
                seg_wts.append(n_f)
            if not seg_w:
                continue
            seg_emb = np.sum(seg_w, axis=0) / max(1, sum(seg_wts))
            nrm = float(np.linalg.norm(seg_emb))
            if nrm > 1e-8:
                seg_emb = seg_emb / nrm

            sims = {s: float(np.dot(seg_emb, ref)) for s, ref in speaker_ref_emb.items()}
            best = max(sims.items(), key=lambda x: x[1])
            if best[1] >= sim_thr:
                reassignments.append((start, end, spk, best[0]))
                if verbose:
                    print(f"[Spurious] {spk}({start:.2f}~{end:.2f}s) → {best[0]} (sim={best[1]:.3f})")
            elif verbose:
                print(f"[Spurious] {spk}({start:.2f}~{end:.2f}s) skip — best sim {best[1]:.3f} < {sim_thr}")

    if verbose:
        print(f"[Spurious] {len(reassignments)} segments reassigned (absorb)")
    return reassignments


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

    # v44: LightASD speaking score 가중치
    # v108+: speaking_pow로 강한 score 더 강조 (silent frame 무력화)
    # face_at_frame[f] = [(tidx, asd_score), ...]
    use_speaking_weight = os.environ.get("LATENTSYNC_FACE_VOTING_SPEAKING_WEIGHT", "1") == "1"
    speaking_min_score = float(os.environ.get("LATENTSYNC_FACE_VOTING_SPEAKING_MIN", "0.0"))
    speaking_pow = float(os.environ.get("LATENTSYNC_FACE_VOTING_SPEAKING_POW", "1.0"))
    face_at_frame = [[] for _ in range(n_frames)]
    for tidx, t in enumerate(tracks):
        if tidx not in track_embeddings:
            continue
        t_frames = t.get("frames", [])
        t_scores = t.get("scores", [1.0] * len(t_frames))
        for fi, f in enumerate(t_frames):
            if 0 <= f < n_frames:
                asd_score = float(t_scores[fi]) if fi < len(t_scores) else 0.0
                face_at_frame[f].append((tidx, asd_score))

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

        # face_freq[tidx] = (frame_count, speaking_weight_sum)
        face_freq = {}
        for f in range(f1, f2):
            for tidx, score in face_at_frame[f]:
                if score < speaking_min_score:
                    continue
                w_speaking = max(0.0, score) if use_speaking_weight else 1.0
                if speaking_pow != 1.0 and w_speaking > 0:
                    w_speaking = w_speaking ** speaking_pow
                cnt, w_sum = face_freq.get(tidx, (0, 0.0))
                face_freq[tidx] = (cnt + 1, w_sum + w_speaking)
        if not face_freq:
            continue
        total_face = sum(cnt for cnt, _ in face_freq.values())
        if total_face < seg_frames * min_share:
            continue

        seg_weighted = []
        seg_weights = []
        for tidx, (cnt, w_speaking_sum) in face_freq.items():
            emb = track_embeddings.get(tidx)
            if emb is None:
                continue
            # v44: speaking-weighted (default) or frame-count weighted
            w = w_speaking_sum if use_speaking_weight else cnt
            if w <= 0:
                continue
            seg_weighted.append(emb * w)
            seg_weights.append(w)
        if not seg_weighted or sum(seg_weights) <= 0:
            continue
        seg_emb = np.sum(seg_weighted, axis=0) / max(1e-6, sum(seg_weights))
        nrm = float(np.linalg.norm(seg_emb))
        if nrm > 1e-8:
            seg_emb = seg_emb / nrm

        sims = {s: float(np.dot(seg_emb, ref)) for s, ref in speaker_ref_emb.items()}
        sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
        best_spk, best_sim = sorted_sims[0]
        second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
        cur_sim = sims.get(spk, 0.0)

        # v115+: segment dominant gender — SPK ref gender와 다르면 reassign 강제
        # v116+: age 추가 — 같은 gender + age 가까운 SPK 우선
        # v120+: face shape (mouth/face geometry) — 같은 gender+age라도 shape distance 멀면 reassign
        if _last_track_genders or _last_track_ages or _last_track_shape_feats:
            from collections import Counter as _C3
            seg_gender_votes = []
            seg_age_votes = []
            seg_shape_list: List[np.ndarray] = []
            seg_shape_w: List[float] = []
            for tidx, (cnt, _) in face_freq.items():
                g = _last_track_genders.get(tidx)
                if g:
                    seg_gender_votes.extend([g] * cnt)
                a = _last_track_ages.get(tidx)
                if a is not None:
                    seg_age_votes.extend([a] * cnt)
                sf = _last_track_shape_feats.get(tidx)
                if sf is not None:
                    seg_shape_list.append(sf * cnt)
                    seg_shape_w.append(float(cnt))
            seg_gender = _C3(seg_gender_votes).most_common(1)[0][0] if seg_gender_votes else None
            seg_age = int(round(sum(seg_age_votes) / len(seg_age_votes))) if seg_age_votes else None
            seg_shape = None
            if seg_shape_list and sum(seg_shape_w) > 0:
                seg_shape = np.sum(seg_shape_list, axis=0) / sum(seg_shape_w)
            # spk별 dominant gender + mean age + mean shape
            cur_spk_gender = None
            cur_spk_age = None
            cur_spk_shape = None
            if speaker_face_count.get(spk):
                g_votes = []
                a_votes = []
                sh_acc: List[np.ndarray] = []
                sh_w: List[float] = []
                for f_idx, n in speaker_face_count[spk].items():
                    g = _last_track_genders.get(f_idx)
                    if g:
                        g_votes.extend([g] * n)
                    a = _last_track_ages.get(f_idx)
                    if a is not None:
                        a_votes.extend([a] * n)
                    sf = _last_track_shape_feats.get(f_idx)
                    if sf is not None:
                        sh_acc.append(sf * n)
                        sh_w.append(float(n))
                if g_votes:
                    cur_spk_gender = _C3(g_votes).most_common(1)[0][0]
                if a_votes:
                    cur_spk_age = int(round(sum(a_votes) / len(a_votes)))
                if sh_acc and sum(sh_w) > 0:
                    cur_spk_shape = np.sum(sh_acc, axis=0) / sum(sh_w)
            # gender mismatch → reassign to same-gender SPK
            need_reassign = False
            reassign_reason = ""
            if seg_gender and cur_spk_gender and seg_gender != cur_spk_gender:
                need_reassign = True
                reassign_reason = "gender"
            # v116+: same gender but age 차이 큼 (>=15세) → reassign 후보
            age_th = int(os.environ.get("LATENTSYNC_FACE_VOTING_AGE_DIFF", "15"))
            if not need_reassign and seg_age is not None and cur_spk_age is not None:
                if abs(seg_age - cur_spk_age) >= age_th:
                    need_reassign = True
                    reassign_reason = "age"
            # v120+: face shape distance — same gender+age라도 mouth/face geometry 다르면 reassign
            # NOTE: 5-point kps geometry는 head pose에 noisy → conservative threshold (default OFF as trigger)
            shape_th = float(os.environ.get("LATENTSYNC_FACE_VOTING_SHAPE_DIST", "9.99"))
            use_shape = os.environ.get("LATENTSYNC_FACE_VOTING_SHAPE", "1") == "1"
            if (use_shape and not need_reassign
                    and seg_shape is not None and cur_spk_shape is not None):
                shape_dist = float(np.linalg.norm(seg_shape - cur_spk_shape))
                if shape_dist >= shape_th:
                    need_reassign = True
                    reassign_reason = f"shape({shape_dist:.3f})"
            if need_reassign:
                # gender 일치 + age 가까운 + shape 가까운 SPK 후보 선택
                best_cand = None
                best_cand_score = float("-inf")
                for cand_spk, cand_sim in sorted_sims:
                    if cand_spk == spk:
                        continue
                    g_v, a_v = [], []
                    sh_acc2: List[np.ndarray] = []
                    sh_w2: List[float] = []
                    for f_idx, n in speaker_face_count.get(cand_spk, {}).items():
                        g = _last_track_genders.get(f_idx)
                        if g:
                            g_v.extend([g] * n)
                        a = _last_track_ages.get(f_idx)
                        if a is not None:
                            a_v.extend([a] * n)
                        sf = _last_track_shape_feats.get(f_idx)
                        if sf is not None:
                            sh_acc2.append(sf * n)
                            sh_w2.append(float(n))
                    cand_g = _C3(g_v).most_common(1)[0][0] if g_v else None
                    cand_a = int(round(sum(a_v) / len(a_v))) if a_v else None
                    cand_sh = None
                    if sh_acc2 and sum(sh_w2) > 0:
                        cand_sh = np.sum(sh_acc2, axis=0) / sum(sh_w2)
                    # gender 일치 필수
                    if seg_gender and cand_g and cand_g != seg_gender:
                        continue
                    # age proximity + shape proximity bonus
                    age_diff = abs(seg_age - cand_a) if (seg_age is not None and cand_a is not None) else 999
                    sh_dist = float(np.linalg.norm(seg_shape - cand_sh)) if (seg_shape is not None and cand_sh is not None) else 0.0
                    # v120: shape weight 추가 — 같은 gender 후보 중 shape 가까운 우선
                    sh_w_coef = float(os.environ.get("LATENTSYNC_FACE_VOTING_SHAPE_WEIGHT", "0.20"))
                    score = cand_sim - 0.005 * age_diff - sh_w_coef * sh_dist
                    if score > best_cand_score:
                        best_cand_score = score
                        best_cand = (cand_spk, cand_sim, cand_g, cand_a, sh_dist)
                if best_cand:
                    cs, csi, cg, ca, csd = best_cand
                    reassignments.append((start, end, spk, cs))
                    if verbose:
                        tag = "G+A+S" if reassign_reason.startswith("shape") else "G+A"
                        print(f"[FaceVoting/{tag}] {start:.2f}~{end:.2f}s {spk}({cur_spk_gender}/{cur_spk_age},sim={cur_sim:.3f}) → "
                              f"{cs}({cg}/{ca},sim={csi:.3f},sh={csd:.3f}) — seg=({seg_gender}/{seg_age}) reason={reassign_reason}", flush=True)
                    continue  # next segment
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
              f"(thr={threshold}, margin={margin}, min_share={min_share}, "
              f"speaking_weight={use_speaking_weight})")
    return reassignments
