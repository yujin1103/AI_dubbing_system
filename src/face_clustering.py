"""face_clustering — LightASD speaking face + face cluster → SPK remap.

검증된 baseline v305f 의 face_clusters 단계 재현. diarize 결과 (RTTM/JSON) 위에
face cluster dominant SPK 매핑을 얹어 잘못 분리된 SPK 를 정정한다.

흐름:
  1) chunk video 에 LightASD 실행 (/opt/Light-ASD/Columbia_test.py subprocess) →
     tracks.pckl + scores.pckl
  2) face embedding (insightface ArcFace) → cluster (cosine greedy, sim ≥ 0.4)
  3) audio diarize segment 시간대 visible face → dominant SPK 추정
  4) speaking_face_spk ≠ audio_spk 이고 dominant ratio ≥ 0.5 → reassign

입력:  paths.chunks_dir (chunk_*.mp4) + paths.diarization_json
출력:  paths.face_clusters_json + paths.diarization_face_matched_json
       report.face_clusters / report.speaker_face_count

config (configs/*.json):
  pipeline.face_clustering:
    enabled: true
    light_asd_dir: /opt/Light-ASD
    venv_python: /usr/bin/python    # face 컨테이너 시스템 python
    face_sim_threshold: 0.4
    min_speak_score: 0.5
    dominant_ratio: 0.5
    min_evidence_frames: 5
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    deep_get,
    ensure_parent,
    get_logger,
    load_json,
    require_value,
    resolve_project_path,
    save_json,
)

logger = get_logger("face_clustering")

LIGHT_ASD_DIR_DEFAULT = os.environ.get("LIGHT_ASD_DIR", "/opt/Light-ASD")
LIGHT_ASD_WEIGHT = "weight/finetuning_TalkSet.model"


def _run_lightasd(
    video_path: str,
    light_asd_dir: str,
    python_bin: str,
    *,
    keep_work_dir: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    # asd_runner.py 의 LightASD subprocess 호출. video_25fps 보존 (face crop 용).
    # 반환: (tracks_dict, video_25fps_path) — work_dir 도 보존 (keep_work_dir=True).
    video_name = Path(video_path).stem
    work_dir = tempfile.mkdtemp(prefix=f"asd_{video_name}_")
    demo_dir = os.path.join(work_dir, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    input_copy = os.path.join(demo_dir, f"{video_name}.mp4")
    if not os.path.exists(input_copy):
        shutil.copy(video_path, input_copy)

    cmd = [
        python_bin, "Columbia_test.py",
        "--videoName", video_name,
        "--videoFolder", demo_dir,
        "--pretrainModel", LIGHT_ASD_WEIGHT,
    ]
    logger.info("LightASD running on %s ...", video_name)
    try:
        result = subprocess.run(
            cmd, cwd=light_asd_dir,
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            logger.error("LightASD failed: %s", result.stderr[-500:])
            if not keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            return None, None
    except subprocess.TimeoutExpired:
        logger.error("LightASD timeout (30min)")
        if not keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return None, None

    pywork = os.path.join(demo_dir, video_name, "pywork")
    tracks_pkl = os.path.join(pywork, "tracks.pckl")
    scores_pkl = os.path.join(pywork, "scores.pckl")
    if not (os.path.exists(tracks_pkl) and os.path.exists(scores_pkl)):
        logger.error("LightASD output files missing: %s", pywork)
        if not keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return None, None

    with open(tracks_pkl, "rb") as f:
        tracks_raw = pickle.load(f)
    with open(scores_pkl, "rb") as f:
        scores_raw = pickle.load(f)

    video_25fps = os.path.join(demo_dir, video_name, "pyavi", "video.avi")
    fps = 25.0
    n_frames = 0
    try:
        import cv2
        cap = cv2.VideoCapture(video_25fps)
        fps_cv = cap.get(cv2.CAP_PROP_FPS)
        n_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps_cv > 0:
            fps = float(fps_cv)
        if n_frames_cv > 0:
            n_frames = n_frames_cv
    except Exception:
        pass

    tracks = []
    for i, t in enumerate(tracks_raw):
        track = t.get("track", {})
        frames = list(track.get("frame", []))
        bboxes = list(track.get("bbox", []))
        scores = list(scores_raw[i]) if i < len(scores_raw) else []
        tracks.append({
            "track_id": i,
            "frames": [int(x) for x in frames],
            "bboxes": [[float(c) for c in b] for b in bboxes],
            "scores": [float(s) for s in scores],
            "work_dir": work_dir,
            "video_25fps": video_25fps,
        })

    return {"fps": fps, "n_frames": n_frames, "tracks": tracks, "work_dir": work_dir}, video_25fps


def _extract_face_embedding(face_app, video_path: str, frame_idx: int, bbox: list[float]):
    # video_25fps 의 frame_idx 위치에서 bbox crop → ArcFace embedding 512-dim.
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    # bbox padding 20% (face_app det가 더 잘 잡음)
    pad = int(0.2 * max(x2 - x1, y2 - y1))
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)
    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None
    faces = face_app.get(crop)
    if not faces:
        return None
    # 가장 큰 face의 embedding
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = best.normed_embedding  # 512-dim, L2-normalized
    return emb


def _cluster_face_tracks(tracks: list[dict], sim_threshold: float) -> dict[int, int]:
    # ArcFace embedding 기반 cluster (cosine greedy, sim ≥ threshold = 같은 인물).
    # 각 track 의 가장 큰 face frame 1개에서 embedding 추출 → 기존 centroid 와 비교.
    try:
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError:
        logger.warning("insightface 없음 — track_id 단독 cluster 로 처리")
        return {t["track_id"]: t["track_id"] for t in tracks}

    # insightface FaceAnalysis 초기화 (GPU)
    try:
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as exc:
        logger.warning("FaceAnalysis init 실패 (%s) — placeholder cluster", exc)
        return {t["track_id"]: t["track_id"] for t in tracks}

    # 각 track 의 대표 embedding 추출 (가장 큰 face area frame)
    track_embs: dict[int, np.ndarray] = {}
    for t in tracks:
        if not t.get("frames"):
            continue
        # 가장 큰 bbox area 의 frame 선택
        best_i = 0
        best_area = -1.0
        for i, b in enumerate(t["bboxes"]):
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area > best_area:
                best_area = area
                best_i = i
        frame_idx = int(t["frames"][best_i])
        bbox = t["bboxes"][best_i]
        video_path = t.get("video_25fps")
        if not video_path or not os.path.exists(video_path):
            continue
        emb = _extract_face_embedding(face_app, video_path, frame_idx, bbox)
        if emb is None:
            continue
        track_embs[t["track_id"]] = np.asarray(emb, dtype=np.float32)

    if not track_embs:
        logger.warning("ArcFace embedding 추출 0 — placeholder cluster")
        return {t["track_id"]: t["track_id"] for t in tracks}

    # cosine greedy clustering
    centroids: list[np.ndarray] = []
    members: list[list[np.ndarray]] = []
    cluster_of: dict[int, int] = {}
    for tid, emb in track_embs.items():
        if not centroids:
            centroids.append(emb)
            members.append([emb])
            cluster_of[tid] = 0
            continue
        sims = [float(np.dot(emb, c)) for c in centroids]
        best = int(np.argmax(sims))
        if sims[best] >= sim_threshold:
            members[best].append(emb)
            cen = np.mean(members[best], axis=0)
            cen = cen / (np.linalg.norm(cen) + 1e-8)
            centroids[best] = cen
            cluster_of[tid] = best
        else:
            centroids.append(emb)
            members.append([emb])
            cluster_of[tid] = len(centroids) - 1

    # embedding 추출 실패한 track 은 별도 cluster
    unique_offset = len(centroids)
    for t in tracks:
        tid = t["track_id"]
        if tid not in cluster_of:
            cluster_of[tid] = unique_offset
            unique_offset += 1

    logger.info(
        "ArcFace clustering: %s tracks → %s clusters (sim ≥ %s)",
        len(tracks), len(centroids), sim_threshold,
    )
    return cluster_of


def _compute_dominant_speaker(
    diarize_segments: list[dict],
    tracks: list[dict],
    face_clusters: dict[int, int],
    fps: float,
    *,
    min_speak_score: float,
    min_evidence_frames: int,
) -> dict[str, tuple[str, float]]:
    # 각 audio SPK 별로 어떤 face cluster 와 가장 자주 같이 나오는지 집계.
    spk_cluster_count: dict[str, Counter] = defaultdict(Counter)
    for seg in diarize_segments:
        start = float(seg.get("start", seg.get("group_start", 0.0)))
        end = float(seg.get("end", seg.get("group_end", 0.0)))
        spk = str(seg.get("speaker", ""))
        f_start = int(start * fps)
        f_end = int(end * fps)
        for t in tracks:
            frames = t.get("frames") or []
            scores = t.get("scores") or []
            if not frames:
                continue
            mask_count = 0
            for fr, sc in zip(frames, scores):
                if f_start <= fr <= f_end and sc >= min_speak_score:
                    mask_count += 1
            if mask_count >= min_evidence_frames:
                cid = face_clusters.get(t["track_id"], -1)
                if cid >= 0:
                    spk_cluster_count[spk][cid] += mask_count
    # SPK → (dominant_cluster, ratio)
    out: dict[str, tuple[int, float]] = {}
    for spk, cnt in spk_cluster_count.items():
        total = sum(cnt.values())
        if total == 0:
            continue
        cid, n = cnt.most_common(1)[0]
        out[spk] = (cid, n / total)
    return out


def cluster_faces_in_run(
    chunks_dir: str | Path,
    diarization_json: str | Path,
    output_face_clusters_json: str | Path,
    output_remapped_json: str | Path,
    *,
    light_asd_dir: str = LIGHT_ASD_DIR_DEFAULT,
    venv_python: str = "/usr/bin/python",
    face_sim_threshold: float = 0.4,
    min_speak_score: float = 0.5,
    dominant_ratio: float = 0.5,
    min_evidence_frames: int = 5,
) -> dict[str, Any]:
    """Run LightASD + face clustering + SPK remap. Returns summary dict."""
    chunks_dir_p = resolve_project_path(chunks_dir)
    chunk_videos = sorted(chunks_dir_p.glob("*.mp4"))
    if not chunk_videos:
        logger.warning("no chunk *.mp4 in %s — skip face_clustering", chunks_dir_p)
        save_json({"face_clusters": {}, "speaker_face_count": {}}, output_face_clusters_json)
        # remapped 는 원본 그대로 복사
        ensure_parent(output_remapped_json)
        diar = load_json(diarization_json)
        save_json(diar, output_remapped_json)
        return {"chunks": 0, "tracks": 0, "remapped_segments": 0}

    diar = load_json(diarization_json)
    segments = diar.get("segments") or diar.get("groups") or diar
    if not isinstance(segments, list):
        raise ValueError(f"diarization_json {diarization_json} segments not a list")

    all_tracks: list[dict] = []
    all_fps = 25.0
    work_dirs: list[str] = []
    for video in chunk_videos:
        asd, _video_25fps = _run_lightasd(str(video), light_asd_dir, venv_python,
                                          keep_work_dir=True)
        if asd is None:
            logger.warning("LightASD failed for %s — skipping its tracks", video)
            continue
        all_fps = float(asd.get("fps") or 25.0)
        if asd.get("work_dir"):
            work_dirs.append(asd["work_dir"])
        for t in asd.get("tracks", []):
            all_tracks.append(t)

    if not all_tracks:
        logger.warning("no tracks produced — passthrough diarize")
        save_json({"face_clusters": {}, "speaker_face_count": {}}, output_face_clusters_json)
        save_json(diar, output_remapped_json)
        return {"chunks": len(chunk_videos), "tracks": 0, "remapped_segments": 0}

    face_clusters = _cluster_face_tracks(all_tracks, face_sim_threshold)
    spk_dominant = _compute_dominant_speaker(
        segments, all_tracks, face_clusters, all_fps,
        min_speak_score=min_speak_score,
        min_evidence_frames=min_evidence_frames,
    )

    # cluster → dominant SPK (보존 v305f report 와 동일 구조)
    cluster_spk: dict[int, Counter] = defaultdict(Counter)
    for spk, (cid, _ratio) in spk_dominant.items():
        cluster_spk[cid][spk] += 1
    cluster_dominant_spk: dict[int, str] = {}
    for cid, cnt in cluster_spk.items():
        if cnt:
            cluster_dominant_spk[cid] = cnt.most_common(1)[0][0]

    # segment 별 best face cluster 매핑 (재사용)
    seg_best_cluster: dict[int, int] = {}
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", seg.get("group_start", 0.0)))
        end = float(seg.get("end", seg.get("group_end", 0.0)))
        cur_spk = str(seg.get("speaker", ""))
        if cur_spk.startswith("SPEAKER_BG"):
            continue
        f_start = int(start * all_fps)
        f_end = int(end * all_fps)
        track_scores = []
        for t in all_tracks:
            frames = t.get("frames") or []
            scores = t.get("scores") or []
            if not frames:
                continue
            mask_n = sum(1 for fr, sc in zip(frames, scores)
                         if f_start <= fr <= f_end and sc >= min_speak_score)
            if mask_n >= min_evidence_frames:
                cid = face_clusters.get(t["track_id"], -1)
                avg_sc = sum(sc for fr, sc in zip(frames, scores) if f_start <= fr <= f_end) \
                         / max(mask_n, 1)
                track_scores.append((avg_sc, cid))
        if track_scores:
            track_scores.sort(key=lambda x: -x[0])
            seg_best_cluster[idx] = track_scores[0][1]

    # SPK split 분석: 한 SPK 의 segments 가 여러 face cluster 에 충분히 분산되면 split
    # 보존 face_cluster_match.py 의 derive_speaker_remap_from_face_clusters 로직.
    from collections import defaultdict as _dd
    spk_cluster_segs: dict[str, dict[int, list[int]]] = _dd(lambda: _dd(list))
    for idx, cid in seg_best_cluster.items():
        spk = str(segments[idx].get("speaker", ""))
        if not spk or spk.startswith("SPEAKER_BG"):
            continue
        spk_cluster_segs[spk][cid].append(idx)

    split_label: dict[int, str] = {}    # seg_idx → new SPK label
    n_split = 0
    min_split = max(2, int(min_evidence_frames // 2))  # cluster 당 최소 segment 수
    for spk, cluster_dist in spk_cluster_segs.items():
        strong = [(cid, idxs) for cid, idxs in cluster_dist.items() if len(idxs) >= min_split]
        if len(strong) < 2:
            continue
        # 가장 큰 cluster 는 원래 라벨 유지, 나머지는 split
        strong.sort(key=lambda x: -len(x[1]))
        for k, (cid, idxs) in enumerate(strong):
            if k == 0:
                continue
            new_label = f"{spk}_v{cid}"
            for i in idxs:
                split_label[i] = new_label
            n_split += len(idxs)
        logger.info("face split: SPK %s → %s sub-speakers (%s clusters)",
                    spk, len(strong), [c for c, _ in strong])

    # 한 segment 내에서 face cluster 가 변경되면 그 시점에 자동 split.
    # 영상 무관 자동 — sim_threshold/min_speak_score/min_evidence_frames 만 사용.
    # min_intra_split_sec = 0.3s (그 이상의 face run 만 의미있는 sub-segment).
    def _intra_face_split(seg_idx: int, seg: dict) -> list[dict]:
        start = float(seg.get("start", seg.get("group_start", 0.0)))
        end = float(seg.get("end", seg.get("group_end", 0.0)))
        if end - start < 1.5:
            return [dict(seg)]  # 짧으면 split 불필요
        f_start = int(start * all_fps)
        f_end = int(end * all_fps)
        timeline: list[tuple[int, int]] = []
        for t in all_tracks:
            frames = t.get("frames") or []
            scores = t.get("scores") or []
            cid = face_clusters.get(t["track_id"], -1)
            if cid < 0:
                continue
            for fr, sc in zip(frames, scores):
                if f_start <= fr <= f_end and sc >= min_speak_score:
                    timeline.append((int(fr), int(cid)))
        if not timeline:
            return [dict(seg)]
        timeline.sort()
        # 같은 cluster 연속 frame 묶음 (≤3 frame gap 허용)
        runs: list[list[int]] = []
        for fr, cid in timeline:
            if runs and runs[-1][0] == cid and fr - runs[-1][2] <= 3:
                runs[-1][2] = fr
            else:
                runs.append([cid, fr, fr])
        # 짧은 run 제거 (< 0.7s) — camera quick cut 무시
        min_intra_frames = max(5, int(0.7 * all_fps))
        runs = [r for r in runs if (r[2] - r[1] + 1) >= min_intra_frames]
        # 동일 cluster 인접 run 합침
        merged: list[list[int]] = []
        for cid, fs, fe in runs:
            if merged and merged[-1][0] == cid:
                merged[-1][2] = fe
            else:
                merged.append([cid, fs, fe])
        if len(merged) < 2:
            return [dict(seg)]
        # 각 run 을 별도 sub-segment 로 (face dominant SPK 자동 할당)
        cur_spk = str(seg.get("speaker", ""))
        out: list[dict] = []
        for i, (cid, fs, fe) in enumerate(merged):
            t_start = start if i == 0 else float(fs) / all_fps
            t_end = end if i == len(merged) - 1 else float(merged[i + 1][1]) / all_fps
            if t_end - t_start < 0.2:
                continue
            dom_spk = cluster_dominant_spk.get(cid, cur_spk)
            ns = dict(seg)
            ns["group_start"] = round(t_start, 3)
            ns["group_end"] = round(t_end, 3)
            if "start" in ns:
                ns["start"] = round(t_start, 3)
            if "end" in ns:
                ns["end"] = round(t_end, 3)
            ns["speaker"] = dom_spk
            ns["from_face_intra_split"] = True
            ns["audio_speaker"] = cur_spk
            out.append(ns)
        return out

    # SPK remap (단일 dominant) + intra-segment face split + split 결합
    remapped: list[dict] = []
    n_reassign = 0
    n_intra = 0
    for idx, seg in enumerate(segments):
        cur_spk = str(seg.get("speaker", ""))
        if cur_spk.startswith("SPEAKER_BG"):
            remapped.append(dict(seg))
            continue
        ns = dict(seg)
        # 1) intra-segment face split (한 segment 내 face cluster 변경)
        intra_subs = _intra_face_split(idx, seg)
        if len(intra_subs) >= 2:
            n_intra += len(intra_subs) - 1
            remapped.extend(intra_subs)
            continue
        # 2) SPK label split (한 SPK 가 여러 face cluster 에 분산)
        if idx in split_label:
            ns["audio_speaker"] = cur_spk
            ns["speaker"] = split_label[idx]
            ns["from_face_split"] = True
            remapped.append(ns)
            continue
        # 3) dominant SPK 와 다르면 reassign
        best_cid = seg_best_cluster.get(idx)
        if best_cid is None:
            remapped.append(ns)
            continue
        dom_spk = cluster_dominant_spk.get(best_cid)
        if dom_spk and dom_spk != cur_spk:
            ns["audio_speaker"] = cur_spk
            ns["speaker"] = dom_spk
            ns["from_face_match"] = True
            remapped.append(ns)
            n_reassign += 1
        else:
            remapped.append(ns)

    # speaker_face_count: SPK → {track_id_str: frame_count} (보존 face_cluster_match
    # 호환 schema). 각 audio SPK 가 어느 face track 의 frame 들에 얼마나 등장했는지.
    speaker_track_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seg in segments:
        start = float(seg.get("start", seg.get("group_start", 0.0)))
        end = float(seg.get("end", seg.get("group_end", 0.0)))
        spk = str(seg.get("speaker", ""))
        if not spk or spk.startswith("SPEAKER_BG"):
            continue
        f_start = int(start * all_fps)
        f_end = int(end * all_fps)
        for t in all_tracks:
            frames = t.get("frames") or []
            scores = t.get("scores") or []
            n_speak = sum(
                1 for fr, sc in zip(frames, scores)
                if f_start <= fr <= f_end and sc >= min_speak_score
            )
            if n_speak > 0:
                speaker_track_count[spk][str(t["track_id"])] += n_speak

    # write outputs (보존 face_cluster_match.py 호환 schema)
    face_summary = {
        "face_clusters": {str(t["track_id"]): int(face_clusters.get(t["track_id"], -1))
                          for t in all_tracks},
        "speaker_face_count": {spk: dict(cnt) for spk, cnt in speaker_track_count.items()},
        "fps": all_fps,
        "n_tracks": len(all_tracks),
        "n_clusters": len(set(face_clusters.values())),
    }
    save_json(face_summary, output_face_clusters_json)

    # remapped diarization
    if isinstance(diar, dict) and "segments" in diar:
        diar["segments"] = remapped
        save_json(diar, output_remapped_json)
    elif isinstance(diar, dict) and "groups" in diar:
        diar["groups"] = remapped
        save_json(diar, output_remapped_json)
    else:
        save_json(remapped, output_remapped_json)

    logger.info(
        "face_clustering: %s chunks, %s tracks, %s clusters, %s SPK reassigned, %s SPK-split segs, %s intra-face splits",
        len(chunk_videos), len(all_tracks), face_summary["n_clusters"], n_reassign, n_split, n_intra,
    )
    # work_dirs cleanup (face embedding 끝나고 video.avi 더 이상 필요 없음)
    for wd in work_dirs:
        shutil.rmtree(wd, ignore_errors=True)
    return {
        "chunks": len(chunk_videos),
        "tracks": len(all_tracks),
        "clusters": face_summary["n_clusters"],
        "remapped_segments": n_reassign,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("chunks_dir")
    ap.add_argument("diarization_json")
    ap.add_argument("--out-face-clusters", required=True)
    ap.add_argument("--out-remapped", required=True)
    ap.add_argument("--light-asd-dir", default=LIGHT_ASD_DIR_DEFAULT)
    ap.add_argument("--venv-python", default="/usr/bin/python")
    ap.add_argument("--face-sim-threshold", type=float, default=0.4)
    ap.add_argument("--min-speak-score", type=float, default=0.5)
    ap.add_argument("--dominant-ratio", type=float, default=0.5)
    ap.add_argument("--min-evidence-frames", type=int, default=5)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    cluster_faces_in_run(
        args.chunks_dir,
        args.diarization_json,
        args.out_face_clusters,
        args.out_remapped,
        light_asd_dir=args.light_asd_dir,
        venv_python=args.venv_python,
        face_sim_threshold=args.face_sim_threshold,
        min_speak_score=args.min_speak_score,
        dominant_ratio=args.dominant_ratio,
        min_evidence_frames=args.min_evidence_frames,
    )


if __name__ == "__main__":
    main()
