"""Face cluster 시간대 cross-check → 같은 audio SPK 안 다른 face cluster 시 강제 split.

Why:
  audio voice cluster 가 acoustic 유사 화자 (예: 두 사람의 외침)을 한 SPK 로
  cluster 했을 때, 각 segment 시간대의 dominant face cluster ID 가 다르면
  실제 다른 사람일 가능성 높음. face evidence 로 강제 SPK 분리.

영상 무관, hardcoding 없음:
  - LightASD tracks.pckl 의 frame → time 변환 (fps 사용)
  - 각 audio segment 시간대 visible face track 의 cluster ID 추출
  - 같은 audio SPK 안 face cluster 가 ≥ 2 → split

알고리즘:
  1. tracks.pckl + face_clusters.json 로드
  2. 각 audio segment 의 시간대 (start-end) 와 overlap 되는 face track 추출
  3. track → face_cluster ID 매핑 → segment 별 dominant cluster ID
  4. 같은 SPK 안 dominant cluster 가 ≥ 2 unique → sub-split
     (suffix: _faceA, _faceB, ...)
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def _track_time_range(track: dict, fps: float) -> tuple[float, float]:
    frames = list(track.get("frame", []))
    if not frames:
        return 0.0, 0.0
    return min(frames) / fps, max(frames) / fps


def face_time_split(
    run_dir: str,
    segments_name: str,
    out_name: str,
    tracks_pkl: str | None = None,
    face_clusters_json: str | None = None,
    fps: float = 25.0,
) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"

    seg_path = meta / segments_name
    data = json.load(open(seg_path, encoding="utf-8"))
    groups = data.get("groups", data.get("segments", []))

    # face_clusters 자동 탐색
    if face_clusters_json is None:
        for cand in ["test*_chunk_000_face_clusters_boost.json",
                     "test*_chunk_000_face_clusters.json",
                     "face_clusters.json"]:
            matches = sorted(meta.glob(cand))
            if matches:
                face_clusters_json = str(matches[0])
                break
    if not face_clusters_json:
        raise SystemExit("face_clusters.json not found")
    fc = json.load(open(face_clusters_json, encoding="utf-8"))
    face_cluster_map = fc.get("face_clusters", {})  # track_id_str → cluster_id
    # fps 정보
    if fc.get("fps"):
        fps = float(fc["fps"])

    # tracks.pckl 자동 탐색
    if tracks_pkl is None:
        # face_clustering 가 work_dir 보존 — /tmp/asd_<chunk_name>_*/demo/<chunk>/pywork/tracks.pckl
        chunk_name = segments_name.split("_segments")[0]
        for cand in Path("/tmp").glob(f"asd_*{chunk_name}*"):
            tp = cand / "demo" / chunk_name / "pywork" / "tracks.pckl"
            if tp.exists():
                tracks_pkl = str(tp)
                break
        # fallback: any asd_<chunk>_* path
        if not tracks_pkl:
            for cand in Path("/tmp").glob(f"asd_*"):
                tp = cand / "demo" / chunk_name / "pywork" / "tracks.pckl"
                if tp.exists():
                    tracks_pkl = str(tp)
                    break
    if not tracks_pkl or not Path(tracks_pkl).exists():
        raise SystemExit(f"tracks.pckl not found (tried /tmp/asd_*/demo/<chunk>/pywork/)")

    with open(tracks_pkl, "rb") as f:
        tracks = pickle.load(f)
    print(f"  tracks: {len(tracks)}, fps: {fps}")

    # track ID → (t_start, t_end, cluster_id)
    track_info = []
    for tid, t in enumerate(tracks):
        track = t.get("track", {})
        t_start, t_end = _track_time_range(track, fps)
        cid = face_cluster_map.get(str(tid), -1)
        if cid >= 0:
            track_info.append((tid, t_start, t_end, cid))

    # 각 audio segment 의 dominant face cluster
    n_split = 0
    spk_cluster_pairs = defaultdict(Counter)
    seg_cluster = {}  # seg idx → dominant cluster
    for i, seg in enumerate(groups):
        sp = str(seg.get("speaker", ""))
        if sp.startswith("SPEAKER_BG"):
            continue
        ss = float(seg.get("group_start", seg.get("start", 0)))
        ee = float(seg.get("group_end", seg.get("end", 0)))
        # overlap 되는 face track 들
        cluster_overlaps = Counter()
        for tid, t_s, t_e, cid in track_info:
            ov = max(0.0, min(ee, t_e) - max(ss, t_s))
            if ov > 0.1:
                cluster_overlaps[cid] += ov
        if cluster_overlaps:
            dom_cid, _ = cluster_overlaps.most_common(1)[0]
            seg_cluster[i] = dom_cid
            spk_cluster_pairs[sp][dom_cid] += 1

    # 같은 SPK 안 face cluster 2+ → sub-split
    MIN_CLUSTER_SUPPORT = 1  # 1+ segment 면 split (face 정확한 evidence)
    splits_info = {}
    for sp, cl_counts in spk_cluster_pairs.items():
        big = [(c, n) for c, n in cl_counts.most_common() if n >= MIN_CLUSTER_SUPPORT]
        if len(big) < 2:
            continue
        # cluster → suffix
        cid_to_suffix = {c: f"face{idx}" for idx, (c, _) in enumerate(big)}
        for i, dom_cid in seg_cluster.items():
            if str(groups[i].get("speaker", "")) != sp:
                continue
            if dom_cid not in cid_to_suffix:
                continue
            suffix = cid_to_suffix[dom_cid]
            if cid_to_suffix[big[0][0]] == suffix:
                continue  # dominant cluster → 기존 SPK 유지
            groups[i]["audio_speaker"] = sp
            groups[i]["speaker"] = f"{sp}_{suffix}"
            groups[i]["face_cluster"] = dom_cid
            groups[i]["from_face_split"] = True
            n_split += 1
        splits_info[sp] = {"face_clusters": [c for c, _ in big], "suffix": cid_to_suffix}
        print(f"  {sp}: face clusters {[c for c, _ in big]} → {len(big)} sub-SPK")

    data["groups"] = groups
    out_path = meta / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved → {out_path} ({n_split} segs split)")
    return {"n_split": n_split, "splits_info": splits_info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segments-name", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--tracks-pkl", help="tracks.pckl path (auto-detect if omitted)")
    ap.add_argument("--face-clusters-json", help="face_clusters.json path")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    r = face_time_split(
        args.run_dir, args.segments_name, args.out_name,
        args.tracks_pkl, args.face_clusters_json, args.fps,
    )
    print(f"summary: {r}")


if __name__ == "__main__":
    main()
