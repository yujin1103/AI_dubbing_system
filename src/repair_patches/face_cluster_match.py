"""face cluster dominant SPK 기반 segment SPK reassign.

원리:
  메인 orch 가 report.json 에 dump 한 face_clusters (track → cluster) +
  speaker_face_count (SPK → {track: frames}) 사용.
  각 face cluster 의 dominant SPK 추정 → segment 시간대 visible face cluster 의
  dominant SPK 가 segment SPK 와 다르면 → 그 SPK 로 reassign.

  voice embedding (ERes2) acoustic ceiling 우회 — face 기반은 emotional shouting 무관.

ASD cache 와 결합:
  - ASD cache: track별 speaking score (높을수록 speaker 가능성 ↑)
  - face_clusters: track → cluster
  - speaker_face_count: SPK → {track: frames}

처리:
  1. cluster 별 SPK 분포 집계 → dominant SPK
  2. 각 segment 시간대 active face tracks 추출
  3. ASD speaking score 가장 높은 track → 그 cluster dominant SPK = "speaking_face_spk"
  4. speaking_face_spk ≠ segment SPK 이고 dominant 비율 충분 → reassign

입력: run_dir
출력: meta/<chunk>_segments_face_matched.json
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE_DIR = Path("/workspace/media/cache/lightasd")
REPORTS_DIR = Path("/workspace/media/reports")
DOMINANT_RATIO = 0.5      # cluster 내 dominant SPK 가 이 비율 이상이어야 신뢰
MIN_SPEAK_SCORE = 0.5     # ASD speaking score 이 이상이어야 speaking


def find_report(chunk_name: str):
    """매칭되는 report.json 찾기."""
    for p in REPORTS_DIR.glob("*.json"):
        try:
            r = json.load(open(p))
            for c in r.get("chunks", []):
                if c.get("name") == chunk_name:
                    return r, c
        except Exception:
            continue
    return None, None


def match_asd_cache(vocals_path: Path):
    audio, sr = sf.read(vocals_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    dur = len(audio) / sr
    best = None
    for p in CACHE_DIR.glob("*.pkl"):
        try:
            d = pickle.load(open(p, "rb"))
            n_f = d.get("n_frames", 0); fps = d.get("fps", 25.0)
            diff = abs(n_f / fps - dur)
            if best is None or diff < best[1]:
                best = (p, diff)
        except Exception:
            continue
    return best[0] if best and best[1] <= 3.0 else None


def compute_cluster_dominant(face_clusters, speaker_face_count):
    """cluster_id → (dominant_spk, ratio)."""
    cluster_spks = defaultdict(Counter)  # cluster_id → Counter(spk)
    for spk, faces in speaker_face_count.items():
        for track_str, cnt in faces.items():
            cluster_id = face_clusters.get(track_str)
            if cluster_id is None:
                continue
            cluster_spks[cluster_id][spk] += cnt
    out = {}
    for cid, cnt in cluster_spks.items():
        total = sum(cnt.values())
        if total == 0:
            continue
        sp, fr = cnt.most_common(1)[0]
        out[cid] = (sp, fr / total)
    return out


def main(run_dir: str):
    rd = Path(run_dir)
    meta = rd / "meta"
    # 우선순위: v190b > v190 > v194 > 원본
    candidates = []
    for sfx in ["_segments_v190b.json", "_segments_v190.json",
                "_segments_v194.json", "_segments.json"]:
        candidates += list(meta.glob(f"*_chunk_*{sfx}"))
    seg_files = []
    seen = set()
    for p in candidates:
        # chunk_name 추출
        n = p.stem
        for sfx in ["_segments_v190b", "_segments_v190",
                    "_segments_v194", "_segments"]:
            if n.endswith(sfx):
                cn = n[:-len(sfx)]
                break
        if cn in seen:
            continue
        seen.add(cn)
        seg_files.append((p, cn))
    if not seg_files:
        raise SystemExit(f"no segments in {meta}")

    for seg_path, chunk_name in seg_files:
        print(f"\n[{chunk_name}] using {seg_path.name}")
        vocals = rd / "vocals" / f"{chunk_name}_clean_vocals.wav"
        if not vocals.exists():
            print(f"  [skip] no vocals")
            continue

        # ASD cache
        cache_pkl = match_asd_cache(vocals)
        if not cache_pkl:
            print(f"  [skip] no ASD cache")
            continue
        asd = pickle.load(open(cache_pkl, "rb"))
        fps = asd["fps"]
        tracks = asd["tracks"]

        # report (face_clusters + speaker_face_count)
        rep, ck = find_report(chunk_name)
        if not ck:
            print(f"  [skip] no report (run main orch first)")
            continue
        face_clusters = ck.get("face_clusters", {})
        speaker_face_count = ck.get("speaker_face_count", {})
        print(f"  face_clusters: {len(face_clusters)} tracks, "
              f"speaker_face_count: {len(speaker_face_count)} SPKs")

        # cluster dominant
        cluster_dom = compute_cluster_dominant(face_clusters, speaker_face_count)
        print(f"  cluster dominant SPK:")
        for cid, (sp, r) in sorted(cluster_dom.items()):
            print(f"    cluster {cid:3d}: {sp} ({r*100:.0f}%)")

        seg_data = json.load(open(seg_path))
        groups = seg_data["groups"]
        n_reassign = 0
        new_segs = []
        for seg in groups:
            ss, se = seg.get("group_start", 0), seg.get("group_end", 0)
            cur_spk = seg["speaker"]
            # BG 화자는 skip
            if cur_spk.startswith("SPEAKER_BG"):
                new_segs.append(dict(seg))
                continue
            # segment 시간대 active face tracks + speaking score
            f_start = int(ss * fps); f_end = int(se * fps)
            track_scores = []
            for ti, t in enumerate(tracks):
                fr = np.array(t.get("frames", []))
                sc = np.array(t.get("scores", []))
                n = min(len(fr), len(sc))   # LightASD track 의 frames/scores 길이 불일치 방어
                if n == 0:
                    continue
                fr = fr[:n]; sc = sc[:n]
                mask = (fr >= f_start) & (fr <= f_end)
                if mask.sum() < 3:
                    continue
                mean_score = float(sc[mask].mean())
                if mean_score < MIN_SPEAK_SCORE:
                    continue
                cid = face_clusters.get(str(ti))
                if cid is None:
                    continue
                if cid not in cluster_dom:
                    continue
                track_scores.append((ti, mean_score, cid))
            if not track_scores:
                new_segs.append(dict(seg))
                continue
            # 가장 speaking 강한 track
            track_scores.sort(key=lambda x: -x[1])
            ti, score, cid = track_scores[0]
            dom_spk, ratio = cluster_dom[cid]
            if ratio < DOMINANT_RATIO:
                new_segs.append(dict(seg))
                continue
            if dom_spk != cur_spk:
                print(f"  REASSIGN [{ss:.2f}-{se:.2f}] {cur_spk} → {dom_spk} "
                      f"(face cluster {cid} dominant {dom_spk} {ratio*100:.0f}%, ASD={score:+.2f})")
                ns = dict(seg)
                ns["speaker"] = dom_spk
                ns["from_face_match"] = True
                new_segs.append(ns)
                n_reassign += 1
            else:
                new_segs.append(dict(seg))

        # consecutive same-spk merge
        new_segs.sort(key=lambda x: x.get("group_start", 0))
        merged = []
        for s in new_segs:
            if merged and merged[-1]["speaker"] == s["speaker"] \
               and s.get("group_start", 0) - merged[-1].get("group_end", 0) <= 0.1:
                merged[-1]["group_end"] = s["group_end"]
                merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
            else:
                merged.append(dict(s))

        from collections import Counter as C
        cnt = C(s["speaker"] for s in merged)
        print(f"\n  [✓] {len(groups)} → {len(merged)} segments (face reassign={n_reassign})")
        print(f"      SPKs: {dict(cnt)}")
        seg_data["groups"] = merged
        out = meta / f"{chunk_name}_segments_face_matched.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, ensure_ascii=False, indent=2)
        print(f"      saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    main(args.run_dir)
