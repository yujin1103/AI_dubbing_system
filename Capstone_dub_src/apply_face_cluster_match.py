"""우리 face_clusters.json + segments_*.json → 보존 face_cluster_match.py 적용.

흐름:
  1) face_clusters.json (face_clustering.py 출력) 로드 — 우리 schema (face_clusters + speaker_face_count)
  2) run_dir 안 segments_*.json 후보 (v190b > v190 > v194 > 원본)
  3) ASD cache 매칭 (보존 코드와 동일 로직 — vocals duration 기반)
  4) cluster dominant SPK + dominant_ratio ≥ DOMINANT_RATIO 시 reassign
  5) BG 화자 skip
  6) 출력: meta/<chunk>_segments_face_matched_preserved.json

영상 무관 default. hardcoding 없음.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE_DIR = Path(os.environ.get("LIGHTASD_CACHE_DIR", "/workspace/media/cache/lightasd"))
DOMINANT_RATIO = 0.5
MIN_SPEAK_SCORE = 0.5
MIN_EVIDENCE_FRAMES = 3


def _match_asd_cache(vocals_path: Path) -> Path | None:
    if not vocals_path.exists():
        return None
    audio, sr = sf.read(vocals_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    dur = len(audio) / sr
    best = None
    for p in CACHE_DIR.glob("*.pkl"):
        try:
            d = pickle.load(open(p, "rb"))
            n_f = d.get("n_frames", 0)
            fps = d.get("fps", 25.0)
            diff = abs(n_f / fps - dur)
            if best is None or diff < best[1]:
                best = (p, diff)
        except Exception:
            continue
    return best[0] if best and best[1] <= 3.0 else None


def _compute_cluster_dominant(face_clusters: dict, speaker_face_count: dict) -> dict[int, tuple[str, float]]:
    cluster_spks: dict[int, Counter] = defaultdict(Counter)
    for spk, tracks in speaker_face_count.items():
        for track_str, cnt in tracks.items():
            cid = face_clusters.get(track_str)
            if cid is None:
                continue
            cluster_spks[int(cid)][spk] += int(cnt)
    out: dict[int, tuple[str, float]] = {}
    for cid, cnt in cluster_spks.items():
        total = sum(cnt.values())
        if total == 0:
            continue
        sp, fr = cnt.most_common(1)[0]
        out[cid] = (sp, fr / total)
    return out


def apply_in_run(
    run_dir: str,
    *,
    dominant_ratio: float = DOMINANT_RATIO,
    min_speak_score: float = MIN_SPEAK_SCORE,
    min_evidence_frames: int = MIN_EVIDENCE_FRAMES,
) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    # face_clusters.json (우리 face_clustering.py 출력)
    face_files = list(meta.glob("face_clusters*.json"))
    if not face_files:
        raise SystemExit(f"no face_clusters.json in {meta}")
    face_data = json.load(open(face_files[0]))
    face_clusters = face_data.get("face_clusters", {})
    # speaker_face_count 는 segments 시간대 ↔ tracks 매핑으로 동적 재계산 (보존 schema).
    # 첫 chunk segments + ASD cache 로 계산 (영상 무관 자동).

    # segments 입력 (v190b > v190 > v194 > 원본 우선순위)
    seg_files = []
    seen = set()
    for sfx in ["_segments_v190b.json", "_segments_v190.json",
                "_segments_v194.json", "_segments.json"]:
        for p in meta.glob(f"*_chunk_*{sfx}"):
            n = p.stem
            for s in ["_segments_v190b", "_segments_v190", "_segments_v194", "_segments"]:
                if n.endswith(s):
                    cn = n[:-len(s)]
                    break
            if cn in seen:
                continue
            seen.add(cn)
            seg_files.append((p, cn))
    if not seg_files:
        raise SystemExit(f"no segments_*.json in {meta}")

    # 모든 chunks 의 segments + tracks 모아서 speaker_face_count 동적 계산
    speaker_face_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seg_path, chunk_name in seg_files:
        seg_data = json.load(open(seg_path))
        groups = seg_data.get("groups", seg_data.get("segments", []))
        vocals = rd / "vocals" / f"{chunk_name}_clean_vocals.wav"
        cache_pkl_init = _match_asd_cache(vocals)
        if not cache_pkl_init:
            continue
        asd_init = pickle.load(open(cache_pkl_init, "rb"))
        fps_init = asd_init["fps"]
        tracks_init = asd_init["tracks"]
        for seg in groups:
            ss = float(seg.get("group_start", seg.get("start", 0)))
            se = float(seg.get("group_end", seg.get("end", 0)))
            spk = str(seg.get("speaker", ""))
            if not spk or spk.startswith("SPEAKER_BG"):
                continue
            f_start = int(ss * fps_init)
            f_end = int(se * fps_init)
            for ti, t in enumerate(tracks_init):
                fr = np.array(t.get("frames", []))
                sc = np.array(t.get("scores", []))
                if len(fr) == 0:
                    continue
                mask = (fr >= f_start) & (fr <= f_end) & (sc >= min_speak_score)
                n_speak = int(mask.sum())
                if n_speak > 0:
                    speaker_face_count[spk][str(ti)] += n_speak

    cluster_dom = _compute_cluster_dominant(face_clusters, speaker_face_count)
    print(f"speaker_face_count: {len(speaker_face_count)} SPKs (auto-computed)")
    print(f"cluster dominant SPK (top):")
    for cid, (sp, r) in sorted(cluster_dom.items())[:20]:
        print(f"  cluster {cid:3d}: {sp} ({r*100:.0f}%)")

    summary = {"n_reassign": 0, "n_segments": 0}
    for seg_path, chunk_name in seg_files:
        print(f"\n[{chunk_name}] using {seg_path.name}")
        vocals = rd / "vocals" / f"{chunk_name}_clean_vocals.wav"
        cache_pkl = _match_asd_cache(vocals)
        if not cache_pkl:
            print(f"  [skip] no ASD cache for {vocals.name}")
            continue
        asd = pickle.load(open(cache_pkl, "rb"))
        fps = asd["fps"]
        tracks = asd["tracks"]

        seg_data = json.load(open(seg_path))
        groups = seg_data.get("groups", seg_data.get("segments", []))
        new_segs = []
        n_reassign = 0
        for seg in groups:
            ss = seg.get("group_start", seg.get("start", 0))
            se = seg.get("group_end", seg.get("end", 0))
            cur_spk = str(seg.get("speaker", ""))
            if cur_spk.startswith("SPEAKER_BG"):
                new_segs.append(dict(seg))
                continue
            f_start = int(float(ss) * fps)
            f_end = int(float(se) * fps)
            track_scores = []
            for ti, t in enumerate(tracks):
                fr = np.array(t.get("frames", []))
                sc = np.array(t.get("scores", []))
                if len(fr) == 0:
                    continue
                mask = (fr >= f_start) & (fr <= f_end)
                if mask.sum() < min_evidence_frames:
                    continue
                mean_score = float(sc[mask].mean())
                if mean_score < min_speak_score:
                    continue
                cid = face_clusters.get(str(ti))
                if cid is None or cid not in cluster_dom:
                    continue
                track_scores.append((ti, mean_score, cid))
            if not track_scores:
                new_segs.append(dict(seg))
                continue
            track_scores.sort(key=lambda x: -x[1])
            _ti, _score, cid = track_scores[0]
            dom_spk, ratio = cluster_dom[cid]
            if ratio >= dominant_ratio and dom_spk != cur_spk:
                ns = dict(seg)
                ns["audio_speaker"] = cur_spk
                ns["speaker"] = dom_spk
                ns["from_face_match_preserved"] = True
                new_segs.append(ns)
                n_reassign += 1
            else:
                new_segs.append(dict(seg))
        # consecutive same-spk merge (보존 동일)
        new_segs.sort(key=lambda x: float(x.get("group_start", x.get("start", 0))))
        merged = []
        for s in new_segs:
            if merged and merged[-1]["speaker"] == s["speaker"]:
                prev_end = float(merged[-1].get("group_end", merged[-1].get("end", 0)))
                cur_start = float(s.get("group_start", s.get("start", 0)))
                if cur_start - prev_end <= 0.1:
                    merged[-1]["group_end"] = s.get("group_end", s.get("end"))
                    if "end" in merged[-1]:
                        merged[-1]["end"] = s.get("group_end", s.get("end"))
                    merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
                    continue
            merged.append(dict(s))

        cnt_spk = Counter(s["speaker"] for s in merged)
        print(f"  [✓] {len(groups)} → {len(merged)} segments (reassign={n_reassign})")
        print(f"      SPKs: {dict(cnt_spk)}")
        seg_data["groups"] = merged
        out = meta / f"{chunk_name}_segments_face_matched_preserved.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, ensure_ascii=False, indent=2)
        print(f"      saved → {out}")
        summary["n_segments"] += len(merged)
        summary["n_reassign"] += n_reassign
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dominant-ratio", type=float, default=DOMINANT_RATIO)
    ap.add_argument("--min-speak-score", type=float, default=MIN_SPEAK_SCORE)
    ap.add_argument("--min-evidence-frames", type=int, default=MIN_EVIDENCE_FRAMES)
    args = ap.parse_args()
    s = apply_in_run(
        args.run_dir,
        dominant_ratio=args.dominant_ratio,
        min_speak_score=args.min_speak_score,
        min_evidence_frames=args.min_evidence_frames,
    )
    print(f"\nsummary: {s}")


if __name__ == "__main__":
    main()
