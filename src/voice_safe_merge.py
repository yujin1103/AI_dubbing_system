"""voice-based safe SPK merge — face cross-evidence + 높은 voice sim 임계값 안전 merge.

영상 무관, hardcoding 없음. SPK 쌍 voice cosine sim 계산 후:
  - face 같은 cluster (cross-evidence) → threshold 0.80+ 합침
  - face 다른/없음 → 0.90+ 매우 높은 sim 만 합침 (false positive 방지)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent / "daemons"))
from eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model


def _l2(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def voice_safe_merge(run_dir: str, *, threshold: float = 0.85, high_conf_threshold: float = 0.90) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    fc_path = meta / "face_clusters.json"
    spk_face_map = {}
    if fc_path.exists():
        fc = json.load(open(fc_path, encoding="utf-8"))
        spk_face_map = fc.get("speaker_face_map", {})

    seg_path = None
    for pat in ["*_segments_gapfilled.json", "*_segments_sandwich.json", "*_segments_v194.json", "*_segments.json"]:
        for cand in sorted(meta.glob(pat)):
            try:
                d = json.load(open(cand, encoding="utf-8"))
                grps = d.get("groups", d.get("segments", []))
                ms = set(str(s.get("speaker", "")) for s in grps if s.get("speaker") and not str(s.get("speaker", "")).startswith("SPEAKER_BG"))
                if len(ms) >= 2:
                    seg_path = cand; break
            except Exception: continue
        if seg_path: break
    if not seg_path:
        raise SystemExit(f"no usable segments in {meta}")
    print(f"using segments: {seg_path.name}")
    data = json.load(open(seg_path))
    segs = data.get("groups", data.get("segments", []))

    vocals_glob = list((rd / "vocals").glob("*_clean_vocals.wav"))
    if not vocals_glob: raise SystemExit("no vocals.wav")
    audio, sr = sf.read(str(vocals_glob[0]))
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    _ = get_eres2netv2_model()

    spk_segs = defaultdict(list)
    for i, s in enumerate(segs):
        sp = str(s.get("speaker", ""))
        if sp and not sp.startswith("SPEAKER_BG"):
            spk_segs[sp].append(i)

    centroids = {}
    for sp, idxs in spk_segs.items():
        best_idx = max(idxs, key=lambda i: float(segs[i].get("group_end", segs[i].get("end", 0))) - float(segs[i].get("group_start", segs[i].get("start", 0))))
        s = segs[best_idx]
        ss = float(s.get("group_start", s.get("start", 0)))
        se = float(s.get("group_end", s.get("end", 0)))
        i0 = int(max(0, (ss - 0.2) * sr)); i1 = int(min(len(audio), (se + 0.2) * sr))
        if i1 <= i0: continue
        try:
            emb = extract_eres2netv2_emb(audio[i0:i1], sr)
            centroids[sp] = _l2(np.asarray(emb, dtype=np.float32))
        except Exception: continue

    print(f"\n=== voice sim matrix (thr={threshold}, high_conf={high_conf_threshold}) ===")
    sp_list = sorted(centroids.keys())
    pair_sims = []
    for i, sa in enumerate(sp_list):
        for sb in sp_list[i + 1:]:
            sim = float(np.dot(centroids[sa], centroids[sb]))
            face_same = False
            if sa in spk_face_map and sb in spk_face_map:
                if spk_face_map[sa].get("dominant_face_cluster") == spk_face_map[sb].get("dominant_face_cluster"):
                    face_same = True
            pair_sims.append((sim, sa, sb, face_same))
    pair_sims.sort(key=lambda x: -x[0])
    for sim, sa, sb, fs in pair_sims[:12]:
        fm = " face_same✓" if fs else ""
        mm = ""
        if sim >= high_conf_threshold: mm = " ★ MERGE(high)"
        elif sim >= threshold and fs: mm = " ★ MERGE(sim+face)"
        elif sim >= threshold: mm = " skip(no face)"
        print(f"  {sa} vs {sb}: sim={sim:.3f}{fm}{mm}")

    merge_map = {}
    for sim, sa, sb, fs in pair_sims:
        do_merge = sim >= high_conf_threshold or (sim >= threshold and fs)
        if not do_merge: continue
        ta = sa
        while ta in merge_map: ta = merge_map[ta]
        tb = sb
        while tb in merge_map: tb = merge_map[tb]
        if ta == tb: continue
        if len(spk_segs[ta]) < len(spk_segs[tb]): merge_map[ta] = tb
        else: merge_map[tb] = ta

    n_reassign = 0
    for s in segs:
        cur = str(s.get("speaker", ""))
        if cur in merge_map:
            t = cur
            while t in merge_map: t = merge_map[t]
            if t != cur:
                s["audio_speaker_orig"] = cur; s["speaker"] = t; s["from_voice_safe_merge"] = True
                n_reassign += 1

    sorted_segs = sorted(segs, key=lambda x: float(x.get("group_start", x.get("start", 0))))
    merged = []
    for s in sorted_segs:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            prev_end = float(merged[-1].get("group_end", merged[-1].get("end", 0)))
            cur_start = float(s.get("group_start", s.get("start", 0)))
            if cur_start - prev_end <= 0.1:
                merged[-1]["group_end"] = s.get("group_end", s.get("end"))
                merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
                continue
        merged.append(dict(s))

    data["groups"] = merged
    out = meta / f"{seg_path.stem}_voice_merged_thr{int(threshold*100):03d}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    cnt = Counter(s["speaker"] for s in merged)
    print(f"\n[✓] {len(segs)} → {len(merged)} segs, {n_reassign} reassigned, {len(merge_map)} merges")
    print(f"    SPKs: {dict(cnt)}")
    print(f"    saved → {out}")
    return {"input": len(segs), "output": len(merged), "reassigns": n_reassign, "merges": len(merge_map), "threshold": threshold, "out_path": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--high-conf-threshold", type=float, default=0.90)
    args = ap.parse_args()
    s = voice_safe_merge(args.run_dir, threshold=args.threshold, high_conf_threshold=args.high_conf_threshold)
    print(f"summary: {s}")


if __name__ == "__main__":
    main()
