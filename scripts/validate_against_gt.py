"""Validate preserved run segments_gapfilled.json against GT.

Computes per-GT-speaker consistency + main-count match + BG detection,
the same metric used by sweep_gt_match.py.

Usage:
  python validate_against_gt.py <segments_gapfilled.json> <gt.json> [<out.json>]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def time_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0.0, min(e1, e2) - max(s1, s2))


def evaluate(detect_segs, gt):
    gt_to_spks: dict[str, list] = defaultdict(list)
    bg_detected = False
    for g in gt["segments"]:
        spks_in_overlap = []
        for d in detect_segs:
            ds = d.get("group_start", d.get("start", 0.0))
            de = d.get("group_end", d.get("end", 0.0))
            ov = time_overlap(g["start"], g["end"], ds, de)
            if ov >= 0.3:
                spks_in_overlap.append((d["speaker"], ov))
        if not spks_in_overlap:
            gt_to_spks[g["speaker"]].append(None)
            continue
        spks_in_overlap.sort(key=lambda x: -x[1])
        best_spk = spks_in_overlap[0][0]
        gt_to_spks[g["speaker"]].append(best_spk)
        if g["speaker"] == "BG":
            bg_detected = True

    consistency_scores: dict[str, float] = {}
    for gt_spk, detected_list in gt_to_spks.items():
        if gt_spk == "BG":
            consistency_scores[gt_spk] = 1.0 if any(d for d in detected_list) else 0.0
            continue
        non_none = [d for d in detected_list if d is not None]
        if not non_none:
            consistency_scores[gt_spk] = 0.0
            continue
        cnt = Counter(non_none)
        _, freq = cnt.most_common(1)[0]
        consistency_scores[gt_spk] = freq / len(detected_list)

    detect_spks = set(
        d["speaker"]
        for d in detect_segs
        if not str(d["speaker"]).startswith("SPEAKER_BG")
    )
    main_match = len(detect_spks) == gt["main_count"]

    avg_consistency = sum(consistency_scores.values()) / max(len(consistency_scores), 1)
    score = (
        avg_consistency
        + (0.2 if main_match else 0.0)
        + (0.1 if bg_detected else 0.0)
    )
    return {
        "score": round(score, 4),
        "avg_consistency": round(avg_consistency, 4),
        "main_count": len(detect_spks),
        "main_count_gt": gt["main_count"],
        "main_match": main_match,
        "bg_detected": bg_detected,
        "per_gt_consistency": {k: round(v, 4) for k, v in consistency_scores.items()},
        "gt_to_detected_speakers": {k: v for k, v in gt_to_spks.items()},
        "n_detect_segments": len(detect_segs),
        "n_gt_segments": len(gt["segments"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments_json")
    ap.add_argument("gt_json")
    ap.add_argument("out_json", nargs="?")
    args = ap.parse_args()

    seg_data = json.load(open(args.segments_json, encoding="utf-8"))
    segs = seg_data.get("groups", seg_data.get("segments", []))
    gt = json.load(open(args.gt_json, encoding="utf-8"))

    print(f"segments: {Path(args.segments_json).name}")
    print(f"GT: main={gt['main_count']}, bg={gt['bg_count']}, n_segments={len(gt['segments'])}")
    print(f"detected: {len(segs)} segments")

    ev = evaluate(segs, gt)
    ev["source"] = {
        "segments_json": str(args.segments_json),
        "gt_json": str(args.gt_json),
    }

    print(f"\n=== RESULT ===")
    print(f"score             : {ev['score']}")
    print(f"avg_consistency   : {ev['avg_consistency']}")
    print(f"main_count        : {ev['main_count']} (GT={ev['main_count_gt']}, match={ev['main_match']})")
    print(f"bg_detected       : {ev['bg_detected']}")
    print(f"per_gt_consistency:")
    for k, v in ev["per_gt_consistency"].items():
        print(f"  {k:20s}: {v}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
        print(f"\nsaved -> {args.out_json}")


if __name__ == "__main__":
    main()
