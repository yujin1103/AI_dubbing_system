"""DER (Diarization Error Rate) + 시간대 정확도 측정.

pyannote.metrics 사용. GT segments 와 detect segments 의 시간 overlap 기반:
  - Confusion: GT 화자와 detect 화자 매핑 (Hungarian optimal)
  - DER = (missed_speech + false_alarm + speaker_error) / total_speech
  - Speaker Error: 같은 시간대에 다른 화자 라벨 (mapping 후)
  - Missed: GT 음성인데 detect 안 한 시간
  - False Alarm: detect 했는데 GT 에 없는 시간

또한 segment-level 정확도:
  - GT segment 별로 detect SPK 매칭 후, 시간 overlap 비율 평균
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, GreedyDiarizationErrorRate


def _to_annotation(segments, *, gt: bool = False) -> Annotation:
    ann = Annotation()
    for s in segments:
        if gt:
            start, end = float(s["start"]), float(s["end"])
            spk = str(s["speaker"])
        else:
            start = float(s.get("group_start", s.get("start", 0.0)))
            end = float(s.get("group_end", s.get("end", 0.0)))
            spk = str(s.get("speaker", "UNK"))
        if end > start and spk:
            ann[Segment(start, end)] = spk
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments_json")
    ap.add_argument("gt_json")
    ap.add_argument("out_json", nargs="?")
    ap.add_argument("--collar", type=float, default=0.25,
                    help="boundary collar seconds (default 0.25)")
    ap.add_argument("--skip-overlap", action="store_true",
                    help="ignore overlapping regions")
    args = ap.parse_args()

    seg_data = json.load(open(args.segments_json, encoding="utf-8"))
    detect_segs = seg_data.get("groups", seg_data.get("segments", []))
    gt = json.load(open(args.gt_json, encoding="utf-8"))
    gt_segs = gt["segments"]

    reference = _to_annotation(gt_segs, gt=True)
    hypothesis = _to_annotation(detect_segs, gt=False)

    print(f"reference: {len(reference)} segments, {len(reference.labels())} speakers, duration {reference.get_timeline().duration():.2f}s")
    print(f"hypothesis: {len(hypothesis)} segments, {len(hypothesis.labels())} speakers, duration {hypothesis.get_timeline().duration():.2f}s")

    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    der_value = metric(reference, hypothesis, detailed=True)
    print(f"\n=== DER (collar={args.collar}s, skip_overlap={args.skip_overlap}) ===")
    print(f"  total speech    : {der_value['total']:.2f}s")
    print(f"  missed detection: {der_value['missed detection']:.2f}s ({100*der_value['missed detection']/max(der_value['total'],1e-9):.1f}%)")
    print(f"  false alarm     : {der_value['false alarm']:.2f}s ({100*der_value['false alarm']/max(der_value['total'],1e-9):.1f}%)")
    print(f"  speaker error   : {der_value['confusion']:.2f}s ({100*der_value['confusion']/max(der_value['total'],1e-9):.1f}%)")
    der_pct = 100 * (der_value['missed detection'] + der_value['false alarm'] + der_value['confusion']) / max(der_value['total'], 1e-9)
    print(f"  DER             : {der_pct:.2f}%")

    # Greedy DER (Hungarian mapping, more forgiving for label permutation)
    greedy_metric = GreedyDiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    greedy_value = greedy_metric(reference, hypothesis, detailed=True)
    greedy_pct = 100 * (greedy_value['missed detection'] + greedy_value['false alarm'] + greedy_value['confusion']) / max(greedy_value['total'], 1e-9)
    print(f"\n=== Greedy DER (Hungarian optimal mapping) ===")
    print(f"  greedy DER      : {greedy_pct:.2f}%")
    print(f"  greedy confusion: {greedy_value['confusion']:.2f}s ({100*greedy_value['confusion']/max(greedy_value['total'],1e-9):.1f}%)")

    # Segment-level: 각 GT segment 의 시간 overlap 비율로 정확도 측정
    print(f"\n=== Segment-level accuracy (per-GT-segment) ===")
    correct_time = 0.0
    total_gt_time = 0.0
    per_spk: dict[str, tuple[float, float]] = {}
    # GT segment 별로 가장 많이 overlap 한 detect SPK
    # Hungarian mapping 후 spk_mapping[gt_spk] = best detect_spk
    # pyannote 4.x: GreedyDiarizationErrorRate.compute_components(reference, hypothesis)
    # uses internal optimal mapping; expose via reference.argmax(hypothesis).
    # 우리는 GT->detect 매핑 수동 계산 (각 GT spk 의 시간 overlap 최대 detect spk).
    from collections import defaultdict as _dd
    overlap_matrix: dict[str, _dd] = _dd(lambda: _dd(float))  # gt_spk -> {detect_spk: ov_sec}
    for g in gt_segs:
        gs, ge = float(g["start"]), float(g["end"])
        gt_spk = str(g["speaker"])
        for d in detect_segs:
            ds = float(d.get("group_start", d.get("start", 0.0)))
            de = float(d.get("group_end", d.get("end", 0.0)))
            d_spk = str(d.get("speaker", ""))
            ov = max(0.0, min(ge, de) - max(gs, ds))
            if ov > 0:
                overlap_matrix[gt_spk][d_spk] += ov
    confusion: dict[str, str] = {}
    for gt_spk, ds in overlap_matrix.items():
        if ds:
            confusion[gt_spk] = max(ds.items(), key=lambda x: x[1])[0]
    inv_mapping = {v: k for k, v in confusion.items()}
    for g in gt_segs:
        gs, ge = float(g["start"]), float(g["end"])
        gt_spk = str(g["speaker"])
        seg_total = ge - gs
        total_gt_time += seg_total
        seg_correct = 0.0
        for d in detect_segs:
            ds = float(d.get("group_start", d.get("start", 0.0)))
            de = float(d.get("group_end", d.get("end", 0.0)))
            d_spk = str(d.get("speaker", ""))
            ov = max(0.0, min(ge, de) - max(gs, ds))
            if ov <= 0:
                continue
            # mapping: GT gt_spk 가 어느 detect spk 에 mapping 됐는지 확인
            mapped = confusion.get(gt_spk)
            if mapped is None:
                continue
            if d_spk == mapped:
                seg_correct += ov
        correct_time += seg_correct
        cur = per_spk.setdefault(gt_spk, (0.0, 0.0))
        per_spk[gt_spk] = (cur[0] + seg_correct, cur[1] + seg_total)
    print(f"  total GT time   : {total_gt_time:.2f}s")
    print(f"  correct time    : {correct_time:.2f}s ({100*correct_time/max(total_gt_time,1e-9):.1f}%)")
    print(f"  per-GT-speaker:")
    for spk, (c, t) in sorted(per_spk.items()):
        pct = 100 * c / max(t, 1e-9)
        print(f"    {spk:20s}: {c:6.2f}/{t:6.2f}s = {pct:5.1f}%  (mapped → {confusion.get(spk, 'UNMAPPED')})")

    result = {
        "der_strict_pct": round(der_pct, 2),
        "der_greedy_pct": round(greedy_pct, 2),
        "missed_pct": round(100*der_value['missed detection']/max(der_value['total'],1e-9), 2),
        "false_alarm_pct": round(100*der_value['false alarm']/max(der_value['total'],1e-9), 2),
        "speaker_error_pct": round(100*der_value['confusion']/max(der_value['total'],1e-9), 2),
        "greedy_confusion_pct": round(100*greedy_value['confusion']/max(greedy_value['total'],1e-9), 2),
        "total_gt_speech_sec": round(der_value['total'], 2),
        "segment_level_correct_pct": round(100*correct_time/max(total_gt_time,1e-9), 2),
        "per_gt_speaker_accuracy_pct": {spk: round(100*c/max(t,1e-9), 1)
                                        for spk, (c, t) in per_spk.items()},
        "speaker_mapping": dict(confusion),
        "collar": args.collar,
        "skip_overlap": args.skip_overlap,
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nsaved → {args.out_json}")


if __name__ == "__main__":
    main()
