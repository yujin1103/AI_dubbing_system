#!/usr/bin/env python3
"""GT-free diarization quality metrics.

영상별 ground-truth가 없는 자동 더빙 시스템 평가용. segments.json만 입력으로 받아
self-consistency / 회귀 감지 신호를 출력. 어떤 영상에든 동일하게 적용 가능.

사용:
    python3 eval_diarization_auto.py <segments_json> [<segments_json> ...]

여러 segments.json을 주면 metric을 나란히 비교 (v26 vs v27 vs v28 회귀 감지에 유용).

신호 (모두 GT 없이 계산 가능):
  1. 화자별 group 수             — over-split 후보 (특정 화자가 비정상적으로 많이 등장)
  2. 화자별 평균 group duration  — 너무 짧으면 boundary 불안정
  3. selfref 사용률              — profile 부족 (높을수록 합성 quality 위험)
  4. 화자별 dominant gap         — 인접 같은 화자 group 사이의 평균 gap
  5. 라벨 fragmentation index     — sentence 내부에서 라벨이 자주 바뀌는 비율
  6. 시간대별 화자 변화 빈도      — 격렬한 대화 vs 가짜 변화 구분
  7. speed/stretch anomaly        — emotion이 아닌데 speed != 1.0 인 segment
  8. min/max group duration       — 극단치 (≤0.5s, ≥15s)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _is_selfref(ref_path: str) -> bool:
    if not ref_path:
        return True  # 빈 경로도 reference 못 만든 것
    name = Path(ref_path).name
    return name.startswith("selfref_") or "/tmp/" in ref_path


def compute_metrics(data: dict) -> dict[str, Any]:
    groups = data.get("groups", [])
    if not groups:
        return {"_empty": True}

    speakers = sorted(set(g["speaker"] for g in groups))
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        by_spk[g["speaker"]].append(g)

    # 1. group count per speaker
    group_counts = {spk: len(by_spk[spk]) for spk in speakers}

    # 2. mean/min/max group duration per speaker
    duration_stats = {}
    for spk in speakers:
        durs = [g["group_end"] - g["group_start"] for g in by_spk[spk]]
        duration_stats[spk] = {
            "mean": round(sum(durs) / len(durs), 2),
            "min": round(min(durs), 2),
            "max": round(max(durs), 2),
            "total": round(sum(durs), 2),
        }

    # 3. selfref usage
    selfref_n = sum(1 for g in groups if _is_selfref(g.get("ref_path", "")))
    selfref_by_spk = {
        spk: sum(1 for g in by_spk[spk] if _is_selfref(g.get("ref_path", "")))
        for spk in speakers
    }

    # 4. dominant gap per speaker — 같은 화자 인접 group 사이 평균 gap
    spk_gaps = {}
    for spk in speakers:
        sorted_g = sorted(by_spk[spk], key=lambda g: g["group_start"])
        gaps = []
        for i in range(1, len(sorted_g)):
            gap = sorted_g[i]["group_start"] - sorted_g[i - 1]["group_end"]
            if gap >= 0:
                gaps.append(gap)
        if gaps:
            spk_gaps[spk] = {
                "n_adjacent": len(gaps),
                "min_gap": round(min(gaps), 2),
                "mean_gap": round(sum(gaps) / len(gaps), 2),
                "max_gap": round(max(gaps), 2),
                # 0.5초 이내 인접: 잘못 쪼개졌을 가능성 높음
                "n_close_lt_0.5s": sum(1 for g in gaps if g < 0.5),
                "n_close_lt_1.0s": sum(1 for g in gaps if g < 1.0),
                "n_close_lt_2.0s": sum(1 for g in gaps if g < 2.0),
            }

    # 5. label fragmentation index — segment_ids 공유 여부로 측정
    # 같은 segment_id가 여러 group에 걸쳐 있으면, 한 sentence가 화자별로 쪼개진 것
    seg_to_groups = defaultdict(list)
    for g in groups:
        for sid in g.get("segment_ids", []):
            seg_to_groups[sid].append(g["speaker"])
    fragmented_segments = []
    for sid, spks in seg_to_groups.items():
        unique = list(dict.fromkeys(spks))  # 순서 보존 dedup
        if len(unique) > 1:
            fragmented_segments.append({"segment_id": sid, "speakers": unique})

    # 6. global speaker transitions
    sorted_all = sorted(groups, key=lambda g: g["group_start"])
    transitions = 0
    rapid_transitions = 0  # < 1s 안에서 화자 바뀜
    for i in range(1, len(sorted_all)):
        if sorted_all[i]["speaker"] != sorted_all[i - 1]["speaker"]:
            transitions += 1
            gap = sorted_all[i]["group_start"] - sorted_all[i - 1]["group_end"]
            if gap < 1.0:
                rapid_transitions += 1

    # 7. extreme group durations
    extreme_short = [
        {
            "spk": g["speaker"],
            "start": g["group_start"],
            "end": g["group_end"],
            "dur": round(g["group_end"] - g["group_start"], 2),
            "text": (g.get("text") or "")[:30],
        }
        for g in groups
        if (g["group_end"] - g["group_start"]) < 0.5
    ]
    extreme_long = [
        {
            "spk": g["speaker"],
            "start": g["group_start"],
            "end": g["group_end"],
            "dur": round(g["group_end"] - g["group_start"], 2),
        }
        for g in groups
        if (g["group_end"] - g["group_start"]) > 15.0
    ]

    # 8. speed anomaly — emotion이 Neutral인데 speed != 1.0
    speed_anomaly = sum(
        1
        for g in groups
        if g.get("emotion") == "Neutral" and abs(g.get("speed", 1.0) - 1.0) > 0.01
    )

    # 9. spurious speaker 후보 — 총 발화 시간이 1.5초 미만
    spurious_candidates = [
        spk for spk in speakers if duration_stats[spk]["total"] < 1.5
    ]

    return {
        "n_groups_total": len(groups),
        "n_speakers": len(speakers),
        "speakers": speakers,
        "group_counts": group_counts,
        "duration_stats": duration_stats,
        "selfref_usage": {
            "total": selfref_n,
            "ratio": round(selfref_n / len(groups), 3),
            "by_speaker": selfref_by_spk,
        },
        "speaker_gaps": spk_gaps,
        "fragmented_segments": {
            "count": len(fragmented_segments),
            "examples": fragmented_segments[:10],
        },
        "global_transitions": {
            "total": transitions,
            "rapid_lt_1s": rapid_transitions,
        },
        "extreme_durations": {
            "short_lt_0.5s": extreme_short,
            "long_gt_15s": extreme_long,
        },
        "speed_anomaly_count": speed_anomaly,
        "spurious_speaker_candidates": spurious_candidates,
    }


def print_summary(name: str, m: dict[str, Any]) -> None:
    if m.get("_empty"):
        print(f"[{name}] empty segments")
        return
    print(f"=== {name} ===")
    print(f"  groups: {m['n_groups_total']}  speakers: {m['n_speakers']} {m['speakers']}")
    print(f"  group_counts: {m['group_counts']}")
    sr = m["selfref_usage"]
    print(f"  selfref: {sr['total']}/{m['n_groups_total']} ({sr['ratio']:.1%})  by_speaker: {sr['by_speaker']}")
    print(f"  transitions: total={m['global_transitions']['total']}  rapid<1s={m['global_transitions']['rapid_lt_1s']}")
    print(f"  fragmented (sentence split across speakers): {m['fragmented_segments']['count']}")
    if m["fragmented_segments"]["examples"]:
        for ex in m["fragmented_segments"]["examples"]:
            print(f"    seg{ex['segment_id']}: {ex['speakers']}")
    print(f"  speed_anomaly (Neutral but speed!=1.0): {m['speed_anomaly_count']}")
    if m["spurious_speaker_candidates"]:
        print(f"  ⚠️  spurious candidates (<1.5s total): {m['spurious_speaker_candidates']}")
    if m["extreme_durations"]["short_lt_0.5s"]:
        print(f"  ⚠️  short <0.5s: {len(m['extreme_durations']['short_lt_0.5s'])} groups")
    print("  per-speaker dominant gaps:")
    for spk, gs in m["speaker_gaps"].items():
        print(
            f"    {spk}: n_adj={gs['n_adjacent']} "
            f"gaps min/mean/max={gs['min_gap']}/{gs['mean_gap']}/{gs['max_gap']}s "
            f"close<0.5s/<1s/<2s={gs['n_close_lt_0.5s']}/{gs['n_close_lt_1.0s']}/{gs['n_close_lt_2.0s']}"
        )


def compare(metrics_list: list[tuple[str, dict]]) -> None:
    """여러 run 비교 — 핵심 metric만 추출해서 표 형식."""
    if len(metrics_list) < 2:
        return
    print("\n=== Comparison ===")
    headers = ["metric"] + [n for n, _ in metrics_list]
    rows = []

    def _row(label, getter):
        return [label] + [str(getter(m)) for _, m in metrics_list]

    rows.append(_row("n_groups", lambda m: m.get("n_groups_total", "-")))
    rows.append(_row("n_speakers", lambda m: m.get("n_speakers", "-")))
    rows.append(_row("selfref %", lambda m: f"{m.get('selfref_usage', {}).get('ratio', 0):.1%}"))
    rows.append(_row("fragmented sent", lambda m: m.get("fragmented_segments", {}).get("count", "-")))
    rows.append(_row("rapid transitions <1s", lambda m: m.get("global_transitions", {}).get("rapid_lt_1s", "-")))
    rows.append(_row("short <0.5s", lambda m: len(m.get("extreme_durations", {}).get("short_lt_0.5s", []))))
    rows.append(_row("spurious cand", lambda m: len(m.get("spurious_speaker_candidates", []))))

    # 화자별 group count (모든 화자 합집합)
    all_spk = sorted(set().union(*(set(m.get("group_counts", {}).keys()) for _, m in metrics_list)))
    for spk in all_spk:
        rows.append(_row(f"  {spk} groups", lambda m, s=spk: m.get("group_counts", {}).get(s, 0)))

    widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def main():
    if len(sys.argv) < 2:
        print("Usage: eval_diarization_auto.py <segments_json> [<segments_json> ...]")
        sys.exit(1)
    metrics_list = []
    for path in sys.argv[1:]:
        try:
            data = _load(path)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        m = compute_metrics(data)
        name = Path(path).stem
        print_summary(name, m)
        print()
        metrics_list.append((name, m))
    compare(metrics_list)


if __name__ == "__main__":
    main()
