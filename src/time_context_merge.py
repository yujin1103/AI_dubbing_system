"""시간 context 기반 SPK merge — 짧은 outlier SPK 가 시간대 인접 main SPK 와 같은 사람일 가능성.

원리:
  - segments 시간 순서에서 (main SPK A) → (짧은 outlier SPK B) → (main SPK A) 패턴
  - 즉 main 사이에 끼인 짧은 outlier 가 sandwich 되어 있으면 → 같은 사람의 emotion shift 로 추정
  - 외침/감정 폭발 시 acoustic shift 로 detect 가 다른 SPK 로 분리한 경우

조건 (영상 무관 default):
  - outlier SPK segments 가 적음 (전체 ≤ 3 segments)
  - 시간 인접 (≤ 5초 내 같은 main SPK 발화 있음)
  - 발화 길이 짧음 (≤ 2초)
  - 양쪽 모두 같은 main SPK 면 강하게 merge (sandwich)
  - 한쪽만 같으면 약하게 merge (edge)

입력: segments_*.json (gap_fill 결과)
출력: segments_time_merged.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def time_context_merge(
    seg_path: str,
    out_path: str,
    *,
    outlier_total_max: int = 3,
    short_dur_max: float = 2.0,
    inter_gap_max: float = 5.0,
    sandwich_only: bool = False,
) -> dict:
    data = json.load(open(seg_path, encoding="utf-8"))
    segs = data.get("groups", data.get("segments", []))
    # sort by time
    segs = sorted(segs, key=lambda s: float(s.get("group_start", s.get("start", 0))))

    # 각 SPK 의 segments count
    spk_counts = Counter(str(s.get("speaker", "")) for s in segs)
    # outlier 후보: SPEAKER_BG 제외 + segments ≤ outlier_total_max
    outlier_spks = {
        spk for spk, cnt in spk_counts.items()
        if cnt <= outlier_total_max
        and not spk.startswith("SPEAKER_BG")
        and spk
    }

    n_merge = 0
    for i, seg in enumerate(segs):
        spk = str(seg.get("speaker", ""))
        if spk not in outlier_spks:
            continue
        dur = float(seg.get("group_end", seg.get("end", 0))) - float(seg.get("group_start", seg.get("start", 0)))
        if dur > short_dur_max:
            continue
        # 양쪽 인접 main SPK (≤ inter_gap_max 거리)
        cur_start = float(seg.get("group_start", seg.get("start", 0)))
        cur_end = float(seg.get("group_end", seg.get("end", 0)))
        prev_spk = None
        next_spk = None
        # prev: 가장 가까운 이전 segment with non-outlier SPK
        for j in range(i - 1, -1, -1):
            ps = str(segs[j].get("speaker", ""))
            if ps not in outlier_spks and ps and not ps.startswith("SPEAKER_BG"):
                pe = float(segs[j].get("group_end", segs[j].get("end", 0)))
                if cur_start - pe <= inter_gap_max:
                    prev_spk = ps
                break
        # next
        for j in range(i + 1, len(segs)):
            ns = str(segs[j].get("speaker", ""))
            if ns not in outlier_spks and ns and not ns.startswith("SPEAKER_BG"):
                nstart = float(segs[j].get("group_start", segs[j].get("start", 0)))
                if nstart - cur_end <= inter_gap_max:
                    next_spk = ns
                break

        # sandwich: prev == next == main → outlier 를 main 으로 reassign
        # edge: 한쪽만 main 매칭 → reassign (sandwich_only=False 면 적용)
        target = None
        if prev_spk and next_spk and prev_spk == next_spk:
            target = prev_spk
        elif not sandwich_only:
            if prev_spk and not next_spk:
                target = prev_spk
            elif next_spk and not prev_spk:
                target = next_spk
            elif prev_spk and next_spk:
                # 양쪽 다른 main — 시간 더 가까운 쪽 선택
                pe = float(segs[i-1].get("group_end", segs[i-1].get("end", 0))) if i > 0 else 0
                nstart = float(segs[i+1].get("group_start", segs[i+1].get("start", 0))) if i < len(segs)-1 else 1e9
                target = prev_spk if (cur_start - pe) <= (nstart - cur_end) else next_spk
        if target and target != spk:
            seg["audio_speaker_orig"] = spk
            seg["speaker"] = target
            seg["from_time_context_merge"] = True
            n_merge += 1
            print(f"  MERGE [{cur_start:.2f}-{cur_end:.2f}] {spk} → {target} (prev={prev_spk}, next={next_spk})")

    # 마지막 consecutive same-SPK merge
    merged = []
    for s in segs:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            prev_end = float(merged[-1].get("group_end", merged[-1].get("end", 0)))
            cur_start = float(s.get("group_start", s.get("start", 0)))
            if cur_start - prev_end <= 0.1:
                merged[-1]["group_end"] = s.get("group_end", s.get("end"))
                merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
                continue
        merged.append(dict(s))

    data["groups"] = merged
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  [✓] {len(segs)} → {len(merged)} segments ({n_merge} time-context merges)")
    return {"input": len(segs), "output": len(merged), "merges": n_merge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("output_json")
    ap.add_argument("--outlier-total-max", type=int, default=3)
    ap.add_argument("--short-dur-max", type=float, default=2.0)
    ap.add_argument("--inter-gap-max", type=float, default=5.0)
    ap.add_argument("--sandwich-only", action="store_true")
    args = ap.parse_args()
    s = time_context_merge(
        args.input_json,
        args.output_json,
        outlier_total_max=args.outlier_total_max,
        short_dur_max=args.short_dur_max,
        inter_gap_max=args.inter_gap_max,
        sandwich_only=args.sandwich_only,
    )
    print(f"summary: {s}")


if __name__ == "__main__":
    main()
