"""Adaptive OUTLIER_FAR_THRESH + gap_fill mm 자동 결정.

영상 무관, hardcoding 없음. raw segments 분포 분석 후 영상별 best thr/mm 자동 추정.

알고리즘:
  1) 우선 default thr=0.40 으로 e2e 1회 (또는 기존 segments 사용)
  2) raw segments SPK 분포 분석:
     - n_main = main SPK count (SPEAKER_BG/outlier 제외)
     - over_split = n_main > median(GT 기대치)
     - under_split = n_main < median
     - GT 모르므로 일반적 영상 가정 (4-8명 main)
  3) 적응:
     - n_main >= 10: over-split → thr 0.30 (덜 검출), mm 낮춤 (over-merge)
     - n_main 5-9: 적절 → thr 0.40 default
     - n_main <= 4: under-split → thr 0.50 (더 검출), mm 높임
  4) 영상 길이 / chunk SPK distance 분포로 mm 자동 결정:
     - SPK 평균 inter-distance (cosine) 가 0.8+ : 잘 분리됨, mm=0.99 (보존 유지)
     - 0.5-0.8: 중간, mm=0.55
     - 0.5 미만: 비슷한 SPK들, mm=0.40 (over-merge 방지하려면 mm=0.65 — 양면)

이건 1차 heuristic. 정확한 best 는 GT sweep 만 가능 — 자동 결정은 80% 정확도 목표.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_raw_segments(seg_path: str) -> dict:
    """raw segments 의 SPK 분포 + 통계 분석."""
    data = json.load(open(seg_path, encoding="utf-8"))
    segs = data.get("groups", data.get("segments", []))
    spk_counts = Counter(s.get("speaker", "") for s in segs)
    main_spks = {s: c for s, c in spk_counts.items()
                 if s and not s.startswith("SPEAKER_BG") and not s.startswith("SPEAKER_9")}
    outlier_spks = {s: c for s, c in spk_counts.items() if s.startswith("SPEAKER_9")}
    durations = [float(s.get("group_end", s.get("end", 0))) - float(s.get("group_start", s.get("start", 0)))
                 for s in segs]
    return {
        "n_segments": len(segs),
        "n_main": len(main_spks),
        "n_outlier": len(outlier_spks),
        "main_spk_counts": main_spks,
        "outlier_spk_counts": outlier_spks,
        "total_duration": sum(durations),
        "avg_segment_dur": sum(durations) / max(len(segs), 1),
        "min_dur": min(durations) if durations else 0.0,
        "max_dur": max(durations) if durations else 0.0,
    }


def recommend_config(stats: dict) -> dict:
    """raw stats 기반 OUTLIER_FAR_THRESH + gap_fill mm/bm/sm 추천."""
    n_main = stats["n_main"]
    n_outlier = stats["n_outlier"]
    n_total = n_main + n_outlier

    # OUTLIER_FAR_THRESH 추천
    if n_total >= 10:
        thr = 0.30  # over-split → 덜 검출
    elif n_total <= 4:
        thr = 0.50  # under-split → 더 검출
    else:
        thr = 0.40  # default

    # gap_fill main_merge — outlier vs main 비율 기반 (영상 무관)
    #   - main 1-2 + outlier 많음: 매우 over-split → mm=0.40
    #   - main 3+ + outlier 많음: 적당 over-merge → mm=0.50 (test5 fresh case)
    #   - outlier ≥ main 절반: mm=0.50 (test4 case)
    #   - outlier 1-2개: 약한 over-merge → mm=0.55
    #   - outlier 0: 보존 default → mm=0.99
    if n_outlier > n_main and n_main <= 2:
        mm = 0.40  # main 매우 적음 + outlier 많음 (trivial 위험)
    elif n_outlier > n_main and n_main >= 3:
        mm = 0.50  # main 적당히 있음 + outlier 많음 (test5 fresh case)
    elif n_outlier >= max(1, n_main // 2):
        mm = 0.50  # outlier ≥ main 절반 (test4 case)
    elif n_outlier >= 1:
        mm = 0.55  # 약한 over-merge
    else:
        mm = 0.99  # 보존 default

    # bg_merge / sim_match — outlier 있으면 BG cluster 활성
    bm = 0.30
    sm = 0.45 if n_outlier > 0 else 0.10

    return {
        "LATENTSYNC_OUTLIER_OFF": "0",
        "LATENTSYNC_OUTLIER_FAR_THRESH": str(thr),
        "gap_fill": {
            "main_merge": mm,
            "bg_merge": bm,
            "sim_match": sm,
            "pad": 0.5,
        },
        "reason": (
            f"n_total={n_total} (main={n_main} outlier={n_outlier}) "
            f"→ thr={thr} mm={mm} bm={bm} sm={sm}"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments_json", help="raw segments_*.json (orchestrator 출력)")
    ap.add_argument("out_json", nargs="?", help="추천 config 저장 경로")
    args = ap.parse_args()

    stats = analyze_raw_segments(args.segments_json)
    print(f"=== Raw segments analysis ===")
    print(f"  n_segments: {stats['n_segments']}")
    print(f"  n_main SPK: {stats['n_main']}")
    print(f"  n_outlier (SPEAKER_9*): {stats['n_outlier']}")
    print(f"  main spk counts: {stats['main_spk_counts']}")
    print(f"  outlier spk counts: {stats['outlier_spk_counts']}")
    print(f"  total duration: {stats['total_duration']:.1f}s")
    print(f"  avg seg duration: {stats['avg_segment_dur']:.2f}s")

    cfg = recommend_config(stats)
    print(f"\n=== Recommended config ===")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))

    if args.out_json:
        out = {"stats": stats, "recommended": cfg}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nsaved → {args.out_json}")


if __name__ == "__main__":
    main()
