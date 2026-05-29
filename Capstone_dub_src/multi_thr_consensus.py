"""multi-thr 합의 — thr 0.30/0.40/0.50 e2e 결과 비교 후 best thr 자동 채택.

영상 무관 (hardcoding 없음). 각 thr 별 e2e 결과 분석:
  - raw SPK 분포: n_main / n_outlier
  - mm 추천 (adaptive_thr) 적용 후 main_count
  - GT 모르므로 main_count plateau / 최대 일관성으로 best 추정

알고리즘:
  1) thr 별 raw stats 수집 (n_main, n_outlier)
  2) adaptive mm 추천 적용 후 main_count + bg_detected 계산 (gap_fill 시뮬레이션 X — 실제 적용 후 측정)
  3) plateau detection:
     - thr 변화에도 main_count 일정 → plateau 영역
     - plateau 의 mid thr 채택 (안정)
  4) plateau 없으면 main_count 최대 + bg_detected=True 우선

영상 무관 default thr 후보: [0.30, 0.40, 0.50]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_run(run_dir: str, segments_pattern: str = "*_chunk_*_segments.json") -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    # raw segments_*.json — exact suffix _segments.json (post-process 결과 제외)
    import re
    all_candidates = sorted(meta.glob("*_chunk_*_segments*.json"))
    candidates = [p for p in all_candidates
                  if re.search(r"_chunk_\d+_segments\.json$", p.name)]
    if not candidates:
        return {"main_count": 0, "bg_detected": False, "n_segments": 0}
    seg_path = candidates[0]
    print(f"  thr {run_dir}: using {seg_path.name}")
    data = json.load(open(seg_path, encoding="utf-8"))
    segs = data.get("groups", data.get("segments", []))
    spk_counts = Counter(str(s.get("speaker", "")) for s in segs)
    main_spks = {s for s in spk_counts if s and not s.startswith("SPEAKER_BG") and not s.startswith("SPEAKER_9")}
    outlier_spks = {s for s in spk_counts if s.startswith("SPEAKER_9")}
    bg = any(str(s).startswith("SPEAKER_BG") for s in spk_counts)
    return {
        "main_count": len(main_spks),
        "outlier_count": len(outlier_spks),
        "n_total_spk": len(main_spks) + len(outlier_spks),
        "bg_detected": bg,
        "n_segments": len(segs),
        "spk_counts": dict(spk_counts),
    }


def consensus(thr_to_run_dir: dict[float, str]) -> dict:
    """thr → run_dir 매핑 → 자동 best thr 결정."""
    results = {}
    for thr, run_dir in thr_to_run_dir.items():
        results[thr] = analyze_run(run_dir)
    print(f"\n=== Multi-thr analysis ===")
    for thr in sorted(results):
        r = results[thr]
        print(f"  thr={thr}: main={r['main_count']} bg={r['bg_detected']} segs={r['n_segments']}")

    # plateau detection — 연속 thr 들의 main_count 같으면 plateau
    sorted_thrs = sorted(results.keys())
    plateaus = []  # [(start_thr, end_thr, main_count)]
    cur_main = None
    cur_start = None
    for thr in sorted_thrs:
        m = results[thr]["main_count"]
        if cur_main is None:
            cur_main = m
            cur_start = thr
        elif m == cur_main:
            continue  # plateau 연장
        else:
            plateaus.append((cur_start, thr, cur_main))
            cur_main = m
            cur_start = thr
    if cur_main is not None:
        plateaus.append((cur_start, sorted_thrs[-1], cur_main))

    # 결정: main_count 최대 + bg_detected=True 우선 (BG 검출 가능하면 더 정확)
    best_thr = None
    best_main = -1
    best_bg = False
    for thr, r in results.items():
        score = r["main_count"] + (0.5 if r["bg_detected"] else 0)
        if score > best_main + (0.5 if best_bg else 0):
            best_main = r["main_count"]
            best_bg = r["bg_detected"]
            best_thr = thr
        elif r["main_count"] == best_main and r["bg_detected"] == best_bg:
            # plateau — 더 낮은 thr 선택 (안정성)
            if thr < best_thr:
                best_thr = thr

    decision = {
        "best_thr": best_thr,
        "best_main_count": best_main,
        "best_bg_detected": best_bg,
        "plateaus": [(round(s, 2), round(e, 2), m) for s, e, m in plateaus],
        "all_results": {str(thr): r for thr, r in results.items()},
    }
    print(f"\n=== Decision ===")
    print(f"  plateaus: {decision['plateaus']}")
    print(f"  → best thr = {best_thr} (main={best_main}, bg={best_bg})")
    return decision


def main():
    ap = argparse.ArgumentParser(description="multi-thr 합의 — thr 별 run_dir → best thr 자동 채택")
    ap.add_argument("--thr-run", nargs="+", required=True,
                    help="thr:run_dir 매핑, 예: 0.30:/path/to/run030 0.40:/path/to/run040 0.50:/path/to/run050")
    ap.add_argument("--out-json", help="결과 저장 경로")
    args = ap.parse_args()
    mapping = {}
    for kv in args.thr_run:
        thr_s, rd = kv.split(":", 1)
        mapping[float(thr_s)] = rd
    decision = consensus(mapping)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        print(f"\nsaved → {args.out_json}")


if __name__ == "__main__":
    main()
