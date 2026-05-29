"""GT 매칭 best config 자동 sweep.

각 config (main-merge × bg-merge × sim-match) 로 gap_fill 적용 →
GT 시간대 별 detect SPK 일관성 측정 → best score config 선택.

Metric:
  - per_gt_speaker_consistency: 같은 GT 화자 = 같은 detect SPK 비율 (1.0 max)
  - main_count_match: detect 메인 화자 수 == GT main_count → +0.2 bonus
  - bg_detected: BG segment overlap (>=0.4) → +0.1
  - total_score = mean(consistency) + bonuses

사용:
  python sweep_gt_match.py <run_dir> <gt_json>
"""
from __future__ import annotations
import argparse
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def load_segments(run_dir: str):
    """가장 최신 gap_fill 결과 (segments_gapfilled.json) 로드."""
    meta = Path(run_dir) / "meta"
    for p in meta.glob("*_chunk_*_segments_gapfilled.json"):
        d = json.load(open(p))
        return d["groups"]
    raise SystemExit(f"no segments_gapfilled.json in {meta}")


def time_overlap(s1, e1, s2, e2):
    return max(0, min(e1, e2) - max(s1, s2))


def evaluate(detect_segs, gt):
    """
    각 GT segment 시간대에 가장 많이 overlap하는 detect SPK 추출 →
    같은 GT 화자 발화들이 같은 SPK로 매핑되는지 consistency 측정.
    """
    gt_to_spks = defaultdict(list)  # gt_speaker → [detect_spk, ...]
    bg_detected = False
    for g in gt["segments"]:
        spks_in_overlap = []
        for d in detect_segs:
            ds = d.get("group_start", 0)
            de = d.get("group_end", 0)
            ov = time_overlap(g["start"], g["end"], ds, de)
            if ov >= 0.3:
                spks_in_overlap.append((d["speaker"], ov))
        if not spks_in_overlap:
            gt_to_spks[g["speaker"]].append(None)
            continue
        # 가장 많이 overlap한 SPK
        spks_in_overlap.sort(key=lambda x: -x[1])
        best_spk = spks_in_overlap[0][0]
        gt_to_spks[g["speaker"]].append(best_spk)
        if g["speaker"] == "BG":
            bg_detected = True

    # consistency 계산
    consistency_scores = {}
    for gt_spk, detected_list in gt_to_spks.items():
        if gt_spk == "BG":
            # BG는 detect만 됐어도 OK (어떤 SPK로든)
            consistency_scores[gt_spk] = 1.0 if any(d for d in detected_list) else 0.0
            continue
        non_none = [d for d in detected_list if d is not None]
        if not non_none:
            consistency_scores[gt_spk] = 0.0
            continue
        # 가장 빈번한 SPK 비율
        cnt = Counter(non_none)
        most_common_spk, freq = cnt.most_common(1)[0]
        consistency_scores[gt_spk] = freq / len(detected_list)

    # detect 메인 화자 수 (BG_xx 제외)
    detect_spks = set(d["speaker"] for d in detect_segs
                      if not d["speaker"].startswith("SPEAKER_BG"))
    main_match = len(detect_spks) == gt["main_count"]

    avg_consistency = sum(consistency_scores.values()) / max(len(consistency_scores), 1)
    score = avg_consistency + (0.2 if main_match else 0.0) + (0.1 if bg_detected else 0.0)
    return {
        "score": score,
        "avg_consistency": avg_consistency,
        "main_count": len(detect_spks),
        "main_match": main_match,
        "bg_detected": bg_detected,
        "per_gt_consistency": consistency_scores,
        "gt_to_spks": dict(gt_to_spks),
    }


def main(run_dir: str, gt_path: str):
    gt = json.load(open(gt_path, encoding='utf-8'))
    print(f"GT: main={gt['main_count']}, bg={gt['bg_count']}, segments={len(gt['segments'])}")

    main_merge_grid = [0.40, 0.45, 0.50, 0.55, 0.99]
    bg_merge_grid = [0.30, 0.40]
    sim_match_grid = [0.10, 0.30, 0.45, 0.55]
    pad_grid = [0.5]

    results = []
    total = len(main_merge_grid) * len(bg_merge_grid) * len(sim_match_grid) * len(pad_grid)
    i = 0
    for mm, bm, sm, pd in itertools.product(main_merge_grid, bg_merge_grid, sim_match_grid, pad_grid):
        i += 1
        print(f"\n[{i}/{total}] mm={mm} bm={bm} sm={sm} pad={pd}")
        # gap_fill 실행
        r = subprocess.run(
            ["/opt/venv_diarizen/bin/python",
             "/workspace/full_dubbing_pipeline/gap_fill.py", run_dir,
             "--main-merge", str(mm), "--bg-merge", str(bm),
             "--sim-match", str(sm), "--pad", str(pd)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"  fail: {r.stderr[-200:]}")
            continue
        segs = load_segments(run_dir)
        ev = evaluate(segs, gt)
        ev["config"] = {"main_merge": mm, "bg_merge": bm,
                        "sim_match": sm, "pad": pd}
        results.append(ev)
        print(f"  score={ev['score']:.3f} consist={ev['avg_consistency']:.2f} "
              f"main={ev['main_count']}({'✓' if ev['main_match'] else '✗'}) "
              f"bg_detect={ev['bg_detected']}")

    # best
    results.sort(key=lambda x: -x["score"])
    print(f"\n{'='*60}\n=== BEST 5 ===")
    for ev in results[:5]:
        c = ev["config"]
        print(f"\n  score={ev['score']:.3f} | mm={c['main_merge']} bm={c['bg_merge']} "
              f"sm={c['sim_match']} pad={c['pad']}")
        print(f"    consistency: {ev['per_gt_consistency']}")
        print(f"    main={ev['main_count']} ({'✓' if ev['main_match'] else '✗ GT='+str(gt['main_count'])}) "
              f"bg_detect={ev['bg_detected']}")

    # save full results
    out = Path(gt_path).parent / "sweep_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nfull results → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("gt_json")
    args = ap.parse_args()
    main(args.run_dir, args.gt_json)
