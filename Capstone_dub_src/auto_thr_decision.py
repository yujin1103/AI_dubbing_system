"""자동 OUTLIER_FAR_THRESH 결정 알고리즘 (영상 무관, hardcoding 없음).

1차 e2e (default thr=0.40) raw segments 를 분석:
  - n_main (SPEAKER_NN, NN < 90) / n_outlier (SPEAKER_9X)
  - voice diversity (SPK centroid 간 cosine sim 분포)
  - segment 평균 길이

heuristic 결정 (retry 여부 + 새 thr):
  - n_total < 5 (under-split): thr 높임 → 0.50 (더 잘게 분리)
  - n_total > 10 (over-split): thr 낮춤 → 0.30 (덜 잘게)
  - voice 다양성 추정 (sim < 0.5 SPK 수) vs detect 비교
  - main:outlier 비율 적정 (1:1 ~ 1:2) → accept
  - 그 외 → retry

영상 무관, sweep 없이 자동.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/workspace")
from patches.eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model


def _l2(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def analyze_and_decide(run_dir: str) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    # raw segments 자동 탐색
    candidates = sorted(meta.glob("*_chunk_*_segments.json"))
    if not candidates:
        raise SystemExit(f"no raw segments_*.json in {meta}")
    seg_path = candidates[0]
    seg = json.load(open(seg_path, encoding="utf-8"))
    segs = seg.get("groups", seg.get("segments", []))

    # SPK 분포
    spk_counts = Counter(str(s.get("speaker", "")) for s in segs)
    real_spks = {s: c for s, c in spk_counts.items()
                 if s and not s.startswith("SPEAKER_BG")}
    main_spks = {s: c for s, c in real_spks.items() if not s.startswith("SPEAKER_9")}
    outlier_spks = {s: c for s, c in real_spks.items() if s.startswith("SPEAKER_9")}
    n_main = len(main_spks)
    n_outlier = len(outlier_spks)
    n_total = n_main + n_outlier

    # voice diversity 계산
    voice_div = 0
    distinct_count = 0
    if real_spks:
        vocals_glob = list((rd / "vocals").glob("*_clean_vocals.wav"))
        if vocals_glob:
            audio, sr = sf.read(str(vocals_glob[0]))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            _ = get_eres2netv2_model()
            centroids = {}
            for sp in real_spks:
                # longest segment of sp
                sp_segs = [s for s in segs if str(s.get("speaker", "")) == sp]
                if not sp_segs:
                    continue
                best = max(sp_segs, key=lambda s: float(s.get("group_end", s.get("end", 0))) - float(s.get("group_start", s.get("start", 0))))
                ss = float(best.get("group_start", best.get("start", 0)))
                se = float(best.get("group_end", best.get("end", 0)))
                i0 = int(max(0, (ss - 0.1) * sr))
                i1 = int(min(len(audio), (se + 0.1) * sr))
                if i1 <= i0:
                    continue
                try:
                    emb = extract_eres2netv2_emb(audio[i0:i1], sr)
                    centroids[sp] = _l2(np.asarray(emb, dtype=np.float32))
                except Exception:
                    continue
            # pair sim < 0.5 = different person (rough)
            sp_list = sorted(centroids.keys())
            if len(sp_list) >= 2:
                sims = []
                for i, sa in enumerate(sp_list):
                    for sb in sp_list[i + 1:]:
                        sims.append(float(np.dot(centroids[sa], centroids[sb])))
                if sims:
                    voice_div = round(1 - np.mean(sims), 3)  # higher = more diverse
                    distinct_count = sum(1 for s in sims if s < 0.5)

    stats = {
        "n_total": n_total,
        "n_main": n_main,
        "n_outlier": n_outlier,
        "voice_diversity": voice_div,
        "distinct_pairs_lt_0.5": distinct_count,
        "n_segments": len(segs),
    }

    # 결정
    if n_total < 5:
        action = "retry_higher_thr"
        new_thr = 0.50
        reason = f"n_total={n_total} too few → raise thr"
    elif n_total > 12:
        action = "retry_lower_thr"
        new_thr = 0.30
        reason = f"n_total={n_total} too many → lower thr"
    elif n_outlier > 2 * n_main and n_main <= 2:
        action = "retry_lower_thr"
        new_thr = 0.30
        reason = f"outlier {n_outlier} >> main {n_main} → lower thr"
    elif n_main <= 4 and voice_div >= 0.4:
        action = "retry_higher_thr"
        new_thr = 0.50
        reason = f"main={n_main} but voice_div={voice_div} suggests more SPKs"
    else:
        action = "accept"
        new_thr = 0.40
        reason = f"n_total={n_total} main={n_main} outlier={n_outlier} OK"

    decision = {"action": action, "thr": new_thr, "reason": reason}
    return {"stats": stats, "decision": decision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out-json", help="저장 경로")
    args = ap.parse_args()
    result = analyze_and_decide(args.run_dir)
    print(f"=== Raw stats ===")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
    print(f"\n=== Decision ===")
    for k, v in result["decision"].items():
        print(f"  {k}: {v}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
