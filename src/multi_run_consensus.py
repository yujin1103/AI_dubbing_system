"""Multi-run consensus — 여러 run 의 segments 시간 단위 majority vote.

영상 1편을 다른 환경변수로 N회 e2e 돌린 후, 각 시점에 어느 detect SPK 가 가장
자주 등장했는지 majority vote 로 결정. DiariZen non-determinism 완화.

알고리즘:
  1) 모든 runs 의 segments 합집합 timeline (frame_dt=0.01s 해상도)
  2) 각 frame 시점에 각 run 의 SPK 라벨 모음
  3) majority vote — 가장 자주 등장한 SPK 가 그 시점 consensus SPK
  4) 연속 같은 SPK frame 들 → segment 로 묶음 (gap_threshold=0.1s 허용)

영상 무관, hardcoding 없음.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_segments(seg_path: str) -> list[tuple[float, float, str]]:
    data = json.load(open(seg_path, encoding="utf-8"))
    segs = data.get("groups", data.get("segments", []))
    out = []
    for s in segs:
        st = float(s.get("group_start", s.get("start", 0)))
        en = float(s.get("group_end", s.get("end", 0)))
        spk = str(s.get("speaker", ""))
        if en > st and spk:
            out.append((st, en, spk))
    return out


def consensus(seg_paths: list[str], frame_dt: float = 0.05, gap_threshold: float = 0.1) -> list[dict]:
    runs = [_load_segments(p) for p in seg_paths]
    if not runs or not any(runs):
        return []
    max_t = max(en for run in runs for _, en, _ in run)
    n_frames = int(max_t / frame_dt) + 1

    # 각 frame 에 각 run 의 SPK 모음 (None 가능)
    # 시간 효율: per-run interval index
    frame_votes: list[Counter] = [Counter() for _ in range(n_frames)]
    for run in runs:
        # run 안 segments 를 timeline 으로 빠르게 매핑
        for st, en, spk in run:
            i0 = int(st / frame_dt)
            i1 = int(en / frame_dt)
            for i in range(i0, min(i1 + 1, n_frames)):
                frame_votes[i][spk] += 1

    # 각 frame consensus SPK (votes 최대)
    consensus_frames = []
    for i, votes in enumerate(frame_votes):
        if not votes:
            consensus_frames.append(None)
            continue
        spk, _ = votes.most_common(1)[0]
        consensus_frames.append(spk)

    # 연속 같은 SPK frame → segment
    out_segs = []
    cur_spk = None
    cur_start_i = None
    last_i = -10
    for i, spk in enumerate(consensus_frames):
        if spk is None:
            # gap — 현재 segment 닫음
            if cur_spk is not None and i - last_i > 1:
                end_t = (last_i + 1) * frame_dt
                start_t = cur_start_i * frame_dt
                if end_t - start_t >= 0.1:
                    out_segs.append({
                        "group_start": round(start_t, 3),
                        "group_end": round(end_t, 3),
                        "speaker": cur_spk,
                        "from_consensus": True,
                    })
                cur_spk = None
                cur_start_i = None
            continue
        if spk == cur_spk and i - last_i <= int(gap_threshold / frame_dt):
            last_i = i
            continue
        # 새 segment 시작 — 이전 종료
        if cur_spk is not None:
            end_t = (last_i + 1) * frame_dt
            start_t = cur_start_i * frame_dt
            if end_t - start_t >= 0.1:
                out_segs.append({
                    "group_start": round(start_t, 3),
                    "group_end": round(end_t, 3),
                    "speaker": cur_spk,
                    "from_consensus": True,
                })
        cur_spk = spk
        cur_start_i = i
        last_i = i
    # 마지막 segment 종료
    if cur_spk is not None:
        end_t = (last_i + 1) * frame_dt
        start_t = cur_start_i * frame_dt
        if end_t - start_t >= 0.1:
            out_segs.append({
                "group_start": round(start_t, 3),
                "group_end": round(end_t, 3),
                "speaker": cur_spk,
                "from_consensus": True,
            })
    return out_segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("segments", nargs="+", help="2+ segments_*.json paths")
    ap.add_argument("--frame-dt", type=float, default=0.05)
    args = ap.parse_args()
    out = consensus(args.segments, frame_dt=args.frame_dt)
    cnt = Counter(s["speaker"] for s in out)
    print(f"consensus: {len(out)} segments, {len(cnt)} unique SPK: {dict(cnt)}")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"groups": out}, f, ensure_ascii=False, indent=2)
    print(f"saved → {args.out_json}")


if __name__ == "__main__":
    main()
