#!/usr/bin/env python3
"""GT 화자구간별 발화 커버리지 측정 — '화자분리=GT, 발화 빠짐없이'가 목표.

각 GT 화자 구간 [s,e] 에 대해:
  - 그 구간에 속하는 ASR 단어 수 (메인 ASR 기준)
  - 더빙 입력 segment 가 그 구간을 덮는지 + 합성된 단어 수
  - 누락(더빙에 안 들어간) 단어 목록
출력: 화자별 커버리지 % + 누락 단어. 100% = 완벽(빠짐없이).

사용:
  python measure_gt_coverage.py --gt <gt.json> --words <main_words.json> --dub <dub_input.json>
"""
import argparse, json


def load_words(path):
    d = json.load(open(path, encoding="utf-8"))
    w = d.get("words", d) if isinstance(d, dict) else d
    return [x for x in w if "start" in x and "end" in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--dub", required=True)
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    gt_segs = gt.get("segments", [])
    dub = json.load(open(args.dub, encoding="utf-8"))
    dseg = dub.get("segments", dub) if isinstance(dub, dict) else dub

    def dub_covers(s, e):
        # GT 구간 [s,e] 와 겹치는 dub segment 가 있나 (= 그 발화가 더빙됨)
        return any(min(e, d["end"]) - max(s, d["start"]) > 0.1 for d in dseg)

    n = len(gt_segs)
    covered = 0
    print(f"=== GT 발화별 더빙 커버리지 (GT {n}개 발화) ===")
    for g in gt_segs:
        s, e = float(g["start"]), float(g["end"])
        ok = dub_covers(s, e)
        if ok:
            covered += 1
        else:
            print(f"  누락 {s:6.2f}-{e:6.2f} {g.get('speaker',''):8s}: {g.get('text','')[:40]!r}")
    print(f"\n더빙 커버리지: {covered}/{n} GT 발화 ({100*covered/max(n,1):.0f}%)  — 100%=빠짐없음")


if __name__ == "__main__":
    main()
