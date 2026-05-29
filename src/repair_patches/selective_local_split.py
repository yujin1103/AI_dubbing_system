"""Selective local fusion — base(DiariZen+NeMo) 위에서 pyannote의 *분할만* 국소 채택.

문제:
  단일 모델로 uniform 정답 불가. test4 = DiariZen 6✓/pyannote 3✗,
  test5 = DiariZen 3✗/pyannote 4✓. pyannote raw 통째 교체는 다른 화자 품질을
  떨어뜨려 net 하락(검증됨). → base 유지하되, pyannote가 한 base 화자 구간을
  ≥2명으로 *나눈* 부분만 국소적으로 채택.

비대칭 규칙 (영상 무관, hardcoding 없음 — 화자 수 강제 X):
  - 한 base 화자 B의 시간 구간 안에서 pyannote가 ≥2명을 충분히(각 dur≥MIN_DUR,
    비율≥MIN_FRAC) 담당하면 → B를 pyannote 경계로 분할 (dominant pyannote 화자는
    B 유지, 나머지는 새 라벨 B__sN).
  - pyannote가 더 coarse하거나(여러 base를 하나로 묶음) 한 명만 담당하면 → B 그대로.
  → pyannote의 *split*만 채택하고 *merge*는 무시 → base보다 화자가 줄지 않음.
    test4(pyannote coarser)는 변화 없음, test5(pyannote가 외침 분리)는 4번째 회복.

I/O:
  입력  meta/<chunk>_segments.json (base = fusion 8918 raw, groups 형식)
  pyannote = 8943 daemon 호출 (vocals/<chunk>_clean_vocals.wav)
  출력  meta/<chunk>_segments_localsplit.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import requests

PYANNOTE_URL = "http://127.0.0.1:8943/diarize"
STEP = 0.02            # grid 해상도 (sec)
MIN_DUR = 0.5          # base 안에서 한 pyannote 화자가 담당해야 할 최소 길이
MIN_FRAC = 0.18        # base 총길이 대비 최소 비율 (둘 다 만족해야 "강한" 분할 후보)


def _segs_of(data):
    return data.get("groups", data.get("segments", []))


def _start(s):
    return float(s.get("group_start", s.get("start", 0.0)))


def _end(s):
    return float(s.get("group_end", s.get("end", 0.0)))


def _grid(segs, n, step):
    g = [None] * n
    for s in segs:
        a = int(_start(s) / step)
        b = int(_end(s) / step)
        for i in range(max(0, a), min(n, b)):
            g[i] = str(s.get("speaker", ""))
    return g


def _pyannote(vocals: str) -> list:
    r = requests.post(PYANNOTE_URL, json={"vocals_wav": vocals, "num_speakers": None,
                                          "min_duration": 0.3}, timeout=600)
    d = r.json()
    if not d.get("success", True) and not d.get("segments"):
        raise SystemExit(f"pyannote failed: {d.get('error')}")
    return [{"start": float(x["start"]), "end": float(x["end"]), "speaker": str(x["speaker"])}
            for x in d.get("segments", [])]


def selective_local_split(base_segs: list, pyan_segs: list, *,
                          step: float = STEP, min_dur: float = MIN_DUR,
                          min_frac: float = MIN_FRAC, confine_frac: float = 0.60):
    dur = max([_end(s) for s in base_segs] + [s["end"] for s in pyan_segs] + [0.0])
    n = int(dur / step) + 1
    bg = _grid(base_segs, n, step)
    pg = _grid(pyan_segs, n, step)
    log = []

    def _main(labels):
        return {l for l in labels if l and not str(l).startswith("SPEAKER_BG")}

    base_n = len(_main(set(bg)))
    pyan_n = len(_main(set(pg)))

    def _passthrough():
        # split 없음 → base 원본 그대로 (grid 양자화로 경계 훼손 방지)
        return [dict(s, group_idx=j) for j, s in enumerate(base_segs)], log

    # 전역 게이트: pyannote 가 base 보다 화자를 더 찾을 때만 split (coarse pyannote 무시)
    if pyan_n <= base_n:
        log.append(f"  no-op: pyannote n={pyan_n} <= base n={base_n} (pyannote가 더 coarse/동등)")
        return _passthrough()
    else:
        # pyannote 화자 P → frame별 base 분포 (confinement 계산)
        pyan_base: dict[str, Counter] = defaultdict(Counter)  # P -> Counter(base)
        for i in range(n):
            p = pg[i]
            if not p:
                continue
            b = bg[i]
            if b and not b.startswith("SPEAKER_BG"):
                pyan_base[p][b] += 1
        # 각 P 를 dominant base 에 귀속 + confinement 검사
        base_to_pyans: dict[str, list] = defaultdict(list)  # B -> [(P, overlap_frames)]
        for p, cnt in pyan_base.items():
            tot = sum(cnt.values())
            if tot == 0:
                continue
            b_dom, ovl = cnt.most_common(1)[0]
            conf = ovl / tot
            if conf >= confine_frac and ovl * step >= min_dur:
                base_to_pyans[b_dom].append((p, ovl))
        # confined pyannote 화자가 ≥2 인 base 만 분할
        relabel: dict[str, dict[str, str]] = {}
        for b, plist in base_to_pyans.items():
            if len(plist) < 2:
                continue
            plist.sort(key=lambda x: -x[1])
            mapping = {p: (b if k == 0 else f"{b}__s{k}") for k, (p, _o) in enumerate(plist)}
            relabel[b] = mapping
            log.append(f"  split base {b}: confined pyannote "
                       f"{[(p, round(o*step,2)) for p,o in plist]} → {sorted(set(mapping.values()))}")
        if not relabel:
            log.append("  no confined multi-speaker base → passthrough")
            return _passthrough()
        ng = [None] * n
        for i in range(n):
            b = bg[i]
            if b is None:
                continue
            if b in relabel:
                ng[i] = relabel[b].get(pg[i], b)  # confined P 만 새 라벨, 그 외 dominant(B)
            else:
                ng[i] = b

    # grid → segments (연속 동일 라벨 병합)
    out = []
    cur = None
    cs = 0
    for i in range(n + 1):
        lab = ng[i] if i < n else None
        if lab != cur:
            if cur is not None:
                out.append({"group_start": round(cs * step, 3),
                            "group_end": round(i * step, 3),
                            "speaker": cur, "segment_ids": [], "text": ""})
            cur = lab
            cs = i
    for j, s in enumerate(out):
        s["group_idx"] = j
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--chunk", required=True, help="e.g. test5_fresh_chunk_000")
    ap.add_argument("--min-dur", type=float, default=MIN_DUR)
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    meta = rd / "meta"
    base = json.load(open(meta / f"{args.chunk}_segments.json", encoding="utf-8"))
    base_segs = _segs_of(base)
    vocals = str(rd / "vocals" / f"{args.chunk}_clean_vocals.wav")
    pyan = _pyannote(vocals)
    print(f"base: {len(base_segs)} segs / {len(set(str(s.get('speaker','')) for s in base_segs))} spk; "
          f"pyannote: {len(pyan)} segs / {len(set(s['speaker'] for s in pyan))} spk")
    out, log = selective_local_split(base_segs, pyan, min_dur=args.min_dur, min_frac=args.min_frac)
    for line in log:
        print(line)
    n_spk = len(set(s["speaker"] for s in out if not s["speaker"].startswith("SPEAKER_BG")))
    outp = meta / f"{args.chunk}_segments_localsplit.json"
    json.dump({"groups": out}, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[✓] {len(out)} segs / {n_spk} main spk → {outp}")


if __name__ == "__main__":
    main()
