#!/usr/bin/env python3
"""dub 입력 생성 v3 — 1_diarize.assign_text 의 "단어 유실 0" 원리 적용.

v1/v2 결함: 단어 중심이 그룹 시간 안에 없으면 누락되거나 엉뚱한 그룹에 묶여
ASR 문장이 조각나고 단어가 유실됐다 (예: "Sean where are you I" → "Sean"만 남음).

v3 (검증된 assign_text 원리):
  1) gapfilled 그룹(speaker, start, end)을 세그먼트로 사용.
  2) 모든 ASR 단어를 배정 — 단어 중심이 겹치는 그룹에, 안 겹치면 **최근접 그룹으로 스냅**.
     → 단어 유실 0. 모든 발화가 어딘가에 반드시 들어간다.
  3) 그룹별 단어를 시간순 결합해 text 채움.
  4) SPEAKER_BG_* 는 기본 제외(배경 화자 번역 skip 규칙). --include-bg 로 포함.
  5) 텍스트 없는 그룹은 출력에서 제외(합성할 게 없음).

사용:
    python build_dub_input_v3.py --gapfilled <gapfilled.json> --words <words.json> --out <dub.json>
"""
import argparse, json
from collections import Counter


def load_words(path):
    d = json.load(open(path, encoding="utf-8"))
    w = d.get("words", d) if isinstance(d, dict) else d
    out = []
    for x in w:
        if not isinstance(x, dict):
            continue
        word = (x.get("word") or x.get("text") or "").strip()
        s, e = x.get("start"), x.get("end")
        if not word or s is None or e is None:
            continue
        out.append({"word": word, "start": float(s), "end": float(e)})
    return sorted(out, key=lambda z: z["start"])


def load_groups(path):
    d = json.load(open(path, encoding="utf-8"))
    g = d.get("groups", d if isinstance(d, list) else [])
    out = []
    for i, x in enumerate(g):
        out.append({
            "idx": i,
            "speaker": x.get("speaker", "SPEAKER_00"),
            "start": float(x.get("group_start", x.get("start", 0))),
            "end": float(x.get("group_end", x.get("end", 0))),
            "words": [],
        })
    return sorted(out, key=lambda z: z["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapfilled", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-bg", action="store_true")
    args = ap.parse_args()

    groups = load_groups(args.gapfilled)
    words = load_words(args.words)
    if not groups:
        raise SystemExit("[v3] ERROR: no groups")
    if not words:
        raise SystemExit("[v3] ERROR: no words")

    # assign_text 원리: 모든 단어를 유실 없이 배정
    assigned = 0
    snapped = 0
    for w in words:
        wc = (w["start"] + w["end"]) / 2.0
        best = None
        for g in groups:
            if g["start"] <= wc <= g["end"]:
                best = g
                break
        if best is None:
            best = min(groups, key=lambda s: min(abs(s["start"] - wc), abs(s["end"] - wc)))
            snapped += 1
        best["words"].append(w)
        assigned += 1

    out_segs, bg_skip, empty = [], 0, 0
    for g in groups:
        if not g["words"]:
            empty += 1
            continue
        if g["speaker"].startswith("SPEAKER_BG") and not args.include_bg:
            bg_skip += 1
            continue
        g["words"].sort(key=lambda x: x["start"])
        text = " ".join(x["word"] for x in g["words"]).strip()
        # 세그먼트 시간은 그룹 경계 유지(화자분리 검증값), 단 단어가 그룹 밖이면 살짝 확장
        st = min(g["start"], g["words"][0]["start"])
        en = max(g["end"], g["words"][-1]["end"])
        out_segs.append({
            "speaker": g["speaker"],
            "start": round(st, 3),
            "end": round(en, 3),
            "text": text,
        })

    out_segs.sort(key=lambda s: s["start"])
    n_spk = len({s["speaker"] for s in out_segs})
    total_words_in = len(words)
    total_words_out = sum(len(s["text"].split()) for s in out_segs)
    json.dump({"segments": out_segs, "n_speakers": n_spk},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[v3] {len(out_segs)} segments, {n_spk} speakers | "
          f"words in={total_words_in} out={total_words_out} snapped={snapped} "
          f"BG-skip={bg_skip} empty-group={empty}")
    print(f"  speakers: {dict(Counter(s['speaker'] for s in out_segs))}")
    if total_words_out < total_words_in * 0.95 and not args.include_bg:
        print(f"  [note] {total_words_in - total_words_out} words dropped (likely BG segments — expected)")


if __name__ == "__main__":
    main()
