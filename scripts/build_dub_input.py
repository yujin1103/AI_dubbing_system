#!/usr/bin/env python3
"""검증된 화자분리(segments_gapfilled.json, groups) + ASR words → dub 입력 diarize-json.

3_dub_pipeline.py / 2_extract_speaker_refs.py 가 기대하는 형식:
    {"segments": [{"speaker","start","end","text"}], "n_speakers": N}

gapfilled groups 의 (speaker, group_start, group_end) 에, ASR words 를 시간으로 슬라이스해
각 group 의 영어 text 를 채운다. (text 없는 segment 는 dub 에서 자동 skip.)

사용:
    python build_dub_input.py --gapfilled <gapfilled.json> --words <words.json> --out <diarize_dub.json>
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
        word = x.get("word") or x.get("text") or ""
        s = x.get("start"); e = x.get("end")
        if s is None or e is None:
            continue
        out.append({"word": str(word), "start": float(s), "end": float(e)})
    return sorted(out, key=lambda z: z["start"])


def text_for_span(words, a, b):
    # word center 가 [a,b] 안이면 포함 (경계 단어 중복 방지)
    toks = [w["word"] for w in words if a <= (w["start"] + w["end"]) / 2.0 <= b]
    return " ".join(t for t in toks if t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapfilled", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.gapfilled, encoding="utf-8"))
    groups = d.get("groups", d if isinstance(d, list) else [])
    words = load_words(args.words)

    segments = []
    for g in groups:
        spk = g.get("speaker", "SPEAKER_00")
        a = float(g.get("group_start", g.get("start", 0.0)))
        b = float(g.get("group_end", g.get("end", 0.0)))
        # 이미 text 가 있으면(번역 전 영어) 우선, 없으면 ASR 슬라이스
        txt = (g.get("text") or "").strip() or text_for_span(words, a, b)
        segments.append({"speaker": spk, "start": round(a, 3), "end": round(b, 3), "text": txt})

    n_spk = len({s["speaker"] for s in segments})
    n_text = sum(1 for s in segments if s["text"])
    json.dump({"segments": segments, "n_speakers": n_spk}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[build_dub_input] {len(segments)} segments, {n_spk} speakers, {n_text} text-filled → {args.out}")
    print(f"  speakers: {dict(Counter(s['speaker'] for s in segments))}")


if __name__ == "__main__":
    main()
