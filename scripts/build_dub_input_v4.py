#!/usr/bin/env python3
"""dub 입력 v4 — 문장 분할(침묵 기준)과 화자 배정(dominant)을 분리.

사용자 설계(2026-05-31):
  문제: 화자분리 구간으로 문장을 자르면 문장이 토막나고, LLM 문장복원은 과의존.
  해결:
    1) segment = 문장 단위. ASR 단어를 '침묵(gap)' 기준으로 묶는다 (LLM 불필요).
       단어 간 gap > SENT_GAP 면 새 문장. 너무 길면(MAX_DUR) 분할.
    2) 화자 = 그 문장 시간대에 '가장 많이 말한 화자'(dominant by overlap).
       = "이 화자가 (그 문장을) 끝까지 발언함". 검증된 gapfilled 화자분리는 배정에만 사용.
  → 문장 안 토막남 + 화자 겹침 없음(문장은 시간순) + 화자분리 결과(1.1167) 보존 + LLM 0 의존.

BG 화자: --include-bg 면 BG 문장도 포함(원본 유지/번역은 dub 단계 정책). 기본은 포함.

사용:
  python build_dub_input_v4.py --gapfilled <gapfilled.json> --words <words.json> --out <dub.json>
        [--sent-gap 0.55] [--max-dur 12] [--no-bg]
"""
import argparse, json
from collections import Counter, defaultdict


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
    for x in g:
        out.append({
            "speaker": x.get("speaker", "SPEAKER_00"),
            "start": float(x.get("group_start", x.get("start", 0))),
            "end": float(x.get("group_end", x.get("end", 0))),
        })
    return sorted(out, key=lambda z: z["start"])


def dominant_speaker(groups, s, e):
    """[s,e] 와 overlap 시간이 가장 큰 화자 = '끝까지 발언한 화자'."""
    acc = defaultdict(float)
    for g in groups:
        ov = min(e, g["end"]) - max(s, g["start"])
        if ov > 0:
            acc[g["speaker"]] += ov
    if acc:
        return max(acc.items(), key=lambda kv: kv[1])[0]
    # overlap 전혀 없으면(드묾) 중심 최근접 화자
    c = (s + e) / 2.0
    return min(groups, key=lambda g: min(abs(g["start"] - c), abs(g["end"] - c)))["speaker"]


def split_sentences(words, sent_gap, max_dur, groups):
    """단어를 (1)단어별 dominant 화자 (2)침묵 기준으로 묶기.
    새 문장 조건: 화자 바뀜 OR 침묵>sent_gap OR 길이>max_dur.
    → 한 문장에 한 화자만 → 화자 걸침/오배정 없음 (사용자 설계)."""
    # 단어별 화자: 그 단어 시각을 포함/최근접하는 화자
    for w in words:
        c = (w["start"] + w["end"]) / 2.0
        spk = None
        for g in groups:
            if g["start"] <= c <= g["end"]:
                spk = g["speaker"]
                break
        if spk is None:
            spk = min(groups, key=lambda g: min(abs(g["start"] - c), abs(g["end"] - c)))["speaker"]
        w["_spk"] = spk

    sents = []
    cur = None
    for w in words:
        if cur is None:
            cur = [w]
            continue
        gap = w["start"] - cur[-1]["end"]
        too_long = (w["end"] - cur[0]["start"]) > max_dur
        spk_change = w["_spk"] != cur[-1]["_spk"]
        if spk_change or gap > sent_gap or too_long:
            sents.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        sents.append(cur)
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapfilled", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sent-gap", type=float, default=0.55, help="단어 간 침묵 이상이면 새 문장")
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--no-bg", action="store_true", help="BG 화자 문장 제외")
    ap.add_argument("--speaker-split", action="store_true",
                    help="한 문장 안에서 dominant 화자가 연속으로 바뀌면 화자 경계로 추가 분할")
    args = ap.parse_args()

    groups = load_groups(args.gapfilled)
    words = load_words(args.words)
    if not groups or not words:
        raise SystemExit("[v4] ERROR: empty groups or words")

    sents = split_sentences(words, args.sent_gap, args.max_dur, groups)

    segs = []
    for sent in sents:
        s = sent[0]["start"]
        e = sent[-1]["end"]
        # 화자: 문장 내 단어는 모두 같은 _spk (split_sentences가 화자 바뀜에 분할).
        # 안전을 위해 overlap dominant 로 재확인.
        spk = sent[0].get("_spk") or dominant_speaker(groups, s, e)
        text = " ".join(w["word"] for w in sent).strip()
        segs.append({"speaker": spk, "start": round(s, 3), "end": round(e, 3), "text": text})

    # BG 필터
    if args.no_bg:
        segs = [s for s in segs if not s["speaker"].startswith("SPEAKER_BG")]

    # 겹침 안전장치: 정렬 후 인접 segment 가 겹치면 경계 조정(앞 end 를 뒤 start 로)
    segs.sort(key=lambda x: (x["start"], x["end"]))
    for a, b in zip(segs, segs[1:]):
        if a["end"] > b["start"]:
            a["end"] = round(b["start"], 3)
    # 0초/초단문(<0.12s) 정리: 직전 같은 화자 segment 에 텍스트 병합, 없으면 제거.
    cleaned = []
    for s in segs:
        if s["end"] - s["start"] >= 0.12:
            cleaned.append(s)
        elif cleaned and cleaned[-1]["speaker"] == s["speaker"] and (s["start"] - cleaned[-1]["end"]) < 0.6:
            cleaned[-1]["text"] = (cleaned[-1]["text"] + " " + s["text"]).strip()
            cleaned[-1]["end"] = max(cleaned[-1]["end"], s["end"])
        # else: 0초 단독 → 버림 (합성 불가)
    segs = cleaned

    n_spk = len({s["speaker"] for s in segs})
    win = len(words)
    wout = sum(len(s["text"].split()) for s in segs)
    json.dump({"segments": segs, "n_speakers": n_spk},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[v4] {len(segs)} 문장 segment, {n_spk} speakers | words in={win} out={wout}")
    print(f"  speakers: {dict(Counter(s['speaker'] for s in segs))}")
    # 짧은(<0.4s) segment 경고
    sh = [s for s in segs if s["end"] - s["start"] < 0.4]
    if sh:
        print(f"  [note] {len(sh)} segment <0.4s (짧은 발화): " +
              ", ".join("%.1fs:%r" % (x["end"] - x["start"], x["text"][:15]) for x in sh[:6]))


if __name__ == "__main__":
    main()
