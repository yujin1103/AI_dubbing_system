#!/usr/bin/env python3
"""Word-driven dub-input builder — 모든 발화 단어를 화자에 배정해 그룹화.

기존 build_dub_input(그룹 기반)은 diarization 그룹을 segment로 쓰고 단어를 시간으로
채워서, 단어가 없는 구간이 "빈 text" segment가 되어 TTS에서 "no candidate"로 실패했다.

v2(단어 기반)는 ASR 단어마다 gapfilled diarization으로 화자를 배정한 뒤 연속 같은-화자
단어를 묶어 segment를 만든다 → 모든 segment가 text를 가지며 빈 segment가 생기지 않는다.
(orchestrator.py 의 검증된 방식과 동일.)

규칙:
  - 단어 화자 = 단어 중심시각을 포함하는 gapfilled 그룹 (없으면 최근접).
  - 화자 바뀌거나 단어 간 gap > GAP_SPLIT 또는 segment 길이 > MAX_DUR 이면 새 segment.
  - SPEAKER_BG_* 는 기본 제외(배경 화자는 번역 skip — 원본 유지 규칙). --include-bg 로 포함.

사용:
    python build_dub_input_v2.py --gapfilled <gapfilled.json> --words <words.json> --out <diarize_dub.json>
"""
import argparse, json
from collections import Counter

GAP_SPLIT = 0.8   # 연속 단어 간 gap 이 이보다 크면 segment 분리
MAX_DUR = 12.0    # segment 최대 길이 (CosyVoice 안정 범위)


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


def speaker_at(groups, t):
    for g in groups:
        if g["start"] <= t <= g["end"]:
            return g["speaker"]
    best, bd = None, 1e9
    for g in groups:
        d = min(abs(g["start"] - t), abs(g["end"] - t))
        if d < bd:
            bd, best = d, g["speaker"]
    return best or "SPEAKER_00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapfilled", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-bg", action="store_true",
                    help="SPEAKER_BG_* 포함 (기본: 제외, 원본 유지)")
    args = ap.parse_args()

    groups = load_groups(args.gapfilled)
    words = load_words(args.words)
    if not words:
        raise SystemExit(f"[build_dub_input_v2] ERROR: no words in {args.words}")

    for w in words:
        w["speaker"] = speaker_at(groups, (w["start"] + w["end"]) / 2.0)

    segs, cur = [], None
    for w in words:
        if (cur and w["speaker"] == cur["speaker"]
                and (w["start"] - cur["end"]) <= GAP_SPLIT
                and (w["end"] - cur["start"]) <= MAX_DUR):
            cur["end"] = w["end"]
            cur["words"].append(w["word"])
        else:
            if cur:
                segs.append(cur)
            cur = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "words": [w["word"]]}
    if cur:
        segs.append(cur)

    out_segs, bg_skipped = [], 0
    for s in segs:
        if s["speaker"].startswith("SPEAKER_BG") and not args.include_bg:
            bg_skipped += 1
            continue
        out_segs.append({
            "speaker": s["speaker"],
            "start": round(s["start"], 3),
            "end": round(s["end"], 3),
            "text": " ".join(s["words"]).strip(),
        })

    n_spk = len({s["speaker"] for s in out_segs})
    json.dump({"segments": out_segs, "n_speakers": n_spk},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    empty = sum(1 for s in out_segs if not s["text"])
    print(f"[build_dub_input_v2] {len(out_segs)} segments, {n_spk} speakers, "
          f"BG-skipped {bg_skipped}, empty-text {empty} → {args.out}")
    print(f"  speakers: {dict(Counter(s['speaker'] for s in out_segs))}")


if __name__ == "__main__":
    main()
