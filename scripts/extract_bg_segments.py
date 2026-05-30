#!/usr/bin/env python3
"""gapfilled diarization에서 BG 화자(SPEAKER_BG_*) 구간만 추출 → bg_segments.json.

3_dub_pipeline.py --bg-segments 입력용. 이 구간들은 합성 대신 원본 오디오(영어)로 채워진다.
(사용자 방침: 짧은 대사/누락 발화를 BG로라도 영상에 넣되 번역은 안 함.)

사용:
    python extract_bg_segments.py --gapfilled <gapfilled.json> --out <bg_segments.json>
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapfilled", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.gapfilled, encoding="utf-8"))
    g = d.get("groups", d if isinstance(d, list) else [])
    bg = []
    for x in g:
        spk = x.get("speaker", "")
        if spk.startswith("SPEAKER_BG"):
            bg.append({
                "speaker": spk,
                "start": round(float(x.get("group_start", x.get("start", 0))), 3),
                "end": round(float(x.get("group_end", x.get("end", 0))), 3),
            })
    bg.sort(key=lambda s: s["start"])
    json.dump({"segments": bg}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[extract_bg] {len(bg)} BG segments → {args.out}")
    for b in bg:
        print("  %.2f-%.2f  %s" % (b["start"], b["end"], b["speaker"]))


if __name__ == "__main__":
    main()
