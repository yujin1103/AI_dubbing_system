#!/usr/bin/env python3
"""전체 구간 boost sub-chunk ASR — 짧은 발화(예: paramedic "Good") 회수.

발견(2026-05-30): 메인 ASR(풀 86초 한 번에)은 짧은 overlap 발화를 옆 단어에 흡수해
놓친다. 같은 오디오를 짧게 잘라 증폭(boost)하면 인식된다 (test4 60.48s "Good" 검출 확인).
화자분리와 무관 — ASR 단계만의 문제이므로 화자분리 결과(1.1167)는 안 건드린다.

처리:
  1. chunk mp4 → 16k mono → volume boost
  2. 0~끝 전체를 WIN=4s/HOP=3s sub-chunk 로 ASR (overlap 1s)
  3. 기존 words.json 과, sub-chunk 끼리 중복 제거 (시간<0.3s & 같은 단어)
  4. 합쳐서 words_full.json 저장 (boost 로 새로 추가된 단어 수 표시)

사용:
    python full_boost_asr.py --chunk <chunk.mp4> --words <words.json> --out <words_full.json> [--vol 3.0]
"""
import argparse, json, subprocess, tempfile, os
from pathlib import Path
import requests

ASR_URL = "http://127.0.0.1:8902/transcribe"
WIN = 4.0
HOP = 3.0
LANG = "English"


def norm(w):
    return w.lower().rstrip(".!?,'\" ")


def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nk=1:nw=1", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--words", required=True, help="기존 메인 ASR words.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vol", type=float, default=3.0)
    args = ap.parse_args()

    existing = []
    if Path(args.words).exists():
        d = json.load(open(args.words, encoding="utf-8"))
        existing = d.get("words", d) if isinstance(d, dict) else d
    print(f"기존 words: {len(existing)}")

    tmp = Path(tempfile.gettempdir())
    raw = tmp / "fb_raw.wav"
    boost = tmp / "fb_boost.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.chunk,
                    "-ar", "16000", "-ac", "1", str(raw)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                    "-af", f"volume={args.vol}", str(boost)], check=True)
    total = probe_dur(raw)
    print(f"전체 {total:.1f}s, boost x{args.vol}, sub-chunk {WIN}s/hop {HOP}s")

    new_words = []
    t = 0.0
    while t < total:
        ln = min(WIN, total - t)
        sub = tmp / f"fb_sub_{int(t*100)}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(boost),
                        "-ss", str(t), "-t", str(ln), str(sub)], check=True)
        try:
            r = requests.post(ASR_URL, json={"audio_path": str(sub), "language": LANG}, timeout=90).json()
        except Exception as ex:
            print(f"  ASR fail @{t:.1f}s: {ex}"); t += HOP; continue
        for w in (r.get("words", []) or []):
            new_words.append({"word": w["word"],
                              "start": float(w["start"]) + t,
                              "end": float(w["end"]) + t})
        try:
            os.remove(sub)
        except OSError:
            pass
        t += HOP

    # 중복 제거: 같은 단어가 시간상 가까우면 제거 (가짜 중복).
    #  - 메인과 같은 단어 0.6s 내 → skip (boost 가 메인 단어 재인식한 것)
    #  - boost 끼리 같은 단어 0.3s 내 → skip
    #  주의: 메인이 "다른 단어"로 점유한 자리(예: Get↔Good)는 보존 — 짧은 발화 회수 우선.
    #        대신 build_dub_input 에서 같은 시각 다른 단어는 둘 다 들어가 한 세그먼트에 병기됨.
    fresh = []
    for w in new_words:
        if any(abs(w["start"] - e["start"]) < 0.6 and norm(w["word"]) == norm(e["word"]) for e in existing):
            continue
        if any(abs(w["start"] - f["start"]) < 0.3 and norm(w["word"]) == norm(f["word"]) for f in fresh):
            continue
        fresh.append(w)

    print(f"\nboost 새 단어 {len(fresh)}개:")
    for w in fresh:
        print(f"  {w['start']:6.2f}-{w['end']:6.2f}  {w['word']!r}")

    combined = list(existing) + fresh
    combined.sort(key=lambda x: float(x["start"]))
    json.dump({"words": combined, "boost_added": len(fresh)},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[✓] {len(combined)} words (+{len(fresh)} boost) → {args.out}")


if __name__ == "__main__":
    main()
