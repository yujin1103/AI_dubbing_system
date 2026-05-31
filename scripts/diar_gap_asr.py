#!/usr/bin/env python3
"""화자분리 span 기반 누락 단어 회수 (완전 자동, 하드코딩 0).

원리: 화자분리(gapfilled)는 '발화'라고 하는데 메인 ASR 단어가 비어있는 구간
      (예: test4 61.12-62.60 SPEAKER_05 = Sean 'Good')을 그 정확한 span 으로
      clean 재-ASR 해 단어를 회수한다. 화자 배정은 화자분리 라벨을 그대로 쓰므로
      별도 지정 불필요(= build_dub_input 이 시각으로 자동 배정).

안전장치:
  - boost 없음(vol=1.0): boost 환각 회피.
  - 에너지 게이트: 침묵 span skip(환각 방지).
  - dedup: 회수 단어가 인접(±DEDUP_WIN) 기존 단어와 같으면 skip(다른 화자 발화 재전사 방지).
  - 단어 span 환각 필터(>max-word-dur, zero-dur) 제거.

사용:
  python diar_gap_asr.py --chunk <mp4> --gapfilled <gapfilled.json> --words <words.json> --out <words_aug.json>
"""
import argparse, json, subprocess, tempfile, os
from pathlib import Path
import requests

ASR_URL = "http://127.0.0.1:8902/transcribe"
LANG = "English"


def norm(w):
    return w.lower().rstrip(".!?,'\" ")


def load_words(path):
    d = json.load(open(path, encoding="utf-8"))
    w = d.get("words", d) if isinstance(d, dict) else d
    return [x for x in w if "start" in x and "end" in x]


def load_groups(path):
    d = json.load(open(path, encoding="utf-8"))
    g = d.get("groups", d if isinstance(d, list) else [])
    out = []
    for x in g:
        out.append((float(x.get("group_start", x.get("start", 0))),
                    float(x.get("group_end", x.get("end", 0))),
                    x.get("speaker", "")))
    return sorted(out, key=lambda z: z[0])


def mean_db(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    for line in (r.stderr or "").splitlines():
        if "mean_volume" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except (ValueError, IndexError):
                pass
    return -99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--gapfilled", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--silence-db", type=float, default=-55.0)
    ap.add_argument("--min-group-dur", type=float, default=0.3, help="이보다 짧은 group 은 skip")
    ap.add_argument("--max-group-dur", type=float, default=3.0,
                    help="이보다 긴 빈 group 은 skip(긴 빈 구간은 보통 침묵/배경)")
    ap.add_argument("--max-word-dur", type=float, default=2.0)
    ap.add_argument("--min-word-dur", type=float, default=0.06)
    ap.add_argument("--dedup-win", type=float, default=1.5,
                    help="회수 단어가 이 시간 내 기존 단어와 같으면 skip(중복 재전사 방지)")
    args = ap.parse_args()

    words = load_words(args.words)
    groups = load_groups(args.gapfilled)
    print(f"기존 words: {len(words)}, groups: {len(groups)}")

    tmp = Path(tempfile.gettempdir())
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nk=1:nw=1", str(args.chunk)],
                                 capture_output=True, text=True).stdout or 0)

    def words_in(s, e):
        return [w for w in words if s <= (float(w["start"]) + float(w["end"])) / 2.0 <= e]

    fresh = []
    skipped_silent = 0
    rejected = []
    for s, e, spk in groups:
        dur = e - s
        if dur < args.min_group_dur or dur > args.max_group_dur:
            continue
        if words_in(s, e):
            continue  # 이미 단어 있음 → 회수 불필요
        a = max(0.0, s - args.pad); b = min(total, e + args.pad)
        sub = tmp / f"dga_{int(a*100)}.wav"
        # 소스 mp4 에서 직접 슬라이스(16k mono). 풀파일→16k→슬라이스 경로는 ASR 결과를 degrade시킴.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(a), "-i", str(args.chunk),
                        "-t", str(b - a), "-ar", "16000", "-ac", "1", str(sub)], check=True)
        if mean_db(sub) < args.silence_db:
            skipped_silent += 1
            try: os.remove(sub)
            except OSError: pass
            continue
        try:
            r = requests.post(ASR_URL, json={"audio_path": str(sub), "language": LANG}, timeout=90).json()
        except Exception as ex:
            print(f"  ASR fail @{a:.1f}s: {ex}"); continue
        for w in (r.get("words", []) or []):
            ws = float(w["start"]) + a; we = float(w["end"]) + a
            wc = (ws + we) / 2.0
            if not (s - 0.05 <= wc <= e + 0.05):
                continue
            wd = we - ws
            if wd > args.max_word_dur or wd < args.min_word_dur:
                rejected.append((ws, we, w["word"])); continue
            # dedup: 인접 기존 단어 또는 이미 회수한 단어와 같으면 skip
            if any(abs(ws - float(x["start"])) < args.dedup_win and norm(w["word"]) == norm(x["word"])
                   for x in words):
                continue
            if any(abs(ws - f["start"]) < args.dedup_win and norm(w["word"]) == norm(f["word"])
                   for f in fresh):
                continue
            fresh.append({"word": w["word"], "start": ws, "end": we, "src": "diar_gap"})
        try: os.remove(sub)
        except OSError: pass

    print(f"\n빈 화자 span 침묵 skip: {skipped_silent}, 회수 단어 {len(fresh)}개:")
    for w in fresh:
        print(f"  {w['start']:6.2f}-{w['end']:6.2f}  {w['word']!r}")
    if rejected:
        print(f"  (환각필터 거부 {len(rejected)}개)")

    combined = sorted(words + fresh, key=lambda x: float(x["start"]))
    json.dump({"words": combined, "diar_gap_added": len(fresh)},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[✓] {len(combined)} words (+{len(fresh)} diar-gap) → {args.out}")


if __name__ == "__main__":
    main()
