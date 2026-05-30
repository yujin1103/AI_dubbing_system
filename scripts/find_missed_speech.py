#!/usr/bin/env python3
"""ASR이 놓친 발화 구간 자동 탐지.

vocals 의 에너지(RMS)로 '음성이 있는' 구간을 찾고, 그중 ASR 단어가 하나도
없는 구간(= ASR 누락 후보)을 출력. 그 구간만 다시 ASR 하면 됨.

사용:
    python find_missed_speech.py --vocals <vocals.wav> --words <words.json> [--start 0 --end 9999]
"""
import argparse, json
import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--win", type=float, default=0.25, help="RMS window sec")
    ap.add_argument("--rms-db", type=float, default=-38.0, help="speech threshold dBFS")
    ap.add_argument("--min-gap", type=float, default=0.4, help="min missed-speech run sec")
    args = ap.parse_args()

    audio, sr = sf.read(args.vocals)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    end = min(args.end, dur)

    words = json.load(open(args.words, encoding="utf-8"))
    words = words.get("words", words) if isinstance(words, dict) else words
    word_iv = [(float(w["start"]), float(w["end"])) for w in words if "start" in w and "end" in w]

    def has_word(t):
        return any(s - 0.15 <= t <= e + 0.15 for s, e in word_iv)

    # RMS per window
    win_n = int(args.win * sr)
    speech_no_word = []  # (t_center, db)
    t = args.start
    while t < end:
        i0 = int(t * sr); i1 = min(i0 + win_n, len(audio))
        if i1 <= i0:
            break
        seg = audio[i0:i1]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        db = 20 * np.log10(rms)
        if db >= args.rms_db and not has_word(t + args.win / 2):
            speech_no_word.append((round(t, 2), round(db, 1)))
        t += args.win

    # 연속 구간 병합
    runs = []
    for tc, db in speech_no_word:
        if runs and tc - runs[-1][1] <= args.win * 1.6:
            runs[-1][1] = tc + args.win
            runs[-1][2] = max(runs[-1][2], db)
        else:
            runs.append([tc, tc + args.win, db])

    print(f"vocals {dur:.1f}s, scan {args.start}-{end:.1f}s, ASR words={len(word_iv)}")
    print("=== 음성 에너지 있으나 ASR 단어 없는 구간 (누락 후보) ===")
    found = [r for r in runs if (r[1] - r[0]) >= args.min_gap]
    for s, e, db in found:
        print("  %.2f-%.2f (%.2fs, peak %.1f dB)" % (s, e, e - s, db))
    if not found:
        print("  (없음 — ASR 누락 구간 미검출)")


if __name__ == "__main__":
    main()
