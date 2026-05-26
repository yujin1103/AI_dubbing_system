"""Per-frame F0-based gender estimation for an audio file.

Output: `audio_f0_gender.json` consumed by
`latentsync/utils/asd_filter.py:AudioGenderTimeline`.

Method:
  - librosa.pyin (probabilistic YIN) over the whole audio.
  - One F0 value per 25fps frame (40ms) using max-vote of pyin frames inside.
  - Gender by adult-speech heuristic:
      F0 < 165 Hz → male
      F0 > 175 Hz → female
      else        → uncertain
    Korean speakers tend to have slightly higher female F0 (avg 200-220Hz)
    and male F0 around 110-130Hz. The 165/175 thresholds give a small dead
    zone so we don't oscillate near 170Hz.

Usage:
    python audio_f0_gender.py \\
        --wav /path/to/dubbed.wav \\
        --fps 25 \\
        --out /path/to/audio_f0_gender.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List

import numpy as np


def _log(msg):
    print(f"[AudioF0] {msg}", flush=True)


def estimate_per_frame_gender(
    wav_path: str,
    fps: float = 25.0,
    male_max: float = 165.0,
    female_min: float = 175.0,
) -> tuple:
    """Return (f0_per_frame, gender_per_frame) as Python lists."""
    try:
        import librosa
    except ImportError:
        # Fallback: a much cruder autocorrelation method
        return _estimate_autocorr(wav_path, fps, male_max, female_min)

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    n_frames = int(np.ceil(duration * fps))
    _log(f"audio {duration:.1f}s @ {sr}Hz, computing pyin (this can take ~5-10s/min of audio)")

    # pyin with default params (suitable for human voice 65–500 Hz)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),  # ~65 Hz
        fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
        sr=sr,
        frame_length=2048,
    )
    # f0 has hop_length=512 by default → ~32ms per frame at 16kHz, ~10ms at 48kHz
    # Get times for each pyin frame
    pyin_times = librosa.times_like(f0, sr=sr)

    # Re-bucket to fps frames
    f0_per_frame = [float("nan")] * n_frames
    for i in range(n_frames):
        t0 = i / fps
        t1 = (i + 1) / fps
        mask = (pyin_times >= t0) & (pyin_times < t1)
        chunk = f0[mask]
        if chunk.size == 0:
            continue
        valid = chunk[~np.isnan(chunk)]
        if valid.size == 0:
            continue
        # robust: median of valid pitch in this frame
        f0_per_frame[i] = float(np.median(valid))

    gender_per_frame = []
    for v in f0_per_frame:
        if np.isnan(v) if isinstance(v, float) else False:
            gender_per_frame.append("unknown")
            continue
        if v != v:  # NaN check fallback
            gender_per_frame.append("unknown")
            continue
        if v < male_max:
            gender_per_frame.append("male")
        elif v > female_min:
            gender_per_frame.append("female")
        else:
            gender_per_frame.append("unknown")

    # Replace NaN with None for JSON cleanliness
    f0_per_frame = [None if (isinstance(v, float) and v != v) else v for v in f0_per_frame]
    return f0_per_frame, gender_per_frame


def _estimate_autocorr(wav_path, fps, male_max, female_min):
    """Fallback when librosa unavailable: crude per-frame autocorrelation."""
    import wave
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        frames_raw = w.readframes(w.getnframes())
        n_ch = w.getnchannels()
        y = np.frombuffer(frames_raw, dtype=np.int16).astype(np.float32)
        if n_ch > 1:
            y = y.reshape(-1, n_ch).mean(axis=1)
    duration = len(y) / sr
    n_frames = int(np.ceil(duration * fps))
    win = int(sr / fps)  # samples per frame at fps
    f0_per_frame = [None] * n_frames
    gender_per_frame = ["unknown"] * n_frames
    low = sr // 400  # 400 Hz max
    high = sr // 50  # 50 Hz min
    for i in range(n_frames):
        start = i * win
        end = start + win
        seg = y[start:end]
        if seg.size < high:
            continue
        seg = seg - seg.mean()
        if np.std(seg) < 100:
            continue
        ac = np.correlate(seg, seg, mode="full")[seg.size - 1 :]
        if high >= ac.size:
            high = ac.size - 1
        if low >= ac.size:
            continue
        peak = low + int(np.argmax(ac[low:high]))
        if peak <= 0:
            continue
        f0 = sr / peak
        f0_per_frame[i] = float(f0)
        if f0 < male_max:
            gender_per_frame[i] = "male"
        elif f0 > female_min:
            gender_per_frame[i] = "female"
    return f0_per_frame, gender_per_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--out", required=True)
    p.add_argument("--male-max", type=float, default=165.0)
    p.add_argument("--female-min", type=float, default=175.0)
    args = p.parse_args()

    if not os.path.isfile(args.wav):
        _log(f"wav not found: {args.wav}")
        return 1

    f0, gender = estimate_per_frame_gender(args.wav, args.fps, args.male_max, args.female_min)
    male_n = gender.count("male")
    female_n = gender.count("female")
    unk_n = gender.count("unknown")
    _log(f"n_frames={len(gender)}  male={male_n}  female={female_n}  unknown={unk_n}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "fps": args.fps,
            "n_frames": len(gender),
            "f0_per_frame": f0,
            "gender_per_frame": gender,
            "stats": {"male": male_n, "female": female_n, "unknown": unk_n},
            "thresholds": {"male_max": args.male_max, "female_min": args.female_min},
        }, f)
    _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
