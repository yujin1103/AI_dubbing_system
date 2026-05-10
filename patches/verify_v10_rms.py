"""leak 의심 시간대 RMS 측정 - 실제 audio 있는지 vs whisper noise."""
import os
import numpy as np
import soundfile as sf

DUBBED = "/workspace/media/output/leak_check_v10/v10_dubbed.wav"

audio, sr = sf.read(DUBBED)
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# leak 시간대들
leak_secs = [1, 8, 11, 24, 28, 33, 38, 41, 42, 43, 45, 46, 51, 53, 56, 57, 59]
clean_korean_secs = [9, 10, 12, 13, 14, 21, 22, 23, 30, 31, 39, 40, 50, 52]

print("=== leak 의심 시간대 RMS ===")
print(f"{'sec':>4} | {'RMS':>8} | {'Peak':>8} | meaning")
print("-" * 50)
for s in leak_secs:
    seg = audio[s*sr:(s+1)*sr]
    rms = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(np.max(np.abs(seg)))
    is_real = "REAL AUDIO" if rms > 0.01 else "silent (whisper noise)"
    print(f"  {s:2d} | {rms:8.5f} | {peak:8.5f} | {is_real}")

print("\n=== 깨끗한 한국어 시간대 RMS (비교) ===")
for s in clean_korean_secs:
    seg = audio[s*sr:(s+1)*sr]
    rms = float(np.sqrt(np.mean(seg ** 2)))
    print(f"  {s:2d} | {rms:8.5f}")
