"""final mp4와 원본 영상의 1.36~8.4s 구간 audio 상세 분석."""
import os
import subprocess
import numpy as np
import soundfile as sf

ORIG_VIDEO = "/workspace/media/input/test2_part1.mp4"
FINAL_MP4 = "/workspace/media/output/test2p1_ko_20260505_123215_test2p1_245243.mp4"
DUBBED_WAV = "/workspace/media/runs/20260505_123215_test2p1_245243/dubbed/test2p1_chunk_000_dubbed.wav"

OUT_DIR = "/workspace/media/output/leak_check"
os.makedirs(OUT_DIR, exist_ok=True)

print("=== 1.36~8.4s 구간: 원본 vs 더빙 mp4 비교 ===\n")

# 원본 영상에서 audio 추출
orig_audio = f"{OUT_DIR}/orig_full.wav"
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", ORIG_VIDEO,
    "-vn", "-ar", "16000", "-ac", "1",
    orig_audio,
])

# 더빙 mp4에서 audio 추출
dub_audio = f"{OUT_DIR}/dubbed_full.wav"
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", FINAL_MP4,
    "-vn", "-ar", "16000", "-ac", "1",
    dub_audio,
])

# RMS 측정 (1초 단위)
def rms_per_second(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n_sec = int(len(audio) / sr)
    rms_list = []
    for i in range(n_sec):
        seg = audio[i*sr:(i+1)*sr]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        rms_list.append(rms)
    return rms_list, sr

orig_rms, orig_sr = rms_per_second(orig_audio)
dub_rms, dub_sr = rms_per_second(dub_audio)
dubbed_rms, _ = rms_per_second(DUBBED_WAV) if os.path.exists(DUBBED_WAV) else ([], 0)

print(f"{'sec':>4} | {'ORIG':>8} | {'FINAL':>8} | {'DUBBED':>8} | 발화?")
print("-" * 55)
for i in range(min(len(orig_rms), len(dub_rms), 30)):
    o = orig_rms[i]
    d = dub_rms[i]
    db = dubbed_rms[i] if i < len(dubbed_rms) else 0
    speech = "🗣 ORIG" if o > 0.05 else "  -  "
    speech_d = "🗣 FINAL" if d > 0.05 else ""
    speech_db = "🇰🇷 KOR" if db > 0.05 else ""
    print(f"  {i:2d}s | {o:8.4f} | {d:8.4f} | {db:8.4f} | {speech} {speech_d} {speech_db}")

# 1.36~8.4s 구간만 export
for label, src in [("orig", ORIG_VIDEO), ("final", FINAL_MP4)]:
    out = f"{OUT_DIR}/{label}_1-9s.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", "1.0", "-to", "9.0",
        "-i", src,
        "-vn", "-ar", "16000", "-ac", "1",
        out,
    ])
    print(f"\n{label} 1~9s wav: {out}")
