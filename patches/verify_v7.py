"""v7 결과 검증 - 외국어/영어 leak 제거 + 감정 적용 확인."""
import os
import subprocess
import torch
from transformers import pipeline
import soundfile as sf
import numpy as np

OUT_DIR = "/workspace/media/output/leak_check_v7"
os.makedirs(OUT_DIR, exist_ok=True)

V7_MP4 = "/workspace/media/output/test2p1v7_ko_20260505_134525_test2p1v7_39535d.mp4"
RUN_DIR = "/workspace/media/runs/20260505_134525_test2p1v7_39535d"
DUBBED_V7 = f"{RUN_DIR}/dubbed/test2p1v7_chunk_000_dubbed.wav"

# 시작 시간 추출
import time
log_path = "/tmp/test2p1_v7.log"
if os.path.exists(log_path):
    with open(log_path) as f:
        first = f.readline().strip()
    print(f"v7 시작: {first}")

for label, src in [("v7_final", V7_MP4), ("v7_dubbed", DUBBED_V7)]:
    out = f"{OUT_DIR}/{label}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-vn", "-ar", "16000", "-ac", "1",
        out,
    ])
    print(f"{label}: {out}")

# whisper 1초 단위 transcribe
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=0 if torch.cuda.is_available() else -1,
)

def transcribe(path, sec_start, sec_end):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    seg = audio[int(sec_start*sr):int(sec_end*sr)]
    if seg.size == 0:
        return ""
    return pipe({"array": seg, "sampling_rate": sr}).get("text", "").strip()

print("\n=== v7 DUBBED (TTS only) 60초 - 외국어/영어 leak 검사 ===\n")
print(f"{'sec':>4} | {'DUBBED text':<60}")
print("-" * 70)

leak_count = 0
for sec in range(60):
    t = transcribe(f"{OUT_DIR}/v7_dubbed.wav", sec, sec+1)
    # 외국어 감지 — Korean이 아닌 글자 비율
    if t and t not in ["you", "."]:
        # 한글이 있는지
        has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 or 0x1100 <= ord(c) <= 0x11FF for c in t)
        # 그 외 (英中日泰 등)
        has_other = any(c.isalpha() and not (0xAC00 <= ord(c) <= 0xD7A3 or 0x1100 <= ord(c) <= 0x11FF) for c in t)
        marker = ""
        if has_other and not has_korean:
            marker = " ⚠️ NON-KOREAN!"
            leak_count += 1
        elif has_other and has_korean:
            marker = " ⚠️ MIXED!"
            leak_count += 1
        print(f"  {sec:2d} | {t[:60]:<60}{marker}")
    else:
        print(f"  {sec:2d} | (silent)")

print(f"\n=== 외국어 leak: {leak_count}/60 seconds ===")
