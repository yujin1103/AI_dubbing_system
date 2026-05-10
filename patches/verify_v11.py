"""v11 결과 검증 - Korean instruction 효과."""
import os
import subprocess
import json
import torch
from transformers import pipeline
import soundfile as sf
import numpy as np

OUT_DIR = "/workspace/media/output/leak_check_v11"
os.makedirs(OUT_DIR, exist_ok=True)

V11_MP4 = "/workspace/media/output/test2p1v11_ko_20260505_143518_test2p1v11_6862be.mp4"
RUN_DIR = "/workspace/media/runs/20260505_143518_test2p1v11_6862be"
DUBBED_V11 = f"{RUN_DIR}/dubbed/test2p1v11_chunk_000_dubbed.wav"
REPORT = "/workspace/media/reports/test2p1v11_ko_20260505_143518_test2p1v11_6862be.json"

print("=== v11 tone (Korean) field ===\n")
with open(REPORT) as f:
    rep = json.load(f)
for chunk in rep["chunks"]:
    for seg in chunk["segments"]:
        sid = seg["id"]
        text = seg["original_text"][:30]
        kor = seg["translated_text"][:35]
        tone = (seg.get("tts_emotion") or "")[:80]
        print(f"  [{sid}] {text:<30} → ko: {kor}")
        print(f"        tone: {tone}")

# audio extract
out = f"{OUT_DIR}/v11_dubbed.wav"
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", DUBBED_V11,
    "-vn", "-ar", "16000", "-ac", "1",
    out,
])

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

print("\n=== v11 DUBBED 1초 단위 ===\n")
print(f"{'sec':>4} | {'text':<60}")
print("-" * 70)
leak_count = 0
silent_count = 0
korean_count = 0
for sec in range(60):
    t = transcribe(out, sec, sec+1)
    if not t or t in ["you", ".", "I", "Bye.", "Hey."]:
        silent_count += 1
        continue
    has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in t)
    has_other = any(c.isalpha() and not (0xAC00 <= ord(c) <= 0xD7A3) and ord(c) > 127 for c in t)
    has_eng = any('a' <= c.lower() <= 'z' for c in t)
    marker = ""
    if has_other:
        marker = " ⚠️ OTHER"
        leak_count += 1
    elif has_eng and not has_korean:
        marker = " ⚠️ ENG"
        leak_count += 1
    elif has_eng and has_korean:
        marker = " ⚠️ MIXED"
        leak_count += 1
    else:
        korean_count += 1
    print(f"  {sec:2d} | {t[:58]:<60}{marker}")

print(f"\n=== leak: {leak_count}/60s, korean: {korean_count}/60s, silent: {silent_count}/60s ===")
print(f"\n비교:")
print(f"  v7  (English wrapping):    17/60s leak")
print(f"  v9  (style+emotion+sit en): 24/60s leak")
print(f"  v10 (parser fix, English): 18/60s leak")
print(f"  v11 (Korean instruction): {leak_count}/60s leak")
