"""v9 결과 검증 + tone field 확인."""
import os
import subprocess
import json
import torch
from transformers import pipeline
import soundfile as sf
import numpy as np

OUT_DIR = "/workspace/media/output/leak_check_v9"
os.makedirs(OUT_DIR, exist_ok=True)

V9_MP4 = "/workspace/media/output/test2p1v9_ko_20260505_141017_test2p1v9_566095.mp4"
RUN_DIR = "/workspace/media/runs/20260505_141017_test2p1v9_566095"
DUBBED_V9 = f"{RUN_DIR}/dubbed/test2p1v9_chunk_000_dubbed.wav"
REPORT = "/workspace/media/reports/test2p1v9_ko_20260505_141017_test2p1v9_566095.json"

# 1. tone field 확인
print("=== v9 tts_emotion (tone) field 확인 ===\n")
with open(REPORT) as f:
    rep = json.load(f)
for chunk in rep["chunks"]:
    for seg in chunk["segments"]:
        sid = seg["id"]
        text = seg["original_text"][:30]
        tone = (seg.get("tts_emotion") or "")[:80]
        print(f"  [{sid}] {text:<35} → tone: {tone}")

# 2. dubbed audio extract
for label, src in [("v9_dubbed", DUBBED_V9), ("v9_final", V9_MP4)]:
    out = f"{OUT_DIR}/{label}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-vn", "-ar", "16000", "-ac", "1",
        out,
    ])

# whisper
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

print("\n=== v9 DUBBED 1초 단위 (외국어 leak 확인) ===\n")
print(f"{'sec':>4} | {'text':<60}")
print("-" * 70)
leak_count = 0
silent_count = 0
for sec in range(60):
    t = transcribe(f"{OUT_DIR}/v9_dubbed.wav", sec, sec+1)
    if not t or t in ["you", ".", "I"]:
        silent_count += 1
        # print(f"  {sec:2d} | (silent)")
        continue
    has_korean = any(0xAC00 <= ord(c) <= 0xD7A3 for c in t)
    has_other_script = any(c.isalpha() and not (0xAC00 <= ord(c) <= 0xD7A3) and ord(c) > 127 for c in t)
    has_english = any('a' <= c.lower() <= 'z' for c in t)
    marker = ""
    if has_other_script:
        marker = " ⚠️ OTHER-SCRIPT"
        leak_count += 1
    elif has_english and not has_korean:
        marker = " ⚠️ ENGLISH"
        leak_count += 1
    elif has_english and has_korean:
        marker = " ⚠️ MIXED"
        leak_count += 1
    print(f"  {sec:2d} | {t[:58]:<60}{marker}")

print(f"\n=== leak: {leak_count}/60s, silent: {silent_count}/60s ===")
