"""v12 결과 검증 - 팀원 setup 100% 재현 (self ref + English instruction)."""
import os
import subprocess
import json
import torch
from transformers import pipeline
import soundfile as sf

OUT_DIR = "/workspace/media/output/leak_check_v12"
os.makedirs(OUT_DIR, exist_ok=True)

V12_MP4 = "/workspace/media/output/test2p1v12_ko_20260505_150244_test2p1v12_ebc736.mp4"
RUN_DIR = "/workspace/media/runs/20260505_150244_test2p1v12_ebc736"
DUBBED_V12 = f"{RUN_DIR}/dubbed/test2p1v12_chunk_000_dubbed.wav"
REPORT = "/workspace/media/reports/test2p1v12_ko_20260505_150244_test2p1v12_ebc736.json"

print("=== v12 tone (English) field ===\n")
with open(REPORT) as f:
    rep = json.load(f)
for chunk in rep["chunks"]:
    for seg in chunk["segments"]:
        sid = seg["id"]
        text = seg["original_text"][:30]
        kor = seg["translated_text"][:35]
        tone = (seg.get("tts_emotion") or "")[:80]
        print(f"  [{sid}] {text:<30}")
        print(f"        ko: {kor}")
        print(f"        tone: {tone}")

# audio extract
out = f"{OUT_DIR}/v12_dubbed.wav"
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", DUBBED_V12,
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

print("\n=== v12 DUBBED 1초 단위 ===\n")
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
print(f"  v7  (English wrapping):           17/60s leak")
print(f"  v9  (English style+emotion+sit):  24/60s leak")
print(f"  v10 (English, parser fix):        18/60s leak")
print(f"  v11 (Korean instruction):         19/60s leak")
print(f"  v12 (self ref + English): {leak_count}/60s leak")
