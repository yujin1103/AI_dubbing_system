"""v4 결과에서 영어 leak 검증 - 1초별 transcribe."""
import os
import subprocess

ORIG_VIDEO = "/workspace/media/input/test2_part1.mp4"
V4_MP4 = "/workspace/media/output/test2p1v4_ko_20260505_132347_test2p1v4_6350a5.mp4"
DUBBED_V4 = "/workspace/media/runs/20260505_132347_test2p1v4_6350a5/dubbed/test2p1v4_chunk_000_dubbed.wav"

OUT_DIR = "/workspace/media/output/leak_check_v4"
os.makedirs(OUT_DIR, exist_ok=True)

# audio 추출
for label, src in [("v4_final", V4_MP4), ("v4_dubbed", DUBBED_V4)]:
    out = f"{OUT_DIR}/{label}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-vn", "-ar", "16000", "-ac", "1",
        out,
    ])
    print(f"{label}: {out}")

# whisper 1초 단위 transcribe
import torch
from transformers import pipeline
import soundfile as sf

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

print("\n=== v4 FINAL mp4 1초 단위 (영어 leak 검사) ===\n")
for sec in range(15):
    text = transcribe(f"{OUT_DIR}/v4_final.wav", sec, sec+1)
    print(f"  {sec}~{sec+1}s: {text[:80]}")

print("\n=== v4 DUBBED.wav (TTS만, 영어 있으면 안 됨) ===\n")
for sec in range(20):
    text = transcribe(f"{OUT_DIR}/v4_dubbed.wav", sec, sec+1)
    print(f"  {sec}~{sec+1}s: {text[:80]}")
