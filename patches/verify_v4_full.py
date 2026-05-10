"""v4 결과 0~60s 전체 1초 단위 transcribe + BGM/dubbed 분리 분석."""
import os
import subprocess
import torch
from transformers import pipeline
import soundfile as sf
import numpy as np

OUT_DIR = "/workspace/media/output/leak_check_v4"
RUN_DIR = "/workspace/media/runs/20260505_132347_test2p1v4_6350a5"

# 추출
files = {
    "v4_final": "/workspace/media/output/test2p1v4_ko_20260505_132347_test2p1v4_6350a5.mp4",
    "v4_dubbed": f"{RUN_DIR}/dubbed/test2p1v4_chunk_000_dubbed.wav",
    "v4_bgm": f"{RUN_DIR}/bgm/test2p1v4_chunk_000_bgm.wav",
    "v4_vocals": f"{RUN_DIR}/vocals/test2p1v4_chunk_000_vocals.wav",
    "v4_clean_vocals": f"{RUN_DIR}/vocals/test2p1v4_chunk_000_clean_vocals.wav",
}
for label, src in files.items():
    out = f"{OUT_DIR}/{label}.wav"
    if not os.path.exists(out):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", src,
            "-vn" if src.endswith(".mp4") else "-c:a", "copy" if not src.endswith(".mp4") else "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            out,
        ])
    if os.path.exists(out):
        print(f"  {label}: OK")

print()

# whisper-base for better accuracy on noisy
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

def rms(path, sec_start, sec_end):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    seg = audio[int(sec_start*sr):int(sec_end*sr)]
    if seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg ** 2)))

# 전체 60s 검사 (특히 20~40s ID 2,3,4 구간)
print("=== 0~60s 1초 단위 검증 (ID 2,3,4 = 20~36s) ===\n")
print(f"{'sec':>4} | {'FINAL':<35} | {'DUBBED':<25} | {'BGM':<25}")
print("-" * 110)

for sec in range(60):
    t_final = transcribe(f"{OUT_DIR}/v4_final.wav", sec, sec+1)
    t_dubbed = transcribe(f"{OUT_DIR}/v4_dubbed.wav", sec, sec+1)
    t_bgm = transcribe(f"{OUT_DIR}/v4_bgm.wav", sec, sec+1)
    print(f"  {sec:2d} | {t_final[:33]:<35} | {t_dubbed[:23]:<25} | {t_bgm[:23]:<25}")
