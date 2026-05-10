"""htdemucs vs BS-Roformer vocals → pyannote diarization 잤 독 + 고유 결과."""
import os
import time
import sys
import subprocess
from pathlib import Path

# 1. htdemucs로 vocals 추출
INPUT = "/workspace/media/output/test3_lipsync/audio_trimmed.wav"
HTDEMUCS_DIR = "/tmp/htdemucs_test"
os.makedirs(HTDEMUCS_DIR, exist_ok=True)

print("="*60)
print("Step 1: htdemucs vocals extraction")
print("="*60)
t0 = time.time()
r = subprocess.run([
    "/opt/venv_asr/bin/python", "-m", "demucs",
    "-n", "htdemucs",
    "-o", HTDEMUCS_DIR,
    "--two-stems=vocals",
    INPUT
], capture_output=True, text=True)
print(f"htdemucs: {time.time()-t0:.1f}s, returncode={r.returncode}")
if r.returncode != 0:
    print(f"stderr: {r.stderr[:500]}")

# htdemucs 출력 위치
htdemucs_vocals = None
for root, dirs, files in os.walk(HTDEMUCS_DIR):
    for f in files:
        if "vocals" in f.lower():
            htdemucs_vocals = os.path.join(root, f)
            break

print(f"htdemucs vocals: {htdememucs_vocals}" if False else f"htdemucs vocals: {htdemucs_vocals}")
if not htdemucs_vocals:
    sys.exit("htdemucs failed")

# 2. BS-Roformer vocals (이미 있음)
bs_vocals = "/tmp/bsroformer_test/audio_trimmed_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav"
print(f"BS-Roformer vocals: {bs_vocals}")

# 3. 그 고고 file size + duration
import soundfile as sf
for label, path in [("htdemucs", htdemucs_vocals), ("BS-Roformer", bs_vocals)]:
    info = sf.info(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {label}: {info.duration:.1f}s, {info.samplerate}Hz, {info.channels}ch, {size_mb:.1f}MB")
