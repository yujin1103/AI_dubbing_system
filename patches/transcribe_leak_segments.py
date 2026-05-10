"""leak 의심 구간 (1.36~8.4s) 별로 transcribe해서 어디에 영어가 있는지 확인."""
import os
import subprocess
import sys

OUT_DIR = "/workspace/media/output/leak_check"

# 기존 ASR worker 활용 (venv_asr)
ASR_PYTHON = "/opt/venv_asr/bin/python"
ASR_WORKER = "/workspace/asr_worker.py"

if not os.path.exists(ASR_WORKER):
    # asr_worker가 없으면 transformers whisper로 직접
    ASR_WORKER = None

# 검사할 파일들
files = {
    "ORIG (원본 영상 audio)": "orig_full.wav",
    "FINAL (더빙 mp4 audio)": "dubbed_full.wav",
}

# 1.36~8.4s 만 transcribe
import soundfile as sf
import numpy as np

def transcribe_region(wav_path: str, start: float, end: float) -> str:
    """간단한 ASR — pyannote가 이미 있으니 transformers Whisper 사용."""
    try:
        import torch
        from transformers import pipeline
        # 작은 모델 (빠름)
        if not hasattr(transcribe_region, "_pipe"):
            transcribe_region._pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=0 if torch.cuda.is_available() else -1,
            )
        # 잘라서 임시 wav로
        audio, sr = sf.read(wav_path)
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        seg = audio[s_idx:e_idx]
        if seg.size == 0:
            return "(empty)"
        # whisper input
        result = transcribe_region._pipe({"array": seg, "sampling_rate": sr})
        return result.get("text", "").strip()
    except Exception as e:
        return f"(error: {e})"

print("=== 1초 단위 transcribe ===\n")
for label, fname in files.items():
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        print(f"{label}: file missing - {path}")
        continue
    print(f"\n[{label}]")
    for sec in range(0, 15):
        text = transcribe_region(path, sec, sec + 1)
        print(f"  {sec}~{sec+1}s: {text[:80]}")
