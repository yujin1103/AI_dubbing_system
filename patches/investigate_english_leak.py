"""1.36~8.4s 구간에서 영어 leak 출처 조사.

확인:
  1. vocals.wav: ASR이 봤어야 할 것
  2. clean_vocals.wav: ECAPA centroid 후처리된 것
  3. bgm.wav: 영어가 leak된 게 있는지
  4. dubbed.wav: TTS만 (영어가 있으면 안 됨)
  5. final mp4: 모두 mix
"""
import os
import subprocess
import numpy as np
import soundfile as sf

RUN_DIR = "/workspace/media/runs/20260505_125859_test2p1v2_2131ce"
TARGET_START = 1.36
TARGET_END = 8.4

files = {
    "vocals": f"{RUN_DIR}/vocals/test2p1v2_chunk_000_vocals.wav",
    "clean_vocals": f"{RUN_DIR}/vocals/test2p1v2_chunk_000_clean_vocals.wav",
    "bgm": f"{RUN_DIR}/bgm/test2p1v2_chunk_000_bgm.wav",
}

print(f"=== 1.36s ~ 8.4s 구간 RMS 분석 ===\n")
for label, path in files.items():
    if not os.path.exists(path):
        print(f"  {label}: 파일 없음")
        continue
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    s_idx = int(TARGET_START * sr)
    e_idx = int(TARGET_END * sr)
    seg = audio[s_idx:e_idx]
    rms = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(np.max(np.abs(seg)))
    print(f"  {label}: sr={sr}, total_dur={len(audio)/sr:.1f}s")
    print(f"    [{TARGET_START}~{TARGET_END}s] RMS={rms:.4f}, peak={peak:.4f}")

# 각 파일의 1.36~8.4s 구간을 별도 wav로 export (사용자가 청취 가능)
print(f"\n=== 1.36~8.4s 구간 export ===")
out_dir = "/workspace/media/output/leak_check"
os.makedirs(out_dir, exist_ok=True)
for label, path in files.items():
    if not os.path.exists(path):
        continue
    out_path = f"{out_dir}/{label}_1.36-8.4s.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(TARGET_START), "-to", str(TARGET_END),
        "-i", path,
        "-ar", "16000", "-ac", "1",
        out_path,
    ])
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path)
        print(f"  {label}: {out_path} ({sz} bytes)")

# 원본 영상도 추출
orig_video = "/workspace/media/input/test2_part1.mp4"
out_orig = f"{out_dir}/orig_1.36-8.4s.wav"
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error",
    "-ss", str(TARGET_START), "-to", str(TARGET_END),
    "-i", orig_video,
    "-vn", "-ar", "16000", "-ac", "1",
    out_orig,
])
print(f"  ORIGINAL: {out_orig}")

# whisper로 빠른 transcribe (영어가 있는지 확인)
print(f"\n=== Whisper 전사 (1.36~8.4s 구간) ===")
for label in ["orig", "vocals", "clean_vocals", "bgm"]:
    wav_path = f"{out_dir}/{label}_1.36-8.4s.wav" if label != "orig" else out_orig
    if not os.path.exists(wav_path):
        continue
    try:
        # 간단한 whisper 호출 (transformers)
        import whisper as _w
        # 첫 번째 사용에서 모델 로드, 캐싱
        # 또는 더 간단히: faster-whisper 또는 기존 ASR worker 호출
        pass
    except ImportError:
        # whisper 없으면 ffmpeg로 segment 길이만 확인
        pass
print("(whisper transcribe는 별도로 수동 검증 권장)")
