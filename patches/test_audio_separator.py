"""audio-separator BS-Roformer test on test3 audio."""
import os
import sys
import time
from pathlib import Path

print("="*60)
print("audio-separator BS-Roformer test")
print("="*60)

try:
    from audio_separator.separator import Separator
    print("[OK] audio_separator imported")
except Exception as e:
    print(f"[FAIL] import: {e}")
    sys.exit(1)

# audio-separator는 model 파일을 자동 다운로드
output_dir = "/tmp/bsroformer_test"
os.makedirs(output_dir, exist_ok=True)

print(f"\n[Init] output_dir={output_dir}")
sep = Separator(
    output_dir=output_dir,
    model_file_dir="/workspace/media/model_cache/audio_separator",
    log_level=20,  # INFO
    use_autocast=True,
)

# BS-Roformer (best for vocals as of 2024-2025)
model_name = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
print(f"\n[Load] {model_name}")
t0 = time.time()
try:
    sep.load_model(model_filename=model_name)
    print(f"[Load] OK ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"[Load] FAIL: {e}")
    sys.exit(1)

# test3 vocals 추출
input_path = "/workspace/media/output/test3_lipsync/audio_trimmed.wav"
if not os.path.exists(input_path):
    input_path = "/workspace/media/output/test3_lipsync/audio_16k.wav"
print(f"\n[Process] {input_path}")
t0 = time.time()
output_files = sep.separate(input_path)
print(f"[Process] OK ({time.time()-t0:.1f}s)")
print(f"[Output files]:")
for f in output_files:
    full_path = os.path.join(output_dir, f)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path) / 1024 / 1024
        print(f"  {full_path} ({size:.1f} MB)")
    else:
        print(f"  {f} (not found)")
