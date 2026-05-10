"""CosyVoice2-0.5B 다운로드."""
import os
from huggingface_hub import snapshot_download

target_dir = "/workspace/media/model_cache/modelscope/hub/FunAudioLLM/CosyVoice2-0___5B"
os.makedirs(os.path.dirname(target_dir), exist_ok=True)

print("CosyVoice2-0.5B 다운로드 시작...")
path = snapshot_download(
    repo_id="FunAudioLLM/CosyVoice2-0.5B",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)
print(f"다운로드 완료: {path}")
print(f"\n파일 목록:")
for root, dirs, files in os.walk(path):
    for f in files:
        full = os.path.join(root, f)
        size_mb = os.path.getsize(full) / 1024 / 1024
        rel = os.path.relpath(full, path)
        print(f"  {rel}: {size_mb:.1f} MB")
