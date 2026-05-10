"""test3 mix만 재실행 (orchestrator 패치 적용된 mix_audio 호출).
이미 dubbed.wav + bgm.wav + chunk.mp4 있으니 mix + concat만.
"""
import sys, os, subprocess
sys.path.insert(0, '/app')

from orchestrator import mix_audio, concat_chunks, OUTPUT_DIR, CHUNKS_DIR, DUBBED_DIR, BGM_DIR

file_name = "test3"
tgt_lang = "ko"

# 기존 final.mp4 + output.mp4 삭제
for f in os.listdir(CHUNKS_DIR):
    if "final" in f:
        os.remove(os.path.join(CHUNKS_DIR, f))
        print(f"[clean] {f}")
old_out = os.path.join(OUTPUT_DIR, f"{file_name}_{tgt_lang}_dubbed.mp4")
if os.path.exists(old_out):
    os.remove(old_out)
    print(f"[clean] {old_out}")

# 각 chunk re-mix (loudnorm 적용된 mix_audio)
chunks = sorted([
    f for f in os.listdir(CHUNKS_DIR)
    if f.startswith(file_name) and f.endswith(".mp4") and "final" not in f
])
print(f"[Remix] {len(chunks)}개 chunk", flush=True)

for chunk_fn in chunks:
    chunk_name = chunk_fn.replace(".mp4", "")
    chunk_path = os.path.join(CHUNKS_DIR, chunk_fn)
    dubbed_path = os.path.join(DUBBED_DIR, f"{chunk_name}_dubbed.wav")
    bgm_path = os.path.join(BGM_DIR, f"{chunk_name}_bgm.wav")
    final_path = os.path.join(CHUNKS_DIR, f"{chunk_name}_final.mp4")

    if not all(os.path.exists(p) for p in [chunk_path, dubbed_path, bgm_path]):
        print(f"[skip] {chunk_name} 파일 부족")
        continue

    print(f"[Remix] {chunk_name}: dubbed_volume=0.65 + loudnorm I=-19", flush=True)
    mix_audio(chunk_path, dubbed_path, bgm_path, final_path,
              dubbed_volume=0.65, bgm_volume=0.5)

# concat
output_path = concat_chunks(file_name, tgt_lang)

# 호스트 복사
import shutil
host_dir = "/workspace/media/output/test3_v12_loudnorm"
os.makedirs(host_dir, exist_ok=True)
shutil.copy(output_path, host_dir)
print(f"[Done] copied to {host_dir}/", flush=True)

# 음량 검증
r = subprocess.run(
    ["ffmpeg", "-i", output_path, "-af", "volumedetect", "-vn", "-f", "null", "-"],
    capture_output=True, text=True
)
for line in r.stderr.split("\n"):
    if "mean_volume" in line or "max_volume" in line or "Duration" in line:
        print(f"[Volume] {line.strip()}", flush=True)
