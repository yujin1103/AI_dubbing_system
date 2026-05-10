"""separate_audio를 BS-Roformer로 교체 (cleaner version)."""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 새 separate_audio 함수
new_func = '''def separate_audio(chunk_path: str) -> Tuple[str, str]:
    """
    BS-Roformer로 청크에서 목소리(vocals)와 배경음(BGM) 분리.

    Why BS-Roformer over Demucs htdemucs_ft:
      - htdemucs_ft: SDR ~9.5 (음악 vocals 학습, 영화 OOD)
      - BS-Roformer (model_bs_roformer_ep_317_sdr_12.9755): SDR 12.97
      - test3 검증: htdemucs 1명 vs BS-Roformer 3명 화자 detect (quiet 화자 보존)

    INPUT:
      chunk_path  : str — /data/chunks/movie_chunk_000.mp4
    OUTPUT:
      vocals_path : str — /data/vocals/movie_chunk_000_vocals.wav
      bgm_path    : str — /data/bgm/movie_chunk_000_bgm.wav
    """
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"청크 파일 없음: {chunk_path}")

    chunk_name = os.path.basename(chunk_path).replace(".mp4", "")
    if CURRENT_RUN_ID:
        temp_base = os.path.join(RUNS_DIR, CURRENT_RUN_ID, "temp")
    else:
        temp_base = tempfile.gettempdir()
    out_dir = os.path.join(temp_base, "bsroformer", chunk_name)
    os.makedirs(out_dir, exist_ok=True)

    # BS-Roformer는 audio-separator 패키지로 실행 (venv_lipsync에 설치됨)
    sep_script = (
        "import os; "
        "from audio_separator.separator import Separator; "
        "sep = Separator(output_dir=os.environ['OUT_DIR'], "
        "model_file_dir='/workspace/media/model_cache/audio_separator', "
        "log_level=30, use_autocast=True); "
        "sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt'); "
        "sep.separate(os.environ['INPUT_PATH'])"
    )

    result = subprocess.run(
        ["/opt/venv_lipsync/bin/python", "-c", sep_script],
        capture_output=True, text=True,
        env={**os.environ,
             "OUT_DIR": out_dir,
             "INPUT_PATH": chunk_path,
             "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}
    )
    if result.returncode != 0:
        raise RuntimeError(f"BS-Roformer 실패:\\n{result.stderr}")

    # BS-Roformer 출력: <name>_(Vocals)_<model>.wav, <name>_(Instrumental)_<model>.wav
    vocals_src = None
    bgm_src = None
    for f in os.listdir(out_dir):
        if "(Vocals)" in f:
            vocals_src = os.path.join(out_dir, f)
        elif "(Instrumental)" in f:
            bgm_src = os.path.join(out_dir, f)

    if not vocals_src or not bgm_src:
        raise FileNotFoundError(f"BS-Roformer 출력 누락: {os.listdir(out_dir)}")

    vocals_dst = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
    bgm_dst    = os.path.join(BGM_DIR,    f"{chunk_name}_bgm.wav")

    shutil.move(vocals_src, vocals_dst)
    shutil.move(bgm_src,    bgm_dst)
    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"[Separate-BSR] vocals: {vocals_dst}")
    print(f"[Separate-BSR] bgm:    {bgm_dst}")
    return vocals_dst, bgm_dst'''

# 기존 separate_audio 함수 찾기 (정확한 시작 + 끝)
import re

# 함수 시작부터 다음 def까지
pattern = re.compile(
    r'def separate_audio\(chunk_path: str\) -> Tuple\[str, str\]:.*?(?=\n\n# ─── Step 3:|\ndef transcribe)',
    re.DOTALL
)

match = pattern.search(src)
if match:
    old_func = match.group(0).rstrip()
    src_new = src.replace(old_func, new_func)
    if src_new != src:
        p.write_text(src_new)
        print(f"OK: separate_audio 교체 완료 ({len(old_func)} → {len(new_func)} chars)")
    else:
        print("WARN: replace 실패 (string mismatch)")
else:
    print("NOT FOUND: separate_audio 함수 못 찾음")
