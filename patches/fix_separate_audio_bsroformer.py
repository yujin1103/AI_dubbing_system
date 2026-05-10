"""separate_audio 함수를 htdemucs_ft → BS-Roformer로 교체.

이유:
  - htdemucs_ft: 음악 vocals 학습, 영화/드라마에서 quiet 화자 손실
  - BS-Roformer: SDR 12.97 (htdemucs ~9.5보다 +3 dB), quiet 화자 보존 우수
  - test3 검증: htdemucs로 1명만 감지된 vs BS-Roformer로 3명 감지

기존 인터페이스 유지:
  Input: chunk_path (mp4)
  Output: (vocals_path, bgm_path) — 동일한 위치 + 파일명
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''def separate_audio(chunk_path: str) -> Tuple[str, str]:
    """
    Demucs로 청크에서 목소리(vocals)와 배경음(BGM) 분리.

    INPUT:
      chunk_path  : str — /data/chunks/movie_chunk_000.mp4

    OUTPUT:
      vocals_path : str — /data/vocals/movie_chunk_000_vocals.wav
      bgm_path    : str — /data/bgm/movie_chunk_000_bgm.wav
    """
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"청크 파일 없음: {chunk_path}")

    chunk_name = os.path.basename(chunk_path).replace(".mp4", "")
    # run-scoped temp 공간 우선 사용 (CURRENT_RUN_ID가 활성화된 경우)
    # 파이프라인 밖에서 호출되는 예외적 상황만 OS tempdir로 폴백
    if CURRENT_RUN_ID:
        temp_base = os.path.join(RUNS_DIR, CURRENT_RUN_ID, "temp")
    else:
        temp_base = tempfile.gettempdir()
    out_dir = os.path.join(temp_base, "demucs", chunk_name)
    os.makedirs(out_dir, exist_ok=True)

    result = subprocess.run(
        ["python", "-m", "demucs", "-n", "htdemucs_ft", "--two-stems=vocals",
         "--out", out_dir, chunk_path],
        capture_output=True, text=True,
        env={**os.environ,
             "TORCHAUDIO_USE_BACKEND_DISPATCHER": "0",
             "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}
)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs 실패:\\n{result.stderr}")

    demucs_out = os.path.join(out_dir, "htdemucs_ft", chunk_name)
    vocals_src = os.path.join(demucs_out, "vocals.wav")
    bgm_src    = os.path.join(demucs_out, "no_vocals.wav")

    if not os.path.exists(vocals_src):
        raise FileNotFoundError("Demucs 출력 파일 없음")

    vocals_dst = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
    bgm_dst    = os.path.join(BGM_DIR,    f"{chunk_name}_bgm.wav")

    shutil.move(vocals_src, vocals_dst)
    shutil.move(bgm_src,    bgm_dst)
    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"[Separate] vocals: {vocals_dst}")
    print(f"[Separate] bgm:    {bgm_dst}")
    return vocals_dst, bgm_dst'''

new = '''def separate_audio(chunk_path: str) -> Tuple[str, str]:
    """
    BS-Roformer로 청크에서 목소리(vocals)와 배경음(BGM) 분리.

    Why BS-Roformer over Demucs htdemucs_ft:
      - htdemucs_ft: SDR ~9.5 (음악 vocals 학습, 영화 OOD)
      - BS-Roformer (model_bs_roformer_ep_317_sdr_12.9755): SDR 12.97
      - test3 검증: htdemucs 1명 감지 vs BS-Roformer 3명 감지 (quiet 화자 보존)

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

    # BS-Roformer는 audio-separator 패키지로 실행 (별도 venv 필요 없음 - venv_lipsync에 설치됨)
    sep_script = '''  # noqa: E501 - 이 inner triple-quote는 separate 스크립트
    sep_script_content = "import sys, os; from audio_separator.separator import Separator; "\\
        "sep = Separator(output_dir=os.environ['OUT_DIR'], "\\
        "model_file_dir='/workspace/media/model_cache/audio_separator', "\\
        "log_level=30, use_autocast=True); "\\
        "sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt'); "\\
        "sep.separate(os.environ['INPUT_PATH'])"

    result = subprocess.run(
        ["/opt/venv_lipsync/bin/python", "-c", sep_script_content],
        capture_output=True, text=True,
        env={**os.environ,
             "OUT_DIR": out_dir,
             "INPUT_PATH": chunk_path,
             "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}
    )
    if result.returncode != 0:
        raise RuntimeError(f"BS-Roformer 실패:\\n{result.stderr}")

    # BS-Roformer 출력 파일 패턴: <name>_(Vocals)_<model>.wav, <name>_(Instrumental)_<model>.wav
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

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: separate_audio → BS-Roformer 교체 완료")
else:
    print("NOT FOUND - check orchestrator.py current state")
