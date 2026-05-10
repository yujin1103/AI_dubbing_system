"""3가지 fix 동시 적용:
1. BGM subtraction: 원본 audio - vocals = bgm_subtracted (Demucs 누락 방지)
2. dubbed_volume 0.65 → 0.5 (한국어 더 작게)
3. segment별 RMS matching: 원본 vocals의 음량 envelope를 한국어에도 적용 (자연스러운 다이내믹스)
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# ============================================================
# 1. dubbed_volume default 0.65 → 0.5
# ============================================================
old1 = '    dubbed_volume: float = 0.65,'
new1 = '    dubbed_volume: float = 0.5,'
if old1 in src:
    src = src.replace(old1, new1)
    print('[1] dubbed_volume 0.5 적용')
else:
    print('[1] not found')

# ============================================================
# 2. BGM subtraction: separate_audio 끝부분에 추가
# ============================================================
old2 = '''    shutil.move(vocals_src, vocals_dst)
    shutil.move(bgm_src,    bgm_dst)
    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"[Separate] vocals: {vocals_dst}")
    print(f"[Separate] bgm:    {bgm_dst}")
    return vocals_dst, bgm_dst'''

new2 = '''    shutil.move(vocals_src, vocals_dst)
    shutil.move(bgm_src,    bgm_dst)
    shutil.rmtree(out_dir, ignore_errors=True)

    # BGM_SUBTRACTION_FIX: 원본 audio - vocals = 누락 없는 진짜 BGM
    # Demucs가 음악/SFX/환경음 일부를 vocals로 잘못 분류하는 한계 우회
    try:
        # 원본 audio 추출
        orig_audio_path = os.path.join(tempfile.gettempdir(), f"orig_{chunk_name}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", chunk_path,
            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
            orig_audio_path
        ], capture_output=True)

        if os.path.exists(orig_audio_path):
            orig, sr_o = sf.read(orig_audio_path)
            voc,  sr_v = sf.read(vocals_dst)

            # sample rate 일치 보장
            if sr_v != sr_o:
                voc = librosa.resample(
                    voc.T if voc.ndim > 1 else voc,
                    orig_sr=sr_v, target_sr=sr_o
                )
                if voc.ndim > 1:
                    voc = voc.T

            # 길이 맞춤
            min_len = min(len(orig), len(voc))
            orig = orig[:min_len]
            voc  = voc[:min_len]

            # subtraction (channel 일치 필요)
            if orig.ndim != voc.ndim:
                if orig.ndim > 1 and voc.ndim == 1:
                    voc = np.stack([voc, voc], axis=1)
                elif orig.ndim == 1 and voc.ndim > 1:
                    voc = voc.mean(axis=1)

            bgm_subtracted = orig - voc

            # 기존 bgm은 backup 후 subtracted로 덮어쓰기
            backup_path = bgm_dst.replace(".wav", "_demucs.wav")
            shutil.move(bgm_dst, backup_path)
            sf.write(bgm_dst, bgm_subtracted, sr_o)
            print(f"[Separate] BGM subtraction 적용 (demucs backup: {os.path.basename(backup_path)})")

            os.unlink(orig_audio_path)
    except Exception as e:
        print(f"[Separate] BGM subtraction 실패 (Demucs BGM 그대로 사용): {e}")

    print(f"[Separate] vocals: {vocals_dst}")
    print(f"[Separate] bgm:    {bgm_dst}")
    return vocals_dst, bgm_dst'''

if old2 in src:
    src = src.replace(old2, new2)
    print('[2] BGM subtraction 추가')
else:
    print('[2] not found')

# ============================================================
# 3. synthesize_chunk에서 원본 vocals RMS matching
# ============================================================
# synthesize_chunk 시작부에 vocals_path 추출 + 원본 vocals 로드
old3 = '''def synthesize_chunk(
    segments: List[Segment],
    profiles: Dict[str, SpeakerProfile],
    chunk_name: str,
    tgt_lang: str,
) -> str:
    """
    청크 전체 더빙 오디오 생성.
    MOS 평가로 품질이 낮은 세그먼트는 자동 재합성 (최대 3회).
    """
    if not segments:
        raise ValueError("segments가 비어 있습니다")'''

new3 = '''def synthesize_chunk(
    segments: List[Segment],
    profiles: Dict[str, SpeakerProfile],
    chunk_name: str,
    tgt_lang: str,
) -> str:
    """
    청크 전체 더빙 오디오 생성.
    MOS 평가로 품질이 낮은 세그먼트는 자동 재합성 (최대 3회).
    각 segment의 한국어 음량을 원본 vocals 음량(RMS)에 매칭 → 자연스러운 다이내믹스.
    """
    if not segments:
        raise ValueError("segments가 비어 있습니다")

    # DYNAMICS_MATCH_FIX: 원본 vocals 로드 (segment별 RMS 매칭용)
    orig_vocals = None
    orig_sr = None
    try:
        vocals_path_orig = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
        if os.path.exists(vocals_path_orig):
            orig_vocals, orig_sr = sf.read(vocals_path_orig)
            if orig_vocals.ndim > 1:
                orig_vocals = orig_vocals.mean(axis=1)
            print(f"[TTS] dynamics matching: 원본 vocals 로드 ({orig_sr}Hz, {len(orig_vocals)/orig_sr:.1f}s)")
    except Exception as e:
        print(f"[TTS] 원본 vocals 로드 실패 (dynamics matching 스킵): {e}")
        orig_vocals = None'''

if old3 in src:
    src = src.replace(old3, new3)
    print('[3a] synthesize_chunk 원본 vocals 로드 추가')
else:
    print('[3a] not found')

# best_audio 결정 후 원본 RMS와 매칭
old4 = '''        if best_audio is None:
            best_audio = np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

        # 세그먼트에 MOS 점수 저장 (JSON용)
        seg._tts_mos = best_mos
        seg._tts_retries = min(max_retries, max(0, max_retries - 1)) if max_retries > 1 else 0'''

new4 = '''        if best_audio is None:
            best_audio = np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

        # DYNAMICS_MATCH: 원본 vocals 같은 시간 RMS 매칭 → 자연스러운 음량 변동
        if orig_vocals is not None and orig_sr is not None:
            o_start = int(seg.start * orig_sr)
            o_end   = int(seg.end * orig_sr)
            orig_seg = orig_vocals[max(0, o_start):min(len(orig_vocals), o_end)]
            if len(orig_seg) > 0:
                orig_rms = float(np.sqrt(np.mean(orig_seg.astype(np.float64) ** 2)))
                dub_rms  = float(np.sqrt(np.mean(best_audio.astype(np.float64) ** 2)))
                if orig_rms > 1e-5 and dub_rms > 1e-5:
                    gain = orig_rms / dub_rms
                    gain = float(np.clip(gain, 0.3, 2.5))  # 너무 극단 방지
                    best_audio = best_audio * gain

        # 세그먼트에 MOS 점수 저장 (JSON용)
        seg._tts_mos = best_mos
        seg._tts_retries = min(max_retries, max(0, max_retries - 1)) if max_retries > 1 else 0'''

if old4 in src:
    src = src.replace(old4, new4)
    print('[3b] segment별 RMS matching 추가')
else:
    print('[3b] not found')

p.write_text(src, encoding='utf-8')
print('[Done] BGM subtract + dubbed 0.5 + dynamics match 적용')
