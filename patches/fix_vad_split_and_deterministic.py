"""3가지 fix 동시 적용:
1. 5분 미만 영상 → split 안 함 (chunk 1개)
2. VAD 기반 chunk split (60s ± 5s 침묵 위치에서 자르기)
3. deterministic CUDA (silhouette 결과 안정화)
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# ============================================================
# 1. import 추가 + deterministic 설정 (top of file)
# ============================================================
old_import = '''import numpy as np
import soundfile as sf
import librosa
import torch'''

new_import = '''import numpy as np
import soundfile as sf
import librosa
import torch

# DETERMINISTIC_FIX: silhouette 화자 분리 결과 안정화
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    np.random.seed(42)
    print("[Init] deterministic mode enabled (silhouette 안정화)")
except Exception as e:
    print(f"[Init] deterministic 일부 실패: {e}")'''

if old_import in src:
    src = src.replace(old_import, new_import)
    print('[1] deterministic 설정 추가')
else:
    print('[1] not found')

# ============================================================
# 2. split_video: 5분 미만 + VAD 기반 split
# ============================================================
old_split = '''    output_pattern = os.path.join(CHUNKS_DIR, f"{file_name}_chunk_%03d.mp4")
    # WHITEFRAME_FIX: -c copy는 GOP 경계 안 맞으면 흰 frame 발생
    # → libx264 re-encode + 매 segment_time마다 강제 keyframe
    cmd = [
        "ffmpeg", "-i", video_path,
        "-map", "0",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_time})",
        "-c:a", "copy",
        "-segment_time", str(segment_time),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern, "-y"
    ]'''

new_split = '''    output_pattern = os.path.join(CHUNKS_DIR, f"{file_name}_chunk_%03d.mp4")

    # SHORT_VIDEO_FIX: 5분 미만 영상은 split 안 함 (single chunk)
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True
        )
        total_dur = float(r.stdout.strip()) if r.stdout.strip() else 0.0
    except Exception:
        total_dur = 0.0

    if 0 < total_dur <= 300.0:  # 5분 이하 → 통째로
        single_chunk_path = os.path.join(CHUNKS_DIR, f"{file_name}_chunk_000.mp4")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-map", "0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            single_chunk_path, "-y"
        ]
        print(f"[Split] 영상 {total_dur:.1f}s ≤ 5분 → 단일 청크 (split 안 함)")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg single chunk 실패:\\n{result.stderr}")
        return [single_chunk_path]

    # VAD_SPLIT_FIX: silero-vad로 segment_time ± 5s 침묵 위치 찾아서 split
    split_times = _find_vad_split_points(video_path, segment_time, total_dur)
    print(f"[Split] VAD 기반 split points: {[round(t,2) for t in split_times]}")

    if split_times:
        # -segment_times 사용 (정확한 시간 지정)
        seg_str = ",".join(f"{t:.3f}" for t in split_times)
        cmd = [
            "ffmpeg", "-i", video_path,
            "-map", "0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-force_key_frames", f"{seg_str}",
            "-c:a", "copy",
            "-f", "segment",
            "-segment_times", seg_str,
            "-reset_timestamps", "1",
            output_pattern, "-y"
        ]
    else:
        # fallback: 기존 방식
        cmd = [
            "ffmpeg", "-i", video_path,
            "-map", "0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-force_key_frames", f"expr:gte(t,n_forced*{segment_time})",
            "-c:a", "copy",
            "-segment_time", str(segment_time),
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern, "-y"
        ]'''

if old_split in src:
    src = src.replace(old_split, new_split)
    print('[2] split_video VAD 적용')
else:
    print('[2] split_video 패턴 못 찾음')

# ============================================================
# 3. _find_vad_split_points 함수 추가 (split_video 직전에)
# ============================================================
vad_helper = '''
def _find_vad_split_points(video_path: str, segment_time: int, total_dur: float, tolerance: float = 5.0) -> list:
    """silero-vad로 segment_time 근처 침묵 위치 찾기.

    각 segment_time 배수 (60s, 120s, ...)에서 ±tolerance 범위 내
    가장 긴 침묵 구간의 중간 시점에서 자름. 발화 잘림 방지.
    """
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
    except ImportError:
        print("[Split] silero-vad 없음 → 일률 split fallback")
        return []

    if total_dur <= segment_time:
        return []

    # audio 추출 (16kHz mono)
    tmp_wav = os.path.join(tempfile.gettempdir(), f"vad_audio_{os.getpid()}.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", tmp_wav
    ], capture_output=True)

    if not os.path.exists(tmp_wav):
        return []

    try:
        model = load_silero_vad()
        wav = read_audio(tmp_wav, sampling_rate=16000)
        speech = get_speech_timestamps(
            wav, model, sampling_rate=16000,
            min_silence_duration_ms=300,
            min_speech_duration_ms=250,
        )
        # speech: [{start: int_samples, end: int_samples}, ...]
        speech_intervals = [(s["start"]/16000.0, s["end"]/16000.0) for s in speech]

        # 각 segment_time 배수에서 ±tolerance 범위 내 가장 긴 silence 찾기
        split_points = []
        candidate_t = segment_time
        while candidate_t < total_dur:
            t_min = candidate_t - tolerance
            t_max = min(candidate_t + tolerance, total_dur - 1.0)
            # speech 사이 silence 구간 후보
            best_split = None
            best_gap = 0.0
            for i in range(len(speech_intervals) - 1):
                _, end_i = speech_intervals[i]
                start_j, _ = speech_intervals[i+1]
                # silence 구간: end_i ~ start_j
                if end_i < t_min or start_j > t_max:
                    continue
                gap = start_j - end_i
                # 후보 split 시점: silence 중간 (t_min~t_max로 clamp)
                mid = (max(end_i, t_min) + min(start_j, t_max)) / 2.0
                if gap > best_gap:
                    best_gap = gap
                    best_split = mid

            if best_split is not None:
                split_points.append(best_split)
                candidate_t = best_split + segment_time
            else:
                # 범위 내 silence 없음 → 그냥 candidate_t에서 자름
                split_points.append(float(candidate_t))
                candidate_t += segment_time

        return split_points
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass


'''

split_marker = "def split_video(video_path: str, file_name: str, segment_time: int = 300) -> List[str]:"
if split_marker in src and "_find_vad_split_points" not in src:
    src = src.replace(split_marker, vad_helper + split_marker)
    print('[3] _find_vad_split_points 추가')
else:
    print('[3] 함수 추가 위치 못 찾음 (이미 있거나)')

p.write_text(src, encoding='utf-8')
print('[Done] 3개 fix 적용 완료')
