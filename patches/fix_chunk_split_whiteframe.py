"""흰 화면 fix: chunk split 시 -c copy 대신 keyframe re-encode.

원인:
  ffmpeg -i input -c copy -segment_time 60 -reset_timestamps 1 chunk_%03d.mp4
  → -c copy는 re-encode 안 하므로 keyframe(GOP) 경계가 60s에 맞지 않으면
    chunk 시작이 P-frame이라 디코더가 reference 못 찾고 흰/회색 frame 출력

해결:
  -force_key_frames "expr:gte(t,n_forced*60)" 입력 영상에 적용해서
  매 60초마다 IDR frame 강제 + re-encode (libx264 빠름)

또는 -ss/-t 로 정확한 컷 + re-encode (각 chunk별로)
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

old = '''    output_pattern = os.path.join(CHUNKS_DIR, f"{file_name}_chunk_%03d.mp4")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-c", "copy", "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern, "-y"
    ]'''

new = '''    output_pattern = os.path.join(CHUNKS_DIR, f"{file_name}_chunk_%03d.mp4")
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

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding='utf-8')
    print('OK: chunk split keyframe re-encode 적용')
else:
    print('NOT FOUND')
