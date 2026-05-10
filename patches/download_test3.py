"""test3.mp4 다운로드 (YouTube, 1080p, 처음 2분, BGM 검증용).
다운로드 후 video/audio length sync 자동 확인 + trim.
"""
import yt_dlp
import os
import subprocess
import json

URL = "https://www.youtube.com/watch?v=IxvAhfsIfUU"
OUT_DIR = "/app/media/input"
NAME = "test3"
DURATION = 120  # 처음 2분

os.makedirs(OUT_DIR, exist_ok=True)
raw_path = f"{OUT_DIR}/{NAME}_raw.mp4"
final_path = f"{OUT_DIR}/{NAME}.mp4"


def custom_ranges(info_dict, ydl):
    return [{'start_time': 0, 'end_time': DURATION}]


ydl_opts = {
    'outtmpl': raw_path,
    'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
    'merge_output_format': 'mp4',
    'download_ranges': custom_ranges,
    'force_keyframes_at_cuts': True,
}

print(f"[Download] {URL} (처음 {DURATION}s, 1080p)", flush=True)
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL])

# 다운로드 후 검증
def get_streams(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', path],
        capture_output=True, text=True
    )
    return json.loads(r.stdout).get('streams', [])

streams = get_streams(raw_path)
v = next((s for s in streams if s['codec_type'] == 'video'), None)
a = next((s for s in streams if s['codec_type'] == 'audio'), None)

v_dur = float(v.get('duration', 0)) if v else 0
a_dur = float(a.get('duration', 0)) if a else 0
print(f"[Check] raw video: {v_dur:.2f}s, audio: {a_dur:.2f}s, diff: {abs(v_dur-a_dur):.2f}s", flush=True)

# sync 보정: 짧은 트랙 기준으로 trim (-shortest)
if abs(v_dur - a_dur) > 0.5:
    target_dur = min(v_dur, a_dur)
    print(f"[Sync] mismatch 발견 → -shortest로 trim ({target_dur:.2f}s)", flush=True)
    subprocess.run([
        'ffmpeg', '-y', '-i', raw_path,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        '-t', f'{target_dur:.3f}',
        '-map', '0:v:0', '-map', '0:a:0',
        final_path
    ], check=True)
    print(f"[Sync] trim 완료 → {final_path}", flush=True)
else:
    print("[Sync] OK, raw → final 그대로 사용", flush=True)
    os.rename(raw_path, final_path)

# 최종 검증
streams = get_streams(final_path)
v = next((s for s in streams if s['codec_type'] == 'video'), None)
a = next((s for s in streams if s['codec_type'] == 'audio'), None)
print(f"[Done] final video: {float(v.get('duration', 0)):.2f}s, audio: {float(a.get('duration', 0)):.2f}s", flush=True)
print(f"[Done] {final_path}", flush=True)

# raw 정리
if os.path.exists(raw_path):
    try:
        os.remove(raw_path)
    except Exception:
        pass
