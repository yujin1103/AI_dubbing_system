"""LatentSync 메모리 절약 패치.

핵심 변경:
  read_video 함수에 max_resolution 파라미터 추가
  → 입력 영상을 720p 또는 540p로 미리 resize
  → 메모리 사용 1080p 대비 50% (720p) 또는 25% (540p)

원본 동작:
  read_video(video_path) → 모든 frames 1080p로 numpy 로드
  메모리: 1080p × 25fps × 120s × 3 = 18 GB ⚠️ OOM

패치 후:
  read_video(video_path) → 720p로 resize 후 로드
  메모리: 720p × 25fps × 120s × 3 = 8 GB ✅
"""
from pathlib import Path

p = Path("/opt/LatentSync/latentsync/utils/util.py")
src = p.read_text(encoding="utf-8")

if "MEMORY_SAVE_PATCH" in src:
    print("이미 적용됨")
    raise SystemExit(0)

old = """def read_video(video_path: str, change_fps=True, use_decord=True):
    if change_fps:
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        command = (
            f"ffmpeg -loglevel error -y -nostdin -i {video_path} -r 25 -crf 18 {os.path.join(temp_dir, 'video.mp4')}"
        )
        subprocess.run(command, shell=True)
        target_video_path = os.path.join(temp_dir, "video.mp4")
    else:
        target_video_path = video_path"""

new = """def read_video(video_path: str, change_fps=True, use_decord=True):
    # MEMORY_SAVE_PATCH: 환경변수 LATENTSYNC_MAX_RESOLUTION으로 입력 영상 resize
    # 1080p (18GB) → 720p (8GB) 또는 540p (4.6GB) 메모리 절약
    max_res = int(os.environ.get("LATENTSYNC_MAX_RESOLUTION", "0"))
    if change_fps or max_res > 0:
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        # scale filter: 짧은 변을 max_res로 (가로 또는 세로 중 작은 쪽 기준)
        if max_res > 0:
            scale_filter = f"-vf scale='if(gt(iw\\,ih)\\,-2\\,{max_res})':'if(gt(iw\\,ih)\\,{max_res}\\,-2)'"
            print(f"[MEMORY_SAVE_PATCH] resize to max_res={max_res} (메모리 절약)", flush=True)
        else:
            scale_filter = ""
        command = (
            f"ffmpeg -loglevel error -y -nostdin -i {video_path} -r 25 -crf 18 {scale_filter} {os.path.join(temp_dir, 'video.mp4')}"
        )
        subprocess.run(command, shell=True)
        target_video_path = os.path.join(temp_dir, "video.mp4")
    else:
        target_video_path = video_path"""

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding="utf-8")
    print("[Memory Save] 패치 적용 완료 ✅")
    print("사용: LATENTSYNC_MAX_RESOLUTION=720 (또는 540) 환경변수로 활성화")
else:
    print("[Memory Save] 패턴 못 찾음")
    raise SystemExit(1)
