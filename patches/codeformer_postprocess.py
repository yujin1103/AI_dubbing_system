"""CodeFormer 후처리 — frame 단위 face restoration.

codeformer-pip 0.0.4의 inference_codeformer.py는 folder 입력을 받음.
ffmpeg로 frame 추출 → CodeFormer → 재조합.

사용:
    /opt/venv_gfpgan/bin/python codeformer_postprocess.py \
        --input lipsync_gfpgan.mp4 \
        --output lipsync_codeformer.mp4 \
        --fidelity 0.7   # 0.5(quality) ~ 1.0(identity)
"""
import argparse
import os
import sys
import subprocess
import shutil
import tempfile


def get_video_fps(video_path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path
    ]).decode().strip()
    if "/" in out:
        n, d = out.split("/")
        return float(n) / float(d)
    return float(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fidelity", type=float, default=0.7,
                        help="0.5(quality 우선) ~ 1.0(identity 우선)")
    parser.add_argument("--upscale", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[CodeFormer] input not found: {args.input}", flush=True)
        sys.exit(1)

    fps = get_video_fps(args.input)
    print(f"[CodeFormer] input: {args.input} (fps={fps:.2f})", flush=True)

    with tempfile.TemporaryDirectory(prefix="codeformer_") as tmpdir:
        frame_dir = os.path.join(tmpdir, "frames")
        result_dir = os.path.join(tmpdir, "results")
        os.makedirs(frame_dir, exist_ok=True)

        print(f"[CodeFormer] extracting frames...", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", args.input,
            os.path.join(frame_dir, "%08d.png")
        ], check=True)

        print(f"[CodeFormer] running CodeFormer (fidelity={args.fidelity}, upscale={args.upscale})...", flush=True)
        # codeformer.inference_codeformer CLI 호출
        cmd = [
            "/opt/venv_gfpgan/bin/python", "-m", "codeformer.inference_codeformer",
            "-i", frame_dir,
            "-o", result_dir,
            "-w", str(args.fidelity),
            "-s", str(args.upscale),
        ]
        subprocess.run(cmd, check=True)

        # CodeFormer가 생성한 final_results 폴더 찾기
        # Output structure: result_dir/<input_name>_<w>/final_results/*.png
        final_dir = None
        for root, dirs, files in os.walk(result_dir):
            if "final_results" in dirs:
                final_dir = os.path.join(root, "final_results")
                break
        if final_dir is None:
            # 또는 단순히 result_dir 안에 직접 png
            png_files = []
            for root, dirs, files in os.walk(result_dir):
                png_files.extend([os.path.join(root, f) for f in files if f.endswith(".png")])
            if png_files:
                # 다 같은 폴더로 모음
                final_dir = os.path.join(tmpdir, "all_results")
                os.makedirs(final_dir, exist_ok=True)
                for p in png_files:
                    shutil.copy(p, final_dir)
            else:
                print(f"[CodeFormer] ❌ no output frames in {result_dir}", flush=True)
                sys.exit(1)

        # 영상 재조합
        print(f"[CodeFormer] assembling video...", flush=True)
        temp_video = args.output + ".tmp.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", os.path.join(final_dir, "*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            temp_video,
        ], check=True)
        # audio merge (원본 영상의 audio)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video, "-i", args.input,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest", args.output,
        ], check=True)
        os.remove(temp_video)
        print(f"[CodeFormer] output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
