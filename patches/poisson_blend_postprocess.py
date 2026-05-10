"""Poisson Seamless Cloning 후처리 — face boundary 자연스러운 blend.

원리:
    OpenCV cv2.seamlessClone (Pérez et al. 2003)
    Gradient domain에서 blend → boundary 색감 자동 매칭
    Color Match (mean/std)보다 훨씬 자연스러움

사용:
    python poisson_blend_postprocess.py \
        --lipsync /path/to/lipsync_gfpgan.mp4 \
        --original /path/to/original.mp4 \
        --output /path/to/output.mp4 \
        --mode normal   # normal / mixed (학계 표준)
"""
import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
import tqdm


def detect_face_box_haar(frame: np.ndarray, prev_box=None) -> tuple:
    """Haar cascade face detect. None 시 prev_box 폴백."""
    if not hasattr(detect_face_box_haar, "cascade"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detect_face_box_haar.cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face_box_haar.cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return tuple(max(faces, key=lambda f: f[2] * f[3]))
    return prev_box


def make_face_mask(frame_shape, face_box, feather=15) -> np.ndarray:
    """face box 기반 ellipse mask. feather (gaussian blur)로 가장자리 부드럽게.

    cv2.seamlessClone은 binary mask (0/255) 받음. blur는 안 함.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_box is None:
        return mask
    x, y, fw, fh = face_box
    cx, cy = x + fw // 2, y + fh // 2
    rx, ry = int(fw * 0.45), int(fh * 0.55)  # face 영역 ellipse (full face)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask


def apply_poisson_blend(orig_frame, lipsync_frame, prev_box=None, mode="normal") -> tuple:
    """Poisson seamless cloning으로 lipsync face → orig frame에 blend.

    Returns: (output, face_box).
    """
    h, w = orig_frame.shape[:2]
    # face detect (lipsync이 더 깨끗할 수 있어서 lipsync에서 시도)
    face_box = detect_face_box_haar(lipsync_frame, prev_box)
    if face_box is None:
        return lipsync_frame, prev_box

    x, y, fw, fh = face_box
    cx, cy = x + fw // 2, y + fh // 2

    mask = make_face_mask(orig_frame.shape, face_box)
    if mask.sum() == 0:
        return lipsync_frame, face_box

    flag = cv2.NORMAL_CLONE if mode == "normal" else cv2.MIXED_CLONE

    try:
        # cv2.seamlessClone: src(lipsync) → dst(orig) 위에 합성
        # mask 안쪽이 src로 채워지고 boundary는 gradient blending
        output = cv2.seamlessClone(
            lipsync_frame, orig_frame, mask, (cx, cy), flag
        )
        return output, face_box
    except cv2.error as e:
        # mask가 frame edge에 너무 가까우면 cv2 에러
        return lipsync_frame, face_box


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
    parser.add_argument("--lipsync", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["normal", "mixed"], default="normal",
                        help="normal=src 그대로, mixed=src+dst gradient mix")
    args = parser.parse_args()

    if not os.path.isfile(args.lipsync) or not os.path.isfile(args.original):
        print(f"[Poisson] input not found")
        sys.exit(1)

    fps = get_video_fps(args.lipsync)
    print(f"[Poisson] mode={args.mode}, fps={fps:.2f}")

    with tempfile.TemporaryDirectory(prefix="poisson_") as tmpdir:
        ls_dir = os.path.join(tmpdir, "ls")
        og_dir = os.path.join(tmpdir, "og")
        out_dir = os.path.join(tmpdir, "out")
        for d in (ls_dir, og_dir, out_dir):
            os.makedirs(d, exist_ok=True)

        print(f"[Poisson] extracting frames...")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.lipsync,
                        os.path.join(ls_dir, "%08d.png")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.original,
                        os.path.join(og_dir, "%08d.png")], check=True)

        ls_files = sorted(Path(ls_dir).glob("*.png"))
        og_files = sorted(Path(og_dir).glob("*.png"))
        n = min(len(ls_files), len(og_files))
        print(f"[Poisson] processing {n} frames...")
        prev_box = None
        for i in tqdm.tqdm(range(n)):
            ls_img = cv2.imread(str(ls_files[i]))
            og_img = cv2.imread(str(og_files[i]))
            if ls_img is None or og_img is None:
                continue
            if og_img.shape[:2] != ls_img.shape[:2]:
                og_img = cv2.resize(og_img, (ls_img.shape[1], ls_img.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4)
            output, prev_box = apply_poisson_blend(og_img, ls_img, prev_box, args.mode)
            cv2.imwrite(os.path.join(out_dir, f"{i+1:08d}.png"), output)

        print(f"[Poisson] assembling...")
        temp = args.output + ".tmp.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(out_dir, "%08d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", temp
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp, "-i", args.lipsync,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest", args.output
        ], check=True)
        os.remove(temp)
        print(f"[Poisson] output: {args.output}")


if __name__ == "__main__":
    main()
