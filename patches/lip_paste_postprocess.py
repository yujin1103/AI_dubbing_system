"""Lip-Only Paste 후처리 — LatentSync 패치 자국 완전 제거.

핵심 아이디어:
    LatentSync는 face_crop (rectangle) 전체를 paste-back → cheek pink + 경계 자국
    → 우리는 INPUT 영상 (cheek 정상) 위에 LipSync OUTPUT의 입 영역만 paste
    → cheek/얼굴 외곽/배경: 원본 그대로 (자국 X)
    → 입: lipsync 결과 (정확한 한국어 발음)

원리:
    1. Face detect (Haar cascade) → face_box
    2. Mouth region = face_box 하단 40% (heuristic)
    3. Ellipse mask + heavy GaussianBlur (boundary 부드럽게)
    4. composite = orig × (1-mask) + lipsync × mask

Trade-off:
    - 입 영역만 lipsync 반영 (cheek pink, mask 자국 X)
    - 턱선 약간 미스매치 가능 (얼굴 회전 시) — minor
    - 말하지 않을 때 입 정지 = 자연스러움 향상

사용:
    /opt/venv_gfpgan/bin/python lip_paste_postprocess.py \\
        --lipsync /workspace/media/output/test15_v58_lora_lipsync.mp4 \\
        --original /workspace/media/output/test15_v27_dubbed_only.mp4 \\
        --output /workspace/media/output/test15_v58_lora_lippaste.mp4
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


def detect_face_box(frame: np.ndarray) -> tuple:
    """Haar cascade face detect. None 반환 가능."""
    if not hasattr(detect_face_box, "cascade"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detect_face_box.cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face_box.cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return tuple(max(faces, key=lambda f: f[2] * f[3]))
    return None


def make_mouth_mask(frame_shape, face_box, mouth_ratio_y=0.55,
                     mouth_height_ratio=0.45, mouth_width_ratio=0.7,
                     feather_ratio=0.25) -> np.ndarray:
    """Face box 하단의 입+턱 영역만 ellipse mask로 추출.

    Args:
        face_box: (x, y, w, h)
        mouth_ratio_y: face_box 상단부터 mouth 시작 비율 (0.55 = 코 아래)
        mouth_height_ratio: mouth 영역 높이 (face height 대비)
        mouth_width_ratio: mouth 너비 (face width 대비)
        feather_ratio: GaussianBlur sigma (face 크기 대비)

    Returns:
        mask (h, w) float32 [0, 1]
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if face_box is None:
        return mask
    x, y, fw, fh = face_box
    # mouth 중심 = face 하단 영역 중앙
    mouth_y_top = y + int(fh * mouth_ratio_y)
    mouth_y_bot = min(y + fh, mouth_y_top + int(fh * mouth_height_ratio))
    cx = x + fw // 2
    cy = (mouth_y_top + mouth_y_bot) // 2
    rx = int(fw * mouth_width_ratio / 2)
    ry = (mouth_y_bot - mouth_y_top) // 2
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    # heavy feather
    feather = max(20, int(min(fw, fh) * feather_ratio))
    ksize = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), feather / 2.0)
    return np.clip(mask, 0.0, 1.0)


def composite_lip(orig_frame: np.ndarray, lipsync_frame: np.ndarray,
                   prev_box=None) -> tuple:
    """입 영역만 lipsync로 교체, 나머지 원본 유지.

    Returns: (composite, face_box).
    """
    h, w = orig_frame.shape[:2]
    # face detect on ORIGINAL (lipsync 안 건드린 frame)
    face_box = detect_face_box(orig_frame)
    if face_box is None:
        # face 미감지 → 원본 100%
        return orig_frame, prev_box

    # resolution 다르면 align
    if lipsync_frame.shape != orig_frame.shape:
        lipsync_frame = cv2.resize(lipsync_frame, (w, h),
                                     interpolation=cv2.INTER_LANCZOS4)

    mask = make_mouth_mask(orig_frame.shape, face_box)
    mask = mask[..., None]  # (h, w, 1)

    composite = (orig_frame.astype(np.float32) * (1.0 - mask) +
                  lipsync_frame.astype(np.float32) * mask).astype(np.uint8)
    return composite, face_box


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
    parser.add_argument("--lipsync", required=True, help="LatentSync 결과 mp4")
    parser.add_argument("--original", required=True, help="원본 (더빙된, lipsync 전) mp4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mouth-ratio-y", type=float, default=0.55,
                         help="face 상단부터 mouth 시작 (0.55 = 코 아래, default)")
    parser.add_argument("--feather-ratio", type=float, default=0.25,
                         help="boundary feather 강도 (0.25 = face 크기의 25%%)")
    args = parser.parse_args()

    if not os.path.isfile(args.lipsync) or not os.path.isfile(args.original):
        print(f"[LipPaste] input not found")
        sys.exit(1)

    fps = get_video_fps(args.lipsync)
    print(f"[LipPaste] lipsync: {args.lipsync} (fps={fps:.2f})")
    print(f"[LipPaste] original: {args.original}")
    print(f"[LipPaste] mouth_ratio_y={args.mouth_ratio_y}, feather_ratio={args.feather_ratio}")

    with tempfile.TemporaryDirectory(prefix="lippaste_") as tmpdir:
        ls_dir = os.path.join(tmpdir, "ls")
        og_dir = os.path.join(tmpdir, "og")
        out_dir = os.path.join(tmpdir, "out")
        for d in (ls_dir, og_dir, out_dir):
            os.makedirs(d, exist_ok=True)

        # 5/7 fix: 두 영상의 framerate 강제 동기화
        # 원본이 30fps, lipsync가 25fps인 경우 frame index가 시간상 다름
        # → "-vf fps={fps}" 로 둘 다 lipsync fps로 resample
        print(f"[LipPaste] extracting frames at {fps:.2f} fps (force sync)...")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.lipsync,
                         "-vf", f"fps={fps}",
                         os.path.join(ls_dir, "%08d.png")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.original,
                         "-vf", f"fps={fps}",
                         os.path.join(og_dir, "%08d.png")], check=True)

        ls_files = sorted(Path(ls_dir).glob("*.png"))
        og_files = sorted(Path(og_dir).glob("*.png"))
        n = min(len(ls_files), len(og_files))
        if len(ls_files) != len(og_files):
            print(f"[LipPaste] frame 수 불일치 (ls={len(ls_files)}, og={len(og_files)}) — 짧은 쪽 기준 {n}")
        else:
            print(f"[LipPaste] processing {n} frames...")

        prev_box = None
        face_count = 0
        for i in tqdm.tqdm(range(n)):
            ls_img = cv2.imread(str(ls_files[i]))
            og_img = cv2.imread(str(og_files[i]))
            if ls_img is None or og_img is None:
                continue
            output, prev_box = composite_lip(og_img, ls_img, prev_box)
            if prev_box is not None:
                face_count += 1
            cv2.imwrite(os.path.join(out_dir, f"{i+1:08d}.png"), output)

        print(f"[LipPaste] {face_count}/{n} frames에 lipsync 입 합성됨 (나머지는 원본)")

        # assemble
        print(f"[LipPaste] assembling...")
        temp = args.output + ".tmp.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(out_dir, "%08d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", temp
        ], check=True)
        # audio from lipsync (한국어 더빙)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp, "-i", args.lipsync,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest", args.output
        ], check=True)
        os.remove(temp)
        print(f"[LipPaste] output: {args.output}")


if __name__ == "__main__":
    main()
