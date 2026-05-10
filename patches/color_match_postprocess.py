"""Color Matching 후처리 — lipsync 자국 (분홍/빨강) fix.

기반: Reinhard color transfer (LAB 색공간 mean/std 매칭).
참고: fffiloni LipSync v3.0 workflow Stage 3 (Color Matching).

원리:
    원본 frame의 face 색감을 추출
    lipsync 결과 face 색감을 원본에 매칭 (전체 face)
    mask boundary에 Gaussian feather → 자연스러운 blend

사용:
    /opt/venv_gfpgan/bin/python color_match_postprocess.py \
        --lipsync /workspace/media/output/lipsync.mp4 \
        --original /workspace/media/output/dubbed.mp4 \
        --output /workspace/media/output/color_matched.mp4 \
        --strength 0.6  # 0.0(원본) ~ 1.0(완전 매칭)
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


def reinhard_color_transfer(source: np.ndarray, target: np.ndarray,
                            strength: float = 0.6) -> np.ndarray:
    """source의 색감을 target에 매칭 (LAB 공간 mean/std).

    strength: 0.0=target 그대로, 1.0=완전 매칭.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean = src_lab.mean(axis=(0, 1), keepdims=True)
    src_std = src_lab.std(axis=(0, 1), keepdims=True) + 1e-8
    tgt_mean = tgt_lab.mean(axis=(0, 1), keepdims=True)
    tgt_std = tgt_lab.std(axis=(0, 1), keepdims=True) + 1e-8

    # Reinhard 변환
    matched = (tgt_lab - tgt_mean) * (src_std / tgt_std) + src_mean

    # strength 적용 (linear interpolation)
    result = tgt_lab * (1.0 - strength) + matched * strength

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def detect_face_box(frame: np.ndarray, prev_box=None) -> tuple:
    """간단한 face detect (Haar cascade). face 미감지 시 None 반환."""
    if not hasattr(detect_face_box, "cascade"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detect_face_box.cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face_box.cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return tuple(max(faces, key=lambda f: f[2] * f[3]))
    return None


def make_diff_mask(orig_frame: np.ndarray, lipsync_frame: np.ndarray,
                    threshold: int = 12, dilate_px: int = 5,
                    feather_px: int = 21) -> np.ndarray:
    """LatentSync가 실제 수정한 픽셀 영역을 mask로 추출.

    원리:
        |lipsync - orig| > threshold = LatentSync의 inpaint 영역
        = mask.png 모양 그대로 (회전된 face에서도 정확)

    Args:
        threshold: 최소 픽셀 차이 (12 = subtle change도 캡처)
        dilate_px: morphological dilate (작은 구멍 메우기)
        feather_px: GaussianBlur sigma (boundary 부드럽게)

    Returns:
        mask (h, w) float32, [0.0, 1.0]
    """
    diff = np.abs(lipsync_frame.astype(np.int16) -
                  orig_frame.astype(np.int16)).astype(np.uint8)
    diff_gray = diff.mean(axis=2).astype(np.uint8)
    binary = (diff_gray > threshold).astype(np.uint8) * 255
    # 작은 구멍 메우기 (close = dilate then erode)
    if dilate_px > 0:
        k = np.ones((dilate_px*2+1, dilate_px*2+1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    # boundary feather
    ksize = feather_px * 2 + 1
    mask = cv2.GaussianBlur(binary.astype(np.float32) / 255.0,
                             (ksize, ksize), feather_px / 2.0)
    return np.clip(mask, 0.0, 1.0)


def apply_color_match(orig_frame: np.ndarray, lipsync_frame: np.ndarray,
                       strength: float, prev_box=None) -> tuple:
    """LatentSync diff mask 영역에만 color match 적용 (5/7 v3 fix).

    개선:
        - rectangle/ellipse 대신 actual diff mask 사용
        - 회전된 얼굴에서도 정확한 위치 (LatentSync 실제 수정 영역)
        - 사각형 자국 완전 제거

    Returns: (output, face_box).
    """
    h, w = orig_frame.shape[:2]
    face_box = detect_face_box(orig_frame, prev_box)
    if face_box is None:
        # face 미감지 → diff mask 사용 못 하니 lipsync 그대로
        return lipsync_frame, None

    x, y, fw, fh = face_box
    pad = max(fw, fh) // 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + fw + pad)
    y2 = min(h, y + fh + pad)

    orig_face = orig_frame[y1:y2, x1:x2]
    lipsync_face = lipsync_frame[y1:y2, x1:x2]
    # color match: LAB Reinhard transfer
    matched_face = reinhard_color_transfer(orig_face, lipsync_face, strength)

    # full-frame matched (face crop만 교체, 나머지는 lipsync)
    matched_full = lipsync_frame.copy()
    matched_full[y1:y2, x1:x2] = matched_face

    # ⭐ DIFF MASK: LatentSync가 실제 수정한 영역만 → 사각형 자국 X
    diff_mask = make_diff_mask(orig_frame, lipsync_frame,
                                 threshold=12, dilate_px=5, feather_px=21)
    diff_mask = diff_mask[..., None]  # (h, w, 1) for broadcast

    # blend: matched_full where lipsync changed, lipsync_frame elsewhere
    final = (matched_full.astype(np.float32) * diff_mask +
             lipsync_frame.astype(np.float32) * (1.0 - diff_mask)).astype(np.uint8)
    return final, face_box


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
    parser.add_argument("--lipsync", required=True, help="lipsync 결과 mp4")
    parser.add_argument("--original", required=True, help="원본 (또는 더빙된) mp4 — 색감 source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=0.6,
                        help="0.0(no match) ~ 1.0(완전 매칭). 권장 0.4-0.7")
    args = parser.parse_args()

    if not os.path.isfile(args.lipsync):
        print(f"[ColorMatch] lipsync not found: {args.lipsync}")
        sys.exit(1)
    if not os.path.isfile(args.original):
        print(f"[ColorMatch] original not found: {args.original}")
        sys.exit(1)

    fps = get_video_fps(args.lipsync)
    print(f"[ColorMatch] lipsync: {args.lipsync} (fps={fps:.2f})")
    print(f"[ColorMatch] original: {args.original}")
    print(f"[ColorMatch] strength: {args.strength}")

    with tempfile.TemporaryDirectory(prefix="colormatch_") as tmpdir:
        # 1. frame 추출
        ls_dir = os.path.join(tmpdir, "lipsync_frames")
        og_dir = os.path.join(tmpdir, "orig_frames")
        out_dir = os.path.join(tmpdir, "out_frames")
        for d in (ls_dir, og_dir, out_dir):
            os.makedirs(d, exist_ok=True)

        print(f"[ColorMatch] extracting frames...")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.lipsync,
                        os.path.join(ls_dir, "%08d.png")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.original,
                        os.path.join(og_dir, "%08d.png")], check=True)

        ls_files = sorted(Path(ls_dir).glob("*.png"))
        og_files = sorted(Path(og_dir).glob("*.png"))

        if len(ls_files) != len(og_files):
            print(f"[ColorMatch] ⚠️  frame 수 불일치 (ls={len(ls_files)}, og={len(og_files)}) — 짧은 쪽 기준")
            n = min(len(ls_files), len(og_files))
        else:
            n = len(ls_files)

        print(f"[ColorMatch] processing {n} frames...")
        prev_box = None
        for i in tqdm.tqdm(range(n)):
            ls_img = cv2.imread(str(ls_files[i]))
            og_img = cv2.imread(str(og_files[i]))
            if ls_img is None or og_img is None:
                continue
            # resolution 다르면 og를 ls 크기로 (lipsync 결과가 GFPGAN으로 upscaled일 수 있음)
            if og_img.shape[:2] != ls_img.shape[:2]:
                og_img = cv2.resize(og_img, (ls_img.shape[1], ls_img.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4)
            output, prev_box = apply_color_match(og_img, ls_img, args.strength, prev_box)
            cv2.imwrite(str(out_dir + f"/{i+1:08d}.png"), output)

        # 2. assemble
        print(f"[ColorMatch] assembling...")
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
        print(f"[ColorMatch] output: {args.output}")


if __name__ == "__main__":
    main()
