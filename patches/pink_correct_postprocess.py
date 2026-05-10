"""HSV Pink 보정 후처리 — LoRA cheek pink artifact 선택적 제거.

원리:
    lipsync 영상만 사용 (원본 paste 없음 → ghost 없음)
    HSV 공간에서 face skin 영역의 pink hue + high saturation 픽셀만 desaturate
    → cheek pink가 자연스러운 살색으로 보정

핵심 차이점 (vs lip_paste):
    - lip_paste: orig face 영상 paste → 위치 불일치 ghost
    - pink_correct: lipsync만 modify → 위치 동일, 색만 보정

사용:
    python pink_correct_postprocess.py \\
        --input /workspace/media/output/test15_v58_lora_lipsync.mp4 \\
        --output /workspace/media/output/test15_v59_pinkfix.mp4 \\
        --strength 0.5
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


def detect_face_box(frame):
    """Haar cascade face detect."""
    if not hasattr(detect_face_box, "cascade"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detect_face_box.cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face_box.cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return tuple(max(faces, key=lambda f: f[2] * f[3]))
    return None


def detect_skin_mask(frame):
    """Asian skin tone HSV detection. Returns binary mask (0 or 255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Skin tone (Asian): two H ranges (red wraps around 0/180)
    # H: 0-25 (warm skin) or 160-179 (rosy)
    # S: 20-180 (avoid neutral grays)
    # V: 70-255 (avoid shadows)
    lower1 = np.array([0, 20, 70], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 20, 70], dtype=np.uint8)
    upper2 = np.array([179, 180, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)


def correct_pink_in_face(frame, face_box, strength=0.5):
    """Face 영역의 pink-tinted skin pixels desaturate.

    Args:
        strength: 0.0 (no correction) ~ 1.0 (max desaturation)
    """
    if face_box is None:
        return frame  # face 미감지 → 원본 유지

    x, y, fw, fh = face_box
    # face crop with padding
    pad = max(fw, fh) // 6
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + fw + pad)
    y2 = min(frame.shape[0], y + fh + pad)

    face_crop = frame[y1:y2, x1:x2].copy()

    # 1. Detect skin in face crop
    skin_mask = detect_skin_mask(face_crop)

    # 2. Detect pink-tinted pixels
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch = hsv[..., 0]
    s_ch = hsv[..., 1]
    # Pink hue: H=0-15 (red-pink) or 160-179 (deep pink)
    # And saturation > 80 (visibly tinted)
    pink_hue = ((h_ch < 15) | (h_ch > 160))
    high_sat = s_ch > 80
    pink_mask = pink_hue & high_sat & (skin_mask > 0)

    # 3. Reduce saturation in pink-skin pixels
    # Linear interpolation: new_S = S × (1 - strength × pink_intensity)
    pink_intensity = (s_ch - 80) / (255 - 80)  # 0 to 1 based on how saturated
    pink_intensity = np.clip(pink_intensity, 0, 1)
    s_ch_corrected = s_ch * (1.0 - strength * pink_intensity)
    hsv[..., 1] = np.where(pink_mask, s_ch_corrected, s_ch)

    # 4. Soft feather to face boundary (avoid sharp transition)
    # gaussian blur on the pink_mask + apply
    smooth_mask = cv2.GaussianBlur(pink_mask.astype(np.float32), (15, 15), 5).astype(np.float32)
    smooth_mask = smooth_mask[..., None]

    corrected_face = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blended_face = (face_crop.astype(np.float32) * (1 - smooth_mask) +
                     corrected_face.astype(np.float32) * smooth_mask).astype(np.uint8)

    # 5. Paste back
    output = frame.copy()
    output[y1:y2, x1:x2] = blended_face
    return output


def get_video_fps(video_path):
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
    parser.add_argument("--input", required=True, help="Lipsync 결과 mp4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=0.5,
                         help="0.0(보정X) ~ 1.0(최대 desaturate). 권장 0.3-0.7")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[PinkFix] input not found: {args.input}")
        sys.exit(1)

    fps = get_video_fps(args.input)
    print(f"[PinkFix] input: {args.input} (fps={fps:.2f})")
    print(f"[PinkFix] strength: {args.strength}")

    with tempfile.TemporaryDirectory(prefix="pinkfix_") as tmpdir:
        in_dir = os.path.join(tmpdir, "in")
        out_dir = os.path.join(tmpdir, "out")
        for d in (in_dir, out_dir):
            os.makedirs(d, exist_ok=True)

        print(f"[PinkFix] extracting frames...")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.input,
                         os.path.join(in_dir, "%08d.png")], check=True)

        in_files = sorted(Path(in_dir).glob("*.png"))
        n = len(in_files)
        print(f"[PinkFix] processing {n} frames...")
        face_count = 0
        for i in tqdm.tqdm(range(n)):
            img = cv2.imread(str(in_files[i]))
            if img is None:
                continue
            face_box = detect_face_box(img)
            if face_box is not None:
                face_count += 1
            corrected = correct_pink_in_face(img, face_box, args.strength)
            cv2.imwrite(os.path.join(out_dir, f"{i+1:08d}.png"), corrected)

        print(f"[PinkFix] {face_count}/{n} frames에 face 검출 + pink 보정")

        print(f"[PinkFix] assembling...")
        temp = args.output + ".tmp.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(out_dir, "%08d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", temp
        ], check=True)
        # audio from input
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp, "-i", args.input,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest", args.output
        ], check=True)
        os.remove(temp)
        print(f"[PinkFix] output: {args.output}")


if __name__ == "__main__":
    main()
