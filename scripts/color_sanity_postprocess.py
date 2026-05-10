"""LatentSync 출력 영상의 cyan/teal blob (잘못된 색) → 원본 frame으로 복원.

문제:
  nf=4 + stage2_512 + 작은 mask 조합에서 모델이 mouth 영역에
  cyan/teal 색을 생성하는 systematic artifact 발생.
  GFPGAN으로는 색 자체를 복구 못함.

해법:
  1. lipsync 출력 영상과 원본 영상 동시에 frame 단위 비교
  2. face detector로 mouth ROI 탐지
  3. mouth ROI 내 평균 hue가 skin tone 범위 (5~50도)를 크게 벗어나면
     "잘못된 색" 판단 → 해당 영역만 원본 frame pixel로 교체 (Gaussian blend)
  4. blob 가장자리 부드럽게 (50px feather)

사용법:
  python color_sanity_postprocess.py \\
    --lipsync /path/to/out_002.mp4 \\
    --original /path/to/v_002.mp4 \\
    --output /path/to/out_002_sanitized.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def is_bad_mouth_color(roi_bgr: np.ndarray, debug: bool = False) -> tuple[bool, float]:
    """ROI가 비정상 색 (cyan/teal/blue)인지 판단.

    HSV 변환 후:
      - skin tone hue: 0~30 (red-orange-yellow) and 160~180 (red wrap)
      - cyan/teal hue: 80~110 (problem zone)
      - 평균 hue가 cyan zone에 있고 saturation 높으면 → bad
    """
    if roi_bgr.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)

    # skin tone mask
    skin_mask = ((h <= 30) | (h >= 160)) & (s >= 20)
    skin_ratio = float(skin_mask.mean())

    # cyan/teal mask (problem zone)
    cyan_mask = (h >= 80) & (h <= 130) & (s >= 30)
    cyan_ratio = float(cyan_mask.mean())

    if debug:
        print(f"  skin_ratio={skin_ratio:.2f}, cyan_ratio={cyan_ratio:.2f}")

    # cyan ratio 30% 이상이면 bad (아주 보수적)
    return cyan_ratio > 0.15, cyan_ratio


def make_feathered_mask(h: int, w: int, cx: int, cy: int, rx: int, ry: int, feather: int = 30) -> np.ndarray:
    """mouth ROI에 부드러운 ellipse mask 생성 (가장자리 Gaussian feather).

    Returns: float mask [0~1], shape (h, w, 1)
    """
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    if feather > 0:
        ksize = max(3, feather * 2 + 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather / 2.0)
    return mask[..., None]


def detect_mouth_roi_simple(face_box: tuple) -> tuple:
    """face bbox에서 mouth ROI 추정 (lower 1/3, center).

    face_box: (x1, y1, x2, y2)
    Returns: (cx, cy, rx, ry)
    """
    x1, y1, x2, y2 = face_box
    fw = x2 - x1
    fh = y2 - y1
    cx = (x1 + x2) // 2
    cy = y1 + int(fh * 0.78)  # 입은 face의 75~80% 위치
    rx = int(fw * 0.30)
    ry = int(fh * 0.16)
    return cx, cy, rx, ry


def process_video(lipsync_path: str, original_path: str, output_path: str,
                  debug_first_n: int = 0, save_debug_frames: int = 0) -> bool:
    """frame 단위로 cyan blob 감지 + 원본 복원."""
    # face detector 로드 (cv2 dnn caffemodel - haarcascade보다 정확)
    use_mp = False
    proto = "/opt/LatentSync/latentsync/utils/deploy.prototxt"
    weights = "/opt/LatentSync/latentsync/utils/res10_300x300_ssd_iter_140000.caffemodel"
    use_dnn = False
    if Path(proto).is_file() and Path(weights).is_file():
        face_det = cv2.dnn.readNetFromCaffe(proto, weights)
        use_dnn = True
        print("[ColorSanity] DNN caffemodel face detector 사용")
    else:
        face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("[ColorSanity] haarcascade face detector 사용 (fallback)")

    cap_lip = cv2.VideoCapture(lipsync_path)
    cap_orig = cv2.VideoCapture(original_path)

    if not cap_lip.isOpened() or not cap_orig.isOpened():
        print(f"[ColorSanity] 영상 못 엶: {lipsync_path} or {original_path}")
        return False

    n_lip = int(cap_lip.get(cv2.CAP_PROP_FRAME_COUNT))
    n_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_lip, n_orig)
    fps = cap_lip.get(cv2.CAP_PROP_FPS)
    w = int(cap_lip.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_lip.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[ColorSanity] {n} frames @ {fps}fps {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_silent = output_path + ".silent.mp4"
    writer = cv2.VideoWriter(tmp_silent, fourcc, fps, (w, h))

    n_replaced = 0
    n_face_fail = 0
    n_no_blob = 0

    for idx in range(n):
        ok_l, frame_l = cap_lip.read()
        ok_o, frame_o = cap_orig.read()
        if not ok_l or not ok_o:
            break

        # face detection (lipsync 영상에서)
        face_box = None
        if use_dnn:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame_l, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            face_det.setInput(blob)
            detections = face_det.forward()
            best_conf = 0
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < 0.5:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    face_box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        else:
            gray = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            faces = face_det.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # 가장 큰 face
                fx, fy, fw, fh = max(faces, key=lambda f: f[2])
                face_box = (fx, fy, fx + fw, fy + fh)

        if face_box is None:
            n_face_fail += 1
            writer.write(frame_l)
            continue

        # mouth ROI 추정
        cx, cy, rx, ry = detect_mouth_roi_simple(face_box)
        x1 = max(0, cx - rx)
        y1 = max(0, cy - ry)
        x2 = min(w, cx + rx)
        y2 = min(h, cy + ry)

        if x2 <= x1 or y2 <= y1:
            writer.write(frame_l)
            continue

        roi_lip = frame_l[y1:y2, x1:x2]
        is_bad, cyan_r = is_bad_mouth_color(roi_lip, debug=(idx < debug_first_n))

        if is_bad:
            # 원본으로 교체 (feather mask로 자연스럽게)
            mask = make_feathered_mask(h, w, cx, cy, int(rx * 1.2), int(ry * 1.2), feather=20)
            frame_blended = (frame_l.astype(np.float32) * (1 - mask) +
                             frame_o.astype(np.float32) * mask)
            frame_l = np.clip(frame_blended, 0, 255).astype(np.uint8)
            n_replaced += 1
        else:
            n_no_blob += 1

        if save_debug_frames and idx < save_debug_frames:
            dbg = frame_l.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0) if not is_bad else (0, 0, 255), 2)
            cv2.putText(dbg, f"cyan={cyan_r:.2f} {'BAD' if is_bad else 'OK'}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255) if is_bad else (0, 255, 0), 2)
            cv2.imwrite(f"/tmp/debug_{idx:04d}.png", dbg)

        writer.write(frame_l)

        if (idx + 1) % 100 == 0:
            print(f"[ColorSanity] {idx+1}/{n} (replaced={n_replaced}, face_fail={n_face_fail}, ok={n_no_blob})", flush=True)

    cap_lip.release()
    cap_orig.release()
    writer.release()

    print(f"[ColorSanity] 완료: {n_replaced}/{n} replaced (face_fail={n_face_fail})")

    # ffmpeg로 오디오 + h264 합치기
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_silent,
        "-i", lipsync_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[ColorSanity] ✅ 출력: {output_path}")
        Path(tmp_silent).unlink()
        return True
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:300] if e.stderr else str(e)
        print(f"[ColorSanity] ffmpeg 실패: {err}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lipsync", required=True, help="LatentSync 출력 mp4")
    parser.add_argument("--original", required=True, help="원본 (lipsync 적용 전) mp4")
    parser.add_argument("--output", required=True, help="복원된 mp4")
    parser.add_argument("--debug-first-n", type=int, default=0, help="첫 N frame은 cyan_ratio 출력")
    parser.add_argument("--save-debug-frames", type=int, default=0, help="첫 N frame을 /tmp/debug_*.png 저장")
    args = parser.parse_args()

    ok = process_video(args.lipsync, args.original, args.output,
                       debug_first_n=args.debug_first_n,
                       save_debug_frames=args.save_debug_frames)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
