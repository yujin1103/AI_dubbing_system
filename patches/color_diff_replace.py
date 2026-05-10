"""LatentSync 출력 영상 - 원본 영상 픽셀 단위 비교 → cyan/teal blob 자동 복원.

face detection 사용 안 함 (haarcascade가 옆모습 못 잡는 문제 우회).

알고리즘:
  1. lipsync 영상 frame과 원본 frame을 동시에 읽음
  2. 두 frame을 HSV로 변환
  3. lipsync에서 cyan zone (H 80~130, S 30+) AND 원본에서는 cyan zone 아님인 픽셀 = "잘못 생성된 cyan"
  4. 그 픽셀들을 morphological closing으로 blob 단위로 묶음
  5. 각 blob을 원본 픽셀로 대체 (Gaussian feather로 부드럽게)

결과:
  - 모델이 잘못 생성한 cyan blob만 원본으로 복원
  - 정상 lipsync (lip color OK)는 영향 없음
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def detect_cyan_blobs(frame_lip: np.ndarray, frame_orig: np.ndarray,
                      h_min: int = 80, h_max: int = 130,
                      s_min: int = 30, dilate: int = 9, min_area: int = 50) -> np.ndarray:
    """lipsync에는 cyan/teal 픽셀이 있고 원본에는 없는 영역 = 잘못된 출력.

    Returns: float mask [0~1], shape (h, w, 1) — Gaussian feather 적용
    """
    hsv_l = cv2.cvtColor(frame_lip, cv2.COLOR_BGR2HSV)
    hsv_o = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2HSV)

    # lipsync 픽셀이 cyan zone
    cyan_l = ((hsv_l[..., 0] >= h_min) & (hsv_l[..., 0] <= h_max) & (hsv_l[..., 1] >= s_min))
    # 원본 픽셀이 cyan zone (이미 cyan이면 lipsync 탓 X)
    cyan_o = ((hsv_o[..., 0] >= h_min) & (hsv_o[..., 0] <= h_max) & (hsv_o[..., 1] >= s_min))
    # lipsync에는 있고 원본에는 없는 cyan만
    bad = cyan_l & ~cyan_o

    if not bad.any():
        return None

    # morphological closing → blob 단위로 묶기
    bad_u8 = bad.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    bad_u8 = cv2.morphologyEx(bad_u8, cv2.MORPH_CLOSE, kernel)

    # 작은 blob 제거 (noise)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bad_u8, connectivity=8)
    final = np.zeros_like(bad_u8)
    for i in range(1, n_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final[labels == i] = 255

    if final.sum() == 0:
        return None

    # 마스크 추가 dilate (blob 가장자리 충분히 덮기)
    dilate_more = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final = cv2.dilate(final, dilate_more)

    # Gaussian feather (가장자리 부드럽게) — 마스크가 1.0 도달하도록 less blur
    final_f = final.astype(np.float32) / 255.0
    final_f = cv2.GaussianBlur(final_f, (15, 15), 3.0)
    # 임계값 0.3 이상은 1.0으로 (안쪽 충분히 교체)
    final_f = np.where(final_f > 0.3, np.minimum(final_f * 2.5, 1.0), final_f * 0.5)
    return final_f[..., None]


def process(lip_path: str, orig_path: str, out_path: str,
            h_min: int = 80, h_max: int = 130, s_min: int = 30,
            dilate: int = 11, min_area: int = 80,
            debug_first_n: int = 0) -> bool:
    cap_l = cv2.VideoCapture(lip_path)
    cap_o = cv2.VideoCapture(orig_path)
    if not cap_l.isOpened() or not cap_o.isOpened():
        print(f"[ColorDiff] open 실패: {lip_path} or {orig_path}")
        return False

    n_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    n_o = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_l, n_o)
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[ColorDiff] {n} frames @ {fps}fps {w}x{h}")
    print(f"[ColorDiff] cyan zone: H={h_min}-{h_max}, S>={s_min}, dilate={dilate}, min_area={min_area}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = out_path + ".silent.mp4"
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))

    n_replaced = 0
    total_pixels_replaced = 0

    for idx in range(n):
        ok_l, fl = cap_l.read()
        ok_o, fo = cap_o.read()
        if not ok_l or not ok_o:
            break

        # crop/resize 차이 보정 (lipsync는 보통 같은 사이즈지만 안전장치)
        if fl.shape != fo.shape:
            fo = cv2.resize(fo, (fl.shape[1], fl.shape[0]))

        mask = detect_cyan_blobs(fl, fo, h_min, h_max, s_min, dilate, min_area)
        if mask is None:
            writer.write(fl)
        else:
            blended = (fl.astype(np.float32) * (1 - mask) +
                       fo.astype(np.float32) * mask)
            fl = np.clip(blended, 0, 255).astype(np.uint8)
            n_replaced += 1
            total_pixels_replaced += int(mask.sum())

            if idx < debug_first_n:
                # 디버그용 마스크 시각화
                vis = (mask[..., 0] * 255).astype(np.uint8)
                cv2.imwrite(f"/tmp/diff_mask_{idx:04d}.png", vis)
                cv2.imwrite(f"/tmp/diff_after_{idx:04d}.png", fl)

        writer.write(fl)

        if (idx + 1) % 100 == 0:
            avg_pix = total_pixels_replaced / max(1, n_replaced)
            print(f"[ColorDiff] {idx+1}/{n} (replaced={n_replaced}, avg pix/frame={avg_pix:.0f})", flush=True)

    cap_l.release()
    cap_o.release()
    writer.release()

    avg_pix = total_pixels_replaced / max(1, n_replaced)
    print(f"[ColorDiff] 완료: {n_replaced}/{n} frames에 cyan 복원 (avg {avg_pix:.0f} px/frame)")

    # ffmpeg로 audio + h264
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp,
        "-i", lip_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        Path(tmp).unlink()
        sz = Path(out_path).stat().st_size / (1024 * 1024)
        print(f"[ColorDiff] ✅ 출력: {out_path} ({sz:.1f}MB)")
        return True
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:300] if e.stderr else str(e)
        print(f"[ColorDiff] ffmpeg 실패: {err}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lipsync", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--h-min", type=int, default=80, help="cyan H min")
    parser.add_argument("--h-max", type=int, default=130, help="cyan H max")
    parser.add_argument("--s-min", type=int, default=30, help="saturation min")
    parser.add_argument("--dilate", type=int, default=11, help="morphology kernel size")
    parser.add_argument("--min-area", type=int, default=80, help="min blob area in pixels")
    parser.add_argument("--debug-first-n", type=int, default=0)
    args = parser.parse_args()

    ok = process(args.lipsync, args.original, args.output,
                 args.h_min, args.h_max, args.s_min,
                 args.dilate, args.min_area,
                 debug_first_n=args.debug_first_n)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
