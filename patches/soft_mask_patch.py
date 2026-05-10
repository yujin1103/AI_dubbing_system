"""LatentSync mask.png를 더 부드러운 Gaussian feather 버전으로 교체.

원본: 하드 binary edge (~5px transition)
새 버전: Gaussian blur sigma=15 (~50px gradient)

효과:
  - 옆모습/회전 시 마스크 경계가 거의 보이지 않음
  - 얼굴 외곽선과 inpainting 결과의 조화 ↑
  - 사용자 보고: "고개 돌릴 때 마스크 그대로 드러남" 해결

원본 백업: mask.png.orig
"""
import shutil
from pathlib import Path

import cv2
import numpy as np


def create_soft_mask(orig_path: Path, out_path: Path, feather: int = 15) -> bool:
    """원본 마스크에 Gaussian blur 적용해서 부드러운 버전 생성.

    feather=15 → ~50px 경계 gradient (256x256 기준).
    값을 더 키우면 더 부드러워짐 but 입 영역이 줄어듦.
    """
    if not orig_path.exists():
        print(f"[SoftMask] 원본 mask 없음: {orig_path}")
        return False

    # 백업
    backup = orig_path.with_suffix(orig_path.suffix + ".orig")
    if not backup.exists():
        shutil.copy(orig_path, backup)
        print(f"[SoftMask] 원본 백업: {backup}")

    # 원본 읽기 (RGB or grayscale 모두 지원)
    img = cv2.imread(str(orig_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[SoftMask] mask.png 읽기 실패")
        return False

    # 다채널이면 첫 채널만 사용 (mask는 흑백 본질)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Gaussian blur로 feather (kernel size = feather*2+1 권장)
    kernel = feather * 2 + 1
    soft = cv2.GaussianBlur(gray, (kernel, kernel), feather)

    # ⭐ 추가 개선: 얼굴 외곽선과 더 잘 맞도록 mask 영역을 약간 확장
    # 원본 mask의 black(0) 영역(=inpainted)을 dilate해서 boundary push out
    # 그 후 다시 Gaussian blur → 경계가 더 안쪽으로 밀려서 잘 안 보임
    binary = (gray < 128).astype(np.uint8) * 255
    dilated = cv2.dilate(binary, np.ones((feather // 3, feather // 3), np.uint8), iterations=1)
    expanded = cv2.GaussianBlur(255 - dilated, (kernel, kernel), feather)
    soft = np.minimum(soft, expanded)  # 더 안전한 (보존 영역 ↓) 쪽으로

    # RGB로 변환 (LatentSync는 mask 3채널 기대)
    soft_rgb = cv2.cvtColor(soft, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(str(out_path), soft_rgb)
    print(f"[SoftMask] 저장: {out_path}")
    print(f"  feather={feather}, gradient ~{feather*3}px")
    return True


def main() -> int:
    mask_path = Path("/opt/LatentSync/latentsync/utils/mask.png")
    if not mask_path.parent.exists():
        print(f"[SoftMask] LatentSync utils 폴더 없음 — SKIP (clone 후 다시)")
        return 0

    ok = create_soft_mask(mask_path, mask_path, feather=15)
    if ok:
        print("[SoftMask] ✅ 적용 완료 (build 또는 컨테이너 시작 시 1회)")
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
