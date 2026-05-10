"""LatentSync face crop padding 증가 — 회전 시 마스크 안전망.

원본: ratio = resolution / 256 * 2.8 (crop_ratio)
       → 얼굴이 256 face_size 안에 빡빡하게 fit
새 버전: ratio × 1.15 (15% 추가 padding)
       → face crop 영역이 약간 넓어져 마스크가 얼굴 밖으로 안 나감
       → 옆모습/회전 시에도 mask boundary가 face 안쪽에 있음

Trade-off:
  - face detail 약간 ↓ (입 영역이 화면에서 약간 작아짐)
  - 마스크 안전성 ↑ (회전/측면 시에도 항상 얼굴 내부)

원본 백업: affine_transform.py.padding.bak
"""
import shutil
from pathlib import Path

p = Path("/opt/LatentSync/latentsync/utils/affine_transform.py")
src = p.read_text(encoding="utf-8")

if "FACE_PADDING_PATCH" in src:
    print("이미 적용됨")
    raise SystemExit(0)

old = "            ratio = resolution / 256 * 2.8"
new = (
    "            # FACE_PADDING_PATCH: ratio 1.15x → face crop 영역 15% 확장\n"
    "            #   회전/측면 시 마스크가 얼굴 외곽선 밖으로 안 나가게 안전망\n"
    "            ratio = resolution / 256 * 2.8 * 1.15"
)

if old not in src:
    print("[ERR] ratio 패턴 못 찾음")
    raise SystemExit(1)

backup = p.with_suffix(".py.padding.bak")
if not backup.exists():
    shutil.copy(p, backup)

src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print(f"[FacePadding] 적용 완료 (ratio × 1.15)")
print(f"  백업: {backup}")
