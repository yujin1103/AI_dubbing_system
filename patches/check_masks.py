"""5개 mask 변형 통계 비교."""
from PIL import Image
import numpy as np

for f in ["mask.png", "mask_orig.png", "mask2.png", "mask3.png", "mask4.png"]:
    try:
        m = np.array(Image.open(f"/opt/LatentSync/latentsync/utils/{f}").convert("L"))
        print(f"{f:20} {m.shape}  mean={m.mean():6.1f}  black%={(m<128).mean()*100:5.1f}")
    except Exception as e:
        print(f"{f}: {e}")
