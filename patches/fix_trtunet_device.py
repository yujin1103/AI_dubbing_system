"""TRTUNet에 dummy parameter 추가 — DiffusionPipeline.device 계산 실패 수정.

문제:
  LipsyncPipeline._execution_device 가 self.device 호출 → diffusers
  DiffusionPipeline.device 가 self.unet.parameters() 의 첫 param.device 를 읽음.
  TRTUNet 은 PyTorch param 이 없어서 (TRT 엔진이 weight 보유) StopIteration → AttributeError.

Fix:
  __init__ 끝에 1-element nn.Parameter 등록 → parameters() 가 하나라도 반환.
멱등.
"""
from pathlib import Path
import re

WRAPPER = Path("/workspace/patches/latentsync_trt_unet.py")
MARKER = "# === TRTUNet_DEVICE_PATCH"


def main():
    src = WRAPPER.read_text()
    if MARKER in src:
        print("[fix_trtunet_device] already patched")
        return 0

    # Insert dummy parameter right after super().__init__() line
    pat = re.search(r"(    def __init__\(self, engine_path: str, original_unet: Optional\[nn\.Module\] = None\):\n        super\(\)\.__init__\(\)\n)", src)
    if not pat:
        print("[fix_trtunet_device] anchor not found")
        return 1
    inject = (
        "\n        # === TRTUNet_DEVICE_PATCH ===\n"
        "        # diffusers DiffusionPipeline.device 가 unet.parameters() 의 첫 device 를 읽음.\n"
        "        # TRT 엔진은 PyTorch param 이 없으므로 dummy 1-element param 으로 device 노출.\n"
        "        self._device_marker = nn.Parameter(\n"
        "            torch.zeros(1, dtype=torch.float16, device=\"cuda\"),\n"
        "            requires_grad=False,\n"
        "        )\n"
        "        # === TRTUNet_DEVICE_PATCH end ===\n"
    )
    src_new = src[:pat.end()] + inject + src[pat.end():]
    WRAPPER.write_text(src_new)
    print("[fix_trtunet_device] OK — dummy device marker added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
