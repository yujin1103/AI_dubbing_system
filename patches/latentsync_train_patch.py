"""LatentSync train_unet.py 빌드 시 자동 패치.

적용:
  1. 8-bit AdamW 옵션 — config.optimizer.use_8bit_adam=true 시 bitsandbytes 사용
     → optimizer state 메모리 ~75% 절감 (RTX 5080 16GB fit 핵심)
  2. gc.collect() + torch.cuda.empty_cache() 5 step마다 — 메모리 단편화 누적 방지

원본 백업: train_unet.py.bak
"""
from __future__ import annotations
import shutil
from pathlib import Path

TRAIN_PY = Path("/opt/LatentSync/scripts/train_unet.py")


def patch_optimizer():
    """torch.optim.AdamW → 옵션부 8-bit AdamW."""
    src = TRAIN_PY.read_text(encoding="utf-8")
    if "ADAMW8BIT_PATCH" in src:
        print("[Patch:8bit] 이미 적용됨")
        return False

    old = "    optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)"
    new = (
        "    # ADAMW8BIT_PATCH: 8-bit AdamW (bitsandbytes) — config.optimizer.use_8bit_adam=True 시\n"
        "    if getattr(config.optimizer, \"use_8bit_adam\", False):\n"
        "        try:\n"
        "            import bitsandbytes as bnb\n"
        "            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=config.optimizer.lr)\n"
        "            print(\"[Optim] bitsandbytes AdamW8bit 사용 (optimizer state ~75% 절감)\", flush=True)\n"
        "        except Exception as _e:\n"
        "            print(f\"[Optim] bnb 로드 실패 ({_e}) -> torch.optim.AdamW 폴백\", flush=True)\n"
        "            optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)\n"
        "    else:\n"
        "        optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)"
    )

    if old not in src:
        print("[Patch:8bit] 원본 패턴 없음")
        return False

    backup = TRAIN_PY.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy(TRAIN_PY, backup)
    TRAIN_PY.write_text(src.replace(old, new), encoding="utf-8")
    print("[Patch:8bit] 적용 완료")
    return True


def patch_empty_cache():
    """매 5 step마다 gc.collect() + torch.cuda.empty_cache() — 단편화 방지."""
    src = TRAIN_PY.read_text(encoding="utf-8")
    if "EMPTY_CACHE_PATCH" in src:
        print("[Patch:empty_cache] 이미 적용됨")
        return False

    # import gc 추가 (없으면)
    if "import gc\n" not in src:
        src = src.replace("import torch\n", "import torch\nimport gc\n", 1)

    # scaler.step(optimizer) 직후에 메모리 정리
    old_step = "scaler.step(optimizer)"
    new_step = (
        "scaler.step(optimizer)\n"
        "                # EMPTY_CACHE_PATCH: 단편화 누적 방지 (5 step마다 GC + empty_cache)\n"
        "                if global_step % 5 == 0:\n"
        "                    gc.collect()\n"
        "                    torch.cuda.empty_cache()"
    )
    if old_step not in src:
        print("[Patch:empty_cache] scaler.step 패턴 없음")
        return False

    src = src.replace(old_step, new_step, 1)
    TRAIN_PY.write_text(src, encoding="utf-8")
    print("[Patch:empty_cache] 적용 완료")
    return True


def main() -> int:
    if not TRAIN_PY.exists():
        print(f"[Patch] {TRAIN_PY} 없음 — SKIP (LatentSync clone 후 다시 실행)")
        return 0
    patch_optimizer()
    patch_empty_cache()
    print("[Patch] LatentSync 학습 패치 완료 ✅")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
