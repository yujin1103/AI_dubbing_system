"""LoRA+ 패치 — train_unet.py optimizer를 A/B matrix 별도 lr 그룹으로.

LoRA+ (2024 paper): A matrix와 B matrix에 다른 lr 적용 → 수렴 ↑ 품질 ↑
  - lora_A (down-projection): 기본 lr
  - lora_B (up-projection): 기본 lr × ratio (보통 16x)

사용:
  config.optimizer.use_lora_plus = true
  config.optimizer.lora_plus_ratio = 16
"""
from pathlib import Path

p = Path("/opt/LatentSync/scripts/train_unet.py")
src = p.read_text(encoding="utf-8")
if "LORA_PLUS_PATCH" in src:
    print("이미 적용됨")
    raise SystemExit(0)

# ADAMW8BIT_PATCH 블록을 찾아서 LoRA+ 분기 추가
old = """    # ADAMW8BIT_PATCH: 8-bit AdamW (bitsandbytes) 옵션 — RTX 5080 16GB optimizer state 절감
    if getattr(config.optimizer, "use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=config.optimizer.lr)
            print("[Optim] bitsandbytes AdamW8bit 사용 (optimizer state ~75% 절감)", flush=True)
        except Exception as _e:
            print(f"[Optim] bnb 로드 실패 ({_e}) → torch.optim.AdamW 폴백", flush=True)
            optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)"""

new = """    # ADAMW8BIT_PATCH + LORA_PLUS_PATCH: 8-bit AdamW + (선택) LoRA+ 분리 lr
    use_8bit = getattr(config.optimizer, "use_8bit_adam", False)
    use_lora_plus = getattr(config.optimizer, "use_lora_plus", False) and getattr(config.run, "use_lora", False)
    lora_plus_ratio = float(getattr(config.optimizer, "lora_plus_ratio", 16.0))

    # LoRA+ : A matrix와 B matrix에 별도 lr (B는 16x 권장)
    if use_lora_plus:
        lora_A_params = [p for n, p in unet.named_parameters() if "lora_A" in n and p.requires_grad]
        lora_B_params = [p for n, p in unet.named_parameters() if "lora_B" in n and p.requires_grad]
        other_params  = [p for n, p in unet.named_parameters()
                         if p.requires_grad and "lora_A" not in n and "lora_B" not in n]
        param_groups = [
            {"params": lora_A_params, "lr": config.optimizer.lr},
            {"params": lora_B_params, "lr": config.optimizer.lr * lora_plus_ratio},
        ]
        if other_params:
            param_groups.append({"params": other_params, "lr": config.optimizer.lr})
        print(f"[LoRA+] A lr={config.optimizer.lr}, B lr={config.optimizer.lr * lora_plus_ratio} "
              f"(ratio={lora_plus_ratio})", flush=True)
        print(f"[LoRA+] |A|={len(lora_A_params)}, |B|={len(lora_B_params)}, |other|={len(other_params)}", flush=True)
        opt_target = param_groups
    else:
        opt_target = trainable_params

    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(opt_target, lr=config.optimizer.lr)
            print("[Optim] bitsandbytes AdamW8bit 사용", flush=True)
        except Exception as _e:
            print(f"[Optim] bnb 로드 실패 ({_e}) -> torch.optim.AdamW 폴백", flush=True)
            optimizer = torch.optim.AdamW(opt_target, lr=config.optimizer.lr)
    else:
        optimizer = torch.optim.AdamW(opt_target, lr=config.optimizer.lr)"""

if old not in src:
    print("[LoRA+] 원본 패턴 없음")
    raise SystemExit(1)

src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("[LoRA+] 패치 적용 완료 ✅")
