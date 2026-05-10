"""LatentSync train_unet.py에 LoRA 모드 추가 패치.

config.run.use_lora=True 시:
  1. UNet 전체 freeze
  2. PEFT LoraConfig로 attn2 + motion_modules 의 to_q/k/v 에 LoRA 부착
  3. 학습 후 LoRA 가중치만 저장 (~16MB vs 5GB)
"""
import re
from pathlib import Path

p = Path("/opt/LatentSync/scripts/train_unet.py")
src = p.read_text(encoding="utf-8")
if "LORA_PATCH" in src:
    print("이미 적용됨")
    raise SystemExit(0)

# 1) trainable_params 설정 부분에 LoRA 분기 추가
old = """    if config.model.use_motion_module:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            for trainable_module_name in config.run.trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    else:
        unet.requires_grad_(True)
        trainable_params = list(unet.parameters())"""

new = """    # LORA_PATCH: config.run.use_lora=True 시 PEFT LoRA 부착
    if getattr(config.run, "use_lora", False):
        from peft import LoraConfig, get_peft_model
        unet.requires_grad_(False)
        lora_r = int(getattr(config.run, "lora_r", 16))
        lora_alpha = int(getattr(config.run, "lora_alpha", lora_r * 2))
        lora_dropout = float(getattr(config.run, "lora_dropout", 0.0))
        # 기본 target: attn2(cross-attn)의 to_q/k/v + motion_modules
        # 사용자 정의: config.run.lora_target_modules (list)
        target = getattr(config.run, "lora_target_modules", None)
        if target is None or len(target) == 0:
            target = ["to_q", "to_k", "to_v"]
        else:
            target = list(target)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target,
            lora_dropout=lora_dropout,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        print(f"[LoRA] 적용 — r={lora_r}, alpha={lora_alpha}, target={target}", flush=True)
        unet.print_trainable_parameters()
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    elif config.model.use_motion_module:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            for trainable_module_name in config.run.trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    else:
        unet.requires_grad_(True)
        trainable_params = list(unet.parameters())"""

if old not in src:
    print("[LoRA] trainable_params 패턴 없음")
    raise SystemExit(1)
src = src.replace(old, new)

# 2) 체크포인트 저장 부분에 LoRA 전용 저장 추가
# train_unet.py에서 unet.save_pretrained(...) 또는 torch.save(...) 호출하는 부분 찾기
save_patterns = [
    r"torch\.save\(unet\.state_dict\(\), [^)]+\)",
    r"unet\.save_pretrained\([^)]+\)",
]

# LoRA mode일 때 체크포인트 저장을 LoRA-only로 바꿈
# 우선 ModelCheckpoint 단계 검색
ckpt_save_str = "torch.save(state_dict, ckpt_path)"
if ckpt_save_str in src:
    new_save = """# LORA_PATCH: LoRA mode면 LoRA 가중치만 저장 (전체 5GB 대신 ~16MB)
                if getattr(config.run, "use_lora", False):
                    lora_state = {k: v for k, v in unet.state_dict().items() if "lora_" in k}
                    torch.save(lora_state, ckpt_path)
                    print(f"[Save] LoRA-only checkpoint: {ckpt_path} ({len(lora_state)} keys)")
                else:
                    torch.save(state_dict, ckpt_path)"""
    src = src.replace(ckpt_save_str, new_save, 1)
    print("[LoRA] ckpt save 패턴 교체 완료")
else:
    print("[LoRA] 체크포인트 save 패턴 없음 — 학습 후 수동 저장 필요할 수 있음")

# 마커 추가
src = src.replace("import torch\n", "import torch\n# LORA_PATCH applied\n", 1)

p.write_text(src, encoding="utf-8")
print("[LoRA] train_unet.py 패치 완료 ✅")
