"""LoRA Resume 패치 — train_unet.py에 LoRA-only ckpt에서 resume 기능 추가.

LoRA-only ckpt 형식 (lora_save_patch가 저장):
  {"global_step": int, "lora_state_dict": dict, "is_lora": True}

흐름:
  1. base UNet 로드 (line 128, 기존)
  2. PEFT LoRA 부착 (line 171, 기존)
  3. NEW: config.ckpt.lora_resume_path 있으면 LoRA weight + global_step 복원

사용:
  config.ckpt.lora_resume_path = "/path/to/checkpoint-5000.pt"
  → 5000 step부터 학습 재개
"""
from pathlib import Path

p = Path("/opt/LatentSync/scripts/train_unet.py")
src = p.read_text(encoding="utf-8")

if "LORA_RESUME_PATCH" in src:
    print("이미 적용됨")
    raise SystemExit(0)

# LoRA 부착 코드 다음에 weight + global_step 복원 로직 추가
old = "        unet = get_peft_model(unet, lora_config)"

new = """        unet = get_peft_model(unet, lora_config)

        # LORA_RESUME_PATCH: LoRA-only ckpt에서 weight + global_step 복원
        lora_resume_path = getattr(config.ckpt, "lora_resume_path", "")
        if lora_resume_path and str(lora_resume_path).strip():
            print(f"[LoRA Resume] Loading from {lora_resume_path}", flush=True)
            ckpt = torch.load(lora_resume_path, map_location="cpu")
            if isinstance(ckpt, dict) and ckpt.get("is_lora"):
                # 저장 형식: {"global_step": int, "lora_state_dict": dict, "is_lora": True}
                lora_state = ckpt["lora_state_dict"]
                missing, unexpected = unet.load_state_dict(lora_state, strict=False)
                # missing: base 모델 keys (LoRA 외) — 기대값 (수천개)
                # unexpected: 0이어야 함
                lora_missing = [k for k in missing if "lora_" in k]
                if lora_missing:
                    print(f"[LoRA Resume] WARN: LoRA key 누락 {len(lora_missing)}개", flush=True)
                if unexpected:
                    print(f"[LoRA Resume] WARN: unexpected keys {len(unexpected)}개", flush=True)
                resume_global_step = ckpt.get("global_step", 0)
                print(f"[LoRA Resume] ✅ Loaded {len(lora_state)} LoRA keys, global_step={resume_global_step}", flush=True)
            else:
                print(f"[LoRA Resume] ERR: not a LoRA ckpt (is_lora missing). 처음부터 시작.", flush=True)"""

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding="utf-8")
    print("[LoRA Resume] 패치 적용 완료 ✅")
else:
    print("[LoRA Resume] 패턴 못 찾음")
    raise SystemExit(1)
