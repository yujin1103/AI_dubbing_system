"""train_unet.py 의 checkpoint save block에 LoRA-only 저장 분기 추가."""
from pathlib import Path

p = Path("/opt/LatentSync/scripts/train_unet.py")
src = p.read_text(encoding="utf-8")

if "LORA_SAVE_PATCH" in src:
    print("이미 적용됨")
    raise SystemExit(0)

old = """            # Save checkpoint and conduct validation
            if is_main_process and (global_step % config.ckpt.save_ckpt_steps == 0):
                model_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")
                state_dict = {
                    "global_step": global_step,
                    "state_dict": unet.module.state_dict(),
                }
                try:
                    torch.save(state_dict, model_save_path)
                    logger.info(f"Saved checkpoint to {model_save_path}")
                except Exception as e:
                    logger.error(f"Error saving model: {e}")"""

new = """            # Save checkpoint and conduct validation
            # LORA_SAVE_PATCH: LoRA mode면 LoRA-only state_dict만 저장 (~16MB vs 5GB)
            if is_main_process and (global_step % config.ckpt.save_ckpt_steps == 0):
                model_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")
                full_state = unet.module.state_dict()
                if getattr(config.run, "use_lora", False):
                    # LoRA 가중치만 추출해서 저장
                    lora_only = {k: v for k, v in full_state.items() if "lora_" in k}
                    state_dict = {
                        "global_step": global_step,
                        "lora_state_dict": lora_only,
                        "is_lora": True,
                    }
                    try:
                        torch.save(state_dict, model_save_path)
                        logger.info(f"Saved LoRA-only checkpoint: {model_save_path} ({len(lora_only)} keys, ~{sum(v.numel() for v in lora_only.values()) * 4 // (1024 * 1024)}MB fp32)")
                    except Exception as e:
                        logger.error(f"Error saving LoRA: {e}")
                else:
                    state_dict = {
                        "global_step": global_step,
                        "state_dict": full_state,
                    }
                    try:
                        torch.save(state_dict, model_save_path)
                        logger.info(f"Saved checkpoint to {model_save_path}")
                    except Exception as e:
                        logger.error(f"Error saving model: {e}")"""

if old not in src:
    print("[ERR] 원본 패턴 못 찾음")
    raise SystemExit(1)
src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("[LoRA Save] 적용 완료 ✅")
