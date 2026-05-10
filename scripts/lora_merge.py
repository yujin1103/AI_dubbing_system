"""LoRA 가중치를 베이스에 merge해서 단일 .pt 파일 생성.

학습 후 사용:
  /opt/venv_lipsync/bin/python /workspace/_lora_merge.py \
      --base /opt/LatentSync/checkpoints/latentsync_unet.pt \
      --lora /workspace/media/training_outputs/.../checkpoint-50000.pt \
      --out /workspace/media/lora/latentsync_ko.pt \
      --config /opt/LatentSync/configs/unet/stage2_efficient.yaml \
      --r 16 --alpha 32

결과:
  - merged 5GB .pt 파일 (베이스와 같은 형식)
  - orchestrator의 resolve_lipsync_ckpt가 자동 인식
  - 추론 시 코드 변경 불필요
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, '/opt/LatentSync')


def merge_lora(base_ckpt: str, lora_ckpt: str, out_ckpt: str,
               config_path: str, r: int = 16, alpha: int = 32,
               target_modules=None):
    import torch
    from omegaconf import OmegaConf
    from peft import LoraConfig, get_peft_model
    from latentsync.models.unet import UNet3DConditionModel

    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v"]

    print(f"=== LoRA Merge ===")
    print(f"  base:   {base_ckpt}")
    print(f"  lora:   {lora_ckpt}")
    print(f"  out:    {out_ckpt}")
    print(f"  config: {config_path}")
    print(f"  r={r}, alpha={alpha}, target={target_modules}")

    print()
    print("[1/4] 베이스 UNet 로드")
    cfg = OmegaConf.load(config_path)
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(cfg.model),
        base_ckpt,
        device="cpu",
    )
    print(f"  베이스 params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

    print()
    print("[2/4] LoRA 어댑터 부착")
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0, bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    print(f"  LoRA params: {sum(p.numel() for p in unet.parameters() if p.requires_grad)/1e6:.2f}M")

    print()
    print("[3/4] 학습된 LoRA 가중치 로드")
    lora_data = torch.load(lora_ckpt, map_location="cpu", weights_only=True)
    if isinstance(lora_data, dict) and "lora_state_dict" in lora_data:
        lora_state = lora_data["lora_state_dict"]
        print(f"  체크포인트 형식: lora_state_dict ({len(lora_state)} keys, "
              f"global_step={lora_data.get('global_step', '?')})")
    elif isinstance(lora_data, dict) and "state_dict" in lora_data:
        # 일반 ckpt — LoRA 키만 추출
        lora_state = {k: v for k, v in lora_data["state_dict"].items() if "lora_" in k}
        print(f"  일반 ckpt에서 LoRA 키 {len(lora_state)}개 추출")
    else:
        lora_state = lora_data
        print(f"  flat dict ({len(lora_state)} keys)")

    missing, unexpected = unet.load_state_dict(lora_state, strict=False)
    n_lora_keys = sum(1 for k in unet.state_dict() if "lora_" in k)
    n_loaded = n_lora_keys - sum(1 for k in missing if "lora_" in k)
    print(f"  로드된 LoRA 키: {n_loaded}/{n_lora_keys}")
    if unexpected:
        print(f"  ⚠️ unexpected keys: {len(unexpected)}")

    print()
    print("[4/4] LoRA → 베이스에 merge (merge_and_unload)")
    merged = unet.merge_and_unload()
    print(f"  merged params: {sum(p.numel() for p in merged.parameters())/1e6:.1f}M")

    print()
    print(f"[저장] {out_ckpt}")
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": merged.state_dict()}, out_ckpt)
    size_gb = Path(out_ckpt).stat().st_size / (1024 ** 3)
    print(f"  파일 크기: {size_gb:.2f} GB")

    print()
    print("✅ Merge 완료. 이 파일을 추론에 사용하면:")
    print(f"   --lipsync-ckpt {out_ckpt}")
    print(f"   또는 latentsync_ko.pt 이름으로 /workspace/media/lora/ 에 두면 자동 인식")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="베이스 .pt")
    parser.add_argument("--lora", required=True, help="학습된 LoRA .pt")
    parser.add_argument("--out", required=True, help="merged 출력 .pt")
    parser.add_argument("--config", default="/opt/LatentSync/configs/unet/stage2_efficient.yaml")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--target", nargs="+", default=["to_q", "to_k", "to_v"])
    args = parser.parse_args()

    merge_lora(args.base, args.lora, args.out, args.config,
               r=args.r, alpha=args.alpha, target_modules=args.target)


if __name__ == "__main__":
    main()
