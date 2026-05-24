"""Merge LoRA adapters into base UNet weights at given scale.

Why: TRT engine is built from base weights. To use LoRA at inference time,
either (a) wrap with PEFT at runtime, or (b) merge B*A into base weights.

This script does (b): merged weights remain a vanilla UNet state_dict,
compatible with the existing inference pipeline and TRT engine (re-export needed).

Math (for each LoRA-targeted Linear layer):
  W_merged = W_base + scale * (B @ A)

Where:
  A ∈ ℝ^(r × in_dim)   (lora_A weight)
  B ∈ ℝ^(out_dim × r)  (lora_B weight)
  scale = lora_alpha / r * user_scale

LoRA target modules (from training config):
  - to_q, to_k, to_v, to_out.0

Usage:
  python merge_lora_into_base.py \
      --base /workspace/media/lora/latentsync_ko_50k.pt \
      --lora /workspace/media/training_outputs/lora_nf4_train/.../checkpoint-45000.pt \
      --scale 0.7 \
      --output /workspace/media/lora/merged_scale_0.7.pt
"""
from __future__ import annotations
import argparse
import os
import time
import torch


def _log(msg):
    print(f"[merge_lora] {msg}", flush=True)


def detect_lora_keys(state_dict):
    """Find all (lora_A, lora_B) pairs in state dict."""
    pairs = {}
    for k in state_dict.keys():
        if "lora_A" in k and k.endswith(".weight"):
            # Extract module path: base_model.model.X.Y.lora_A.default.weight
            # → X.Y is the base module
            base_key = k.replace(".lora_A.default.weight", "")
            base_key = base_key.replace("base_model.model.", "")
            b_key = k.replace("lora_A", "lora_B")
            if b_key in state_dict:
                pairs[base_key] = (k, b_key)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base UNet checkpoint")
    ap.add_argument("--lora", required=True, help="LoRA checkpoint (PEFT state_dict)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="LoRA merge scale (0.0=base, 1.0=full LoRA)")
    ap.add_argument("--lora-r", type=int, default=32,
                    help="LoRA rank (from training config)")
    ap.add_argument("--lora-alpha", type=float, default=16.0,
                    help="LoRA alpha (from training config)")
    ap.add_argument("--output", required=True, help="Output merged checkpoint")
    args = ap.parse_args()

    _log(f"loading base: {args.base}")
    t0 = time.time()
    base_ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    if isinstance(base_ckpt, dict) and "state_dict" in base_ckpt:
        base_sd = base_ckpt["state_dict"]
    else:
        base_sd = base_ckpt
    _log(f"  base keys: {len(base_sd)}")

    _log(f"loading LoRA ckpt: {args.lora}")
    lora_ckpt = torch.load(args.lora, map_location="cpu", weights_only=False)
    lora_sd = lora_ckpt["state_dict"] if isinstance(lora_ckpt, dict) and "state_dict" in lora_ckpt else lora_ckpt
    is_lora = lora_ckpt.get("is_lora", False) if isinstance(lora_ckpt, dict) else False
    _log(f"  lora keys: {len(lora_sd)}, is_lora={is_lora}")

    # Detect LoRA pairs
    lora_pairs = detect_lora_keys(lora_sd)
    _log(f"  found {len(lora_pairs)} LoRA pairs")
    if not lora_pairs:
        _log("ERROR: no LoRA pairs found")
        return 1

    # Compute effective scale
    eff_scale = args.scale * (args.lora_alpha / args.lora_r)
    _log(f"effective scale = user({args.scale}) * (alpha={args.lora_alpha}/r={args.lora_r}) "
         f"= {eff_scale:.4f}")

    # Apply merge: W_merged = W_base + eff_scale * (B @ A)
    n_merged = 0
    n_missing = 0
    n_motion_modules = 0
    merged_sd = {}

    # Start with base weights
    for k, v in base_sd.items():
        merged_sd[k] = v.clone() if isinstance(v, torch.Tensor) else v

    # Also handle motion_modules + other trainable weights from LoRA ckpt
    # (these are NOT LoRA, they're full-tune motion_modules — copy directly)
    for k, v in lora_sd.items():
        if "lora_" in k:
            continue
        # Strip PEFT prefix
        clean_key = k.replace("base_model.model.", "")
        if "motion_modules" in clean_key:
            if clean_key in merged_sd:
                merged_sd[clean_key] = v.clone() if isinstance(v, torch.Tensor) else v
                n_motion_modules += 1

    # Apply LoRA merge for each pair
    for base_key, (a_key, b_key) in lora_pairs.items():
        weight_key = f"{base_key}.weight"
        if weight_key not in merged_sd:
            # Maybe stored without .weight in base?
            if base_key in merged_sd:
                weight_key = base_key
            else:
                n_missing += 1
                continue

        A = lora_sd[a_key].float()  # (r, in_dim)
        B = lora_sd[b_key].float()  # (out_dim, r)
        delta = eff_scale * (B @ A)  # (out_dim, in_dim)

        W = merged_sd[weight_key].float()
        if W.shape != delta.shape:
            _log(f"  shape mismatch {weight_key}: W={W.shape} vs delta={delta.shape}")
            n_missing += 1
            continue

        merged_sd[weight_key] = (W + delta).to(merged_sd[weight_key].dtype)
        n_merged += 1

    _log(f"merged: {n_merged}/{len(lora_pairs)} LoRA pairs, "
         f"motion_modules updates: {n_motion_modules}, missing: {n_missing}")

    # Save in same format as base
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = {"state_dict": merged_sd}
    torch.save(output_dict, args.output)
    _log(f"saved to {args.output}")
    _log(f"  size: {os.path.getsize(args.output)/1e9:.2f} GB")
    _log(f"  total time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
