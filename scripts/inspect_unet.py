"""LatentSync UNet 구조 분석 — LoRA 타겟 layer 식별."""
import sys
sys.path.insert(0, '/opt/LatentSync')
import torch
from omegaconf import OmegaConf
from latentsync.models.unet import UNet3DConditionModel
from collections import Counter

cfg = OmegaConf.load('/opt/LatentSync/configs/unet/stage2_efficient.yaml')
print('=== UNet3DConditionModel from_pretrained ===')
unet, _ = UNet3DConditionModel.from_pretrained(
    OmegaConf.to_container(cfg.model),
    cfg.ckpt.resume_ckpt_path,  # 베이스 가중치
    device="cpu",  # 분석만
)

print()
print('=== Module 종류별 개수 ===')
mod_counts = Counter()
for name, mod in unet.named_modules():
    mod_counts[type(mod).__name__] += 1
for name, cnt in mod_counts.most_common(15):
    print(f'  {cnt:5d} × {name}')

print()
print('=== attn2 (cross-attn with audio) Linear layer 종류 ===')
attn2_linears = Counter()
for name, mod in unet.named_modules():
    if 'attn2' in name and type(mod).__name__ == 'Linear':
        last_name = name.rsplit('.', 1)[-1]
        attn2_linears[last_name] += 1
print(f'고유 이름 + 개수: {dict(attn2_linears)}')

print()
print('=== motion_modules Linear layer 종류 ===')
mm_linears = Counter()
for name, mod in unet.named_modules():
    if 'motion_modules' in name and type(mod).__name__ == 'Linear':
        last_name = name.rsplit('.', 1)[-1]
        mm_linears[last_name] += 1
print(f'고유 이름 + 개수: {dict(mm_linears)}')

print()
print('=== 파라미터 통계 ===')
total = sum(p.numel() for p in unet.parameters())
attn2_p = sum(p.numel() for n, p in unet.named_parameters() if 'attn2' in n)
mm_p = sum(p.numel() for n, p in unet.named_parameters() if 'motion_modules' in n)
print(f'전체 UNet: {total/1e6:.1f}M params')
print(f'attn2:     {attn2_p/1e6:.1f}M params ({100*attn2_p/total:.2f}%)')
print(f'motion_modules: {mm_p/1e6:.1f}M params ({100*mm_p/total:.2f}%)')
print(f'attn2 + motion: {(attn2_p+mm_p)/1e6:.1f}M params ({100*(attn2_p+mm_p)/total:.1f}%)')

print()
print('=== LoRA r=16 시 attn2 일부 layer 예상 크기 ===')
shown = 0
for n, m in unet.named_modules():
    if 'attn2' in n and type(m).__name__ == 'Linear':
        if any(k in n for k in ['to_q', 'to_k', 'to_v']):
            in_f = m.in_features
            out_f = m.out_features
            orig = in_f * out_f
            lora_r = 16
            lora_p = in_f * lora_r + lora_r * out_f
            print(f'  {n[:80]}: {in_f}x{out_f} = {orig:,} → LoRA: {lora_p:,} ({100*lora_p/orig:.1f}%)')
            shown += 1
            if shown >= 3:
                break

print()
print('=== 권장 LoRA target_modules ===')
print('attn2 의 to_q/to_k/to_v: 한국어 viseme 매핑 핵심')
print('to_out (선택): 출력 변환, 추가 capacity')
print('motion_modules to_q/k/v (선택): 시간 일관성 조정')
