#!/bin/bash
# loopfix + brightness 패치 적용된 상태로 chunks 0, 1, 2, 3 재실행
cd /opt/LatentSync

for i in 000 001 002 003; do
    echo ">>> chunk $i ($(date +%T))"
    /opt/venv_lipsync/bin/python -m scripts.inference \
        --unet_config_path configs/unet/stage2_512.yaml \
        --inference_ckpt_path checkpoints/latentsync_unet.pt \
        --inference_steps 20 --guidance_scale 1.5 --enable_deepcache \
        --video_path /workspace/media/output/test3_lipsync/chunks/v_${i}.mp4 \
        --audio_path /workspace/media/output/test3_lipsync/chunks/a_${i}.wav \
        --video_out_path /workspace/media/output/test3_lipsync/chunks/out_${i}_loopfix.mp4 \
        2>&1 | grep -E "Loop|Restore|Skip|Brightness|Error|Traceback" | tail -10
done
echo ">>> ALL DONE"
