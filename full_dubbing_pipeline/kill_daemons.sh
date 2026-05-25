#!/bin/bash
# Stop all diarize + CosyVoice daemons (release GPU before lipsync stage).
echo "==> Killing diarize + CosyVoice daemons"
pkill -9 -f diarize_daemon || true
pkill -9 -f nemo_diarize || true
pkill -9 -f pyannote_diarize || true
pkill -9 -f fusion_diarize || true
pkill -9 -f cosyvoice_daemon || true
sleep 3

# Wait for GPU release (zombie cleanup may take time)
for i in 1 2 3 4 5; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "  GPU used: ${USED}MiB"
    if [ "$USED" -lt 2000 ]; then break; fi
    sleep 2
done
echo "==> Done. GPU should be free for lipsync stage."
