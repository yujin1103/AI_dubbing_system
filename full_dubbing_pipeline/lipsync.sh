#!/bin/bash
# Stage 4 — Lipsync (run AFTER kill_daemons.sh to free GPU memory).
# Single-video lipsync via parallel_lipsync_orchestrator.py (from main repo patches).
#
# Usage: bash lipsync.sh <input_video> <dub_audio.wav> <output_lipsync.mp4>
set -e

VIDEO="${1:?usage: lipsync.sh <video> <audio> <out_mp4>}"
AUDIO="${2:?usage: lipsync.sh <video> <audio> <out_mp4>}"
OUT="${3:?usage: lipsync.sh <video> <audio> <out_mp4>}"

LIPSYNC_PYBIN=${LIPSYNC_PYBIN:-/opt/venv_lipsync/bin/python}
ORCHESTRATOR=${ORCHESTRATOR:-/workspace/patches/parallel_lipsync_orchestrator.py}
CKPT=${LIPSYNC_CKPT:-/workspace/media/lora/merged/merged_scale_0.7.pt}
ENGINE=${LIPSYNC_TRT:-/workspace/trt_work/engines/unet_lora07_fp16.trt}

# Speaker face filter (optional — better lipsync if you have a previous run with ASD/profile data)
ASD_RUN_DIR=${ASD_RUN_DIR:-}

ENV_ARGS=""
[ -n "$ASD_RUN_DIR" ] && ENV_ARGS="$ENV_ARGS LATENTSYNC_ASD_FILTER_RUN_DIR=$ASD_RUN_DIR"
[ -n "$ASD_RUN_DIR" ] && [ -f "$ASD_RUN_DIR/meta/audio_f0_gender.json" ] && \
    ENV_ARGS="$ENV_ARGS LATENTSYNC_AUDIO_F0_GENDER_PATH=$ASD_RUN_DIR/meta/audio_f0_gender.json"

echo "==> Lipsync (LoRA 0.7 + balanced ASD filter + chunk 30s)"
echo "  video : $VIDEO"
echo "  audio : $AUDIO"
echo "  out   : $OUT"

LATENTSYNC_USE_NVENC=0 \
LATENTSYNC_INFERENCE_CKPT="$CKPT" \
LATENTSYNC_TRT_ENGINE="$ENGINE" \
LATENTSYNC_ASD_THRESHOLD=${LATENTSYNC_ASD_THRESHOLD:-0.20} \
LATENTSYNC_ASD_IOU_MATCH=${LATENTSYNC_ASD_IOU_MATCH:-0.30} \
LIPSYNC_MIN_GPU_GB=3 LIPSYNC_GPU_WAIT_S=10 \
$ENV_ARGS \
PYTHONUNBUFFERED=1 $LIPSYNC_PYBIN $ORCHESTRATOR \
    --input "$VIDEO" --audio "$AUDIO" --output "$OUT" \
    --chunk-seconds 30 --no-enhance --no-parallel

echo
echo "==> Lipsync done: $OUT"
