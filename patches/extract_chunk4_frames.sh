#!/bin/bash
for f in 230 235 240 245 248; do
    ffmpeg -y -loglevel error -i /workspace/media/output/test3_lipsync/chunks/v_004.mp4 -vf "select=eq(n\,$f)" -frames:v 1 /tmp/v4_orig_idx${f}.png
done
ls -la /tmp/v4_orig_idx*.png
