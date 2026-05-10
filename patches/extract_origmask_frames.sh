#!/bin/bash
src=/workspace/media/output/test3_lipsync/chunks/out_004_orig_mask.mp4
mkdir -p /tmp/origmask
for f in 50 100 150 200 220 240 248; do
    ffmpeg -y -loglevel error -i $src -vf "select=eq(n\,$f)" -frames:v 1 /tmp/origmask/idx${f}.png
done
ls /tmp/origmask/
