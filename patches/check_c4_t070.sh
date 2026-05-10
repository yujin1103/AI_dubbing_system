#!/bin/bash
src=/workspace/media/output/test3_lipsync/chunks/out_004_t070.mp4
mkdir -p /tmp/c4t070
for f in 100 200 220 230 240 245 248; do
    ffmpeg -y -loglevel error -i $src -vf "select=eq(n\,$f)" -frames:v 1 /tmp/c4t070/idx${f}.png
done
ls /tmp/c4t070/
