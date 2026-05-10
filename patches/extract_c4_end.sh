#!/bin/bash
# chunk 4 lipsync 출력의 마지막 frames (title card 전환 지점)
src=/workspace/media/output/test3_lipsync/chunks/out_004_diffrepl_v2.mp4
mkdir -p /tmp/c4end
for f in 200 210 220 230 240 245 248 249; do
    ffmpeg -y -loglevel error -i $src -vf "select=eq(n\,$f)" -frames:v 1 /tmp/c4end/c4_idx${f}.png
done
ls /tmp/c4end/
