#!/bin/bash
src=/workspace/media/output/test3_lipsync/chunks/out_004_loopfix.mp4
mkdir -p /tmp/c4lp
for f in 200 220 235 240 245 248 290; do
    ffmpeg -y -loglevel error -i $src -vf "select=eq(n\,$f)" -frames:v 1 /tmp/c4lp/idx${f}.png
done
ls /tmp/c4lp/
