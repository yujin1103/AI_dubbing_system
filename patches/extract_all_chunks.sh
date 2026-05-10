#!/bin/bash
# 각 chunk 원본의 시작/중간/끝 frame 추출
mkdir -p /tmp/inspect
for i in 000 001 002 003 004 005; do
    src=/workspace/media/output/test3_lipsync/chunks/v_${i}.mp4
    n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nw=1:nk=1 $src)
    mid=$((n / 2))
    end=$((n - 5))
    ffmpeg -y -loglevel error -i $src -vf "select=eq(n\,0)" -frames:v 1 /tmp/inspect/v${i}_start.png
    ffmpeg -y -loglevel error -i $src -vf "select=eq(n\,$mid)" -frames:v 1 /tmp/inspect/v${i}_mid.png
    ffmpeg -y -loglevel error -i $src -vf "select=eq(n\,$end)" -frames:v 1 /tmp/inspect/v${i}_end.png
    echo "v${i}.mp4: ${n} frames (mid=${mid}, end=${end})"
done
ls /tmp/inspect/v*.png | head -30
