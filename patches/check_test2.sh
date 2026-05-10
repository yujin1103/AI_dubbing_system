#!/bin/bash
for f in /workspace/media/input/test2.mp4 /workspace/media/input/test2_part1.mp4; do
    echo "=== $f ==="
    ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f"
    ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 "$f" | head -5
done
