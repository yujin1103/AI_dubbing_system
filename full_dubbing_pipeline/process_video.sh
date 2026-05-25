#!/bin/bash
# Process one video — stages 1-3 (diarize + refs + dub).
# Assumes daemons already running (via start_daemons.sh).
# After all videos: bash kill_daemons.sh + lipsync.sh per video.
#
# Usage: bash process_video.sh <input_video> <output_dir>
# Env:
#   FUSION_URL=http://127.0.0.1:8903   (set by start_daemons.sh)
#   COSYVOICE_URL=http://127.0.0.1:8901
#   HF_TOKEN, VECTORENGINE_API_KEY/BASE/MODEL
set -e

INPUT="${1:-input.mp4}"
OUTDIR="${2:-./output}"
HERE=$(dirname "$0")
LANG="${LANG_CODE:-en}"
TARGET="${TARGET_LANG:-Korean}"

if [ -z "$FUSION_URL" ]; then
    echo "⚠ FUSION_URL not set — falling back to standalone pyannote 3.1 (~2 speakers only)"
    echo "  Run 'bash start_daemons.sh' first for accurate 4-way fusion."
fi

mkdir -p "$OUTDIR"

# Stage 1 — diarize (uses fusion daemon if FUSION_URL set)
echo "==> Stage 1: diarize"
python $HERE/1_diarize.py --input "$INPUT" --output "$OUTDIR/diarize.json" --language "$LANG" \
    ${FUSION_URL:+--fusion-url "$FUSION_URL"}

# Stage 2 — speaker references
echo "==> Stage 2: extract refs"
python $HERE/2_extract_speaker_refs.py \
    --video "$INPUT" --diarize-json "$OUTDIR/diarize.json" --out-dir "$OUTDIR/refs"

# ==== USER EDIT POINT ====
# Between Stage 2 and 3, edit $OUTDIR/diarize.json:
#   - segments[].text   : translation source
#   - segments[].emotion (if added): emotion hint per segment
# Then re-run from Stage 3 onward.

# Stage 3 — dub
echo "==> Stage 3: dub"
python $HERE/3_dub_pipeline.py \
    --video "$INPUT" \
    --diarize-json "$OUTDIR/diarize.json" \
    --refs-manifest "$OUTDIR/refs/manifest.json" \
    --out-dir "$OUTDIR/dub" \
    --target-lang "$TARGET" \
    ${SPEAKER_CONFIG:+--speaker-config "$SPEAKER_CONFIG"}

echo
echo "==================================================="
echo "Stages 1-3 done:"
echo "  diarize: $OUTDIR/diarize.json"
echo "  refs   : $OUTDIR/refs/"
echo "  dub    : $OUTDIR/dub/dubbed.mp4 + dub_audio.wav"
echo
echo "Next (after ALL videos processed):"
echo "  bash kill_daemons.sh           # release GPU"
echo "  bash lipsync.sh $INPUT $OUTDIR/dub/dub_audio.wav $OUTDIR/lipsync.mp4"
echo "==================================================="
