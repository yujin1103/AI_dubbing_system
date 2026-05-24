#!/usr/bin/env bash
# End-to-end dubbing pipeline example
# Usage: ./run_pipeline.sh <input_video> <output_dir>
set -euo pipefail

INPUT="${1:-input.mp4}"
OUTDIR="${2:-./output}"
LANG="${LANG_CODE:-en}"         # source language for ASR
TARGET="${TARGET_LANG:-Korean}" # target language for translation

mkdir -p "$OUTDIR"

# Stage 1 — diarization + ASR
echo "==> Stage 1: diarization"
python 1_diarize.py --input "$INPUT" --output "$OUTDIR/diarize.json" --language "$LANG"

# Stage 2 — speaker references
echo "==> Stage 2: extract speaker refs"
python 2_extract_speaker_refs.py \
  --video "$INPUT" \
  --diarize-json "$OUTDIR/diarize.json" \
  --out-dir "$OUTDIR/refs"

# Stage 3 — dub (VAD boundary refine + LLM translate + CosyVoice synth + composite)
echo "==> Stage 3: dub"
python 3_dub_pipeline.py \
  --video "$INPUT" \
  --diarize-json "$OUTDIR/diarize.json" \
  --refs-manifest "$OUTDIR/refs/manifest.json" \
  --out-dir "$OUTDIR/dub" \
  --target-lang "$TARGET" \
  ${SPEAKER_CONFIG:+--speaker-config "$SPEAKER_CONFIG"}

echo "==> Done: $OUTDIR/dub/dubbed.mp4"
echo "Next step (your turn): apply lip-sync on $OUTDIR/dub/dubbed.mp4"
