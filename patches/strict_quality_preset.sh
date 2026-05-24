#!/bin/bash
# Source this file to enable all quality gates at stricter levels.
# Usage: . /workspace/patches/strict_quality_preset.sh
# Or:    source /workspace/patches/strict_quality_preset.sh

# ─── Wrong-person lipsync mitigation ────────────────────────────
# Profile gate: embedding cosine threshold (default 0.5 → 0.55 stricter)
export LATENTSYNC_PROFILE_MATCH_THRESHOLD=0.55
# Profile gate: treat "no opinion" (None) as skip → wrong-face never enters
export LATENTSYNC_PROFILE_STRICT_NONE=1
# Yaw threshold: skip face with > 0.30 yaw (front + slight side)
export LATENTSYNC_PROFILE_THRESHOLD=0.30
# ASD audio-speech detection: minimum confidence
export LATENTSYNC_ASD_THRESHOLD=0.30
# Face detector strict mode: det_score 0.85 + roll/landmark sanity
export LATENTSYNC_FACE_STRICT=1
# Skip very small faces (where artifacts most visible)
export LATENTSYNC_FACE_DIAG_MIN_RATIO=0.10

# ─── Mouth artifact mitigation ──────────────────────────────────
# Larger feather (smoother boundary)
export LATENTSYNC_FEATHER_SIGMA=6
# Temporal mask smoothing (5-frame rolling avg → no flicker)
export LATENTSYNC_MOUTH_TEMPORAL_SMOOTH=5
# Color histogram match (mouth ↔ surrounding skin)
export LATENTSYNC_MOUTH_COLOR_MATCH=1
# Mask erosion (2px inward before feather)
export LATENTSYNC_MOUTH_ERODE_PX=2

echo "[strict-quality] envs applied:"
env | grep -E '^LATENTSYNC_(PROFILE|ASD|FACE|MOUTH|FEATHER)' | sort
