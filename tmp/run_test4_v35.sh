#!/bin/bash
# test4 v35 — FACE_ID_SPEAKER_SHARE 60% → 40% (정확한 env var)
#
# v34 진단: LATENTSYNC_FACE_ID_CLUSTER_SHARE 변경했지만 실제 skip threshold는
#           LATENTSYNC_FACE_ID_SPEAKER_SHARE (default 0.60)였음.
# v35: 정확한 env var 변경 → SPK_01 (51%) + SPK_03 (45%) 둘 다 remap 발동 기대.
# 변경: LATENTSYNC_FACE_ID_SPEAKER_SHARE=0.40

set -e
INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v35_$(date +%Y%m%d_%H%M%S)
echo "==================================================="
echo "  test4 v35 - FACE_ID_SPEAKER_SHARE=0.40 (정확한 knob)"
echo "  Run ID: $RUN_ID"
echo "==================================================="

export LATENTSYNC_RESUME=1
unset LATENTSYNC_POSTPROC_SAME_SPK_GAP
export LATENTSYNC_SAME_SPK_GAP=0.0
export LATENTSYNC_SAME_SPK_MERGE_CAP=14.0
export LATENTSYNC_SENT_MERGE_CAP=14.0

unset LATENTSYNC_FACE_ID_OFF
# === v35 정확한 변경 ===
export LATENTSYNC_FACE_ID_SPEAKER_SHARE=0.40

export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.50
export LATENTSYNC_DIARIZE_SHORT_TURN=2.0
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.5
export LATENTSYNC_PREDICT_THRESHOLD=1.10
unset LATENTSYNC_PROTECT_LAST_GROUP
export LATENTSYNC_PROTECT_LAST_MERGE=1
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=8
export LATENTSYNC_OUTLIER_OFF=1
export LATENTSYNC_REF_MIN_DUR=6.0
export LATENTSYNC_REF_MAX_DUR=15.0
export LATENTSYNC_REF_FALLBACK_MIN=4.0
export LATENTSYNC_REF_FALLBACK_MAX=25.0
export LATENTSYNC_SPLIT_MAJORITY_TH=0.80
unset LATENTSYNC_AV_REASSIGN_OFF
export LATENTSYNC_AV_REASSIGN_DOMINANT=0.85
export LATENTSYNC_AV_REASSIGN_SHARE=0.75
export LATENTSYNC_AV_REASSIGN_MAX_DUR=10.0
export LATENTSYNC_REF_EXCLUDE_REASSIGNED=1
export LATENTSYNC_SYL_RANGE=0.5
export LATENTSYNC_CAPEL_SHORT=1
export LATENTSYNC_MAX_STRETCH=1.15
export LATENTSYNC_MIN_STRETCH=0.90
export LATENTSYNC_TOL_LATE=0.20
export LATENTSYNC_SENT_MIN_WORDS=3
export LATENTSYNC_SENT_MIN_DURATION=1.5
unset LATENTSYNC_SPLIT_BY_SPEAKER_OFF
export LATENTSYNC_SPLIT_MIN_SUB_WORDS=3
export LATENTSYNC_QG_OFF=1
export LATENTSYNC_REF_SINGLE_PER_SPEAKER=1
export LATENTSYNC_AV_MERGE_OFF=1
export LATENTSYNC_SEP_ENSEMBLE=1
export LATENTSYNC_FORCE_LLM_DESC=1
export LATENTSYNC_EMOTION_INTENSITY=moderate
export COSY_DAEMON_URL=http://127.0.0.1:8901
export ASR_DAEMON_URL=http://127.0.0.1:8902
export DIARIZE_DAEMON_URL=http://127.0.0.1:8903

cd /opt/LatentSync
START=$(date +%s)
echo "[$(date '+%H:%M:%S')] 시작 (SPEAKER_SHARE=0.40)"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" --name "test4_v35" --run-id "$RUN_ID" \
    --lang ko --content-type drama 2>&1 | tee /tmp/test4_v35_${RUN_ID}.log | \
    grep -E "Resume skip|DiariZen daemon|최종 화자 수|FaceID|cluster|speaker remap|AV-Reassign\] [0-9]|Segments\] [0-9]+개|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED" | head -200

ELAPSED=$(($(date +%s) - START))
echo
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))

echo
echo "=== 모든 segments ==="
SEG=$(ls -t /workspace/media/runs/test4_v35_*/meta/test4_v35_chunk_000_segments.json 2>/dev/null | head -1)
python3 << PYEOF
import json
data = json.load(open("$SEG"))
print(f"[total] {len(data['groups'])} groups")
for g in sorted(data["groups"], key=lambda x: x["group_start"]):
    d = g["group_end"] - g["group_start"]
    print(f"  [{g['group_idx']:>2}] {g['speaker']} {g['group_start']:>6.2f}~{g['group_end']:>6.2f}s ({d:.2f}s) '{g.get('text','')[:35]}'")
print()
from collections import Counter
c = Counter(g["speaker"] for g in data["groups"])
print(f"counts: {dict(sorted(c.items()))}")
PYEOF

echo
echo "=== Auto metric v26/v31/v33/v34/v35 ==="
V26=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
V31=$(ls -t /workspace/media/runs/test4_v31_*/meta/test4_v31_chunk_000_segments.json 2>/dev/null | head -1)
V33=$(ls -t /workspace/media/runs/test4_v33_*/meta/test4_v33_chunk_000_segments.json 2>/dev/null | head -1)
V34=$(ls -t /workspace/media/runs/test4_v34_*/meta/test4_v34_chunk_000_segments.json 2>/dev/null | head -1)
python3 /workspace/tmp/eval_diarization_auto.py "$V26" "$V31" "$V33" "$V34" "$SEG" 2>&1 | tail -25
