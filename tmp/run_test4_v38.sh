#!/bin/bash
# test4 v38 — DiariZen ahc_threshold 0.6 → 0.5 (cluster 더 적극적 merge)
#
# v37 baseline + DiariZen native param.
# 가설: ahc_threshold 낮추면 audio diarization 자체에서 화자 더 적극적 통합.
#       SPK_4와 SPK_5가 같은 cluster로 합쳐질 수도 (face_id 없이 audio만으로).
# 단점: 다른 다른 화자도 잘못 합쳐질 위험.
#
# daemon 재시작 필요 (env var는 daemon 시작 시 적용됨).

set -e
INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v38_$(date +%Y%m%d_%H%M%S)
echo "==================================================="
echo "  test4 v38 - DiariZen ahc_threshold=0.5"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v38 NEW: DiariZen native param (daemon env var) ===
# 단, daemon 재시작 별도 진행됨 (이 script 외부에서 docker exec 로)

export LATENTSYNC_RESUME=1
unset LATENTSYNC_POSTPROC_SAME_SPK_GAP
export LATENTSYNC_SAME_SPK_GAP=0.0
export LATENTSYNC_SAME_SPK_MERGE_CAP=14.0
export LATENTSYNC_SENT_MERGE_CAP=14.0

unset LATENTSYNC_FACE_ID_OFF
export LATENTSYNC_AV_REASSIGN_OFF=1  # v37 유지: FaceVoting과 충돌 방지

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
echo "[$(date '+%H:%M:%S')] 시작 (ahc_threshold=0.5)"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" --name "test4_v38" --run-id "$RUN_ID" \
    --lang ko --content-type drama 2>&1 | tee /tmp/test4_v38_${RUN_ID}.log | \
    grep -E "Resume skip|DiariZen daemon|최종 화자 수|FaceID|FaceVoting|speaker remap|AV-Reassign|Segments\] [0-9]+개|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED" | head -200

ELAPSED=$(($(date +%s) - START))
echo
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))

echo
echo "=== 모든 segments ==="
SEG=$(ls -t /workspace/media/runs/test4_v38_*/meta/test4_v38_chunk_000_segments.json 2>/dev/null | head -1)
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
echo "=== Auto metric v26/v33/v37/v38 ==="
V26=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
V33=$(ls -t /workspace/media/runs/test4_v33_*/meta/test4_v33_chunk_000_segments.json 2>/dev/null | head -1)
V37=$(ls -t /workspace/media/runs/test4_v37_*/meta/test4_v37_chunk_000_segments.json 2>/dev/null | head -1)
python3 /workspace/tmp/eval_diarization_auto.py "$V26" "$V33" "$V37" "$SEG" 2>&1 | tail -25
