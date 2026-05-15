#!/bin/bash
# test4 v33 — FACE_ID 활성 (ArcFace face cluster로 audio over-detect 보정)
#
# v32 진단: face owner가 audio diarization에 의존 → audio 오류 재현. f36/f41이
#            audio SPK_05라서 face owner도 SPK_05 → AV-Reassign 효과 없음.
# v33 가설: face_id_embedder가 ArcFace로 face cluster 생성 (audio 독립).
#            같은 cluster의 audio speaker 다르면 통일 → SPK_05 over-detect 자동 회복.
# 변경: LATENTSYNC_FACE_ID_OFF=0 (활성)
#       face_id sim/share 임계값은 default (0.50/0.50)
# baseline: v31 (SAME_SPK_GAP=0.0 라벨 회복) + v32 (AV_REASSIGN_MAX_DUR=10)

set -e
INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v33_$(date +%Y%m%d_%H%M%S)
echo "==================================================="
echo "  test4 v33 - FACE_ID 활성 (ArcFace cluster)"
echo "  Run ID: $RUN_ID"
echo "==================================================="

export LATENTSYNC_RESUME=1
unset LATENTSYNC_POSTPROC_SAME_SPK_GAP
export LATENTSYNC_SAME_SPK_GAP=0.0
export LATENTSYNC_SAME_SPK_MERGE_CAP=14.0
export LATENTSYNC_SENT_MERGE_CAP=14.0

# === v33 CHANGE: FACE_ID 활성 ===
unset LATENTSYNC_FACE_ID_OFF
# face_id 임계값 (default 사용)
# export LATENTSYNC_FACE_ID_SIM=0.50
# export LATENTSYNC_FACE_ID_CLUSTER_SHARE=0.50

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
echo "[$(date '+%H:%M:%S')] 시작 (FACE_ID 활성)"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" --name "test4_v33" --run-id "$RUN_ID" \
    --lang ko --content-type drama 2>&1 | tee /tmp/test4_v33_${RUN_ID}.log | \
    grep -E "Resume skip|DiariZen daemon|최종 화자 수|FaceID|face cluster|speaker remap|AV-Reassign|Segments\] [0-9]+개|TTS\] \[SPEAKER_(04|05)|Pipeline\] 완료|Traceback|ERROR|FAILED" | head -200

ELAPSED=$(($(date +%s) - START))
echo
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))

echo
echo "=== FaceID remap 발동 횟수 ==="
grep "FaceID\]" /tmp/test4_v33_${RUN_ID}.log 2>/dev/null

echo
echo "=== 59-72s 영역 ==="
SEG=$(ls -t /workspace/media/runs/test4_v33_*/meta/test4_v33_chunk_000_segments.json 2>/dev/null | head -1)
python3 - << PYEOF
import json
data = json.load(open("$SEG"))
print(f"[total] {len(data[\"groups\"])} groups")
for g in sorted(data["groups"], key=lambda g: g["group_start"]):
    if 58 <= g["group_start"] <= 73:
        d = g["group_end"] - g["group_start"]
        print(f"  [{g['group_idx']:>2}] {g['speaker']} {g['group_start']:>6.2f}~{g['group_end']:>6.2f}s ({d:.2f}s) '{g['text'][:30]}'")
print()
from collections import Counter
c = Counter(g["speaker"] for g in data["groups"])
print(f"counts: {dict(sorted(c.items()))}")
spk5 = sorted([g for g in data["groups"] if g["speaker"] == "SPEAKER_05"], key=lambda g: g["group_start"])
print(f"\nSPEAKER_05 ({len(spk5)} groups) — 목표: 1개")
for g in spk5:
    print(f"  [{g['group_idx']:>2}] {g['group_start']:>6.2f}~{g['group_end']:>6.2f}s '{g['text'][:25]}'")
PYEOF
