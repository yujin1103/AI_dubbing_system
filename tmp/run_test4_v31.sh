#!/bin/bash
# test4 v31 — _merge_same_speaker_adjacent 비활성 검증
#
# 변경: LATENTSYNC_SAME_SPK_GAP=0.0  → gap < 0.0 항상 false → 합치지 않음
#       (함수 자체는 호출되지만 effect 없음)
# 가설: v26→v27 라벨 회귀가 이 함수의 부작용이라면, v31에서 v26 라벨 복원됨.
#       v31 segments == v28 면 이 함수가 원인 아님 → SENT_MERGE_CAP 등 다른 변수.
#
# selfref concat fix는 v30에서 검증된 그대로 (코드 변경 영구적용)

set -e
INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v31_$(date +%Y%m%d_%H%M%S)
echo "==================================================="
echo "  test4 v31 - SAME_SPK_GAP=0 (merge 함수 비활성)"
echo "  Run ID: $RUN_ID"
echo "==================================================="

export LATENTSYNC_RESUME=1
unset LATENTSYNC_POSTPROC_SAME_SPK_GAP

# === v31 CHANGE: SAME_SPK_GAP 0.0 (effectively disables _merge_same_speaker_adjacent) ===
export LATENTSYNC_SAME_SPK_GAP=0.0
export LATENTSYNC_SAME_SPK_MERGE_CAP=14.0

# 나머지 v29 동일
export LATENTSYNC_SENT_MERGE_CAP=14.0
export LATENTSYNC_FACE_ID_OFF=1
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
export LATENTSYNC_AV_REASSIGN_MAX_DUR=2.0
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
echo "[$(date '+%H:%M:%S')] 시작 (SAME_SPK_GAP=0.0)"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" --name "test4_v31" --run-id "$RUN_ID" \
    --lang ko --content-type drama 2>&1 | tee /tmp/test4_v31_${RUN_ID}.log | \
    grep -E "Resume skip|DiariZen daemon|최종 화자 수|같은 화자 인접|Segments\] [0-9]+개|TTS\] \[SPEAKER|self-ref|Pipeline\] 완료|Traceback|ERROR|FAILED" | head -150

ELAPSED=$(($(date +%s) - START))
echo
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))

echo
echo "=== 60.48-68.96s 영역 라벨 (v26: SPK_05, v28/29: SPK_04) ==="
SEG=$(ls -t /workspace/media/runs/test4_v31_*/meta/test4_v31_chunk_000_segments.json 2>/dev/null | head -1)
python3 - << PYEOF
import json
data = json.load(open("$SEG"))
for g in sorted(data["groups"], key=lambda g: g["group_start"]):
    if 58 <= g["group_start"] <= 73:
        d = g["group_end"] - g["group_start"]
        print(f"  [{g['group_idx']:>2}] {g['speaker']} {g['group_start']:>6.2f}~{g['group_end']:>6.2f}s ({d:.2f}s) '{g['text']}'")
print()
from collections import Counter
c = Counter(g["speaker"] for g in data["groups"])
print(f"전체 group counts: {dict(sorted(c.items()))}")
PYEOF

echo
echo "=== Auto metric v26 vs v27 vs v28 vs v29 vs v30 vs v31 ==="
V26=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
V27=$(ls -t /workspace/media/runs/test4_v27_*/meta/test4_v27_chunk_000_segments.json 2>/dev/null | head -1)
V28=$(ls -t /workspace/media/runs/test4_v28_*/meta/test4_v28_chunk_000_segments.json 2>/dev/null | head -1)
V29=$(ls -t /workspace/media/runs/test4_v29_*/meta/test4_v29_chunk_000_segments.json 2>/dev/null | head -1)
V30=$(ls -t /workspace/media/runs/test4_v30_*/meta/test4_v30_chunk_000_segments.json 2>/dev/null | head -1)
python3 /workspace/tmp/eval_diarization_auto.py "$V26" "$V27" "$V28" "$V29" "$V30" "$SEG" 2>&1 | tail -20
