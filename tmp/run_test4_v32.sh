#!/bin/bash
# test4 v32 — AV_REASSIGN_MAX_DUR 늘려서 긴 segment face 기반 교정 허용
#
# 사용자 GT (2026-05-15):
#   - SPK_05 = 영상 전체에서 "거기로 가" 1발화만
#   - v31의 SPK_05 5개 중 4개 오분류 (실제 SPK_04 등)
#   - 60.48-69.36 "좋아"는 face owner = SPK_04
#
# v32 가설: AV-Reassign이 face owner 기반 교정하지만 현재 MAX_DUR=2.0 때문에
#           긴 segment (8.88s) skip됨. MAX_DUR=10.0 으로 늘리면 reassign 시도.
#           face owner 정확하면 audio diarization 오류 자동 교정.
#
# v31 baseline + 변경: LATENTSYNC_AV_REASSIGN_MAX_DUR 2.0 → 10.0

set -e
INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v32_$(date +%Y%m%d_%H%M%S)
echo "==================================================="
echo "  test4 v32 - AV_REASSIGN_MAX_DUR=10 (긴 segment face 교정)"
echo "  Run ID: $RUN_ID"
echo "==================================================="

export LATENTSYNC_RESUME=1
unset LATENTSYNC_POSTPROC_SAME_SPK_GAP

# v31 유지: merge 함수 비활성 (라벨 회복)
export LATENTSYNC_SAME_SPK_GAP=0.0
export LATENTSYNC_SAME_SPK_MERGE_CAP=14.0
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

# === v32 CHANGE: 긴 segment도 face 기반 교정 허용 ===
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
echo "[$(date '+%H:%M:%S')] 시작 (AV_REASSIGN_MAX_DUR=10.0)"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" --name "test4_v32" --run-id "$RUN_ID" \
    --lang ko --content-type drama 2>&1 | tee /tmp/test4_v32_${RUN_ID}.log | \
    grep -E "Resume skip|DiariZen daemon|최종 화자 수|AV-Reassign|face owner|같은 화자 인접|Segments\] [0-9]+개|self-ref|TTS\] \[SPEAKER_(04|05)|Pipeline\] 완료|Traceback|ERROR|FAILED" | head -150

ELAPSED=$(($(date +%s) - START))
echo
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))

echo
echo "=== AV-Reassign 발동 횟수 ==="
grep -c "AV-Reassign\] [0-9]" /tmp/test4_v32_${RUN_ID}.log 2>/dev/null
grep "AV-Reassign\]" /tmp/test4_v32_${RUN_ID}.log 2>/dev/null

echo
echo "=== 59-72s 영역 ==="
SEG=$(ls -t /workspace/media/runs/test4_v32_*/meta/test4_v32_chunk_000_segments.json 2>/dev/null | head -1)
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
print(f"\nSPEAKER_05 ({len(spk5)} groups) — 목표: 1개 (영상 GT)")
for g in spk5:
    print(f"  [{g['group_idx']:>2}] {g['group_start']:>6.2f}~{g['group_end']:>6.2f}s '{g['text'][:25]}'")
PYEOF

echo
echo "=== Auto metric v26-v32 ==="
V26=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
V27=$(ls -t /workspace/media/runs/test4_v27_*/meta/test4_v27_chunk_000_segments.json 2>/dev/null | head -1)
V28=$(ls -t /workspace/media/runs/test4_v28_*/meta/test4_v28_chunk_000_segments.json 2>/dev/null | head -1)
V29=$(ls -t /workspace/media/runs/test4_v29_*/meta/test4_v29_chunk_000_segments.json 2>/dev/null | head -1)
V30=$(ls -t /workspace/media/runs/test4_v30_*/meta/test4_v30_chunk_000_segments.json 2>/dev/null | head -1)
V31=$(ls -t /workspace/media/runs/test4_v31_*/meta/test4_v31_chunk_000_segments.json 2>/dev/null | head -1)
python3 /workspace/tmp/eval_diarization_auto.py "$V26" "$V31" "$SEG" 2>&1 | tail -15
