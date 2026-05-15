#!/bin/bash
# test4 v29 — 결정성 검증 (v28 env 100% 동일, vocals 재사용)
#
# 목적: v26→v27/v28의 라벨 차이가 (a) 환경변수 효과인지 (b) 비결정성인지 isolate.
#   - vocals는 v28 결과 재사용 (LATENTSYNC_RESUME=1) → BS-RoFormer 비결정성 배제
#   - 환경변수는 v28과 100% 동일 → DiariZen/ECAPA가 같은 입력에 같은 결과 내는지 검증
#
# 비교:
#   v28 segments == v29 segments  → 파이프라인 deterministic, v26 차이는 다른 원인
#   v28 segments != v29 segments  → CUDA non-determinism, 자동 시스템에 random seed 필요

set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v29_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v29 - 결정성 검증 (v28 env + RESUME=1)"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v29 NEW: vocals/bgm 재사용 (BS-RoFormer 비결정성 배제) ===
export LATENTSYNC_RESUME=1

# === v28과 100% 동일한 환경변수 ===
unset LATENTSYNC_POSTPROC_SAME_SPK_GAP
export LATENTSYNC_SAME_SPK_GAP=1.0
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
echo "[$(date '+%H:%M:%S')] 시작 (LATENTSYNC_RESUME=1)"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" \
    --name "test4_v29" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v29_${RUN_ID}.log | \
    grep -E "Separate|Resume skip|DiariZen|최종 화자 수|Segments\] [0-9]+개|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|같은 화자 인접" | head -120

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
echo "==================================================="

echo
echo "=== 결정성 검증: v28 vs v29 segments diff ==="
V28=$(ls -t /workspace/media/runs/test4_v28_*/meta/test4_v28_chunk_000_segments.json 2>/dev/null | head -1)
V29=$(ls -t /workspace/media/runs/test4_v29_*/meta/test4_v29_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$V28" ] && [ -n "$V29" ]; then
    python3 - << PYEOF
import json
v28 = json.load(open("$V28"))
v29 = json.load(open("$V29"))
g28 = v28["groups"]
g29 = v29["groups"]
print(f"v28 groups: {len(g28)}, v29 groups: {len(g29)}")
print()
if len(g28) != len(g29):
    print(f"❌ NON-DETERMINISTIC: group count differs!")
diffs = 0
for i in range(min(len(g28), len(g29))):
    a, b = g28[i], g29[i]
    if (a["speaker"] != b["speaker"]
        or abs(a["group_start"] - b["group_start"]) > 0.01
        or abs(a["group_end"] - b["group_end"]) > 0.01):
        diffs += 1
        print(f"  [g{i}] v28: {a['speaker']} {a['group_start']:.2f}-{a['group_end']:.2f}  "
              f"v29: {b['speaker']} {b['group_start']:.2f}-{b['group_end']:.2f}")
        if diffs >= 10:
            print("  ... (10개 초과 — 생략)")
            break
print()
if diffs == 0 and len(g28) == len(g29):
    print("✅ DETERMINISTIC: v28 == v29 (모든 segment 일치)")
else:
    print(f"❌ NON-DETERMINISTIC: {diffs} groups differ")
PYEOF
fi

echo
echo "=== Auto metric: v26 vs v27 vs v28 vs v29 ==="
V26=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
V27=$(ls -t /workspace/media/runs/test4_v27_*/meta/test4_v27_chunk_000_segments.json 2>/dev/null | head -1)
python3 /workspace/tmp/eval_diarization_auto.py "$V26" "$V27" "$V28" "$V29" 2>&1 | tail -30
