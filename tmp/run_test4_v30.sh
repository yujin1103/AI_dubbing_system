#!/bin/bash
# test4 v30 — selfref concat fix 검증 (env는 v29와 동일)
#
# 변경: orchestrator.py SELF_REF_FALLBACK 블록 — 같은 화자의 모든 segment를 concat 해서
#       reference 생성 (기존: 단일 segment + padding).
# 효과 가설:
#   - SPK_05 4개 segment (0.64+1.76+0.64+0.64 = 3.68s) → 충분한 ref 길이
#   - SPK_00 3개 segment (0.88+1.04+1.04 = 2.96s) → 충분한 ref 길이
#   - zero-shot voice cloning 안정성 ↑ → 기계음 감소
# 검증:
#   - v30 log에서 "self-ref concat" 메시지 등장 확인
#   - 출력 영상 청취 (사용자) — SPK_05, SPK_00 segment 음질 비교
#   - segments.json은 v29와 동일해야 (코드 변경이 라벨 결정에 영향 X)
#
# env: v29와 100% 동일 (LATENTSYNC_RESUME=1 로 vocals/diarize 결과 isolate)

set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v30_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v30 - selfref concat fix"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v29 env 동일 (vocals 재사용) ===
export LATENTSYNC_RESUME=1
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
echo "[$(date '+%H:%M:%S')] 시작 (selfref concat fix, RESUME=1)"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" \
    --name "test4_v30" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v30_${RUN_ID}.log | \
    grep -E "Resume skip|DiariZen daemon|최종 화자 수|같은 화자 인접|Segments\] [0-9]+개|self-ref|TTS\] \[SPEAKER_(00|05)|Pipeline\] 완료|Traceback|ERROR|FAILED" | head -150

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
echo "==================================================="

echo
echo "=== selfref concat 발동 횟수 ==="
grep -c "self-ref concat" /tmp/test4_v30_${RUN_ID}.log 2>/dev/null
echo
grep "self-ref" /tmp/test4_v30_${RUN_ID}.log 2>/dev/null

echo
echo "=== v29 vs v30 segments diff (라벨 동일해야 정상) ==="
V29=$(ls -t /workspace/media/runs/test4_v29_*/meta/test4_v29_chunk_000_segments.json 2>/dev/null | head -1)
V30=$(ls -t /workspace/media/runs/test4_v30_*/meta/test4_v30_chunk_000_segments.json 2>/dev/null | head -1)
python3 - << PYEOF
import json
v29 = json.load(open("$V29"))
v30 = json.load(open("$V30"))
g29, g30 = v29["groups"], v30["groups"]
print(f"v29 groups: {len(g29)}, v30 groups: {len(g30)}")
diffs = 0
for i in range(min(len(g29), len(g30))):
    a, b = g29[i], g30[i]
    if (a["speaker"] != b["speaker"]
        or abs(a["group_start"] - b["group_start"]) > 0.01):
        diffs += 1
        print(f"  [g{i}] v29: {a['speaker']} {a['group_start']:.2f}-{a['group_end']:.2f}  "
              f"v30: {b['speaker']} {b['group_start']:.2f}-{b['group_end']:.2f}")
print()
if diffs == 0 and len(g29) == len(g30):
    print("✅ segments identical (selfref concat은 라벨에 영향 X — 정상)")
else:
    print(f"⚠️  {diffs} groups differ — 예상 외 영향")
PYEOF

echo
echo "=== Auto metric (selfref % 비교 — v30이 더 낮아야 정상) ==="
V26=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
V27=$(ls -t /workspace/media/runs/test4_v27_*/meta/test4_v27_chunk_000_segments.json 2>/dev/null | head -1)
V28=$(ls -t /workspace/media/runs/test4_v28_*/meta/test4_v28_chunk_000_segments.json 2>/dev/null | head -1)
python3 /workspace/tmp/eval_diarization_auto.py "$V26" "$V27" "$V28" "$V29" "$V30" 2>&1 | tail -20
