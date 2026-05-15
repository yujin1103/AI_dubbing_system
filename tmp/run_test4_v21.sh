#!/bin/bash
# test4 v21 — v20 회귀 즉시 fix (SPEAKER_99/98/97 가짜 speaker 양산 차단)
#
# ─ v20 회귀 원인 ─
#   OUTLIER_FAR_THRESH 0.20 → 0.25 잘못 올림
#   → post_process_diarization의 outlier 검출이 cosine<0.25 segment 만나면
#     SPEAKER_99 → 98 → 97 가짜 라벨 양산
#   → 화자 수 6 → 9
#
# ─ v21 fix ─
#   1) LATENTSYNC_OUTLIER_OFF=1 (outlier 검출 자체 비활성)
#      가짜 SPEAKER_99/98/97 회귀 차단
#   2) DIARIZE_MIN_DUR_CENTROID 1.0 → 0.5 (1.0 너무 strict, 짧은 화자 centroid 못 만듦)
#   3) 나머지 v20 설정 유지 (ref 6초+, fade-out, split smoothing, AV-Reassign max_dur 2.0)
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v21_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v21 - v20 OUTLIER 회귀 fix"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v21 NEW: OUTLIER 검출 자체 비활성 (가짜 speaker 양산 차단) ===
export LATENTSYNC_OUTLIER_OFF=1
unset LATENTSYNC_OUTLIER_FAR_THRESH

# === v21: DiariZen MIN_DUR_CENTROID 보수화 (v20 1.0 너무 strict) ===
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.5    # 1.0 → 0.5
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.55
export LATENTSYNC_DIARIZE_SHORT_TURN=1.5

# === v20 유지: ref 6초+ ===
export LATENTSYNC_REF_MIN_DUR=6.0
export LATENTSYNC_REF_MAX_DUR=15.0
export LATENTSYNC_REF_FALLBACK_MIN=4.0
export LATENTSYNC_REF_FALLBACK_MAX=25.0

# === v20 유지: split smoothing ===
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=6
export LATENTSYNC_SPLIT_MAJORITY_TH=0.80

# === v20 유지: AV-Reassign 보수화 + max_dur 2.0 ===
unset LATENTSYNC_AV_REASSIGN_OFF
export LATENTSYNC_AV_REASSIGN_DOMINANT=0.85
export LATENTSYNC_AV_REASSIGN_SHARE=0.75
export LATENTSYNC_AV_REASSIGN_MAX_DUR=2.0
export LATENTSYNC_REF_EXCLUDE_REASSIGNED=1

# === C 유지: LLM soft + CAPEL + stretch ===
export LATENTSYNC_SYL_RANGE=0.5
export LATENTSYNC_CAPEL_SHORT=1
export LATENTSYNC_MAX_STRETCH=1.15
export LATENTSYNC_MIN_STRETCH=0.90
export LATENTSYNC_TOL_LATE=0.20
export LATENTSYNC_PREDICT_THRESHOLD=1.20
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
echo "[$(date '+%H:%M:%S')] 시작"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" \
    --name "test4_v21" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v21_${RUN_ID}.log | \
    grep -E "Separate|Ensemble|chunk_|DiariZen|최종 화자 수|Outlier|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Fusion|AV-Reassign|Profiles\]|face owner|fade-out|trim" | head -200

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v21_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v21_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== Speaker 검증 ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v21_*/meta/test4_v21_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi
