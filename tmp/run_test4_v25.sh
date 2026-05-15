#!/bin/bash
# test4 v24 — v23 회귀 fix: Face Recognition 임계 강화 + 임베딩 추출률 개선
#
# ─ v23 회귀 ─
#   화자 수 6 → 4 (over-merge)
#   원인: 임베딩 추출률 30% (19/62) + threshold 0.50 관대 + remap 안전장치 없음
#
# ─ v24 변경 ─
#   [임베딩 추출률 개선]
#   FACE_DET_SIZE 640 → 320 (작은 얼굴/측면 얼굴 검출률 ↑)
#   FACE_DET_THRESH 0.50 → 0.30 (낮은 confidence 수용)
#   FACE_SAMPLES 5 → 7 (트랙당 sampling 더 많이)
#   FACE_PADDING 0.20 → 0.40 (bbox 패딩 확대)
#
#   [임계 강화 — false merge 차단]
#   FACE_ID_SIM 0.50 → 0.60 (ArcFace cosine 더 strict)
#   FACE_ID_MIN_TRACKS 신규 = 2 (단일 track cluster는 noisy, remap 사용 안 함)
#   FACE_ID_CLUSTER_SHARE 0.50 → 0.70 (cluster dominant speaker 70%+)
#   FACE_ID_SPEAKER_SHARE 신규 = 0.60 (speaker face frame의 60%+가 한 cluster에 집중)
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v25_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v25 - Face threshold 0.45 (cluster 분산 해소)"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v24 NEW: 임베딩 추출률 개선 ===
export LATENTSYNC_FACE_DET_SIZE=320
export LATENTSYNC_FACE_DET_THRESH=0.30
export LATENTSYNC_FACE_SAMPLES=7
export LATENTSYNC_FACE_PADDING=0.40

# === v24 NEW: 임계 강화 (false merge 차단) ===
unset LATENTSYNC_FACE_ID_OFF
export LATENTSYNC_FACE_ID_SIM=0.45                  # 0.60 → 0.45 (cluster 통합 강화)
export LATENTSYNC_FACE_ID_MIN_TRACKS=2               # 신규 (단일 track cluster skip)
export LATENTSYNC_FACE_ID_CLUSTER_SHARE=0.70         # 0.50 → 0.70
export LATENTSYNC_FACE_ID_SPEAKER_SHARE=0.60         # 신규

# === v22 안전 유지 ===
export LATENTSYNC_PREDICT_THRESHOLD=1.10
export LATENTSYNC_SENT_MERGE_CAP=10.0
unset LATENTSYNC_PROTECT_LAST_GROUP
export LATENTSYNC_PROTECT_LAST_MERGE=1
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=8

# === v21 유지 ===
export LATENTSYNC_OUTLIER_OFF=1
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.5
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.55
export LATENTSYNC_DIARIZE_SHORT_TURN=1.5

# === v20 유지 ===
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

# === C 유지 ===
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
echo "[$(date '+%H:%M:%S')] 시작"

/opt/venv_lipsync/bin/python /workspace/orchestrator.py \
    --input "$INPUT" \
    --name "test4_v25" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v25_${RUN_ID}.log | \
    grep -E "Separate|chunk_|DiariZen|최종 화자 수|Outlier|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Reassign|FaceID|Profiles\]|fade-out|trim|cluster" | head -250

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v25_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v25_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v24 Face Recognition 효과 ==="
echo "[임베딩 추출률 (목표: 60%+)]"
grep "FaceID\] [0-9]*/[0-9]* face tracks" /tmp/test4_v25_${RUN_ID}.log 2>/dev/null
echo
echo "[face cluster 수 (목표: 6명에 가까움)]"
grep "face clusters" /tmp/test4_v25_${RUN_ID}.log 2>/dev/null
echo
echo "[skip된 cluster/speaker (안전장치 발동)]"
grep -E "cluster_[0-9]+ skip|remap skip" /tmp/test4_v25_${RUN_ID}.log 2>/dev/null | head -10
echo
echo "[적용된 speaker remap]"
grep -A 5 "speaker remap (face" /tmp/test4_v25_${RUN_ID}.log 2>/dev/null | head -10
echo
echo "[remap 결과 화자 수]"
grep "speaker remap 적용" /tmp/test4_v25_${RUN_ID}.log 2>/dev/null

echo
echo "=== Speaker 검증 ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v25_*/meta/test4_v25_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi
