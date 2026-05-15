#!/bin/bash
# test4 v26 — Face Recognition 비활성 + audio diarization 수치 튜닝 + CAPEL 버그 fix
#
# ─ v23~v25 결론 ─
#   Face Recognition으론 본 영상의 화자 분리 추가 개선 불가능
#   (drama 영상 face_id 신호 약함, 안전장치 모두 skip)
#   → 비활성하고 audio diarization 수치 직접 튜닝으로 방향 전환
#
# ─ v26 변경 ─
#   [Face Recognition 비활성]
#   LATENTSYNC_FACE_ID_OFF=1 (cluster 무용지물)
#
#   [CAPEL 버그 fix — 코드]
#   _parse 함수에 LLM 응답의 <N> 마커 정규식 제거 추가
#   (v25에서 "<3>거<2>기<1>로<0>가" 같은 출력 발생)
#
#   [audio diarization 수치 튜닝 — SPEAKER_05 과검출 fix 시도]
#   DIARIZE_MERGE_THRESHOLD 0.55 → 0.50 (더 적극 centroid 병합)
#   DIARIZE_SHORT_TURN 1.5 → 2.0 (더 적극 짧은 turn 재할당)
#   DIARIZE_MIN_DUR_CENTROID 0.5 유지
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v26_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v26 - FACE_ID OFF + audio 튜닝 + CAPEL fix"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v26 NEW: Face Recognition 비활성 (효과 없음 입증됨) ===
export LATENTSYNC_FACE_ID_OFF=1
unset LATENTSYNC_FACE_ID_SIM
unset LATENTSYNC_FACE_ID_MIN_TRACKS
unset LATENTSYNC_FACE_ID_CLUSTER_SHARE
unset LATENTSYNC_FACE_ID_SPEAKER_SHARE
unset LATENTSYNC_FACE_DET_SIZE
unset LATENTSYNC_FACE_DET_THRESH
unset LATENTSYNC_FACE_SAMPLES
unset LATENTSYNC_FACE_PADDING

# === v26 NEW: audio diarization 수치 튜닝 ===
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.50    # 0.55 → 0.50 (더 적극 병합)
export LATENTSYNC_DIARIZE_SHORT_TURN=2.0           # 1.5 → 2.0 (더 적극 재할당)
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.5     # 유지

# === v22 유지 ===
export LATENTSYNC_PREDICT_THRESHOLD=1.10
export LATENTSYNC_SENT_MERGE_CAP=10.0
unset LATENTSYNC_PROTECT_LAST_GROUP
export LATENTSYNC_PROTECT_LAST_MERGE=1
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=8

# === v21 유지 ===
export LATENTSYNC_OUTLIER_OFF=1

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
    --name "test4_v26" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v26_${RUN_ID}.log | \
    grep -E "Separate|chunk_|DiariZen|최종 화자 수|Outlier|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Reassign|FaceID|Profiles\]|fade-out|trim|병합" | head -200

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v26_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v26_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v26 효과 ==="
echo "[DiariZen 병합 결과]"
grep -E "병합된 쌍|merge_threshold|화자 수:" /tmp/test4_v26_${RUN_ID}.log 2>/dev/null | head -10
echo
echo "[CAPEL 마커 누출 확인 (없어야 정상)]"
grep -E "<[0-9]>" /tmp/test4_v26_${RUN_ID}.log 2>/dev/null | head -5 || echo "(CAPEL 누출 없음 ✅)"
echo
echo "[trim 발생]"
grep -c "trim" /tmp/test4_v26_${RUN_ID}.log 2>/dev/null

echo
echo "=== Speaker 검증 ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi
