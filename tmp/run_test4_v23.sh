#!/bin/bash
# test4 v23 — 사용자 핵심 통찰 반영: 화자 분리 근본 개선
#
#   사용자: "화자 분리를 더 잘하면 짧은 말이어도 상관 없는건데
#           이 부분은 어떻게 안되나? 화자 분리가 제일 중요한 관건인거같은데"
#
# ─ v23 핵심 신규 기능: Face Recognition cluster ─
#   동영상 얼굴은 결정적 (같은 얼굴 = 같은 사람).
#   LightASD가 face track 만들고, 우리는 거기에 ArcFace face_id 임베딩 추가.
#
#   파이프라인:
#     1) face track마다 5개 frame sampling → InsightFace 임베딩 추출
#     2) cosine sim ≥ 0.50 → 동일인 cluster (Union-Find)
#     3) cluster마다 대표 audio speaker 결정 (점유율 50%+)
#     4) 같은 cluster의 다른 audio speaker는 대표로 통합 (over-split fix)
#     5) audio diarization Annotation 갱신 + fusion 데이터 갱신
#     6) downstream AV-Reassign + ref bank가 깨끗한 라벨 사용
#
# ─ 환경변수 ─
#   LATENTSYNC_FACE_ID_OFF=1 → 비활성
#   LATENTSYNC_FACE_ID_SIM (default 0.50) — ArcFace cosine 임계
#   LATENTSYNC_FACE_ID_CLUSTER_SHARE (default 0.50) — 대표 speaker 최소 점유율
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v23_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v23 - Face Recognition cluster 화자 통합"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v23 NEW: Face Recognition ===
unset LATENTSYNC_FACE_ID_OFF                         # ON
export LATENTSYNC_FACE_ID_SIM=0.50                    # ArcFace cosine 임계 (0.5 = 안전)
export LATENTSYNC_FACE_ID_CLUSTER_SHARE=0.50          # cluster 대표 50%+

# === v22 안전 재설계 유지 ===
export LATENTSYNC_PREDICT_THRESHOLD=1.10
export LATENTSYNC_SENT_MERGE_CAP=10.0
unset LATENTSYNC_PROTECT_LAST_GROUP                   # 제거됨 (위험)
export LATENTSYNC_PROTECT_LAST_MERGE=1                # 안전 조건 (같은 화자+짧은 gap)만
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=8

# === v21 유지: OUTLIER OFF + DiariZen ===
export LATENTSYNC_OUTLIER_OFF=1
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.5
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.55
export LATENTSYNC_DIARIZE_SHORT_TURN=1.5

# === v20 유지: ref 6초+ ===
export LATENTSYNC_REF_MIN_DUR=6.0
export LATENTSYNC_REF_MAX_DUR=15.0
export LATENTSYNC_REF_FALLBACK_MIN=4.0
export LATENTSYNC_REF_FALLBACK_MAX=25.0

# === v20 유지: split smoothing ===
export LATENTSYNC_SPLIT_MAJORITY_TH=0.80

# === v20 유지: AV-Reassign ===
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
    --name "test4_v23" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v23_${RUN_ID}.log | \
    grep -E "Separate|chunk_|DiariZen|최종 화자 수|Outlier|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Reassign|FaceID|Profiles\]|fade-out|trim|cluster" | head -250

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v23_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v23_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v23 Face Recognition 효과 ==="
echo "[face track 임베딩 추출]"
grep "FaceID\]" /tmp/test4_v23_${RUN_ID}.log 2>/dev/null | grep -E "face tracks 임베딩|face clusters" | head -3
echo
echo "[face cluster 분포]"
grep "cluster_[0-9]" /tmp/test4_v23_${RUN_ID}.log 2>/dev/null | head -10
echo
echo "[face_id 기반 speaker remap]"
grep "FaceID\]" /tmp/test4_v23_${RUN_ID}.log 2>/dev/null | grep -E "speaker remap|remap 적용|remap 없음" | head -5
echo
echo "[trim 발생]"
grep -c "trim" /tmp/test4_v23_${RUN_ID}.log 2>/dev/null

echo
echo "=== Speaker 검증 (SPEAKER_05 1회 기대) ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v23_*/meta/test4_v23_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi
