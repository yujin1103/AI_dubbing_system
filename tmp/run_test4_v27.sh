#!/bin/bash
# test4 v27 — 분할 회복 (사용자 핵심 분석 반영)
#
# 사용자 통찰: "문장 길이 제한 때문에 SPEAKER_05가 5번 나오는 거 아닌가"
# 데이터로 확인: SPEAKER_05 [59.20~72.00]가 한 발화인데 우리 코드가 gap 0.64/0.56s에서 3분할
#
# ─ v27 변경 (A+B+C 동반) ─
#   [A. 코드 추가] _merge_same_speaker_adjacent
#     같은 화자 + gap < 1.0s + 합쳐도 14s 이내면 무조건 안전 병합
#     build_segments에서 _merge_short_sentences 다음에 호출
#
#   [B. 코드 추가] post_process_diarization 인접 같은 화자 gap 환경변수화
#     0.3s 하드코딩 → LATENTSYNC_POSTPROC_SAME_SPK_GAP=1.0
#
#   [C. 환경변수] SENT_MERGE_CAP 10 → 14 (같은 화자 합치기 한계 확대)
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v27_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v27 - 같은 화자 인접 발화 안전 병합 (분할 회복)"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v27 NEW: 같은 화자 인접 sentence 병합 ===
export LATENTSYNC_SAME_SPK_GAP=1.0                    # 1.0s 이내 gap만 병합
export LATENTSYNC_SAME_SPK_MERGE_CAP=14.0              # 합쳐도 14s 이내

# === v27 NEW: post_process 인접 같은 화자 gap (was 하드코딩 0.3s) ===
export LATENTSYNC_POSTPROC_SAME_SPK_GAP=1.0

# === v27 NEW: SENT_MERGE_CAP 키움 ===
export LATENTSYNC_SENT_MERGE_CAP=14.0                  # v22~v26 10.0 → 14.0

# === v26 유지: FACE_ID OFF ===
export LATENTSYNC_FACE_ID_OFF=1

# === v26 유지: audio diarization 튜닝 ===
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.50
export LATENTSYNC_DIARIZE_SHORT_TURN=2.0
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.5

# === v22 유지: trim 줄임 ===
export LATENTSYNC_PREDICT_THRESHOLD=1.10
unset LATENTSYNC_PROTECT_LAST_GROUP
export LATENTSYNC_PROTECT_LAST_MERGE=1
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=8

# === v21 유지: OUTLIER OFF ===
export LATENTSYNC_OUTLIER_OFF=1

# === v20 유지: ref 6초+ ===
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
    --name "test4_v27" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v27_${RUN_ID}.log | \
    grep -E "Separate|chunk_|DiariZen|turn 수|최종 화자 수|Outlier|Segments\]|Merge\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Reassign|FaceID|Profiles\]|fade-out|trim" | head -200

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v27_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v27_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v27 분할 회복 효과 ==="
echo "[같은 화자 인접 sentence 안전 병합 (목표: 발동)]"
grep "같은 화자 인접 sentence" /tmp/test4_v27_${RUN_ID}.log 2>/dev/null
echo
echo "[post_process turn 병합 (전→후)]"
grep "turn 수:" /tmp/test4_v27_${RUN_ID}.log 2>/dev/null
echo
echo "[trim 발생 (v26: 5)]"
grep -c "trim" /tmp/test4_v27_${RUN_ID}.log 2>/dev/null

echo
echo "=== Speaker 검증 (SPEAKER_05 1-2 기대) ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v27_*/meta/test4_v27_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi

echo
echo "=== SPEAKER_05 영역 상세 (목표: 59.20~72.00s 한 group) ==="
python3 - << "PYEOF"
import json, glob
files = sorted(glob.glob("/workspace/media/runs/test4_v27_*/meta/test4_v27_chunk_000_segments.json"))
if files:
    data = json.load(open(files[-1]))
    spk5 = sorted([g for g in data['groups'] if g['speaker'] == 'SPEAKER_05'], key=lambda g: g['group_start'])
    print(f"SPEAKER_05: {len(spk5)} groups")
    for g in spk5:
        dur = g['group_end'] - g['group_start']
        print(f"  [g{g['group_idx']}] {g['group_start']:.2f}~{g['group_end']:.2f}s ({dur:.2f}s) '{g['text']}'")
PYEOF
