#!/bin/bash
# test4 v28 — POSTPROC_SAME_SPK_GAP 회귀 fix + daemon fade
#
# v27 분석 (2026-05-15):
#   - count metric (SPK_05 5→4 groups) 개선 같았지만 실제 라벨은 회귀
#   - 59-72s 영역 v26 [SPK_05 SPK_05 SPK_05] → v27 [SPK_05 SPK_04 SPK_03 SPK_05]
#   - 원인: LATENTSYNC_POSTPROC_SAME_SPK_GAP 0.3→1.0 변경이 ECAPA centroid를 흔들어
#           짧은 turn 재할당이 잘못된 화자로 가는 부작용
#
# v28 변경:
#   [A] LATENTSYNC_POSTPROC_SAME_SPK_GAP 언셋 (= 0.3 default 복원)
#       → post_process_diarization centroid 안정성 회복 (v26 라벨링 복원)
#   [B] LATENTSYNC_SAME_SPK_GAP / LATENTSYNC_SAME_SPK_MERGE_CAP 그대로 유지
#       → sentence-level merge는 안전 (같은 화자만 합침, label 변경 X)
#       → 이 단계가 v26→v28에서 SPK_05 3개 group → 1개로 회복 시킬 것 (의도된 부분)
#   [C] cosyvoice_daemon.py fade-in/out 패치 적용됨 (별도 commit)
#       → 모든 segment의 cold-start click + abrupt cut 사라짐
#       → 데몬 재시작 필수
#
# 기대 결과:
#   - SPK_05 = 3 groups (1 main merged in 59-72s + 84.40 + 95.92)
#   - 59-72s 모든 라벨이 SPK_05 (v26 수준 회복)
#   - 모든 segment 음질 개선 (기계음 첫/끝 ramp)

set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v28_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v28 - POSTPROC 회귀 fix + daemon fade"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v28 CHANGE A: POSTPROC_SAME_SPK_GAP 언셋 (default 0.3 복원) ===
unset LATENTSYNC_POSTPROC_SAME_SPK_GAP

# === v27 유지 (B): sentence-level same-spk merge ===
# 같은 화자 + gap<1.0s + 합쳐도 14s 이내면 안전 병합 (라벨 변경 X)
export LATENTSYNC_SAME_SPK_GAP=1.0
export LATENTSYNC_SAME_SPK_MERGE_CAP=14.0

# === v27 유지: SENT_MERGE_CAP ===
export LATENTSYNC_SENT_MERGE_CAP=14.0

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
    --name "test4_v28" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v28_${RUN_ID}.log | \
    grep -E "Separate|chunk_|DiariZen|turn 수|최종 화자 수|Outlier|Segments\]|Merge\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Reassign|FaceID|Profiles\]|fade-out|trim" | head -200

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v28_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v28_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v28 효과 (회귀 fix 확인) ==="
echo "[post_process turn 병합 (default 0.3s gap, 결과 안정성 회복 기대)]"
grep "turn 수:" /tmp/test4_v28_${RUN_ID}.log 2>/dev/null
echo
echo "[같은 화자 인접 sentence 안전 병합 (목표: SPK_05 3개 → 1개 회복)]"
grep "같은 화자 인접 sentence" /tmp/test4_v28_${RUN_ID}.log 2>/dev/null

echo
echo "=== Speaker 검증 (기존) ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v28_*/meta/test4_v28_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi

echo
echo "=== Auto Metric (GT-free, v26/v27/v28 비교) ==="
V26_JSON=$(ls -t /workspace/media/runs/test4_v26_*/meta/test4_v26_chunk_000_segments.json 2>/dev/null | head -1)
V27_JSON=$(ls -t /workspace/media/runs/test4_v27_*/meta/test4_v27_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ] && [ -f /workspace/tmp/eval_diarization_auto.py ]; then
    ARGS=""
    [ -n "$V26_JSON" ] && ARGS="$ARGS $V26_JSON"
    [ -n "$V27_JSON" ] && ARGS="$ARGS $V27_JSON"
    ARGS="$ARGS $SEG_JSON"
    python3 /workspace/tmp/eval_diarization_auto.py $ARGS
fi

echo
echo "=== SPEAKER_05 영역 상세 (목표: 59.20~72.00s 모두 SPK_05) ==="
python3 - << "PYEOF"
import json, glob
files = sorted(glob.glob("/workspace/media/runs/test4_v28_*/meta/test4_v28_chunk_000_segments.json"))
if files:
    data = json.load(open(files[-1]))
    spk5 = sorted([g for g in data['groups'] if g['speaker'] == 'SPEAKER_05'], key=lambda g: g['group_start'])
    print(f"SPEAKER_05: {len(spk5)} groups")
    for g in spk5:
        dur = g['group_end'] - g['group_start']
        print(f"  [g{g['group_idx']}] {g['group_start']:.2f}~{g['group_end']:.2f}s ({dur:.2f}s) '{g['text']}'")
    print()
    print("=== 59-72s 영역 모든 segments ===")
    region = sorted([g for g in data['groups'] if 58.0 <= g['group_start'] <= 73.0],
                    key=lambda g: g['group_start'])
    for g in region:
        dur = g['group_end'] - g['group_start']
        print(f"  [g{g['group_idx']}] {g['speaker']} {g['group_start']:.2f}~{g['group_end']:.2f}s ({dur:.2f}s) '{g['text']}'")
PYEOF
