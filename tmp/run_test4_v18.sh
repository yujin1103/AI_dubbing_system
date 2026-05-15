#!/bin/bash
# test4 v18 — B 작업 (AV 화자분리 강화: per-segment 재할당)
#
# ─ v17(C) 위에 쌓는 변경 ─
#   1) LATENTSYNC_AV_REASSIGN_OFF=0 (default) ★ NEW
#      → face track owner ≠ audio speaker 인 segment 자동 재할당
#      → SPEAKER_3 → 5/0/1 오인 패턴 fix (v1~v15 미해결)
#   2) LATENTSYNC_AV_REASSIGN_DOMINANT=0.60 (face owner 결정 임계)
#   3) LATENTSYNC_AV_REASSIGN_SHARE=0.50 (segment dominant face 점유율 임계)
#
#   4) LATENTSYNC_AV_MERGE_OFF=0 (재활성) ★ — reassign과 상호보완
#      LATENTSYNC_AV_MERGE_FRAMES=20 (보수적, was default 10)
#      LATENTSYNC_AV_MERGE_RATIO=0.50 (보수적, was default 0.30)
#      → 1회 발화 speaker가 후반에 잘못 재출현하는 케이스 fix
#
#   5) v17의 모든 C 설정 그대로 (LLM soft range, CAPEL, stretch 한계)
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v18_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v18 - AV per-segment 재할당 + 보수적 merge"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v18 NEW: AV reassign + 보수적 merge ===
unset LATENTSYNC_AV_REASSIGN_OFF              # reassign ON (default)
export LATENTSYNC_AV_REASSIGN_DOMINANT=0.60    # face owner 임계 (60%+ 점유)
export LATENTSYNC_AV_REASSIGN_SHARE=0.50       # segment dominant face 임계 (50%+)

unset LATENTSYNC_AV_MERGE_OFF                 # merge ON (v14~v17 OFF였음)
export LATENTSYNC_AV_MERGE_FRAMES=20           # 보수적 (default 10)
export LATENTSYNC_AV_MERGE_RATIO=0.50          # 보수적 (default 0.30)

# === v17 C 설정 + 기계음 완화 미세조정 ===
# id 13("나도 몰라") 기계음 분석: speed=0.90 retry + atempo 1.15 *연쇄 누적* 원인
# → MAX_STRETCH 1.10으로 rubberband 안전 영역 복귀, MIN_STRETCH 0.92로 0.90 retry 영역 축소
export LATENTSYNC_SYL_RANGE=0.5
export LATENTSYNC_CAPEL_SHORT=1
export LATENTSYNC_MAX_STRETCH=1.10        # v17 1.15 → 1.10 (기계음 완화)
export LATENTSYNC_MIN_STRETCH=0.92         # v17 0.90 → 0.92 (CosyVoice 0.9 retry 영역 축소)
export LATENTSYNC_TOL_LATE=0.20
export LATENTSYNC_PREDICT_THRESHOLD=1.15   # MAX_STRETCH와 매칭 (was 1.20)
export LATENTSYNC_SENT_MIN_WORDS=3
export LATENTSYNC_SENT_MIN_DURATION=1.5
unset LATENTSYNC_SPLIT_BY_SPEAKER_OFF
export LATENTSYNC_SPLIT_MIN_SUB_WORDS=3
export LATENTSYNC_QG_OFF=1
export LATENTSYNC_REF_SINGLE_PER_SPEAKER=1
export LATENTSYNC_DIARIZE_SHORT_TURN=1.5
export LATENTSYNC_OUTLIER_FAR_THRESH=0.20
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.55
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.3
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
    --name "test4_v18" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v18_${RUN_ID}.log | \
    grep -E "Separate|Ensemble|Split\]|chunk_|DiariZen|최종 화자 수|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Fusion|AV-Reassign|face owner|spurious|자동 병합" | head -150

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v18_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v18_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v18 AV 효과 통계 ==="
echo "[AV-Reassign 재할당 횟수]"
grep -c "AV-Reassign\] [0-9]" /tmp/test4_v18_${RUN_ID}.log 2>/dev/null || echo 0
echo
echo "[face owner 결정 결과]"
grep "face owners:" /tmp/test4_v18_${RUN_ID}.log 2>/dev/null | head -5
echo
echo "[화자 자동 병합 결과]"
grep "자동 병합:" /tmp/test4_v18_${RUN_ID}.log 2>/dev/null || echo "병합 없음"
echo
echo "[spurious 제거]"
grep "spurious 화자 제거" /tmp/test4_v18_${RUN_ID}.log 2>/dev/null || echo "spurious 없음"

python3 - << "PYEOF"
import json, glob
files = sorted(glob.glob("/workspace/media/runs/test4_v18_*/meta/test4_v18_chunk_000_segments.json"))
if files:
    data = json.load(open(files[-1]))
    groups = data["groups"]
    speakers = set(g["speaker"] for g in groups)
    print(f"\nv18: 총 {len(groups)} groups, 화자 {len(speakers)}: {sorted(speakers)}")
    # 화자별 발화 횟수
    from collections import Counter
    cnt = Counter(g["speaker"] for g in groups)
    for s, c in sorted(cnt.items()):
        print(f"  {s}: {c} groups")
PYEOF
