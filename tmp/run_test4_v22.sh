#!/bin/bash
# test4 v22 — 사용자 우선순위 3개 fix (안전 재설계)
#   ① trim/발화 끝 잘림 (1순위)
#   ② 마지막 토끼 split (2순위)
#   ③ 기계음 (3순위)
#
# ─ 사용자 지적 (v22 1차) ─
#   "단어를 병합하면 같은 화자라 인식할 경우 다른 말까지 합쳐지잖아"
#   → 마지막 sentence 무조건 병합 / 마지막 group 절대 보호는 *위험*
#   → 안전 조건 (같은 화자 + gap < 1초) 추가 후에만 병합
#
# ─ v22 (재설계) 변경점 ─
#   [코드 변경 — 안전화]
#   1. _split_groups_by_speaker: 마지막 group 절대 보호 *제거*
#      (단일 화자면 어차피 split 안 일어남, 화자 섞이면 정상 split이 옳음)
#   2. _merge_short_sentences: 마지막 sentence 안전 병합
#      - 조건: 같은 화자 + gap < 1.0s + 4단어/2초 이하
#      - diarization 필수, 그 외엔 병합 skip
#      - LATENTSYNC_PROTECT_LAST_MERGE=1 (명시적 enable)
#
#   [환경변수 변경 — trim 감소]
#   3. PREDICT_THRESHOLD 1.20 → 1.10 (사전 재번역 더 적극)
#   4. SENT_MERGE_CAP 12.0 → 10.0 (긴 segment 자체 방지)
#   5. SPLIT_NO_SHORT_WORDS 6 → 8 (짧은 group 보호 강화)
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v22_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v22 - trim 줄임 + 마지막 split 절대 보호"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v22 NEW: trim 줄이기 ===
export LATENTSYNC_PREDICT_THRESHOLD=1.10           # 1.20 → 1.10 (사전 재번역 적극)
export LATENTSYNC_SENT_MERGE_CAP=10.0              # 12.0 → 10.0 (긴 segment 자체 방지)

# === v22 NEW: 마지막 split 안전 보호 (재설계) ===
# PROTECT_LAST_GROUP은 제거됨 — 마지막 group에 두 화자 섞이면 정상 split 필요
unset LATENTSYNC_PROTECT_LAST_GROUP
export LATENTSYNC_PROTECT_LAST_MERGE=1              # 안전 조건 (같은 화자+짧은 gap)만 병합
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=8            # 6 → 8 (짧은 group 보호 강화)

# === v21 유지: OUTLIER OFF (가짜 화자 차단) ===
export LATENTSYNC_OUTLIER_OFF=1

# === v21 유지: DiariZen ===
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.5
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.55
export LATENTSYNC_DIARIZE_SHORT_TURN=1.5

# === v20 유지: ref 6초+ ===
export LATENTSYNC_REF_MIN_DUR=6.0
export LATENTSYNC_REF_MAX_DUR=15.0
export LATENTSYNC_REF_FALLBACK_MIN=4.0
export LATENTSYNC_REF_FALLBACK_MAX=25.0

# === v20 유지: split smoothing 강화 ===
export LATENTSYNC_SPLIT_MAJORITY_TH=0.80

# === v20 유지: AV-Reassign 보수화 + max_dur 2.0 ===
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
    --name "test4_v22" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v22_${RUN_ID}.log | \
    grep -E "Separate|chunk_|DiariZen|최종 화자 수|Outlier|Segments\]|Merge\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Reassign|Profiles\]|fade-out|trim|overflow|축약" | head -200

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v22_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v22_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v22 핵심 효과 ==="
echo "[trim 발생 횟수 (목표: 줄어듦)]"
grep -c "trim" /tmp/test4_v22_${RUN_ID}.log 2>/dev/null
echo
echo "[큰 trim (>1초) 케이스]"
grep "[0-9]\.[0-9][0-9]s trim" /tmp/test4_v22_${RUN_ID}.log 2>/dev/null | grep -oE "[1-9]\.[0-9]+s trim" | head -10
echo
echo "[축약 재번역 발동]"
grep -c "축약됨" /tmp/test4_v22_${RUN_ID}.log 2>/dev/null
echo
echo "[마지막 sentence 보호]"
grep "마지막 sentence 강제 병합" /tmp/test4_v22_${RUN_ID}.log 2>/dev/null || echo "(병합 발동 안 함 — 이미 1 sentence)"
echo
echo "=== Speaker 검증 ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v22_*/meta/test4_v22_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi
echo
echo "[마지막 group 확인 — 'stop being a stupid rabbit' 한 문장 유지?]"
python3 - << "PYEOF"
import json, glob
files = sorted(glob.glob("/workspace/media/runs/test4_v22_*/meta/test4_v22_chunk_000_segments.json"))
if files:
    data = json.load(open(files[-1]))
    groups = data["groups"]
    print(f"총 {len(groups)} groups")
    for g in groups[-3:]:
        dur = g["group_end"] - g["group_start"]
        print(f"  [g{g['group_idx']}] {g['group_start']:.2f}~{g['group_end']:.2f}s ({dur:.2f}s) [{g['speaker']}/{g['emotion']}]: '{g['text']}'")
PYEOF
