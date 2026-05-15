#!/bin/bash
# test4 v20 — v19 + 검색 기반 기계음 대책 + 사용자 피드백 fix
#
# ─ v19 회귀 fix 확인 (SPEAKER_02 톤, 기계음, ref 오염) — 유지 ─
# ─ v20 신규 변경 ─
#   [기계음 해결 검색 결과 적용]
#   B. instruct_text prefix 'You are a helpful assistant.<|endofprompt|>' 통일
#      (이미 cosyvoice_daemon.py에 적용됨 — 검증)
#   A. ref audio 길이 6초+ 하한 (CosyVoice 공식 권장, BGM leak 영향 줄임)
#   D. frontend_zero_shot prompt_text override 패치 (Issue #1400 보호)
#
#   [사용자 피드백 fix]
#   ① 발화 끝 잘림 → fade-out 30ms 추가 (TTS 출력 + trim 발생 시)
#   ② 마지막 토끼 문장 split → _split_groups_by_speaker 강화
#       - 6 단어 이하 group split 비활성
#       - 2-word smoothing (A B B A → A A A A)
#       - majority 80%+ 면 group 전체 통일
#   ③ SPEAKER_05 과검출 (1회만 발화인데 4회 잡힘) → DiariZen 수치 strict
#       - MIN_DUR_CENTROID 0.3 → 1.0 (centroid 산정 최소 길이 확대)
#       - OUTLIER_FAR_THRESH 0.20 → 0.25 (outlier 더 적극)
#       - AV_REASSIGN_MAX_DUR 1.0 → 2.0 (긴 segment도 face owner 기반 교정)
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v20_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v20 - 기계음 검색 + 사용자 피드백 fix"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v20 NEW: ref audio 6초+ 하한 (CosyVoice 공식 권장) ===
export LATENTSYNC_REF_MIN_DUR=6.0           # 3.0 → 6.0
export LATENTSYNC_REF_MAX_DUR=15.0
export LATENTSYNC_REF_FALLBACK_MIN=4.0      # 2.0 → 4.0
export LATENTSYNC_REF_FALLBACK_MAX=25.0

# === v20 NEW: split smoothing 강화 ===
export LATENTSYNC_SPLIT_NO_SHORT_WORDS=6     # 6 단어 이하 split 비활성
export LATENTSYNC_SPLIT_MAJORITY_TH=0.80     # majority 80%+ 통일

# === v20 NEW: DiariZen strict (SPEAKER_05 과검출 방지) ===
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=1.0   # 0.3 → 1.0
export LATENTSYNC_OUTLIER_FAR_THRESH=0.25         # 0.20 → 0.25
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.55
export LATENTSYNC_DIARIZE_SHORT_TURN=1.5

# === v20 NEW: AV-Reassign max_dur 확대 + 보수화 유지 ===
unset LATENTSYNC_AV_REASSIGN_OFF
export LATENTSYNC_AV_REASSIGN_DOMINANT=0.85       # 0.90 → 0.85 (약간 완화)
export LATENTSYNC_AV_REASSIGN_SHARE=0.75          # 0.80 → 0.75
export LATENTSYNC_AV_REASSIGN_MAX_DUR=2.0          # 1.0 → 2.0 (더 긴 segment도 fix)
export LATENTSYNC_REF_EXCLUDE_REASSIGNED=1         # ref 오염 방지

# === C 유지 (LLM soft range + CAPEL) ===
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
    --name "test4_v20" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v20_${RUN_ID}.log | \
    grep -E "Separate|Ensemble|chunk_|DiariZen|최종 화자 수|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Fusion|AV-Reassign|Profiles\]|face owner|fade-out|trim" | head -200

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v20_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v20_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v20 핵심 효과 ==="
echo "[ref audio 길이 정책]"
grep -E "Profiles\] 6초\+ ref|Profiles\] AV-Reassign" /tmp/test4_v20_${RUN_ID}.log 2>/dev/null
echo
echo "[AV-Reassign 결과]"
grep "AV-Reassign\]" /tmp/test4_v20_${RUN_ID}.log 2>/dev/null | grep "→ SPEAKER\|재할당" | head -10
echo
echo "[trim + fade-out 발생]"
grep "fade-out" /tmp/test4_v20_${RUN_ID}.log 2>/dev/null | head -5

echo
echo "=== Speaker 검증 (SPEAKER_05 1회 기대) ==="
SEG_JSON=$(ls -t /workspace/media/runs/test4_v20_*/meta/test4_v20_chunk_000_segments.json 2>/dev/null | head -1)
if [ -n "$SEG_JSON" ]; then
    python3 /workspace/tmp/eval_speaker_count.py "$SEG_JSON" /workspace/media/gt/test4_gt.json
fi
