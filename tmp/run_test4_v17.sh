#!/bin/bash
# test4 v17 — 커뮤니티 검증 방향 적용 (C: TTS speed isochrony)
#
# ─ v15까지의 잘못된 방향 (HARD CONSTRAINT) 폐기 ─
#   학계+업계 합의 (VideoDubber AAAI 2023, CAPEL arxiv 2508.13805):
#     · hard syllable constraint는 증명된 실패 패턴
#     · 길이는 LLM이 아닌 TTS speed + time-stretch가 흡수
#     · 의미 보존 > 음절 정확도
#
# ─ v17 변경점 ─
#   1) LLM prompt soft range 재작성 (HARD CONSTRAINT 제거, "meaning > syllables")
#   2) CAPEL countdown 마커 — 짧은 utterance(<1.5s, target≤4syl) 한정으로
#      LLM 길이 준수율 30%→95% 향상 ('Good' → '<2>_<1>_<0>' 가이드)
#   3) SYL_RANGE 0.3 → 0.5 복원 (v15 첫부분 잘림 직접 원인 제거)
#   4) MAX_STRETCH 1.07 → 1.15 (rubberband WSOLA 자연 한계 활용)
#   5) MIN_STRETCH 0.95 → 0.90 (atempo 늘림 폭 확장)
#   6) TOL_LATE 0.15s → 0.20s (LLM soft range와 매칭)
#   7) PREDICT_THRESHOLD MAX_STRETCH 분리 → 1.20 (재번역 과적극 방지)
#   8) retranslate_shorter 결과 검증 (target * 0.5 미만 거부 — 의미 손상 방지)
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v17_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v17 - community-verified soft range + CAPEL"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === 핵심 v17: soft range + CAPEL + stretch 한계 확장 ===
export LATENTSYNC_SYL_RANGE=0.5               # v14 복원, soft range
export LATENTSYNC_CAPEL_SHORT=1                # CAPEL countdown for short utt (<1.5s, ≤4syl)
export LATENTSYNC_MAX_STRETCH=1.15             # rubberband 자연 한계 (was 1.07)
export LATENTSYNC_MIN_STRETCH=0.90             # 늘림 폭 확장 (was 0.95)
export LATENTSYNC_TOL_LATE=0.20                # tolerance 확장 (was 0.15)
export LATENTSYNC_PREDICT_THRESHOLD=1.20       # 재번역 trigger 완화 (was 1.07)
export LATENTSYNC_SENT_MIN_WORDS=3             # v14 복원 (v15 5 too aggressive)
export LATENTSYNC_SENT_MIN_DURATION=1.5        # v14 복원

# === split ON + 같은 화자만 병합 (v15 회귀 차단) ===
unset LATENTSYNC_SPLIT_BY_SPEAKER_OFF          # split ON (v14 off는 다른 화자 강제병합 야기)
export LATENTSYNC_SPLIT_MIN_SUB_WORDS=3        # 같은 화자 짧은 sub만 병합 (v16 검증된 로직)

# === QG OFF + 단일 ref 유지 (긍정 효과) ===
export LATENTSYNC_QG_OFF=1
export LATENTSYNC_REF_SINGLE_PER_SPEAKER=1

# === 화자 분리 (v11 검증된 값) ===
export LATENTSYNC_DIARIZE_SHORT_TURN=1.5
export LATENTSYNC_OUTLIER_FAR_THRESH=0.20
export LATENTSYNC_DIARIZE_MERGE_THRESHOLD=0.55
export LATENTSYNC_DIARIZE_MIN_DUR_CENTROID=0.3
export LATENTSYNC_AV_MERGE_OFF=1

# === 기타 ===
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
    --name "test4_v17" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v17_${RUN_ID}.log | \
    grep -E "Separate|Ensemble|Split\]|chunk_|DiariZen|최종 화자 수|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|축약|countdown|lip-sync|ratio|atempo" | head -120

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v17_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v17_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v17 통계: 짧은 group + 길이 mismatch ==="
python3 - << "PYEOF"
import json, glob, re
files = sorted(glob.glob("/workspace/media/runs/test4_v17_*/meta/test4_v17_chunk_000_segments.json"))
if files:
    data = json.load(open(files[-1]))
    groups = data["groups"]
    short = sum(1 for g in groups if (g["group_end"]-g["group_start"]) < 1.5 or len(g["text"].split()) < 4)
    print(f"v17: 총 {len(groups)} groups, 짧은 {short} ({short*100//max(1,len(groups))}%)")
    # 첫 5개 group + 가장 짧음
    for g in groups[:5]:
        dur = g["group_end"]-g["group_start"]
        print(f"  [g{g['group_idx']:>2}] {dur:.2f}s [{g['speaker']}/{g['emotion']}]: \"{g['text'][:50]}\"")
    shortest = min(groups, key=lambda g: g["group_end"]-g["group_start"])
    sd = shortest["group_end"]-shortest["group_start"]
    print(f"가장 짧음: {sd:.2f}s [{shortest['speaker']}] \"{shortest['text'][:40]}\"")
PYEOF

echo
echo "=== v17 ratio 분포 (TTS speed retry 발동 빈도) ==="
grep -oE "ratio[=:][0-9.]+|atempo [0-9.]+x|↳" /tmp/test4_v17_${RUN_ID}.log 2>/dev/null | sort | uniq -c | sort -rn | head -20
