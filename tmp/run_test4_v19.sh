#!/bin/bash
# test4 v19 — v18 회귀 fix + B 재구조 + 안전장치
#
# ─ v18 회귀 4가지 + 원인 ─
#   ① SPEAKER_02 여성 → 남성처럼
#      원인: AV-Reassign이 SPEAKER_01 (남성) segment 3개를 SPEAKER_02 (여성)로
#      재할당 → 그 segments가 SPEAKER_02 ref bank에 섞임 → 남성톤 합성
#   ② 기계음 더 심해짐
#      원인: MAX_STRETCH 1.10이 *오히려* ratio 1.07~1.10 영역의 자연스러운
#      segment를 atempo 압축 강제 → 짧은 클립 metallic ring
#   ③ id 8 늦게 나와 잘림
#      원인: AV-Reassign으로 segment timing 누적 → 다음 segment 침범
#   ④ 마지막 발화 다시 나눠짐
#      원인: AV-Reassign 후 split 로직 재발동
#
# ─ v19 처방 ─
#   1) MAX_STRETCH 1.15 / MIN_STRETCH 0.90 복원 (v17 값, 기계음 회복)
#   2) PREDICT_THRESHOLD 1.20 복원 (재번역 과적극 방지)
#   3) AV-Reassign DOMINANT 0.60 → 0.90 (face owner 매우 확실할 때만)
#   4) AV-Reassign SHARE 0.50 → 0.80 (segment dominant face 80%+)
#   5) AV-Reassign MAX_DUR 1.0s (긴 발화는 audio 신뢰)
#   6) REF_EXCLUDE_REASSIGNED=1 (재할당 영역 ref bank 제외) ★ 안전장치
set -e

INPUT=/workspace/media/input/test4.mp4
RUN_ID=test4_v19_$(date +%Y%m%d_%H%M%S)

echo "==================================================="
echo "  test4 v19 - v18 회귀 fix + B 재구조 + ref 안전장치"
echo "  Run ID: $RUN_ID"
echo "==================================================="

# === v19: 기계음 회복 (v17 값 복원) ===
export LATENTSYNC_MAX_STRETCH=1.15
export LATENTSYNC_MIN_STRETCH=0.90
export LATENTSYNC_TOL_LATE=0.20
export LATENTSYNC_PREDICT_THRESHOLD=1.20

# === v19: AV-Reassign 보수화 + ref 오염 차단 ===
unset LATENTSYNC_AV_REASSIGN_OFF                  # reassign ON
export LATENTSYNC_AV_REASSIGN_DOMINANT=0.90        # 0.60→0.90 (false positive 차단)
export LATENTSYNC_AV_REASSIGN_SHARE=0.80           # 0.50→0.80
export LATENTSYNC_AV_REASSIGN_MAX_DUR=1.0          # 짧은 segment만 (1초 이하)
export LATENTSYNC_REF_EXCLUDE_REASSIGNED=1         # ref bank 오염 방지 ★

# === AV merge OFF (v17까지 OFF였음, v18은 ON했다가 회귀 — 다시 OFF) ===
export LATENTSYNC_AV_MERGE_OFF=1

# === C 유지 (LLM soft range + CAPEL) ===
export LATENTSYNC_SYL_RANGE=0.5
export LATENTSYNC_CAPEL_SHORT=1
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
    --name "test4_v19" \
    --run-id "$RUN_ID" \
    --lang ko \
    --content-type drama 2>&1 | tee /tmp/test4_v19_${RUN_ID}.log | \
    grep -E "Separate|Ensemble|Split\]|chunk_|DiariZen|최종 화자 수|Segments\]|TTS\] \[SPEAKER|Pipeline\] 완료|Traceback|ERROR|FAILED|완료:|AV-Fusion|AV-Reassign|Profiles\]|face owner|spurious|ratio<|atempo" | head -180

ELAPSED=$(($(date +%s) - START))
echo
echo "==================================================="
printf "  완료: %ds (%dm %ds)\n" $ELAPSED $((ELAPSED / 60)) $((ELAPSED % 60))
RESULT=/workspace/media/output/test4_v19_ko_${RUN_ID}.mp4
if [ -f "$RESULT" ]; then
    echo "  결과: $RESULT"
    echo "  호스트: E:\\TTS_capstone\\media\\output\\test4_v19_ko_${RUN_ID}.mp4"
fi
echo "==================================================="

echo
echo "=== v19 핵심 효과 ==="
echo "[AV-Reassign 재할당 (보수화 후)]"
grep "AV-Reassign\]" /tmp/test4_v19_${RUN_ID}.log 2>/dev/null | grep "→ SPEAKER" || echo "재할당 없음 (보수화 효과)"
echo
echo "[ref bank 제외된 segments]"
grep "AV-Reassign 영역" /tmp/test4_v19_${RUN_ID}.log 2>/dev/null || echo "제외 없음"
echo
echo "[atempo 압축 빈도 (기계음 지표)]"
grep -c "atempo 1\." /tmp/test4_v19_${RUN_ID}.log 2>/dev/null || echo 0

python3 - << "PYEOF"
import json, glob
files = sorted(glob.glob("/workspace/media/runs/test4_v19_*/meta/test4_v19_chunk_000_segments.json"))
if files:
    data = json.load(open(files[-1]))
    groups = data["groups"]
    from collections import Counter
    cnt = Counter(g["speaker"] for g in groups)
    print(f"\nv19: 총 {len(groups)} groups, 화자 {len(cnt)}")
    for s, c in sorted(cnt.items()):
        print(f"  {s}: {c} groups")
PYEOF
