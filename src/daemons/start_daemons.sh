#!/bin/bash
# 모든 데몬 시작 스크립트 (background)
# 경로는 스크립트 자기 위치에서 자동 도출 — 컨테이너 재시작/리마운트(/workspace vs /workspace/project)에도 안전.
# 첫 실행: TTS 60-90초, ASR 30-45초, Diarize 20-30초 모델 로딩 / 이후 HTTP 호출 < 1초.
#
# 사용:
#   bash start_daemons.sh            # 3개 모두 시작 (이미 떠 있으면 skip)
#   bash start_daemons.sh diarize    # diarize(8903) 만 — 16GB GPU 에서 단독 권장
#   bash stop_daemons.sh             # 중지
#   curl http://127.0.0.1:8903/health
#
# ※ 16GB GPU 에 3개 동시 상주는 빠듯하다. 화자분리만 필요하면 'diarize' 인자로 8903 만 띄워라.

set -e

# --- 경로 자동 도출 (이 스크립트는 <project>/src/daemons/ 에 있다) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

ONLY="${1:-all}"   # all | cosy | asr | diarize

start_one() {
    local name="$1" port="$2" venv="$3" script="$4"
    if curl -s "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        echo "[$name] already running (:$port)"
        return
    fi
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "[$name] SKIP — $SCRIPT_DIR/$script 없음"
        return
    fi
    echo "[$name] starting on $port ..."
    nohup "/opt/$venv/bin/python" "$SCRIPT_DIR/$script" --port "$port" \
        > "$LOG_DIR/${name}_daemon.log" 2>&1 &
    echo "[$name] PID $!  log=$LOG_DIR/${name}_daemon.log"
}

echo "=== 데몬 시작 (project=$PROJECT_ROOT, target=$ONLY) ==="

[ "$ONLY" = "all" ] || [ "$ONLY" = "cosy" ]    && start_one cosy    8901 venv_lipsync  cosyvoice_daemon.py
[ "$ONLY" = "all" ] || [ "$ONLY" = "asr" ]     && start_one asr     8902 venv_asr       asr_daemon.py
[ "$ONLY" = "all" ] || [ "$ONLY" = "diarize" ] && start_one diarize 8903 venv_diarizen  diarize_daemon.py

echo ""
echo "=== 모델 로딩 대기 (30-90초) ==="
echo "확인: curl http://127.0.0.1:8901/health  (cosy)"
echo "      curl http://127.0.0.1:8902/health  (asr)"
echo "      curl http://127.0.0.1:8903/health  (diarize)"
