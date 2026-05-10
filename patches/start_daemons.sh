#!/bin/bash
# 모든 데몬 시작 스크립트 (background)
# 첫 실행: TTS 60-90초, ASR 30-45초, Diarize 20-30초 모델 로딩
# 이후: HTTP 호출 < 1초 응답
#
# 사용:
#   bash start_daemons.sh        # 시작
#   bash stop_daemons.sh         # 중지
#   curl http://127.0.0.1:8901/health  # 상태 확인

set -e

LOG_DIR="/workspace/media/logs"
mkdir -p "$LOG_DIR"

echo "=== 데몬 시작 ==="

# 1. CosyVoice3 daemon (port 8901)
if curl -s http://127.0.0.1:8901/health > /dev/null 2>&1; then
    echo "[Cosy] already running"
else
    echo "[Cosy] starting on 8901..."
    nohup /opt/venv_cosy/bin/python /workspace/patches/cosyvoice_daemon.py --port 8901 \
        > "$LOG_DIR/cosy_daemon.log" 2>&1 &
    echo "[Cosy] PID $!"
fi

# 2. ASR daemon (port 8902)
if curl -s http://127.0.0.1:8902/health > /dev/null 2>&1; then
    echo "[ASR] already running"
else
    echo "[ASR] starting on 8902..."
    nohup /opt/venv_asr/bin/python /workspace/patches/asr_daemon.py --port 8902 \
        > "$LOG_DIR/asr_daemon.log" 2>&1 &
    echo "[ASR] PID $!"
fi

# 3. Diarize daemon (port 8903)
if curl -s http://127.0.0.1:8903/health > /dev/null 2>&1; then
    echo "[Diarize] already running"
else
    echo "[Diarize] starting on 8903..."
    nohup /opt/venv_diarizen/bin/python /workspace/patches/diarize_daemon.py --port 8903 \
        > "$LOG_DIR/diarize_daemon.log" 2>&1 &
    echo "[Diarize] PID $!"
fi

echo ""
echo "=== 모델 로딩 대기 (30-90초) ==="
echo "확인: curl http://127.0.0.1:8901/health"
echo "      curl http://127.0.0.1:8902/health"
echo "      curl http://127.0.0.1:8903/health"
echo ""
echo "log: $LOG_DIR/{cosy,asr,diarize}_daemon.log"
