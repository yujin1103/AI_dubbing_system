#!/bin/bash
# pyannote-3.1 daemon(8943) on-demand 제어 — 화자분리 단계에만 켜서 GPU 점유 최소화.
#
# 왜: pyannote-3.1 을 상시 켜두면 이후 립싱크/TTS 단계에서 GPU OOM 위험.
#     화자분리(selective_local_split) 때만 켜고, 끝나면 꺼서 GPU 를 반환한다.
#
# 사용:
#   bash pyannote_ctl.sh start   # diarize 단계 시작 시
#   bash pyannote_ctl.sh stop    # diarize 단계 끝나고 (cut_chunks/asr/tts/lipsync 전)
#   bash pyannote_ctl.sh status
#
# 핵심: pyannote 는 venv_pyann + LD_LIBRARY_PATH="" 에서만 로드됨
#       (venv_diarizen=lightning 불일치, 시스템 cuDNN 9.19 vs torch 번들 9.20 충돌 회피).
set -u
PORT=8943
PYBIN=/opt/venv_pyann/bin/python
DAEMON=/workspace/src/daemons/pyannote_diarize_daemon.py
LOG=/workspace/media/logs/pyann_8943_ctl.log
: "${HF_TOKEN:=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep '^HF_TOKEN=' | cut -d= -f2)}"

is_up() { curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true'; }

case "${1:-status}" in
  start)
    if is_up; then echo "[pyannote] already loaded on $PORT"; exit 0; fi
    pkill -f "pyannote_diarize_daemon.py --port $PORT" 2>/dev/null; sleep 2
    echo "[pyannote] starting on $PORT (venv_pyann, LD_LIBRARY_PATH cleared)..."
    PYANNOTE_MODEL=pyannote/speaker-diarization-3.1 \
    HF_HOME=/workspace/media/model_cache/huggingface \
    HF_TOKEN="$HF_TOKEN" LD_LIBRARY_PATH="" \
      nohup "$PYBIN" "$DAEMON" --port "$PORT" > "$LOG" 2>&1 &
    echo "[pyannote] pid $!  — 모델 로드 대기 (~10-30s). 확인: curl 127.0.0.1:$PORT/health"
    for i in $(seq 1 30); do sleep 2; if is_up; then echo "[pyannote] loaded ✓"; exit 0; fi; done
    echo "[pyannote] WARN: 30s 내 미로드. 로그: $LOG"; tail -5 "$LOG"; exit 1
    ;;
  stop)
    pkill -f "pyannote_diarize_daemon.py --port $PORT" 2>/dev/null && echo "[pyannote] stopped (GPU 반환)" || echo "[pyannote] not running"
    ;;
  status)
    if is_up; then echo "[pyannote] UP (loaded)"; else echo "[pyannote] DOWN/not-loaded"; fi
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1;;
esac
