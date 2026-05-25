#!/bin/bash
# Start 4-way fusion diarize daemons + CosyVoice TTS daemon (keep alive).
# Run once; subsequent video processing reuses these daemons (fast).
# Stop with: ./kill_daemons.sh
#
# Critical env vars for v195-accurate diarization (from main repo orchestrator.py):
#   LATENTSYNC_OUTLIER_OFF=1     # 가짜 SPK_99/98/97 outlier 라벨 방지
# (이 외에 v178/v190/v194 env는 메인 repo orchestrator만 적용 가능 — standalone은 자체 refiner 사용)
#
# Usage:
#   bash start_daemons.sh                            # Default ports
#   FUSION_PORT=8903 bash start_daemons.sh           # Custom port

PATCHES_DIR=${PATCHES_DIR:-/workspace/patches}
COSYVOICE_PORT=${COSYVOICE_PORT:-8901}
FUSION_PORT=${FUSION_PORT:-8903}

# Backend daemon ports
DZ_PORT=8913
NEMO_PORT=8923
PYANN_C1_PORT=8933
PYANN_31_PORT=8943

mkdir -p /tmp/daemon_logs

echo "==> Starting 4 diarize backends + fusion proxy + CosyVoice TTS"
nohup /opt/venv_diarizen/bin/python $PATCHES_DIR/diarize_daemon.py --port $DZ_PORT > /tmp/daemon_logs/dz.log 2>&1 &
nohup /opt/venv_diarizen/bin/python $PATCHES_DIR/nemo_diarize_daemon.py --port $NEMO_PORT > /tmp/daemon_logs/nemo.log 2>&1 &
PYANNOTE_MODEL=pyannote/speaker-diarization-community-1 \
  nohup /opt/venv_pyann/bin/python $PATCHES_DIR/pyannote_diarize_daemon.py --port $PYANN_C1_PORT > /tmp/daemon_logs/py1.log 2>&1 &
PYANNOTE_MODEL=pyannote/speaker-diarization-3.1 \
  nohup /opt/venv_pyann/bin/python $PATCHES_DIR/pyannote_diarize_daemon.py --port $PYANN_31_PORT > /tmp/daemon_logs/py2.log 2>&1 &
sleep 5

FUSION_DIARIZEN_URL=http://127.0.0.1:$DZ_PORT \
FUSION_NEMO_URL=http://127.0.0.1:$NEMO_PORT \
FUSION_PYANNOTE_URL=http://127.0.0.1:$PYANN_C1_PORT \
FUSION_PYANNOTE2_URL=http://127.0.0.1:$PYANN_31_PORT \
nohup /opt/venv_diarizen/bin/python $PATCHES_DIR/fusion_diarize_daemon.py --port $FUSION_PORT > /tmp/daemon_logs/fusion.log 2>&1 &

nohup /opt/venv_lipsync/bin/python $PATCHES_DIR/cosyvoice_daemon.py --port $COSYVOICE_PORT > /tmp/daemon_logs/cosy.log 2>&1 &

echo "==> Waiting for all daemons to become ready (models loading ~5-10min)..."
wait_health() {
    local url=$1 name=$2 max=$3
    for i in $(seq 1 $max); do
        if curl -s -o /dev/null -w "%{http_code}" $url/health 2>/dev/null | grep -q 200; then
            echo "  ✓ $name ready (${i}*3s)"
            return 0
        fi
        sleep 3
    done
    echo "  ✗ $name FAILED to become ready"
    return 1
}

wait_health http://127.0.0.1:$DZ_PORT       DiariZen   200
wait_health http://127.0.0.1:$NEMO_PORT     NeMo       200
wait_health http://127.0.0.1:$PYANN_C1_PORT pyann_c1   200
wait_health http://127.0.0.1:$PYANN_31_PORT pyann_3.1  200
wait_health http://127.0.0.1:$FUSION_PORT   Fusion     30
wait_health http://127.0.0.1:$COSYVOICE_PORT CosyVoice 120

echo
echo "==================================================="
echo "All daemons ready. Use these in process_video.sh:"
echo "  export FUSION_URL=http://127.0.0.1:$FUSION_PORT"
echo "  export COSYVOICE_URL=http://127.0.0.1:$COSYVOICE_PORT"
echo
echo "GPU usage: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "Stop with: bash kill_daemons.sh"
echo "==================================================="
