#!/bin/bash
# 모든 데몬 중지 스크립트
echo "=== 데몬 중지 ==="
pkill -f cosyvoice_daemon && echo "[Cosy] stopped" || echo "[Cosy] not running"
pkill -f asr_daemon && echo "[ASR] stopped" || echo "[ASR] not running"
pkill -f diarize_daemon && echo "[Diarize] stopped" || echo "[Diarize] not running"
sleep 2
echo "=== 확인 ==="
ps aux | grep -E 'cosy|asr|diarize' | grep -v grep | grep daemon || echo "all stopped"
