#!/bin/bash
# Multi-angle 전처리 — A 100% + B/C/F/G 50%
# 추출(extract_aihub_all.sh) 완료 후 실행
# 사용: bash scripts/preprocess_multi_angle.sh

# Git Bash MSYS path 자동 변환 비활성화 (docker exec 인자 보호)
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

set -e

CONTAINER="dubbing_pipeline"
JSON_ROOT="/workspace/media/aihub_extracted/labels_train"
VIDEO_ROOT="/workspace/media/aihub_extracted/video_train"
OUT_HOST="E:/TTS_capstone/media/aihub_processed/train"
OUT_CONT="/workspace/media/aihub_processed/train"
LOG="E:/TTS_capstone/logs/preprocess_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT_HOST" "$(dirname $LOG)"

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"
}

run_angle() {
    local ANGLE=$1
    local MAX=$2          # 0 = no limit
    local START=$(date +%s)

    if [ "$MAX" -eq 0 ]; then
        log "=== Angle $ANGLE 시작 (전체) ==="
        docker exec $CONTAINER /opt/venv_lipsync/bin/python /workspace/scripts/aihub_face_crop.py \
            --json_root $JSON_ROOT --video_root $VIDEO_ROOT --out_dir $OUT_CONT \
            --filter_angle $ANGLE 2>&1 | tee -a "$LOG"
    else
        log "=== Angle $ANGLE 시작 (max=$MAX) ==="
        docker exec $CONTAINER /opt/venv_lipsync/bin/python /workspace/scripts/aihub_face_crop.py \
            --json_root $JSON_ROOT --video_root $VIDEO_ROOT --out_dir $OUT_CONT \
            --filter_angle $ANGLE --max_videos $MAX 2>&1 | tee -a "$LOG"
    fi

    local END=$(date +%s)
    local ELAPSED=$((END - START))
    local COUNT=$(ls "$OUT_HOST" 2>/dev/null | wc -l)
    log "=== Angle $ANGLE 완료 (${ELAPSED}초, 누적 $COUNT 영상) ==="
    echo "" | tee -a "$LOG"
}

log "========================================"
log "Multi-angle 전처리 시작"
log "예상: A 100%(120) + B/C/F/G 50%(60×4=240) = 360 영상"
log "========================================"

# 컨테이너 살아있는지
if ! docker ps --filter name=$CONTAINER --format "{{.Names}}" | grep -q $CONTAINER; then
    log "[ERROR] 컨테이너 $CONTAINER 죽어있음. docker compose up -d 실행 필요"
    exit 1
fi

# A 100% (정면, 가장 중요한 데이터)
run_angle A 0

# B/C/F/G 50% (측면, catastrophic forgetting 방지용 적당량)
run_angle B 60
run_angle C 60
run_angle F 60
run_angle G 60

log "========================================"
log "✅ 전처리 완료"
log "========================================"
TOTAL=$(ls "$OUT_HOST" 2>/dev/null | wc -l)
log "총 mp4 파일 수: $TOTAL"
log "fileslist 위치: E:/TTS_capstone/media/aihub_processed/fileslist_train.txt"
