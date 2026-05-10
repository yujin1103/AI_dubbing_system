#!/bin/bash
# AIHub TS1~TS50 mp4 추출 + 즉시 .tar.part 정리
# 디스크 안전 모드: 각 TS 추출 성공 후 원본 .tar.part 즉시 삭제
# 사용: bash scripts/extract_aihub_all.sh

set -e

ROOT="E:/Download/009.립리딩(입모양)_음성인식_데이터/01.데이터"
OUT="E:/TTS_capstone/media/aihub_extracted"
LOG="E:/TTS_capstone/logs/extract_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT/video_train" "$(dirname $LOG)"

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"
}

extract_tar() {
    local DIR="$1"
    local NAME="$2"
    local DEST="$3"

    log "=== $NAME 추출 시작 → $DEST ==="

    local PARTS=$(ls "$DIR"/${NAME}.tar.part* 2>/dev/null | awk -F'.tar.part' '{ printf "%020d %s\n", $2, $0 }' | sort -n | awk '{ print $2 }')

    if [ -z "$PARTS" ]; then
        log "[SKIP] $NAME 청크 없음 (이미 처리됨?)"
        return 0
    fi

    local COUNT=$(echo "$PARTS" | wc -l)
    local TOTAL_SIZE=$(du -cb $PARTS 2>/dev/null | tail -1 | awk '{print $1}')
    local SIZE_GB=$(echo "scale=2; $TOTAL_SIZE / 1073741824" | bc)
    log "[Info] $COUNT parts, ${SIZE_GB}GB"

    # 추출
    local START=$(date +%s)
    cat $PARTS | tar -xf - -C "$DEST"
    local END=$(date +%s)
    local ELAPSED=$((END - START))
    log "[OK] $NAME 추출 완료 (${ELAPSED}초)"

    # 추출 결과 확인
    local EXTRACTED_DIR="$DEST/$NAME"
    if [ -d "$EXTRACTED_DIR" ]; then
        local MP4_COUNT=$(find "$EXTRACTED_DIR" -name "*.mp4" | wc -l)
        local DIR_SIZE=$(du -sh "$EXTRACTED_DIR" | awk '{print $1}')
        log "[Verify] $MP4_COUNT mp4 files, $DIR_SIZE in $EXTRACTED_DIR"

        if [ "$MP4_COUNT" -eq 0 ]; then
            log "[ERROR] $NAME mp4 0개! .tar.part 보존하고 중단"
            exit 1
        fi
    else
        log "[ERROR] 추출 폴더 없음: $EXTRACTED_DIR"
        exit 1
    fi

    # 추출 성공 → .tar.part 즉시 삭제 (디스크 확보)
    rm -f "$DIR"/${NAME}.tar.part*
    log "[CLEAN] ${NAME}.tar.part* 삭제됨"

    # 디스크 상태
    log "[Disk] $(df -h /e | tail -1 | awk '{print "Used="$3", Free="$4", Use="$5}')"
    echo "" | tee -a "$LOG"
}

log "========================================"
log "AIHub TS 추출 시작 (TS1, 10, 20, 30, 40, 50)"
log "========================================"
log "[Disk] 시작: $(df -h /e | tail -1 | awk '{print "Used="$3", Free="$4}')"
echo "" | tee -a "$LOG"

# 작은 것부터 (TS50 75GB → TS10 80GB → TS30 81GB → TS20 86GB → TS1 92GB → TS40 94GB)
for TS in TS50 TS10 TS30 TS20 TS1 TS40; do
    extract_tar "$ROOT/1.Training/원천데이터" "$TS" "$OUT/video_train"
done

log "========================================"
log "✅ 모든 TS 추출 완료"
log "========================================"
log "[Final disk] $(df -h /e | tail -1)"
log ""
log "=== 추출된 폴더 ==="
du -sh "$OUT/video_train"/* | tee -a "$LOG"
log ""
log "=== mp4 총 개수 ==="
find "$OUT/video_train" -name "*.mp4" | wc -l | xargs -I {} log "Total mp4 files: {}"
