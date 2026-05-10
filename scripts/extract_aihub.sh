#!/bin/bash
# AIHub Innorix 청크 분할 tar 합치기 + 추출 (Git Bash on host)

set -e

ROOT="E:/Download/009.립리딩(입모양)_음성인식_데이터/01.데이터"
OUT="E:/TTS_capstone/media/aihub_extracted"  # 컨테이너에서 /workspace/media/aihub_extracted로 보임

mkdir -p "$OUT"

extract_tar() {
    local DIR="$1"
    local NAME="$2"
    local DEST="$3"

    echo "=== $NAME 추출 시작 → $DEST ==="
    mkdir -p "$DEST"

    # part 파일들을 byte offset 순으로 정렬
    local PARTS=$(ls "$DIR"/${NAME}.tar.part* 2>/dev/null | awk -F'.tar.part' '{ printf "%020d %s\n", $2, $0 }' | sort -n | awk '{ print $2 }')

    if [ -z "$PARTS" ]; then
        echo "[ERROR] $NAME 청크 없음 in $DIR"
        return 1
    fi

    local COUNT=$(echo "$PARTS" | wc -l)
    local TOTAL_SIZE=$(du -cb $PARTS 2>/dev/null | tail -1 | awk '{print $1}')
    local SIZE_GB=$(echo "scale=2; $TOTAL_SIZE / 1073741824" | bc)
    echo "[Info] $COUNT 청크, 총 ${SIZE_GB}GB"

    # cat으로 직접 파이프 → 중간 파일 안 만듦
    cat $PARTS | tar -xf - -C "$DEST"
    echo "[OK] $NAME 추출 완료"
    echo ""
}

# 1. 라벨 먼저 (작아서 빠름)
extract_tar "$ROOT/1.Training/라벨링데이터" "TL174" "$OUT/labels_train"
extract_tar "$ROOT/2.Validation/라벨링데이터" "VL11" "$OUT/labels_val"

# 2. 영상 (큼)
extract_tar "$ROOT/1.Training/원천데이터" "TS174" "$OUT/video_train"
extract_tar "$ROOT/2.Validation/원천데이터" "VS11" "$OUT/video_val"

echo ""
echo "=== 모두 완료 — 추출된 데이터 ==="
du -sh "$OUT"/*
echo ""
echo "=== 폴더 구조 (3 depth) ==="
find "$OUT" -maxdepth 3 -type d | head -30
