#!/bin/bash
# Update PROGRESS.md (Korean) and PROGRESS_EN.md (English) and push to GitHub.
#
# Usage:
#   ./update_progress.sh "## YYYY-MM-DD — Title (Korean)" "## YYYY-MM-DD — Title (English)"
#
# Or interactive (single language):
#   ./update_progress.sh
#   (it will prompt which language)
#
# Or specify file with --en or --ko:
#   ./update_progress.sh --ko "## ..."
#   ./update_progress.sh --en "## ..."
#   ./update_progress.sh --both "## ... Korean ..." "## ... English ..."

set -e
cd "$(dirname "$0")"

KO_ENTRY=""
EN_ENTRY=""

usage() {
    cat <<EOF
Usage:
  $0 --ko "<korean entry>"
  $0 --en "<english entry>"
  $0 --both "<korean entry>" "<english entry>"
  $0 "<korean entry>" "<english entry>"   # both, positional

Entry format starts with "## YYYY-MM-DD — Title".
EOF
    exit 1
}

# Parse arguments
case "$1" in
    --ko)
        KO_ENTRY="$2"
        ;;
    --en)
        EN_ENTRY="$2"
        ;;
    --both)
        KO_ENTRY="$2"
        EN_ENTRY="$3"
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        # Positional: KO then EN
        KO_ENTRY="$1"
        EN_ENTRY="$2"
        ;;
esac

if [ -z "$KO_ENTRY" ] && [ -z "$EN_ENTRY" ]; then
    echo "ERROR: empty entries"
    usage
fi

insert_top() {
    local file="$1"
    local new_entry="$2"
    [ -z "$new_entry" ] && return 0
    [ ! -f "$file" ] && {
        echo "ERROR: $file not found"
        return 1
    }
    # Preserve header (first 3 lines: # title, blank, > quote)
    local header=$(head -3 "$file")
    local body=$(tail -n +4 "$file")
    {
        echo "$header"
        echo
        echo "$new_entry"
        echo
        echo "---"
        echo
        echo "$body"
    } > "$file.tmp"
    mv "$file.tmp" "$file"
    echo "[ok] updated $file"
}

# Update files
insert_top "PROGRESS.md"     "$KO_ENTRY"
insert_top "PROGRESS_EN.md"  "$EN_ENTRY"

# Title for commit
COMMIT_TITLE=""
if [ -n "$KO_ENTRY" ]; then
    COMMIT_TITLE=$(echo "$KO_ENTRY" | head -1 | sed 's/^## *//')
fi
if [ -n "$EN_ENTRY" ] && [ -z "$COMMIT_TITLE" ]; then
    COMMIT_TITLE=$(echo "$EN_ENTRY" | head -1 | sed 's/^## *//')
fi

# Stage changed files
git add PROGRESS.md PROGRESS_EN.md 2>/dev/null || true

# Check if anything to commit
if git diff --cached --quiet; then
    echo "[skip] no changes to commit"
    exit 0
fi

git commit -m "PROGRESS: $COMMIT_TITLE

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"

git push origin main 2>&1 | tail -3

echo "[ok] PROGRESS updated and pushed to GitHub"
