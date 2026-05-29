"""Post-process: fix segments.json text using words.json (word-level reassign).

Why: 메인 orchestrator는 화자 segment에 ASR utterance를 통째 복사 (word-level
filter 누락 bug). 이 script가 segment timing 기준으로 정확한 word만 모아
text를 재구성.

Usage:
  python postprocess_reassign_text.py <run_dir>

Input (run_dir/meta/):
  - <chunk>_segments.json (groups with wrong "text" field)
  - <chunk>_words.json    (Qwen3-ASR word-level timestamps)

Output:
  - <chunk>_segments_fixed.json  (corrected "text" per group)
"""
import argparse
import json
from pathlib import Path


def reassign_text(seg_start: float, seg_end: float, words: list, margin: float = 0.05) -> str:
    """Filter words whose center falls inside [seg_start - margin, seg_end + margin]."""
    out = []
    for w in words:
        center = (w["start"] + w["end"]) / 2
        if seg_start - margin <= center <= seg_end + margin:
            out.append(w["word"])
    return " ".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Pipeline run directory (contains meta/)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="Word boundary tolerance in seconds (default 0.05)")
    args = ap.parse_args()

    meta = Path(args.run_dir) / "meta"
    if not meta.is_dir():
        raise SystemExit(f"meta/ not found: {meta}")

    seg_files = sorted(meta.glob("*_chunk_*_segments.json"))
    if not seg_files:
        raise SystemExit(f"no *_segments.json in {meta}")

    for seg_path in seg_files:
        chunk_name = seg_path.stem.replace("_segments", "")
        words_path = meta / f"{chunk_name}_words.json"
        if not words_path.exists():
            print(f"[skip] {chunk_name}: no words.json")
            continue

        with open(seg_path) as f:
            seg_data = json.load(f)
        with open(words_path) as f:
            words_data = json.load(f)
        words = words_data.get("words", [])

        n_fixed = 0
        n_empty = 0
        for g in seg_data.get("groups", []):
            old = g.get("text", "")
            new = reassign_text(g["group_start"], g["group_end"], words, args.margin)
            if new and new != old:
                g["text"] = new
                n_fixed += 1
            elif not new:
                n_empty += 1
                # Keep old text as fallback (don't blank out)
        out_path = seg_path.parent / f"{chunk_name}_segments_fixed.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, ensure_ascii=False, indent=2)
        print(f"[✓] {chunk_name}: {len(seg_data['groups'])} groups, "
              f"fixed={n_fixed}, empty_fallback={n_empty} → {out_path}")


if __name__ == "__main__":
    main()
