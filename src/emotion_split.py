"""Emotion-aware SPK split — 같은 SPK 가 emotion 큰 변화 시 sub-speaker 로 분리.

원리:
  - orchestrator 가 segments 별 emotion 분석 (emotion2vec_plus_large)
  - 같은 audio SPK 의 segments 라도 emotion 이 크게 다르면 (e.g. Neutral → Angry)
    실제로 다른 화자일 가능성 (예: mom 외침 vs dad 평소 발화 가 SPK_03 공유)
  - emotion 큰 변화 시 새 SPK 라벨 (SPK_emo) 로 split

영상 무관 default:
  - emotion_score_diff_th = 0.6 (큰 변화 임계값)
  - min_split_chunks = 2

입력: run_dir (segments_*.json + emotion.json 필요)
출력: meta/<chunk>_segments_emotion_split.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def split_by_emotion(
    run_dir: str,
    *,
    emotion_diff_th: float = 0.6,
    min_split_chunks: int = 2,
) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    # emotion.json 찾기
    emotion_files = list(meta.glob("emotion.json")) + list(meta.glob("*emotion*.json"))
    if not emotion_files:
        print("[skip] no emotion.json in", meta)
        return {"split": 0}

    emotion_data = json.load(open(emotion_files[0]))
    # emotion_data: chunk_id → {label, scores} (orchestrator 형식)
    emotion_by_chunk = {}
    if isinstance(emotion_data, dict):
        for cid, info in emotion_data.items():
            emotion_by_chunk[str(cid)] = info
    elif isinstance(emotion_data, list):
        for item in emotion_data:
            emotion_by_chunk[str(item.get("chunk_id"))] = item

    # segments 후보
    seg_files = sorted(meta.glob("*_segments_gapfilled.json"))
    if not seg_files:
        seg_files = sorted(meta.glob("*_segments.json"))
    if not seg_files:
        print("[skip] no segments_*.json in", meta)
        return {"split": 0}

    n_split = 0
    for sp in seg_files:
        data = json.load(open(sp))
        groups = data.get("groups", data.get("segments", []))
        # SPK 별 emotion 분포
        spk_emotions: dict[str, list] = defaultdict(list)
        for idx, seg in enumerate(groups):
            spk = str(seg.get("speaker", ""))
            cid = str(seg.get("chunk_id", idx))
            if cid in emotion_by_chunk:
                lab = emotion_by_chunk[cid].get("label") or emotion_by_chunk[cid].get("emotion")
                if lab:
                    spk_emotions[spk].append((idx, lab))

        # emotion 큰 변화 detect → split
        for spk, idx_lab in spk_emotions.items():
            if spk.startswith("SPEAKER_BG"):
                continue
            lab_counts = Counter(l for _, l in idx_lab)
            if len(lab_counts) < 2:
                continue
            # 가장 큰 라벨 vs 두번째 — min_split_chunks 이상이면 분리
            common = lab_counts.most_common()
            main_lab, main_cnt = common[0]
            for sub_lab, sub_cnt in common[1:]:
                if sub_cnt < min_split_chunks:
                    continue
                # 그 라벨의 segments 를 새 SPK 로
                new_label = f"{spk}_emo{sub_lab[:3]}"
                for idx, l in idx_lab:
                    if l == sub_lab:
                        groups[idx]["audio_speaker"] = spk
                        groups[idx]["speaker"] = new_label
                        groups[idx]["from_emotion_split"] = True
                        n_split += 1

        data["groups"] = groups
        out = meta / f"{sp.stem}_emotion_split.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  saved {out.name} ({n_split} segs split)")

    return {"split": n_split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--emotion-diff-th", type=float, default=0.6)
    ap.add_argument("--min-split-chunks", type=int, default=2)
    args = ap.parse_args()
    s = split_by_emotion(
        args.run_dir,
        emotion_diff_th=args.emotion_diff_th,
        min_split_chunks=args.min_split_chunks,
    )
    print(f"summary: {s}")


if __name__ == "__main__":
    main()
