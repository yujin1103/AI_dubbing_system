"""UI metadata 생성 — 각 화자의 얼굴 thumbnail + segments + 감정 + 번역.

UI 가 표시할 데이터 구조:
{
  "speakers": {
    "SPEAKER_00": {
      "face_thumbnail": "face_thumbnails/cluster_001.jpg",
      "n_segments": 7,
      "total_duration_sec": 20.5,
      "alt_thumbnails": ["face_thumbnails/cluster_002.jpg", ...],
      "segments": [
        {
          "start": 0.66, "end": 6.90,
          "text_src": "...", "text_translated": "...",
          "emotion": "Neutral", "emotion_scores": {...},
          "wav": "dubbed/group_000.wav"
        }, ...
      ]
    }, ...
  },
  "timeline": [...]  # 시간 순 전체 segments
}

입력:
  - segments_*.json (현재 best, 화자 라벨 포함)
  - face_clusters.json (speaker_face_map + cluster_thumbnails)
  - asr_*.json (text_src)
  - translated_*.json (text_translated)
  - emotion_*.json (감정 label + scores)

출력: ui_metadata.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_metadata(
    run_dir: str,
    *,
    segments_file: str | None = None,
    out_json: str | None = None,
) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"

    # 입력 파일 자동 탐색
    if segments_file is None:
        # 우선순위: sandwich > time_merged > gapfilled > segments
        for pat in ["*_segments_sandwich.json", "*_segments_time_merged.json",
                    "*_segments_gapfilled.json", "*_segments.json"]:
            matches = sorted(meta.glob(pat))
            if matches:
                segments_file = str(matches[0])
                break
    if not segments_file:
        raise SystemExit(f"no segments_*.json in {meta}")
    print(f"using segments: {Path(segments_file).name}")

    # face_clusters.json
    face_data = {}
    fcs = list(meta.glob("face_clusters*.json"))
    if fcs:
        face_data = json.load(open(fcs[0], encoding="utf-8"))

    # ASR + translated + emotion (선택)
    asr_data = {}
    for pat in ["*words.json", "*asr.json"]:
        for f in meta.glob(pat):
            try:
                d = json.load(open(f, encoding="utf-8"))
                if isinstance(d, dict) and "words" in d:
                    asr_data["words"] = d["words"]
                elif isinstance(d, list):
                    asr_data["rows"] = d
            except Exception:
                continue

    # segments 로드
    segs = json.load(open(segments_file, encoding="utf-8"))
    seg_list = segs.get("groups", segs.get("segments", []))

    # 화자별 grouping
    speakers: dict[str, dict] = defaultdict(lambda: {
        "face_thumbnail": None,
        "n_segments": 0,
        "total_duration_sec": 0.0,
        "alt_thumbnails": [],
        "segments": [],
    })
    spk_face_map = face_data.get("speaker_face_map", {})
    cluster_thumbnails = face_data.get("cluster_thumbnails", {})

    for seg in seg_list:
        spk = str(seg.get("speaker", ""))
        if not spk:
            continue
        start = float(seg.get("group_start", seg.get("start", 0)))
        end = float(seg.get("group_end", seg.get("end", 0)))
        dur = end - start
        speakers[spk]["n_segments"] += 1
        speakers[spk]["total_duration_sec"] += dur
        # face thumbnail
        if speakers[spk]["face_thumbnail"] is None and spk in spk_face_map:
            speakers[spk]["face_thumbnail"] = spk_face_map[spk].get("face_thumbnail")
            speakers[spk]["dominant_confidence"] = spk_face_map[spk].get("dominant_confidence")
            alt_cids = spk_face_map[spk].get("alt_clusters", {})
            speakers[spk]["alt_thumbnails"] = [
                cluster_thumbnails.get(cid) for cid in alt_cids.keys()
                if cluster_thumbnails.get(cid)
            ]
        # segment 정보
        seg_info = {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(dur, 3),
            "text": seg.get("text", ""),
        }
        # emotion / translated (있으면)
        for key in ("emotion", "emotion_scores", "text_src", "text_translated",
                    "tts_instruct_text", "wav"):
            if key in seg:
                seg_info[key] = seg[key]
        speakers[spk]["segments"].append(seg_info)

    # 시간 순 timeline
    timeline = sorted([
        {"start": s["start"], "end": s["end"], "speaker": spk, "text": s.get("text", "")}
        for spk, info in speakers.items()
        for s in info["segments"]
    ], key=lambda x: x["start"])

    metadata = {
        "speakers": dict(speakers),
        "timeline": timeline,
        "n_speakers": len(speakers),
        "total_segments": len(seg_list),
        "source": {
            "segments_file": Path(segments_file).name,
            "face_clusters_file": str(fcs[0].name) if fcs else None,
        },
    }

    if out_json is None:
        out_json = str(meta / "ui_metadata.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"saved UI metadata → {out_json}")
    print(f"  speakers: {len(speakers)}, segments: {len(seg_list)}")
    for spk, info in speakers.items():
        thumb = info.get("face_thumbnail", "(none)")
        print(f"    {spk}: {info['n_segments']} segs, {info['total_duration_sec']:.1f}s, face={thumb}")
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segments", help="override segments file")
    ap.add_argument("--out", help="output ui_metadata.json path")
    args = ap.parse_args()
    build_metadata(args.run_dir, segments_file=args.segments, out_json=args.out)


if __name__ == "__main__":
    main()
