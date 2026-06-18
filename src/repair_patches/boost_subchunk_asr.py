"""짧은 외침 detect 위한 boost + sub-chunk ASR 후처리.

Why:
  메인 ASR (풀 86초 vocals) 은 짧고 약한 외침 ("Adam!", "out") 을 cut.
  Raw chunk audio (BS-RoFormer 분리 전, 더 풍부) + 3x volume boost + 4-5s sub-chunk
  → "Adam! Adam!" 같은 외침 detect 가능.

처리:
  1. chunk mp4 → raw 16k wav 추출
  2. ffmpeg 3x volume boost
  3. 0-30s 영역만 sub-chunking (overlap 1s) — 외침 대부분 초반에 집중
  4. 각 sub-chunk ASR 호출 → words (global time 보정)
  5. 기존 words.json 과 중복 제거 (시간 거리 < 0.3s)
  6. 새 boosted_words 추가해서 words_v2.json 저장

화자 분리 embedding 영향 X — raw vocals 그대로 사용 (이 script 는 ASR 만 boost).

입력: run_dir
출력: meta/<chunk>_words_v2.json
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import requests

import os
ASR_URL = "http://127.0.0.1:8902/transcribe"
# 전체 구간 boost (env 로 조절). 짧은 발화는 영상 어디서나 나오므로 0~끝 전체.
# BOOST_END=0 (기본) 이면 영상 전체 길이로 자동 설정.
BOOST_START = float(os.environ.get("BOOST_START", "0.0"))
BOOST_END = float(os.environ.get("BOOST_END", "0.0"))   # 0 → 전체
BOOST_VOL = float(os.environ.get("BOOST_VOL", "3.0"))
WIN = 4.0                  # sub-chunk 길이
HOP = 3.0                  # 다음 sub-chunk 시작점 (overlap = WIN - HOP)
LANG = "English"


def main(run_dir: str):
    rd = Path(run_dir)
    meta = rd / "meta"
    chunks_dir = rd / "chunks"
    seg_files = sorted(meta.glob("*_chunk_*_segments.json"))
    if not seg_files:
        raise SystemExit(f"no segments.json in {meta}")

    for seg_path in seg_files:
        chunk_name = seg_path.stem.replace("_segments", "")
        chunk_mp4 = chunks_dir / f"{chunk_name}.mp4"
        words_path = meta / f"{chunk_name}_words.json"
        if not chunk_mp4.exists():
            print(f"[skip] {chunk_name}: no chunk mp4")
            continue

        # 기존 words 로드
        existing = []
        if words_path.exists():
            existing = json.load(open(words_path)).get("words", [])
        print(f"\n[{chunk_name}] 기존 words: {len(existing)}")

        # raw audio + boost
        raw_wav = Path("/tmp") / f"{chunk_name}_raw.wav"
        boost_wav = Path("/tmp") / f"{chunk_name}_boost3.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(chunk_mp4),
                        "-ar", "16000", "-ac", "1", str(raw_wav)],
                       capture_output=True, check=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(raw_wav),
                        "-af", "volume=3.0", str(boost_wav)],
                       capture_output=True, check=True)

        # sub-chunk ASR
        new_words = []
        t = BOOST_AREA[0]
        while t < BOOST_AREA[1]:
            ln = min(WIN, BOOST_AREA[1] - t)
            sub_wav = Path("/tmp") / f"sub_{chunk_name}_{int(t*100)}.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(boost_wav),
                            "-ss", str(t), "-t", str(ln), str(sub_wav)],
                           capture_output=True, check=True)
            try:
                r = requests.post(ASR_URL,
                                  json={"audio_path": str(sub_wav), "language": LANG},
                                  timeout=60).json()
            except Exception as ex:
                print(f"  ASR fail at {t:.1f}s: {ex}")
                t += HOP
                continue
            ws = r.get("words", []) or []
            for w in ws:
                gw_start = float(w["start"]) + t
                gw_end = float(w["end"]) + t
                new_words.append({"word": w["word"], "start": gw_start, "end": gw_end})
            t += HOP

        # 중복 제거 (기존 words 와 시간 거리 < 0.3s 면 skip)
        fresh = []
        for w in new_words:
            if any(abs(w["start"] - ew["start"]) < 0.3
                   and w["word"].lower().rstrip('.!?,') == ew["word"].lower().rstrip('.!?,')
                   for ew in existing):
                continue
            # sub-chunk 끼리 중복도 제거
            if any(abs(w["start"] - f["start"]) < 0.2
                   and w["word"].lower().rstrip('.!?,') == f["word"].lower().rstrip('.!?,')
                   for f in fresh):
                continue
            fresh.append(w)
        print(f"  sub-chunk ASR: {len(new_words)} → {len(fresh)} fresh (중복 제거 후)")
        for w in fresh:
            print(f"    [{w['start']:5.2f}-{w['end']:5.2f}] {w['word']!r}")

        # 통합 (시간 순 정렬)
        combined = existing + fresh
        combined.sort(key=lambda x: float(x["start"]))
        out_path = meta / f"{chunk_name}_words_v2.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_name": chunk_name, "detected_lang": LANG,
                       "words": combined, "boost_added": len(fresh)},
                      f, ensure_ascii=False, indent=2)
        print(f"  [✓] {len(combined)} words (+{len(fresh)} boost) → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    main(args.run_dir)
