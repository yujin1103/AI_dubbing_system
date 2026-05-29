"""Boost diarize — 짧은 외침 자동 검출 보강.

Why:
  메인 fusion diarize 는 짧고 약한 외침 ("Adam!", "Help!") 을 segmentation 단계에서
  drop. 0-30s 영역만 3x volume boost 후 재 diarize 하면 외침 segment 검출 ↑.

영상 무관, hardcoding 없음:
  - 외침은 영상 도입부에 집중 → 0-30s 영역만 boost (다른 구간은 메인 segmentation 이 잘 잡음)
  - 3x volume (saturation 미만, 가청 향상)
  - boost 결과는 raw segments 와 **merge** (override 아님) — 새 SPK / 새 시간대만 추가

처리:
  1. clean_vocals.wav 0-30s slice → 3x volume → clean_vocals_boost.wav
  2. fusion daemon /diarize 호출 (boost vocals)
  3. 원본 raw segments + boost segments 비교
     - boost 에서 새 시간대 (overlap < 0.5 with raw) → 추가
     - 같은 시간대 dual detect → raw 우선 (보존)
  4. merged segments → segments_boost_merged.json
  5. 후속 patches (focused_nemo / face_match / gap_fill) 가 이 파일 사용

자동 적용: orchestrator → fusion 직후 호출 (영상 무관).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import requests

FUSION_URL = "http://127.0.0.1:8918/diarize"
BOOST_END = 30.0      # 0-30s 영역만 boost
BOOST_GAIN = 3.0      # ffmpeg volume gain
OVERLAP_THR = 0.5     # raw seg 와 시간 겹침 비율 — 0.5+ → 중복


def _boost_vocals(vocals_path: Path, out_path: Path) -> bool:
    if not vocals_path.exists():
        print(f"  [skip] no vocals: {vocals_path}")
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(vocals_path),
        "-af", f"atrim=0:{BOOST_END},volume={BOOST_GAIN}",
        "-ar", "16000", "-ac", "1", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [err] ffmpeg: {e.stderr.decode()[:200]}")
        return False


def _diarize(vocals_path: str) -> list:
    try:
        r = requests.post(FUSION_URL, json={
            "vocals_wav": vocals_path,
            "num_speakers": None,
            "min_duration": 0.2,
        }, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("segments", [])
    except Exception as e:
        print(f"  [err] fusion: {e}")
        return []


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """a 가 b 와 겹치는 비율 (a 길이 대비)."""
    o = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    a_len = max(1e-6, a_end - a_start)
    return o / a_len


def merge_boost(raw_segs: list, boost_segs: list) -> list:
    """raw + boost 결합. boost 에서 raw 와 안 겹치는 (또는 0-30s 새 SPK) segment 만 추가."""
    merged = list(raw_segs)
    for bs in boost_segs:
        bs_start = float(bs.get("group_start", bs.get("start", 0)))
        bs_end = float(bs.get("group_end", bs.get("end", 0)))
        if bs_start >= BOOST_END:
            continue  # boost 영역 밖
        # raw seg 와 0.5+ 겹치면 skip (중복)
        max_overlap = 0.0
        for rs in raw_segs:
            rs_start = float(rs.get("group_start", rs.get("start", 0)))
            rs_end = float(rs.get("group_end", rs.get("end", 0)))
            ov = _overlap_ratio(bs_start, bs_end, rs_start, rs_end)
            if ov > max_overlap:
                max_overlap = ov
        if max_overlap >= OVERLAP_THR:
            continue
        # 새 segment — SPK 이름 충돌 방지 위해 BOOST_ prefix
        new_bs = dict(bs)
        sp = bs.get("speaker", "")
        new_bs["speaker"] = sp  # 그대로 (downstream face/voice 가 cluster 정리)
        new_bs["source"] = "boost"
        merged.append(new_bs)
    merged.sort(key=lambda s: float(s.get("group_start", s.get("start", 0))))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out-suffix", default="_segments_boost_merged.json")
    args = ap.parse_args()

    rd = Path(args.run_dir)
    meta = rd / "meta"
    vocals_dir = rd / "vocals"
    seg_files = sorted(meta.glob("*_chunk_*_segments.json"))
    if not seg_files:
        raise SystemExit(f"no segments.json in {meta}")

    for seg_path in seg_files:
        chunk_name = seg_path.stem.replace("_segments", "")
        vocals_path = vocals_dir / f"{chunk_name}_clean_vocals.wav"
        boost_vocals = vocals_dir / f"{chunk_name}_clean_vocals_boost.wav"

        print(f"[{chunk_name}] boost diarize:")
        if not _boost_vocals(vocals_path, boost_vocals):
            continue

        boost_segs = _diarize(str(boost_vocals))
        if not boost_segs:
            print(f"  [skip] no boost segments")
            continue

        raw_data = json.load(open(seg_path, encoding="utf-8"))
        raw_segs = raw_data.get("groups", raw_data.get("segments", []))

        # boost segments → groups 형식 변환
        boost_groups = []
        for i, s in enumerate(boost_segs):
            boost_groups.append({
                "group_idx": len(raw_segs) + i,
                "speaker": s.get("speaker", ""),
                "group_start": float(s.get("start", 0)),
                "group_end": float(s.get("end", 0)),
                "segment_ids": [],
                "text": "",
            })

        merged = merge_boost(raw_segs, boost_groups)
        new_added = len(merged) - len(raw_segs)
        print(f"  raw={len(raw_segs)} + boost_new={new_added} = {len(merged)} merged")

        out_path = meta / f"{chunk_name}{args.out_suffix}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"groups": merged}, f, ensure_ascii=False, indent=2)
        print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
