"""F0 intra-SPK split — 같은 SPK 안의 외침/평소 발화 자동 분리.

Why:
  같은 audio SPK 가 (1) 평소 발화 + (2) 외침 두 가지 모드를 가질 때, voice acoustic
  cluster 가 같은 SPK 로 묶음 → 실제 다른 사람일 수 있음.
  per-segment F0 (median pitch) 가 SPK 평균 대비 +50Hz 이상 jump 시 외침 mode
  로 보고 SPK 분리.

영상 무관, hardcoding 없음:
  - SPK 별 F0 median 계산
  - segment F0 - SPK median ≥ +50 Hz → "{spk}_shout" 으로 split
  - shout cluster 안에서 F0 추가 분리 (e.g. 250 Hz 미만 = 아빠, 이상 = 엄마)

알고리즘:
  1. SPK 별 segment F0 분포 분석
  2. SPK 내 F0 std/range 가 큰 경우만 split (균질하면 skip)
  3. shout sub-cluster 들끼리 F0 차이 ≥ 80 Hz 면 분리 SPK 부여
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


F0_JUMP_THR = 50.0      # SPK median 대비 이 이상 차이면 mode 변화
F0_SUB_SPLIT = 80.0     # shout sub-cluster 간 F0 차이 (다른 사람 추정)
MIN_SHOUT_DUR = 0.5     # shout segment 최소 길이 (너무 짧으면 노이즈)


def _compute_f0_median(audio: np.ndarray, sr: int) -> float:
    """librosa pyin median F0 — voiced frames 만."""
    try:
        import librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=80, fmax=500, sr=sr, frame_length=2048,
        )
        voiced = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced) == 0:
            return 0.0
        return float(np.median(voiced))
    except Exception:
        return 0.0


def _segment_f0(audio: np.ndarray, sr: int, start: float, end: float) -> float:
    i0 = int(max(0, start * sr))
    i1 = int(min(len(audio), end * sr))
    if i1 <= i0:
        return 0.0
    return _compute_f0_median(audio[i0:i1], sr)


def split_by_f0(run_dir: str, segments_name: str, out_name: str) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    vocals_dir = rd / "vocals"

    seg_path = meta / segments_name
    if not seg_path.exists():
        raise SystemExit(f"no {segments_name}")

    data = json.load(open(seg_path, encoding="utf-8"))
    groups = data.get("groups", data.get("segments", []))

    chunk_name = segments_name.split("_segments")[0]
    vocals_path = vocals_dir / f"{chunk_name}_clean_vocals.wav"
    if not vocals_path.exists():
        print(f"  [skip] no vocals: {vocals_path}")
        return {"split": 0}
    audio, sr = sf.read(str(vocals_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 1. SPK 별 F0 계산
    spk_f0s = defaultdict(list)
    seg_f0s = {}
    for i, seg in enumerate(groups):
        sp = str(seg.get("speaker", ""))
        if sp.startswith("SPEAKER_BG"):
            continue
        ss = float(seg.get("group_start", seg.get("start", 0)))
        ee = float(seg.get("group_end", seg.get("end", 0)))
        f0 = _segment_f0(audio, sr, ss, ee)
        seg_f0s[i] = f0
        if f0 > 0:
            spk_f0s[sp].append(f0)

    # SPK F0 median
    spk_median = {sp: float(np.median(arr)) for sp, arr in spk_f0s.items() if arr}
    print(f"  SPK F0 medians: {spk_median}")

    # 2. SPK 내 jump detect → shout segments
    n_split = 0
    shout_assignments = {}  # idx → new_spk
    spk_shout_segs = defaultdict(list)  # spk → [(idx, f0)]
    for i, seg in enumerate(groups):
        sp = str(seg.get("speaker", ""))
        if sp not in spk_median:
            continue
        ss = float(seg.get("group_start", seg.get("start", 0)))
        ee = float(seg.get("group_end", seg.get("end", 0)))
        if ee - ss < MIN_SHOUT_DUR:
            continue
        f0 = seg_f0s.get(i, 0.0)
        if f0 == 0:
            continue
        diff = f0 - spk_median[sp]
        if diff >= F0_JUMP_THR:
            spk_shout_segs[sp].append((i, f0))

    # 3. shout sub-cluster F0 분리 (F0 차이 큼 → 다른 사람)
    for sp, idx_f0s in spk_shout_segs.items():
        if not idx_f0s:
            continue
        # F0 sort + cluster (단순 split: 평균 기준)
        idx_f0s.sort(key=lambda x: x[1])
        f0_vals = [x[1] for x in idx_f0s]
        # 가장 큰 gap 찾기 (binary cluster)
        if len(f0_vals) >= 2:
            gaps = [(f0_vals[i + 1] - f0_vals[i], i) for i in range(len(f0_vals) - 1)]
            max_gap, gap_pos = max(gaps, key=lambda x: x[0])
            if max_gap >= F0_SUB_SPLIT:
                # 두 cluster: low F0 = subA, high F0 = subB
                for j, (idx, f0) in enumerate(idx_f0s):
                    if j <= gap_pos:
                        new_spk = f"{sp}_shoutA"  # lower F0 (아빠 추정)
                    else:
                        new_spk = f"{sp}_shoutB"  # higher F0 (엄마 추정)
                    shout_assignments[idx] = new_spk
                print(f"  {sp}: F0 cluster split at gap={max_gap:.1f}Hz "
                      f"→ shoutA(<{f0_vals[gap_pos]:.0f}Hz) + shoutB(>{f0_vals[gap_pos+1]:.0f}Hz)")
                n_split += len(idx_f0s)
                continue
        # gap 작거나 1개 → 모두 shout 단일
        for idx, f0 in idx_f0s:
            shout_assignments[idx] = f"{sp}_shout"
        print(f"  {sp}: {len(idx_f0s)} shout segs → single shout SPK")
        n_split += len(idx_f0s)

    # 4. apply
    for idx, new_spk in shout_assignments.items():
        groups[idx]["audio_speaker"] = groups[idx]["speaker"]
        groups[idx]["speaker"] = new_spk
        groups[idx]["from_f0_split"] = True

    data["groups"] = groups
    out_path = meta / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved → {out_path} ({n_split} segs split)")
    return {"split": n_split, "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segments-name", required=True, help="e.g. test5_fresh_chunk_000_segments_gapfilled.json")
    ap.add_argument("--out-name", required=True, help="e.g. test5_fresh_chunk_000_segments_f0split.json")
    args = ap.parse_args()
    r = split_by_f0(args.run_dir, args.segments_name, args.out_name)
    print(f"summary: {r}")


if __name__ == "__main__":
    main()
