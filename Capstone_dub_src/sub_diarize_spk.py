"""Sub-diarize within a single SPK — over-cluster 된 SPK 자동 분리.

Why:
  1차 diarize 가 acoustic 유사 화자들 (예: 외침 동시 발생 + 같은 음역대 발화)을
  한 SPK 로 cluster. 그 SPK 의 segments 만 모아서 NeMo titanet 재 diarize 시
  내부 sub-cluster 검출 → 실제 다른 화자 분리 가능.

영상 무관, hardcoding 없음:
  - 자동 후보 선택: segment 수 ≥ 5 + total duration ≥ 10s 인 SPK
  - 각 후보 SPK 의 segments concat (0.3s silence padding 으로 분리 보존)
  - fusion daemon /diarize 호출 → sub-cluster 검출
  - n_speakers ≥ 2 → SPK 분리 (suffix _a, _b, ...)

알고리즘:
  1. raw segments 의 SPK 별 segment 수 + duration 집계
  2. 후보 SPK 선택 (auto threshold)
  3. SPK 별 audio concat
  4. /diarize sub-cluster
  5. concat time → original time mapping
  6. sub-SPK 적용 (예: SPEAKER_02 → SPEAKER_02_a / SPEAKER_02_b)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


FUSION_URL = "http://127.0.0.1:8918/diarize"
MIN_SEG_COUNT = 4
MIN_TOTAL_DUR = 8.0
SILENCE_PAD = 0.3  # 분리 보존용 무음


def _build_concat(audio: np.ndarray, sr: int, seg_times: list) -> tuple[np.ndarray, list]:
    """SPK segments concat — (concat_audio, mapping). mapping: [(concat_start, concat_end, orig_idx)]."""
    silence = np.zeros(int(SILENCE_PAD * sr))
    out_parts = []
    mapping = []
    cur_t = 0.0
    for orig_idx, (ss, ee) in seg_times:
        i0 = int(max(0, ss * sr))
        i1 = int(min(len(audio), ee * sr))
        if i1 <= i0:
            continue
        part = audio[i0:i1]
        dur = (i1 - i0) / sr
        mapping.append((cur_t, cur_t + dur, orig_idx))
        out_parts.append(part)
        out_parts.append(silence)
        cur_t += dur + SILENCE_PAD
    if not out_parts:
        return np.array([]), []
    return np.concatenate(out_parts), mapping


def _concat_time_to_orig(seg_start: float, seg_end: float, mapping: list) -> int | None:
    """concat 시간 → original segment idx (가장 큰 overlap)."""
    best_idx = None
    best_ov = 0.0
    for cs, ce, orig_idx in mapping:
        ov = max(0.0, min(seg_end, ce) - max(seg_start, cs))
        if ov > best_ov:
            best_ov = ov
            best_idx = orig_idx
    return best_idx


def sub_diarize(run_dir: str, segments_name: str, out_name: str) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    vocals_dir = rd / "vocals"
    tmp_dir = rd / "temp"
    tmp_dir.mkdir(exist_ok=True)

    seg_path = meta / segments_name
    data = json.load(open(seg_path, encoding="utf-8"))
    groups = data.get("groups", data.get("segments", []))

    chunk_name = segments_name.split("_segments")[0]
    vocals_path = vocals_dir / f"{chunk_name}_clean_vocals.wav"
    if not vocals_path.exists():
        raise SystemExit(f"no vocals: {vocals_path}")
    audio, sr = sf.read(str(vocals_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 1. SPK 별 집계
    spk_segs = {}
    for i, seg in enumerate(groups):
        sp = str(seg.get("speaker", ""))
        if sp.startswith("SPEAKER_BG") or sp.startswith("SPEAKER_9"):
            continue
        ss = float(seg.get("group_start", seg.get("start", 0)))
        ee = float(seg.get("group_end", seg.get("end", 0)))
        spk_segs.setdefault(sp, []).append((i, (ss, ee)))

    n_splits = 0
    splits_info = {}
    for sp, idx_times in spk_segs.items():
        total_dur = sum(ee - ss for _, (ss, ee) in idx_times)
        if len(idx_times) < MIN_SEG_COUNT or total_dur < MIN_TOTAL_DUR:
            continue
        print(f"\n[{sp}] {len(idx_times)} segs, total={total_dur:.1f}s — sub-diarize:")

        concat_audio, mapping = _build_concat(audio, sr, idx_times)
        if len(concat_audio) == 0:
            continue
        # 임시 wav 저장
        concat_wav = tmp_dir / f"sub_{sp}.wav"
        sf.write(str(concat_wav), concat_audio, sr)

        # fusion daemon 호출
        try:
            # 컨테이너 내부 경로 변환
            container_path = str(concat_wav).replace("E:\\TTS_capstone", "/workspace").replace("\\", "/")
            r = requests.post(FUSION_URL, json={
                "vocals_wav": container_path,
                "num_speakers": None,
                "min_duration": 0.2,
            }, timeout=180)
            r.raise_for_status()
            sub_data = r.json()
        except Exception as e:
            print(f"  [err] {e}")
            continue

        sub_segs = sub_data.get("segments", [])
        sub_n_spk = sub_data.get("n_speakers", 1)
        if sub_n_spk < 2:
            print(f"  sub n_speakers={sub_n_spk} — no split")
            continue
        print(f"  sub n_speakers={sub_n_spk}, {len(sub_segs)} sub-segs")

        # sub-SPK → suffix 매핑
        sub_spk_names = sorted({s.get("speaker") for s in sub_segs})
        suffix_map = {ssp: f"{sp}_{chr(ord('a') + i)}" for i, ssp in enumerate(sub_spk_names)}
        print(f"  sub SPK rename: {suffix_map}")

        # 각 original segment 의 가장 큰 overlap sub-SPK 채택
        reassigned = 0
        for sub_seg in sub_segs:
            ss = float(sub_seg.get("start", 0))
            ee = float(sub_seg.get("end", 0))
            orig_idx = _concat_time_to_orig(ss, ee, mapping)
            if orig_idx is None:
                continue
            new_sp = suffix_map[sub_seg.get("speaker")]
            # 이미 다른 sub-SPK 받았으면 longest overlap 우선
            cur_sp = groups[orig_idx].get("speaker")
            if cur_sp == sp or cur_sp.startswith(f"{sp}_"):
                # 동일 SPK 의 sub split 만 적용
                if groups[orig_idx]["speaker"] != new_sp:
                    groups[orig_idx]["audio_speaker"] = sp
                    groups[orig_idx]["speaker"] = new_sp
                    groups[orig_idx]["from_sub_diarize"] = True
                    reassigned += 1
        print(f"  {reassigned} original segs reassigned")
        n_splits += reassigned
        splits_info[sp] = {"n_sub": sub_n_spk, "reassigned": reassigned, "suffix_map": suffix_map}

    data["groups"] = groups
    out_path = meta / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nsaved → {out_path} ({n_splits} segs split)")
    return {"n_splits": n_splits, "splits_info": splits_info, "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segments-name", required=True)
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()
    r = sub_diarize(args.run_dir, args.segments_name, args.out_name)
    print(f"summary: {r}")


if __name__ == "__main__":
    main()
