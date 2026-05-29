"""v190 visual ASD reassign — segment 안에서 speaking face track 검출 후 sub-region split.

알고리즘:
1. ASD cache 자동 매칭 (vocals duration ≈ n_frames/fps)
2. 각 face track 의 speaking 구간 (score >= ASD_TH) 추출
3. 각 segment 에 대해 speaking 구간이 ≥ MIN_SPEAK_DUR 인 sub-region 검사
4. sub-region 의 ERes2 embedding → 가장 가까운 메인 SPK centroid 검색
5. best_sim > current_sim + MARGIN AND best_sim ≥ MIN_SIM → split & reassign

[[baseline-v190-6-speakers]] 의 visual ASD trigger 일반화.
입력: run_dir (v190/v194/원본 segments + vocals)
출력: meta/<chunk>_segments_v190b.json
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "daemons"))
from eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model

CACHE_DIR = Path("/workspace/media/cache/lightasd")
ASD_TH = 0.0             # LightASD speaking score threshold (raw scores 보통 -1~1, 0 이상=speak)
MIN_SPEAK_DUR = 0.5      # face track 의 연속 speaking 구간 최소 길이 (sec)
MERGE_GAP = 0.2          # speaking frame gap 이 이 이상이면 별도 구간
EMB_PAD = 0.1
SIM_MIN = 0.40           # best_sim 이 이 이상이어야 reassign
SIM_MARGIN = 0.10        # best_sim > current_sim + margin


def match_asd_cache(vocals_path: Path) -> Path | None:
    audio, sr = sf.read(vocals_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    dur = len(audio) / sr
    best = None
    for p in CACHE_DIR.glob("*.pkl"):
        try:
            d = pickle.load(open(p, "rb"))
            n_f = d.get("n_frames", 0)
            fps = d.get("fps", 25.0)
            diff = abs(n_f / fps - dur)
            if best is None or diff < best[1]:
                best = (p, diff)
        except Exception:
            continue
    if best is None or best[1] > 3.0:  # 3초 이상 차이 → 매칭 실패
        return None
    return best[0]


def speaking_runs(track: dict, fps: float, th: float = ASD_TH, merge_gap: float = MERGE_GAP):
    """track 의 speaking frame 들을 연속 구간으로 묶기. Returns list of (t0, t1) sec."""
    frames = list(track.get("frames", []))
    scores = list(track.get("scores", []))
    if not frames or not scores:
        return []
    # frame_idx → speak (bool) 정렬
    spk = [(int(f), float(s) >= th) for f, s in zip(frames, scores)]
    spk.sort(key=lambda x: x[0])
    runs = []
    cur_start = None
    cur_end = None
    for f, on in spk:
        if not on:
            if cur_start is not None:
                runs.append((cur_start / fps, (cur_end + 1) / fps))
                cur_start = None
            continue
        if cur_start is None:
            cur_start = f
            cur_end = f
        elif f - cur_end <= merge_gap * fps:
            cur_end = f
        else:
            runs.append((cur_start / fps, (cur_end + 1) / fps))
            cur_start = f
            cur_end = f
    if cur_start is not None:
        runs.append((cur_start / fps, (cur_end + 1) / fps))
    return [(s, e) for s, e in runs if e - s >= MIN_SPEAK_DUR]


def emb_at(audio, sr, t0, t1, pad=EMB_PAD):
    s = max(0, t0 - pad)
    e = min(len(audio) / sr, t1 + pad)
    c = audio[int(s*sr):int(e*sr)]
    if len(c) < int(0.08 * sr):
        return None
    return extract_eres2netv2_emb(c, sr=sr)


def main(run_dir: str):
    rd = Path(run_dir)
    meta = rd / "meta"
    # 우선순위: v190 > v194 > 원본
    v190 = sorted(meta.glob("*_chunk_*_segments_v190.json"))
    v194 = sorted(meta.glob("*_chunk_*_segments_v194.json"))
    base = v190 if v190 else (v194 if v194 else sorted(meta.glob("*_chunk_*_segments.json")))
    base = [p for p in base if not any(p.name.endswith(x) for x in
            ["_fixed.json", "_gapfilled.json", "_v190b.json"])]
    if not base:
        raise SystemExit(f"no segments in {meta}")
    get_eres2netv2_model()

    for seg_path in base:
        chunk_name = seg_path.stem.replace("_segments_v190", "") \
                                  .replace("_segments_v194", "") \
                                  .replace("_segments", "")
        vocals = rd / "vocals" / f"{chunk_name}_clean_vocals.wav"
        if not vocals.exists():
            print(f"[skip] {chunk_name}: no vocals")
            continue
        cache_pkl = match_asd_cache(vocals)
        if cache_pkl is None:
            print(f"[skip] {chunk_name}: no matching ASD cache")
            continue
        print(f"\n[{chunk_name}] ASD cache: {cache_pkl.name}")
        asd = pickle.load(open(cache_pkl, "rb"))
        fps = asd.get("fps", 25.0)
        tracks = asd.get("tracks", [])
        print(f"  {len(tracks)} face tracks loaded")

        seg_data = json.load(open(seg_path))
        groups = seg_data["groups"]
        audio, sr = sf.read(vocals)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        total_dur = len(audio) / sr

        # SPK centroids
        spk_embs = defaultdict(list)
        for s in groups:
            if s["group_end"] - s["group_start"] < 0.5:
                continue
            e = emb_at(audio, sr, s["group_start"], s["group_end"])
            if e is not None:
                spk_embs[s["speaker"]].append(e)
        spk_cent = {}
        for spk, es in spk_embs.items():
            c = np.mean(np.stack(es), axis=0)
            c = c / max(np.linalg.norm(c), 1e-9)
            spk_cent[spk] = c
        print(f"  SPK centroids: {sorted(spk_cent)}")

        # 모든 face track speaking 구간 통합
        all_speak_runs = []
        for ti, track in enumerate(tracks):
            for (s, e) in speaking_runs(track, fps):
                all_speak_runs.append((s, e, ti))
        all_speak_runs.sort()
        print(f"  total speaking runs (≥{MIN_SPEAK_DUR}s): {len(all_speak_runs)}")

        # 각 segment 에 대해 speaking runs overlap 검사 → sub-region split
        new_segs = []
        n_split = 0
        for seg in groups:
            ss, se = seg["group_start"], seg["group_end"]
            spk = seg["speaker"]
            cur_cent = spk_cent.get(spk)
            if cur_cent is None:
                new_segs.append(dict(seg))
                continue
            # segment 안에 들어가는 speaking runs
            inside = []
            for (rs, re, ti) in all_speak_runs:
                ov_s = max(ss, rs)
                ov_e = min(se, re)
                if ov_e - ov_s >= 0.4:
                    inside.append((ov_s, ov_e, ti))
            best_alt = None  # (sub_start, sub_end, alt_spk, alt_sim, cur_sim)
            for (sub_s, sub_e, ti) in inside:
                e = emb_at(audio, sr, sub_s, sub_e)
                if e is None:
                    continue
                cur_sim = float(np.dot(e, cur_cent))
                sims = {sp: float(np.dot(e, c)) for sp, c in spk_cent.items()}
                bspk, bsim = max(sims.items(), key=lambda x: x[1])
                if bspk != spk and bsim >= SIM_MIN and bsim > cur_sim + SIM_MARGIN:
                    if best_alt is None or bsim - cur_sim > best_alt[3] - best_alt[4]:
                        best_alt = (sub_s, sub_e, bspk, bsim, cur_sim)
            if best_alt is None:
                new_segs.append(dict(seg))
                continue
            sub_s, sub_e, alt_spk, alt_sim, cur_sim = best_alt
            print(f"  SPLIT [{ss:.2f}-{se:.2f}] {spk}(cur={cur_sim:.2f}) "
                  f"→ ASD-region [{sub_s:.2f}-{sub_e:.2f}] {alt_spk}({alt_sim:.2f})")
            # 3 sub-segments: pre / asd-region / post (단 비어있으면 skip)
            text = seg.get("text", "")
            words = text.split()
            total_len = se - ss
            pre_len = sub_s - ss
            asd_len = sub_e - sub_s
            post_len = se - sub_e
            def _wcut(start_frac, end_frac):
                if not words:
                    return ""
                i0 = max(0, int(round(len(words) * start_frac)))
                i1 = max(i0, int(round(len(words) * end_frac)))
                return " ".join(words[i0:i1])
            if pre_len >= 0.2:
                new_segs.append({"group_start": ss, "group_end": sub_s, "speaker": spk,
                                 "text": _wcut(0, pre_len/total_len)})
            new_segs.append({"group_start": sub_s, "group_end": sub_e, "speaker": alt_spk,
                             "text": _wcut(pre_len/total_len, (pre_len+asd_len)/total_len),
                             "from_v190b_asd": True})
            if post_len >= 0.2:
                new_segs.append({"group_start": sub_e, "group_end": se, "speaker": spk,
                                 "text": _wcut((pre_len+asd_len)/total_len, 1.0)})
            n_split += 1

        # === gap-aware: segment 사이 gap 안의 speaking run 도 추가 ===
        new_segs.sort(key=lambda x: x["group_start"])
        gaps = []
        if new_segs and new_segs[0]["group_start"] > 0.3:
            gaps.append((0.0, new_segs[0]["group_start"]))
        for i in range(len(new_segs)-1):
            g0, g1 = new_segs[i]["group_end"], new_segs[i+1]["group_start"]
            if g1 - g0 >= 0.5:
                gaps.append((g0, g1))
        if new_segs and total_dur - new_segs[-1]["group_end"] >= 0.5:
            gaps.append((new_segs[-1]["group_end"], total_dur))

        n_gap_added = 0
        for (gs_, ge_) in gaps:
            # 이 gap 안의 speaking runs
            for (rs, re, ti) in all_speak_runs:
                ov_s = max(gs_, rs)
                ov_e = min(ge_, re)
                if ov_e - ov_s < MIN_SPEAK_DUR:
                    continue
                e = emb_at(audio, sr, ov_s, ov_e)
                if e is None:
                    continue
                sims = {sp: float(np.dot(e, c)) for sp, c in spk_cent.items()}
                bspk, bsim = max(sims.items(), key=lambda x: x[1])
                if bsim < SIM_MIN:
                    continue
                print(f"  GAP-ADD [{ov_s:.2f}-{ov_e:.2f}] (track {ti}) → {bspk}(sim={bsim:.2f})")
                new_segs.append({"group_start": float(ov_s), "group_end": float(ov_e),
                                 "speaker": bspk, "text": "",
                                 "from_v190b_asd_gap": True})
                n_gap_added += 1
        if n_gap_added:
            print(f"  gap-add total: {n_gap_added}")

        # consecutive same-SPK merge
        new_segs.sort(key=lambda x: x["group_start"])
        merged = []
        for s in new_segs:
            if merged and merged[-1]["speaker"] == s["speaker"] \
               and s["group_start"] - merged[-1]["group_end"] <= 0.1:
                merged[-1]["group_end"] = s["group_end"]
                merged[-1]["text"] = (merged[-1].get("text","") + " " + s.get("text","")).strip()
            else:
                merged.append(dict(s))

        cnt = Counter(s["speaker"] for s in merged)
        print(f"\n  [✓] {len(groups)} → {len(merged)} segments (visual ASD splits={n_split})")
        print(f"      SPKs: {dict(cnt)}")
        seg_data["groups"] = merged
        out = meta / f"{chunk_name}_segments_v190b.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, ensure_ascii=False, indent=2)
        print(f"      saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    main(args.run_dir)
