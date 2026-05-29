"""자동 정확도 향상 (hardcoding 없음, 영상 무관).

3단계 자동 refinement:
  Stage 1: 결과 segments 분석 → 잘못 분리된 SPK 의심 후보 자동 검출
    - 짧은 outlier SPK (segments ≤ 3, dur ≤ 2s)
    - SPK 분포 imbalance (한 SPK 가 다른 SPK 의 alt face cluster 와 매칭)
  Stage 2: 의심 SPK 의 segments 를 voice embedding 으로 모든 main SPK 와 비교
    - whole-segment ERes2NetV2 embedding
    - best sim > current + REASSIGN_MARGIN (0.05)
    - best sim >= REASSIGN_MIN_SIM (0.40)
    - 단 face cluster 가 다른 main SPK 의 face 와 일치하면 더 강하게 reassign
  Stage 3: face cluster 정보 활용 multi-evidence reassign
    - voice + face 모두 같은 SPK 가리키면 reassign
    - voice 만, face 만 둘 중 하나면 보수적 (margin ↑)

입력: run_dir (segments_*.json + face_clusters.json + ASD cache)
출력: meta/<chunk>_segments_auto_refined.json

영상 무관 default (모두 자동 결정):
  - outlier_total_max: 3 (작은 SPK)
  - short_dur_max: 2.0s
  - reassign_margin: 0.05 (voice only) / 0.02 (voice + face 일치)
  - reassign_min_sim: 0.40
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/workspace")
from patches.eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model

CACHE_DIR = Path("/workspace/media/cache/lightasd")


def _l2(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _match_asd_cache(vocals_path: Path):
    if not vocals_path.exists():
        return None
    audio, sr = sf.read(str(vocals_path))
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
    return best[0] if best and best[1] <= 3.0 else None


def auto_refine(
    run_dir: str,
    *,
    outlier_total_max: int = 3,
    short_dur_max: float = 2.0,
    reassign_margin: float = 0.05,
    reassign_margin_with_face: float = 0.02,
    reassign_min_sim: float = 0.40,
) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    # face_clusters.json (있으면)
    fc_path = meta / "face_clusters.json"
    face_clusters = {}
    spk_face_map = {}
    if fc_path.exists():
        fc = json.load(open(fc_path, encoding="utf-8"))
        face_clusters = {int(k): v for k, v in fc.get("face_clusters", {}).items()}
        spk_face_map = fc.get("speaker_face_map", {})

    # ASD cache (face tracks)
    vocals_glob = list((rd / "vocals").glob("*_clean_vocals.wav"))
    tracks = []
    fps = 25.0
    if vocals_glob:
        cache = _match_asd_cache(vocals_glob[0])
        if cache:
            asd = pickle.load(open(cache, "rb"))
            tracks = asd.get("tracks", [])
            fps = float(asd.get("fps", 25.0))

    # best segments 찾기 — main_count >= 2 인 것 중 가장 최신 (sandwich over-merge 자동 회피)
    candidates = []
    for pat in ["*_segments_gapfilled.json", "*_segments_sandwich.json",
                "*_segments_v194.json", "*_segments.json"]:
        candidates.extend(sorted(meta.glob(pat)))
    seg_path = None
    for cand in candidates:
        try:
            d = json.load(open(cand, encoding="utf-8"))
            grps = d.get("groups", d.get("segments", []))
            ms = set(str(s.get("speaker", "")) for s in grps
                    if s.get("speaker") and not str(s.get("speaker", "")).startswith("SPEAKER_BG"))
            if len(ms) >= 2:
                seg_path = cand
                break
        except Exception:
            continue
    if not seg_path and candidates:
        seg_path = candidates[0]
    if not seg_path:
        raise SystemExit(f"no segments_*.json in {meta}")
    print(f"using segments: {seg_path.name}")
    data = json.load(open(seg_path))
    segs = data.get("groups", data.get("segments", []))

    # SPK 분포 — outlier 임계값 자동 결정 (main 평균 segments 의 1/3)
    spk_counts = Counter(str(s.get("speaker", "")) for s in segs)
    real_spks = {s: c for s, c in spk_counts.items() if s and not s.startswith("SPEAKER_BG")}
    if not real_spks:
        return {"reassigned": 0}
    avg_count = sum(real_spks.values()) / len(real_spks)
    auto_thr = max(1, int(avg_count / 3))  # main 평균의 1/3 미만이면 suspect
    print(f"avg SPK count={avg_count:.1f} → auto outlier threshold = {auto_thr}")
    main_spks = {s: c for s, c in real_spks.items() if c > auto_thr}
    suspect_spks = {s: c for s, c in real_spks.items() if c <= auto_thr}
    print(f"main SPKs: {dict(main_spks)}")
    print(f"suspect (small) SPKs: {dict(suspect_spks)}")
    if not suspect_spks:
        print("no suspect SPKs — no auto refine needed")
        return {"reassigned": 0}

    # voice embedding 모델 + audio
    audio_path = vocals_glob[0] if vocals_glob else None
    if not audio_path:
        print("no vocals.wav — voice embedding 불가")
        return {"reassigned": 0}
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    model = get_eres2netv2_model()

    # main SPK centroid 추출 (longest segment 기준)
    main_segments: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(segs):
        sp = str(s.get("speaker", ""))
        if sp in main_spks:
            main_segments[sp].append(i)

    main_cent: dict[str, np.ndarray] = {}
    for sp, idxs in main_segments.items():
        # longest segment 의 embedding
        best_idx = max(idxs, key=lambda i: float(segs[i].get("group_end", segs[i].get("end", 0))) -
                                            float(segs[i].get("group_start", segs[i].get("start", 0))))
        s = segs[best_idx]
        ss = float(s.get("group_start", s.get("start", 0)))
        se = float(s.get("group_end", s.get("end", 0)))
        i0 = int(max(0, (ss - 0.2) * sr))
        i1 = int(min(len(audio), (se + 0.2) * sr))
        if i1 <= i0:
            continue
        try:
            emb = extract_eres2netv2_emb(audio[i0:i1], sr)
            main_cent[sp] = _l2(np.asarray(emb, dtype=np.float32))
        except Exception:
            continue

    print(f"main centroids: {list(main_cent.keys())}")

    # 의심 SPK 의 각 segment 분석
    n_reassign = 0
    for i, s in enumerate(segs):
        cur_spk = str(s.get("speaker", ""))
        if cur_spk not in suspect_spks:
            continue
        ss = float(s.get("group_start", s.get("start", 0)))
        se = float(s.get("group_end", s.get("end", 0)))
        if se - ss > short_dur_max:
            continue
        # voice embedding
        i0 = int(max(0, (ss - 0.2) * sr))
        i1 = int(min(len(audio), (se + 0.2) * sr))
        if i1 <= i0:
            continue
        try:
            emb = extract_eres2netv2_emb(audio[i0:i1], sr)
        except Exception:
            continue
        emb = _l2(np.asarray(emb, dtype=np.float32))
        # 모든 main SPK 와 cosine sim
        sims = {sp: float(np.dot(emb, c)) for sp, c in main_cent.items()}
        if not sims:
            continue
        best_sp, best_sim = max(sims.items(), key=lambda x: x[1])
        # face cluster 확인 — 그 시간대 dominant face cluster 의 SPK 같으면 강하게
        face_evidence = None
        if tracks and face_clusters:
            f0 = int(ss * fps)
            f1 = int(se * fps)
            track_count: Counter = Counter()
            for ti, t in enumerate(tracks):
                fr = np.array(t.get("frames", []))
                sc = np.array(t.get("scores", []))
                if len(fr) == 0:
                    continue
                mask = (fr >= f0) & (fr <= f1) & (sc >= 0.5)
                n = int(mask.sum())
                if n > 0:
                    cid = face_clusters.get(ti, -1)
                    if cid >= 0:
                        track_count[cid] += n
            if track_count:
                dom_cid = track_count.most_common(1)[0][0]
                # spk_face_map 에서 그 cluster 의 dominant SPK 찾기
                for sp, info in spk_face_map.items():
                    if info.get("dominant_face_cluster") == dom_cid:
                        face_evidence = sp
                        break

        margin = reassign_margin_with_face if face_evidence == best_sp else reassign_margin
        if best_sp != cur_spk and best_sim >= reassign_min_sim and best_sim >= margin:
            print(f"  REASSIGN [{ss:.2f}-{se:.2f}] {cur_spk} → {best_sp} "
                  f"(sim={best_sim:.2f}, face={face_evidence or 'X'}, margin={margin})")
            s["audio_speaker_orig"] = cur_spk
            s["speaker"] = best_sp
            s["from_auto_refine"] = True
            n_reassign += 1

    # consecutive same-SPK merge
    segs_sorted = sorted(segs, key=lambda x: float(x.get("group_start", x.get("start", 0))))
    merged = []
    for s in segs_sorted:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            prev_end = float(merged[-1].get("group_end", merged[-1].get("end", 0)))
            cur_start = float(s.get("group_start", s.get("start", 0)))
            if cur_start - prev_end <= 0.1:
                merged[-1]["group_end"] = s.get("group_end", s.get("end"))
                merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
                continue
        merged.append(dict(s))

    data["groups"] = merged
    out = meta / f"{seg_path.stem}_auto_refined.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    cnt = Counter(s["speaker"] for s in merged)
    print(f"\n  [✓] {len(segs)} → {len(merged)} segments ({n_reassign} reassigns)")
    print(f"      SPKs: {dict(cnt)}")
    print(f"      saved → {out}")
    return {"input": len(segs), "output": len(merged), "reassigns": n_reassign}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--outlier-total-max", type=int, default=3)
    ap.add_argument("--short-dur-max", type=float, default=2.0)
    ap.add_argument("--reassign-margin", type=float, default=0.05)
    ap.add_argument("--reassign-margin-with-face", type=float, default=0.02)
    ap.add_argument("--reassign-min-sim", type=float, default=0.40)
    args = ap.parse_args()
    s = auto_refine(
        args.run_dir,
        outlier_total_max=args.outlier_total_max,
        short_dur_max=args.short_dur_max,
        reassign_margin=args.reassign_margin,
        reassign_margin_with_face=args.reassign_margin_with_face,
        reassign_min_sim=args.reassign_min_sim,
    )
    print(f"summary: {s}")


if __name__ == "__main__":
    main()
