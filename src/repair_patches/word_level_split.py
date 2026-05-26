"""v194 word-level intra-segment split (일반화).

핵심: 한 segment 안에서 prefix vs suffix word embedding 비교 → 화자 바뀜 검출 → 분할.
v194 patch ([[baseline-v194-word-level]]) 일반화.

입력: run_dir
처리:
  Stage A: SPK centroid 추출 (long >=3s 우선, fallback longest)
  Stage B: 각 segment ≥2 words → prefix/suffix split 시도
     boundary 조건: F0 jump ≥100Hz + LR_cos < 0.35 + 양쪽 sim ≥0.40
                  OR LR_cos < 0.25 + 양쪽 sim ≥0.40
  Stage C: short segment (<=4 words, <=2.5s) whole-voice reassign
     best_sim > current_sim + 0.03 AND best_sim ≥ 0.35
  Stage D: consecutive same-SPK merge (gap ≤ 0.1s)

출력: meta/<chunk>_segments_v194.json
"""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/workspace")
from patches.eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model

WORD_DIFF_TH = 0.30      # LR_cos < 이면 split 후보
F0_JUMP_TH = 100         # F0 차 Hz
LR_COS_STRICT = 0.35     # F0 jump 함께면 이 threshold
SIM_BOTH_MIN = 0.40      # 양쪽 sim 모두 이 이상
REASSIGN_MARGIN = 0.03   # whole-seg reassign best > cur + this
REASSIGN_MIN_SIM = 0.35


def emb_at(audio, sr, t0, t1, pad=0.2):
    s = max(0, t0 - pad)
    e = min(len(audio) / sr, t1 + pad)
    c = audio[int(s*sr):int(e*sr)]
    if len(c) < int(0.08 * sr):
        return None
    return extract_eres2netv2_emb(c, sr=sr)


def f0_median(f0_arr, ftimes, t0, t1):
    mask = (ftimes >= t0) & (ftimes < t1)
    v = f0_arr[mask]
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) >= 3 else None


def main(run_dir: str):
    rd = Path(run_dir)
    meta = rd / "meta"
    seg_files = sorted(meta.glob("*_chunk_*_segments.json"))
    seg_files = [p for p in seg_files
                 if not any(p.name.endswith(x) for x in
                            ["_fixed.json", "_gapfilled.json", "_v194.json"])]
    if not seg_files:
        raise SystemExit(f"no segments.json in {meta}")
    get_eres2netv2_model()

    import librosa

    for seg_path in seg_files:
        chunk_name = seg_path.stem.replace("_segments", "")
        words_path = meta / f"{chunk_name}_words.json"
        vocals = rd / "vocals" / f"{chunk_name}_clean_vocals.wav"
        if not words_path.exists() or not vocals.exists():
            print(f"[skip] {chunk_name}: missing words.json or vocals")
            continue

        seg_data = json.load(open(seg_path))
        groups = seg_data["groups"]
        words = json.load(open(words_path))["words"]
        valid_words = [w for w in words
                       if w.get("start") is not None and w.get("end") is not None]

        audio, sr = sf.read(vocals)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        total_dur = len(audio) / sr

        # Stage A: SPK centroid (long seg 우선, fallback longest)
        spk_embs = defaultdict(list)
        for s in groups:
            dur = s["group_end"] - s["group_start"]
            if dur < 3.0:
                continue
            e = emb_at(audio, sr, s["group_start"], s["group_end"])
            if e is not None:
                spk_embs[s["speaker"]].append(e)
        for s in groups:
            if s["speaker"] in spk_embs and spk_embs[s["speaker"]]:
                continue
            if s["group_end"] - s["group_start"] < 0.4:
                continue
            e = emb_at(audio, sr, s["group_start"], s["group_end"])
            if e is not None:
                spk_embs[s["speaker"]].append(e)
        spk_cent = {}
        for spk, es in spk_embs.items():
            c = np.mean(np.stack(es), axis=0)
            c = c / max(np.linalg.norm(c), 1e-9)
            spk_cent[spk] = c
        print(f"\n[{chunk_name}] SPK centroids: {sorted(spk_cent)}")

        # F0 (전체 audio)
        print("  computing F0 (pyin)...")
        f0_arr, _, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=2048,
        )
        ftimes = librosa.times_like(f0_arr, sr=sr)

        # Stage B: word-level intra-segment split
        new_segs = []
        n_split = 0
        for seg in groups:
            seg_words = [w for w in valid_words
                         if seg["group_start"] - 0.05 <= (w["start"]+w["end"])/2 <= seg["group_end"] + 0.05]
            if len(seg_words) < 2:
                new_segs.append(dict(seg))
                continue
            word_embs = [emb_at(audio, sr, w["start"], w["end"]) for w in seg_words]
            if not all(e is not None for e in word_embs):
                new_segs.append(dict(seg))
                continue

            best_boundary = None
            for k in range(1, len(word_embs)):
                la = np.mean(np.stack(word_embs[:k]), axis=0)
                la /= max(np.linalg.norm(la), 1e-9)
                ra = np.mean(np.stack(word_embs[k:]), axis=0)
                ra /= max(np.linalg.norm(ra), 1e-9)
                ls = {sp: float(np.dot(la, c)) for sp, c in spk_cent.items()}
                rs = {sp: float(np.dot(ra, c)) for sp, c in spk_cent.items()}
                lb, lv = max(ls.items(), key=lambda x: x[1])
                rb, rv = max(rs.items(), key=lambda x: x[1])
                if lb == rb:
                    continue
                sim_lr = float(np.dot(la, ra))
                lf0 = f0_median(f0_arr, ftimes, seg_words[0]["start"], seg_words[k-1]["end"])
                rf0 = f0_median(f0_arr, ftimes, seg_words[k]["start"], seg_words[-1]["end"])
                f0d = abs((lf0 or 0) - (rf0 or 0)) if (lf0 and rf0) else 0
                has_f0_jump = (lf0 is not None and rf0 is not None and f0d >= F0_JUMP_TH)
                strong_voice = (sim_lr < 0.25 and lv >= SIM_BOTH_MIN and rv >= SIM_BOTH_MIN)
                if (has_f0_jump and lv >= SIM_BOTH_MIN and rv >= SIM_BOTH_MIN and sim_lr < LR_COS_STRICT) \
                   or strong_voice:
                    score = sim_lr - 0.001 * f0d
                    if best_boundary is None or score < best_boundary[-1]:
                        best_boundary = (k, lb, rb, lv, rv, sim_lr, score)
            if best_boundary is None:
                new_segs.append(dict(seg))
                continue

            k, lb, rb, lv, rv, sim_lr, _ = best_boundary
            print(f"  SPLIT [{seg['group_start']:.2f}-{seg['group_end']:.2f}] {seg['speaker']} "
                  f"@word#{k} ({seg_words[k-1].get('word','')}|{seg_words[k].get('word','')}) "
                  f"L→{lb}({lv:.2f}) R→{rb}({rv:.2f}) LR={sim_lr:.2f}")
            # left half
            lw = seg_words[:k]
            la = np.mean(np.stack(word_embs[:k]), axis=0)
            la /= max(np.linalg.norm(la), 1e-9)
            l_best = max(spk_cent.items(), key=lambda x: float(np.dot(la, x[1])))[0]
            new_segs.append({
                "group_start": lw[0]["start"], "group_end": lw[-1]["end"],
                "speaker": l_best, "text": " ".join(w["word"].strip() for w in lw),
                "from_v194_split": True, "word_count": len(lw),
            })
            # right half
            rw = seg_words[k:]
            ra = np.mean(np.stack(word_embs[k:]), axis=0)
            ra /= max(np.linalg.norm(ra), 1e-9)
            r_best = max(spk_cent.items(), key=lambda x: float(np.dot(ra, x[1])))[0]
            new_segs.append({
                "group_start": rw[0]["start"], "group_end": rw[-1]["end"],
                "speaker": r_best, "text": " ".join(w["word"].strip() for w in rw),
                "from_v194_split": True, "word_count": len(rw),
            })
            n_split += 1

        # Stage C: short segment whole-voice reassign
        n_reassign = 0
        for s in new_segs:
            if s.get("word_count", 99) > 4:
                continue
            if s["group_end"] - s["group_start"] > 2.5:
                continue
            e = emb_at(audio, sr, s["group_start"], s["group_end"])
            if e is None:
                continue
            cur = spk_cent.get(s["speaker"])
            cur_sim = float(np.dot(e, cur)) if cur is not None else 0
            sims = {sp: float(np.dot(e, c)) for sp, c in spk_cent.items()}
            best_sp, best_sim = max(sims.items(), key=lambda x: x[1])
            if best_sp != s["speaker"] and best_sim > cur_sim + REASSIGN_MARGIN \
               and best_sim >= REASSIGN_MIN_SIM:
                print(f"  REASSIGN [{s['group_start']:.2f}-{s['group_end']:.2f}] "
                      f"{s['speaker']} → {best_sp} (cur={cur_sim:.2f} best={best_sim:.2f})")
                s["speaker"] = best_sp
                n_reassign += 1

        # Stage D: merge consecutive same-SPK
        new_segs.sort(key=lambda x: x["group_start"])
        merged = []
        for s in new_segs:
            if merged and merged[-1]["speaker"] == s["speaker"] \
               and s["group_start"] - merged[-1]["group_end"] <= 0.1:
                merged[-1]["group_end"] = s["group_end"]
                merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
            else:
                merged.append(dict(s))

        cnt = Counter(s["speaker"] for s in merged)
        print(f"\n  [✓] {len(groups)} → {len(merged)} segments (splits={n_split}, reassigns={n_reassign})")
        print(f"      SPKs: {dict(cnt)}")
        seg_data["groups"] = merged
        out = meta / f"{chunk_name}_segments_v194.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, ensure_ascii=False, indent=2)
        print(f"      saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    main(args.run_dir)
