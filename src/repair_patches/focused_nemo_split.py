"""v190 generalized: focused NeMo re-diarize for cross-speaker drift.

핵심: long segment (≥ MIN_DUR_FOCUS) 마다 NeMo daemon 에 짧은 audio chunk 던져
num_speakers=2 force 로 호출 → 2 SPK 검출되면 boundary 에서 split,
양쪽을 가장 가까운 메인 SPK centroid 로 라벨링.

v190 baseline ([[baseline-v190-6-speakers]]) 의 visual ASD trigger 부분은 생략
(generic: 모든 long segment 에 적용). LightASD cache 의존 X.

입력: run_dir
출력: meta/<chunk>_segments_v190.json
"""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "daemons"))
from eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model

NEMO_URL = "http://127.0.0.1:8923"
MIN_DUR_FOCUS = 1.5      # 이 길이 이상 segment 만 focused NeMo 호출
GAP_MIN_NEMO = 2.0       # gap (segment 사이) 이 길이 이상이면 NeMo 호출 (mom 잡기)
F0_FEMALE_MIN = 180      # 여성/외침 F0 최소 (Hz)
MARGIN = 0.5             # NeMo audio chunk window padding (sec)
MIN_NEMO_DUR = 0.4       # NeMo가 보고한 sub-segment 최소 길이 (sec) — 너무 짧으면 skip
EMB_PAD = 0.2            # ERes2 embedding chunk padding


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
    # v194 결과 우선, else segments
    v194 = sorted(meta.glob("*_chunk_*_segments_v194.json"))
    base = v194 if v194 else sorted(meta.glob("*_chunk_*_segments.json"))
    base = [p for p in base if not any(p.name.endswith(x) for x in
            ["_fixed.json", "_gapfilled.json", "_v190.json"])]
    if not base:
        raise SystemExit(f"no segments in {meta}")
    get_eres2netv2_model()

    for seg_path in base:
        chunk_name = seg_path.stem.replace("_segments_v194", "").replace("_segments", "")
        vocals = rd / "vocals" / f"{chunk_name}_clean_vocals.wav"
        if not vocals.exists():
            print(f"[skip] {chunk_name}: no vocals")
            continue

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
        print(f"\n[{chunk_name}] SPK centroids: {sorted(spk_cent)}")

        new_segs = []
        n_split = 0
        for seg in groups:
            dur = seg["group_end"] - seg["group_start"]
            if dur < MIN_DUR_FOCUS:
                new_segs.append(dict(seg))
                continue

            # focused NeMo call
            t0 = max(0.0, seg["group_start"] - MARGIN)
            t1 = min(total_dur, seg["group_end"] + MARGIN)
            chunk = audio[int(t0*sr):int(t1*sr)]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, chunk, sr)
                tmp_wav = tf.name
            try:
                r = requests.post(f"{NEMO_URL}/diarize",
                                  json={"vocals_wav": tmp_wav,
                                        "num_speakers": 2,
                                        "min_duration": 0.3},
                                  timeout=60).json()
            except Exception as ex:
                print(f"  NeMo fail [{seg['group_start']:.2f}-{seg['group_end']:.2f}]: {ex}")
                new_segs.append(dict(seg))
                continue
            sub_segs = r.get("segments", [])
            n_spk = r.get("n_speakers", 0)
            if n_spk < 2 or len(sub_segs) < 2:
                new_segs.append(dict(seg))
                continue

            # NeMo sub-boundary → global time + 양쪽 embedding → 메인 SPK 매핑
            sub_local = sorted(sub_segs, key=lambda s: s["start"])
            # 통합: 같은 SPK 라벨 연속이면 merge (NeMo 내부 라벨)
            merged_local = []
            for s in sub_local:
                if merged_local and merged_local[-1]["speaker"] == s["speaker"] \
                   and s["start"] - merged_local[-1]["end"] <= 0.2:
                    merged_local[-1]["end"] = s["end"]
                else:
                    merged_local.append(dict(s))
            if len(merged_local) < 2:
                new_segs.append(dict(seg))
                continue

            # 각 sub-seg 를 global time 으로 + 길이 < MIN_NEMO_DUR 면 skip
            new_sub = []
            for s in merged_local:
                gs = s["start"] + t0
                ge = s["end"] + t0
                # original segment 경계로 clip
                gs = max(gs, seg["group_start"])
                ge = min(ge, seg["group_end"])
                if ge - gs < MIN_NEMO_DUR:
                    continue
                e = emb_at(audio, sr, gs, ge)
                if e is None:
                    continue
                # 가장 가까운 메인 SPK centroid
                best = max(spk_cent.items(), key=lambda x: float(np.dot(e, x[1])))[0]
                new_sub.append({"group_start": float(gs), "group_end": float(ge),
                                "speaker": best, "text": "",
                                "from_v190_focused": True})
            if len(new_sub) < 2:
                new_segs.append(dict(seg))
                continue

            # 모든 sub_segs 가 같은 SPK 면 split 의미 없음
            sub_spks = set(s["speaker"] for s in new_sub)
            if len(sub_spks) < 2:
                new_segs.append(dict(seg))
                continue

            # 텍스트 재할당 (원본 segment 의 text 를 시간 비율로 분배 — 정확도 한계)
            orig_text = seg.get("text", "")
            words_in_text = orig_text.split()
            if words_in_text and len(new_sub) > 0:
                total_new = sum(s["group_end"] - s["group_start"] for s in new_sub)
                cursor = 0
                for i, s in enumerate(new_sub):
                    d = s["group_end"] - s["group_start"]
                    if i == len(new_sub) - 1:
                        s["text"] = " ".join(words_in_text[cursor:])
                    else:
                        take = max(1, round(len(words_in_text) * d / max(total_new, 1e-9)))
                        s["text"] = " ".join(words_in_text[cursor:cursor + take])
                        cursor += take

            print(f"  SPLIT [{seg['group_start']:.2f}-{seg['group_end']:.2f}] {seg['speaker']} "
                  f"({dur:.2f}s) → {len(new_sub)} sub: " +
                  " | ".join(f"[{s['group_start']:.2f}-{s['group_end']:.2f}]{s['speaker']}" for s in new_sub))
            new_segs.extend(new_sub)
            n_split += 1

        # === gap-aware NeMo: segment 사이 gap 에 NeMo 호출 (mom 외침 잡기) ===
        new_segs.sort(key=lambda x: x["group_start"])
        # F0 계산 (전체 audio, 짧은 frame_length)
        import librosa
        f0_arr, _, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=2048,
        )
        ftimes = librosa.times_like(f0_arr, sr=sr)
        def _f0_med(t0, t1):
            mask = (ftimes >= t0) & (ftimes < t1)
            v = f0_arr[mask]; v = v[~np.isnan(v)]
            return float(np.median(v)) if len(v) >= 3 else None
        # 각 SPK centroid 의 F0 평균 (메인 orch 가 보낸 segments 중 long segment 사용)
        spk_f0 = {}
        for sp in spk_cent:
            f0s = []
            for s in groups:
                if s["speaker"] == sp and s["group_end"] - s["group_start"] >= 1.0:
                    f = _f0_med(s["group_start"], s["group_end"])
                    if f: f0s.append(f)
            if f0s: spk_f0[sp] = float(np.median(f0s))
        print(f"  SPK F0 median: { {sp: round(f,1) for sp, f in spk_f0.items()} }")

        gaps = []
        if new_segs and new_segs[0]["group_start"] >= GAP_MIN_NEMO:
            gaps.append((0.0, new_segs[0]["group_start"]))
        for i in range(len(new_segs)-1):
            g0, g1 = new_segs[i]["group_end"], new_segs[i+1]["group_start"]
            if g1 - g0 >= GAP_MIN_NEMO:
                gaps.append((g0, g1))
        if new_segs and total_dur - new_segs[-1]["group_end"] >= GAP_MIN_NEMO:
            gaps.append((new_segs[-1]["group_end"], total_dur))

        n_gap_split = 0
        for (gs_, ge_) in gaps:
            chunk = audio[int(gs_*sr):int(ge_*sr)]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, chunk, sr)
                tmp_wav = tf.name
            try:
                r = requests.post(f"{NEMO_URL}/diarize",
                                  json={"vocals_wav": tmp_wav, "num_speakers": 2,
                                        "min_duration": 0.3},
                                  timeout=60).json()
            except Exception as ex:
                continue
            subs = r.get("segments", [])
            if not subs:
                continue
            # 각 sub-segment 의 SPK 결정 (voice + F0 결합)
            for s in subs:
                gs2 = s["start"] + gs_
                ge2 = s["end"] + gs_
                if ge2 - gs2 < 0.4:
                    continue
                e = emb_at(audio, sr, gs2, ge2)
                if e is None:
                    continue
                # voice candidate
                sims = {sp: float(np.dot(e, c)) for sp, c in spk_cent.items()}
                voice_best, voice_sim = max(sims.items(), key=lambda x: x[1])
                # F0 candidate (가장 가까운 F0)
                seg_f0 = _f0_med(gs2, ge2)
                f0_best, f0_diff = None, 1e9
                if seg_f0 is not None:
                    for sp, sf0 in spk_f0.items():
                        d = abs(seg_f0 - sf0)
                        if d < f0_diff:
                            f0_diff, f0_best = d, sp
                # 결정: F0 가 모든 메인 SPK 보다 훨씬 높으면 (외침/여성) → F0 best 우선
                # 그렇지 않으면 voice best
                chosen = voice_best
                rationale = f"voice({voice_sim:.2f})"
                if seg_f0 is not None and spk_f0:
                    max_main_f0 = max(spk_f0.values())
                    # 외침/여성: F0 가 모든 메인 F0 + 30Hz 이상이면 F0 best 우선
                    if seg_f0 > max_main_f0 + 30 and f0_best != voice_best:
                        chosen = f0_best
                        rationale = f"F0({seg_f0:.0f}Hz max_main={max_main_f0:.0f}) → {f0_best}"
                print(f"  GAP-NEMO [{gs2:.2f}-{ge2:.2f}] → {chosen} ({rationale}) "
                      f"[seg_f0={seg_f0}, voice_best={voice_best}({voice_sim:.2f}), f0_best={f0_best}({f0_diff:.0f}Hz)]")
                new_segs.append({"group_start": float(gs2), "group_end": float(ge2),
                                 "speaker": chosen, "text": "",
                                 "from_v190_gap_nemo": True})
                n_gap_split += 1

        # merge consecutive same-SPK
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
        print(f"\n  [✓] {len(groups)} → {len(merged)} segments (focused splits={n_split})")
        print(f"      SPKs: {dict(cnt)}")
        seg_data["groups"] = merged
        out = meta / f"{chunk_name}_segments_v190.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(seg_data, f, ensure_ascii=False, indent=2)
        print(f"      saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    main(args.run_dir)
