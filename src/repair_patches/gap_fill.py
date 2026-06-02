"""Gap-fill v2: gap 외침 detect + BG inter-cluster merge + 메인 over-merge.

목표: test5 → 메인 4명 + BG 2명 = 6명

처리:
  1. ERes2NetV2 centroid 추출 (메인 화자별 longest segment)
  2. diarize gap (≥1.5s) 짧게 잘라 ASR → fresh word
  3. 각 fresh word ± PAD chunk → embedding → 기존 메인과 cosine
     - sim ≥ MAIN_MATCH → 메인 라벨
     - else → 임시 BG_RAW_N
  4. 모든 BG_RAW_N 끼리 다시 cosine cluster → BG_00 / BG_01 ... merge (sim ≥ BG_MERGE)
  5. 메인 화자 over-merge: sim ≥ MAIN_MERGE 면 두 메인 화자 통합
  6. segments_gapfilled.json + 화자 수 report
"""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "daemons"))
from eres2netv2_helper import extract_eres2netv2_emb, get_eres2netv2_model

ASR_URL = "http://127.0.0.1:8902/transcribe"
GAP_MIN = 1.5
PAD = 1.0            # word 주변 padding (sec). 외침은 짧으므로 크게
SIM_MAIN_MATCH = 0.50  # cosine ≥ 이면 메인 화자로 매칭
BG_MERGE = 0.40        # BG 화자들끼리 sim ≥ 이면 같은 BG로 merge
MAIN_MERGE = 0.80      # 메인 화자끼리 sim ≥ 이면 over-merge
LANG = "English"

# === 짧은 segment voice 재배정 (v178 ERes2NetV2 short reassign) ===
# face(화면) 재배정은 리액션샷/저신뢰 클러스터에서 메인 대화를 망가뜨리지만,
# voice 는 짧은 발화도 본인 화자에 강하게 매칭(검증 2026-05-31: test4 'Good' 0.76s → sean 0.59 vs dad 0.29).
SHORT_REASSIGN = True   # 짧은 발화를 voice centroid 로 본인 화자에 자동 재배정
SHORT_DUR = 1.5         # 이보다 짧은 segment 만 대상 (긴 turn 은 안 건드림)
SHORT_MARGIN = 0.15     # best_sim - current_sim ≥ 이 값일 때만 재배정 (명확할 때만)
SHORT_MIN_SIM = 0.45    # best_sim ≥ 이 값일 때만 (잡음/단역 오배정 방지)


def cosine(a, b):
    return float(np.dot(a, b))


def cluster_bgs(bgs):
    """bgs: list of (id, emb). Returns dict {old_id: new_label} 매핑."""
    if not bgs:
        return {}
    labels = list(range(len(bgs)))  # union-find
    def find(i):
        while labels[i] != i:
            labels[i] = labels[labels[i]]
            i = labels[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            labels[max(ri,rj)] = min(ri,rj)
    for i in range(len(bgs)):
        for j in range(i+1, len(bgs)):
            if cosine(bgs[i][1], bgs[j][1]) >= BG_MERGE:
                union(i, j)
    root_to_label = {}
    out = {}
    for i, (bid, _) in enumerate(bgs):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = f"SPEAKER_BG_{len(root_to_label):02d}"
        out[bid] = root_to_label[r]
    return out


def main(run_dir: str):
    rd = Path(run_dir)
    meta = rd / "meta"
    # 우선순위: v190b (visual ASD) > v190 (focused NeMo) > v194 (word-split) > 원본
    v190b = sorted(meta.glob("*_chunk_*_segments_v190b.json"))
    v190 = sorted(meta.glob("*_chunk_*_segments_v190.json"))
    v194 = sorted(meta.glob("*_chunk_*_segments_v194.json"))
    seg_files = (v190b if v190b else (v190 if v190 else
                 (v194 if v194 else sorted(meta.glob("*_chunk_*_segments.json")))))
    seg_files = [p for p in seg_files if not any(p.name.endswith(x)
                 for x in ["_fixed.json", "_gapfilled.json"])]
    if not seg_files:
        raise SystemExit(f"no segments.json in {meta}")
    get_eres2netv2_model()

    for seg_path in seg_files:
        # _segments_v194 / _segments 둘 다 처리
        chunk_name = (seg_path.stem.replace("_segments_v190b", "")
                                   .replace("_segments_v190", "")
                                   .replace("_segments_v194", "")
                                   .replace("_segments", ""))
        # words_v2 (boost+sub-chunk 적용된 풍부한 words) 우선
        words_v2 = meta / f"{chunk_name}_words_v2.json"
        words_path = words_v2 if words_v2.exists() else meta / f"{chunk_name}_words.json"
        chunk_mp4 = rd / "chunks" / f"{chunk_name}.mp4"
        if not chunk_mp4.exists() or not words_path.exists():
            print(f"[skip] {chunk_name}")
            continue

        seg_data = json.load(open(seg_path))
        groups = seg_data["groups"]
        existing_words = json.load(open(words_path))["words"]

        # audio
        audio_wav = Path("/tmp") / f"{chunk_name}_full.wav"
        if not audio_wav.exists():
            subprocess.run(["ffmpeg", "-y", "-i", str(chunk_mp4),
                            "-ar", "16000", "-ac", "1", str(audio_wav)],
                           capture_output=True, check=True)
        audio, sr = sf.read(audio_wav)
        total_dur = len(audio) / sr
        def slice(s, e):
            i0, i1 = max(0, int(s*sr)), min(len(audio), int(e*sr))
            return audio[i0:i1]

        # 메인 화자 centroids — 모든 SPK 포함 (짧으면 padding 으로 ≥0.4s 보장)
        PAD_CENT = 0.5
        spk_best = {}
        for g in groups:
            spk = g["speaker"]
            dur = g["group_end"] - g["group_start"]
            if spk not in spk_best or dur > spk_best[spk][0]:
                spk_best[spk] = (dur, g["group_start"], g["group_end"])
        centroids = {}
        for spk, (d, s, e) in spk_best.items():
            s2 = max(0.0, s - PAD_CENT)
            e2 = min(total_dur, e + PAD_CENT)
            if e2 - s2 < 0.4:
                continue
            emb = extract_eres2netv2_emb(slice(s2, e2), sr=sr)
            if emb is not None:
                centroids[spk] = emb
        print(f"\n[{chunk_name}] 메인 centroids: {sorted(centroids.keys())}")

        # === 메인 over-merge: 메인 화자끼리 sim ≥ MAIN_MERGE 면 통합 ===
        main_keys = sorted(centroids.keys())
        merge_map = {k: k for k in main_keys}
        for i in range(len(main_keys)):
            for j in range(i+1, len(main_keys)):
                a, b = main_keys[i], main_keys[j]
                if cosine(centroids[a], centroids[b]) >= MAIN_MERGE:
                    # b → a 로 흡수
                    target = merge_map[a]
                    src = merge_map[b]
                    merge_map = {k: (target if v == src else v) for k, v in merge_map.items()}
        # apply
        for g in groups:
            g["speaker"] = merge_map.get(g["speaker"], g["speaker"])
        # centroids 다시
        main_after = sorted(set(merge_map.values()))
        print(f"  메인 over-merge 후: {main_after}  (merge_map={merge_map})")

        # === 짧은 segment voice 재배정 (v178 short reassign) ===
        # 짧은 발화가 인접 화자 턴에 흡수돼 잘못 배정된 경우, voice centroid 로 본인 화자에 되돌림.
        # 예: test4 'Good'(sean) 이 diarization 턴 경계 때문에 dad(03) 턴에 들어간 것 → sean(04) 으로 자동 교정.
        if SHORT_REASSIGN:
            vr_best = {}
            for g in groups:
                spk = g["speaker"]; d = g["group_end"] - g["group_start"]
                if spk not in vr_best or d > vr_best[spk][0]:
                    vr_best[spk] = (d, g["group_start"], g["group_end"])
            vr_cent = {}
            for spk, (d, s, e) in vr_best.items():
                s2 = max(0.0, s - PAD_CENT); e2 = min(total_dur, e + PAD_CENT)
                if e2 - s2 < 0.4:
                    continue
                em = extract_eres2netv2_emb(slice(s2, e2), sr=sr)
                if em is not None:
                    vr_cent[spk] = em
            n_vr = 0
            for g in groups:
                if g["group_end"] - g["group_start"] > SHORT_DUR:
                    continue
                cur = g["speaker"]
                # 본인 화자 centroid 출처(최장 segment)면 skip
                if cur in vr_best and abs(vr_best[cur][1] - g["group_start"]) < 1e-6:
                    continue
                s2 = max(0.0, g["group_start"] - 0.1); e2 = min(total_dur, g["group_end"] + 0.1)
                em = extract_eres2netv2_emb(slice(s2, e2), sr=sr)
                if em is None:
                    continue
                sims = {sp: cosine(em, c) for sp, c in vr_cent.items()}
                if not sims:
                    continue
                best = max(sims, key=sims.get)
                cur_sim = sims.get(cur, -1.0)
                if best != cur and sims[best] >= SHORT_MIN_SIM and (sims[best] - cur_sim) >= SHORT_MARGIN:
                    print(f"  [voice-reassign] {g['group_start']:.2f}-{g['group_end']:.2f} {cur}→{best} (sim {cur_sim:.2f}→{sims[best]:.2f})")
                    g["speaker"] = best
                    n_vr += 1
            if n_vr:
                print(f"  voice 짧은segment 재배정 {n_vr}건")

        # === gap 검출 ===
        gaps = []
        if groups[0]["group_start"] > GAP_MIN:
            gaps.append((0.0, groups[0]["group_start"]))
        for i in range(len(groups)-1):
            g0, g1 = groups[i]["group_end"], groups[i+1]["group_start"]
            if g1 - g0 >= GAP_MIN:
                gaps.append((g0, g1))
        if total_dur - groups[-1]["group_end"] >= GAP_MIN:
            gaps.append((groups[-1]["group_end"], total_dur))

        new_segs = []
        raw_bgs = []   # [(temp_id, emb)] for cluster
        temp_id_counter = 0

        for s, e in gaps:
            gap_wav = Path("/tmp") / f"gap_{chunk_name}_{int(s*100)}_{int(e*100)}.wav"
            sf.write(gap_wav, slice(s, e), sr)
            try:
                r = requests.post(ASR_URL,
                                  json={"audio_path": str(gap_wav), "language": LANG},
                                  timeout=120).json()
            except Exception as ex:
                print(f"  gap {s:.2f}-{e:.2f} ASR fail: {ex}")
                continue
            gap_words = r.get("words", []) or []
            fresh = []
            for w in gap_words:
                gws = w["start"] + s
                if any(abs(gws - ew["start"]) < 0.3 for ew in existing_words):
                    continue
                fresh.append({"word": w["word"], "start": gws, "end": w["end"]+s})
            if not fresh:
                continue
            print(f"  gap {s:.2f}-{e:.2f}: {len(fresh)} fresh {[w['word'] for w in fresh]}")
            for w in fresh:
                # word ± PAD — gap 경계 무시 (chunk 충분히 확보)
                c0 = max(0.0, w["start"] - PAD)
                c1 = min(total_dur, w["end"] + PAD)
                if c1 - c0 < 0.4:
                    pad2 = (0.4 - (c1-c0))/2 + 0.05
                    c0 = max(s, c0 - pad2)
                    c1 = min(e, c1 + pad2)
                emb = extract_eres2netv2_emb(slice(c0, c1), sr=sr) if c1-c0 >= 0.4 else None
                # 메인 매칭
                assigned = None
                if emb is not None:
                    best = None
                    for spk, cent in centroids.items():
                        # 이미 over-merge된 메인 라벨 확인
                        target_spk = merge_map.get(spk, spk)
                        sim = cosine(emb, cent)
                        if best is None or sim > best[1]:
                            best = (target_spk, sim)
                    if best and best[1] >= SIM_MAIN_MATCH:
                        assigned = best[0]
                        print(f"    {w['start']:.2f}-{w['end']:.2f} {w['word']!r} → {assigned} (sim={best[1]:.2f})")
                if assigned is None:
                    # 임시 BG (나중에 cluster)
                    tid = f"_raw_bg_{temp_id_counter:02d}"
                    temp_id_counter += 1
                    if emb is not None:
                        raw_bgs.append((tid, emb))
                    assigned = tid  # placeholder
                    print(f"    {w['start']:.2f}-{w['end']:.2f} {w['word']!r} → {tid} (will cluster)")
                new_segs.append({"group_start": float(c0), "group_end": float(c1),
                                 "speaker": assigned, "text": w["word"],
                                 "from_gap_fill": True})

        # === BG inter-cluster ===
        bg_map = cluster_bgs(raw_bgs)
        print(f"\n  BG cluster: {temp_id_counter} raw → {len(set(bg_map.values()))} unique BG")
        for ns in new_segs:
            if ns["speaker"].startswith("_raw_bg_"):
                ns["speaker"] = bg_map.get(ns["speaker"], "SPEAKER_BG_UNK")

        # === 통합 ===
        all_seg = groups + new_segs
        all_seg.sort(key=lambda x: x["group_start"])
        seg_data["groups"] = all_seg
        out = meta / f"{chunk_name}_segments_gapfilled.json"
        json.dump(seg_data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # 최종 화자 수
        from collections import Counter
        cnt = Counter(g["speaker"] for g in all_seg)
        main_n = sum(1 for s in cnt if not s.startswith("SPEAKER_BG"))
        bg_n = sum(1 for s in cnt if s.startswith("SPEAKER_BG"))
        print(f"\n  [✓] {len(all_seg)} groups (+{len(new_segs)} gap-fill)")
        print(f"      메인 {main_n}명 + BG {bg_n}명 = {main_n+bg_n}명  (목표: 4+2=6)")
        print(f"      counts: {dict(cnt)}")
        print(f"      saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--main-merge", type=float, default=MAIN_MERGE)
    ap.add_argument("--bg-merge", type=float, default=BG_MERGE)
    ap.add_argument("--sim-match", type=float, default=SIM_MAIN_MATCH)
    ap.add_argument("--pad", type=float, default=PAD)
    args = ap.parse_args()
    MAIN_MERGE = args.main_merge
    BG_MERGE = args.bg_merge
    SIM_MAIN_MATCH = args.sim_match
    PAD = args.pad
    main(args.run_dir)
