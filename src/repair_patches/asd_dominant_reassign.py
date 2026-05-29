"""ASD-dominant fragment reassign — 과분할된 짧은 조각을 ASD로 흡수.

문제:
  word_split / focused_nemo 등이 만든 짧은 spurious 화자(ASD speaking 지지 ~0)가
  실제로는 그 시간대에 화면에서 말하는 major 화자의 조각인 경우가 많다
  (예: SPEAKER_96@6.16 "That was a mistake" = man1 조각).

원리 (영상 무관, hardcoding 없음 — 화자 수 강제 X):
  1. audio 화자별 ASD speaking 지지도(speaker_face_count 합) 계산.
  2. 지지도 내림차순에서 **가장 큰 배수 gap**으로 major 화자 자동 분리
     (예: [543,419,202,181 | 6,3,0...] → major 4명). 임계 hardcode 없음.
  3. major 가 아닌 화자의 각 segment 에 대해:
       - 그 시간창에서 ASD speaking(score>=th) 인 face track 들을 face_cluster →
         dominant audio 화자로 환산해 집계.
       - 집계 1위가 **major** 이고 충분한 frame(>=MIN_FR)이면 그 major 로 재배정 (조각 흡수).
       - 그 시간창에 speaking face 가 없으면(off-camera) **그대로 보존**
         → mom 처럼 화면에 안 잡히는 진짜 화자를 spurious 와 구분해 살린다.

즉 "화면에서 실제로 말하는 사람"이 있으면 그쪽으로 붙이고, 아무도 안 말하면
(off-camera) 건드리지 않는다 — ASD 를 화자 정체가 아니라 '지금 누가 말하나'에 사용.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE_DIR = Path("/workspace/media/cache/lightasd")
SPEAK_TH = 0.5        # ASD score >= 이면 speaking frame
MIN_FRAMES = 3        # 시간창 내 major face 의 최소 speaking frame (이 미만이면 재배정 안 함)
MAJOR_GAP = 3.0       # 지지도 내림차순에서 이 배수 이상 떨어지면 그 위가 major (gap fallback)


def match_asd_cache(vocals_path: Path):
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
    if best is None or best[1] > 3.0:
        return None
    return best[0]


def _cluster_dominant_spk(face_clusters: dict, speaker_face_count: dict) -> dict:
    # face_cluster_id → 가장 많이 speaking 한 audio 화자.
    cluster_spk: dict[int, Counter] = defaultdict(Counter)
    for spk, tc in speaker_face_count.items():
        for tid_s, cnt in tc.items():
            cid = face_clusters.get(int(tid_s))
            if cid is None:
                continue
            cluster_spk[int(cid)][spk] += cnt
    return {cid: cnt.most_common(1)[0][0] for cid, cnt in cluster_spk.items() if cnt}


def _detect_majors(support: dict) -> set:
    # 지지도 내림차순에서 가장 큰 배수 gap 위쪽을 major 로 (화자 수 hardcode 없음).
    items = sorted(((s, sp) for sp, s in support.items() if s > 0), reverse=True)
    if not items:
        return set()
    if len(items) == 1:
        return {items[0][1]}
    best_gap = 1.0
    cut = len(items)
    for i in range(len(items) - 1):
        hi = items[i][0]
        lo = items[i + 1][0]
        ratio = hi / max(lo, 1e-6)
        if ratio >= best_gap:
            best_gap = ratio
            cut = i + 1
    if best_gap < MAJOR_GAP:
        # 뚜렷한 gap 없음 → 모두 major (아무것도 흡수 안 함, 안전)
        return {sp for _, sp in items}
    return {sp for _, sp in items[:cut]}


def main(run_dir: str, out_suffix: str = "_segments_asd_reassigned.json"):
    rd = Path(run_dir)
    meta = rd / "meta"
    fc_path = meta / "face_clusters.json"
    if not fc_path.exists():
        raise SystemExit(f"no face_clusters.json in {meta}")
    fc = json.load(open(fc_path, encoding="utf-8"))
    face_clusters = {int(k): int(v) for k, v in fc.get("face_clusters", {}).items()}
    speaker_face_count = fc.get("speaker_face_count", {})
    cluster_dom = _cluster_dominant_spk(face_clusters, speaker_face_count)
    support = {sp: sum(tc.values()) for sp, tc in speaker_face_count.items()}
    majors = _detect_majors(support)
    print(f"  ASD support: {dict(sorted(support.items(), key=lambda x:-x[1]))}")
    print(f"  → majors (auto gap): {sorted(majors)}")

    # gapfilled 입력
    seg_files = sorted(meta.glob("*_chunk_*_segments_gapfilled.json"))
    if not seg_files:
        raise SystemExit(f"no gapfilled in {meta}")
    for seg_path in seg_files:
        chunk_name = seg_path.stem.replace("_segments_gapfilled", "")
        cache = match_asd_cache(rd / "vocals" / f"{chunk_name}_clean_vocals.wav")
        if cache is None:
            print(f"  [skip] {chunk_name}: no ASD cache")
            continue
        asd = pickle.load(open(cache, "rb"))
        fps = float(asd.get("fps", 25.0))
        tracks = asd.get("tracks", [])

        data = json.load(open(seg_path, encoding="utf-8"))
        segs = data.get("groups", data.get("segments", []))

        n_reassign = 0
        n_kept_offcam = 0
        for s in segs:
            spk = str(s.get("speaker", ""))
            if spk.startswith("SPEAKER_BG") or spk in majors:
                continue
            ss = float(s.get("group_start", s.get("start", 0)))
            se = float(s.get("group_end", s.get("end", 0)))
            f0 = int(ss * fps)
            f1 = int(se * fps)
            # 이 시간창에서 speaking 하는 face → dominant audio 화자 집계
            spk_frames: Counter = Counter()
            for t in tracks:
                cid = face_clusters.get(int(t.get("track_id", -1)))
                if cid is None:
                    continue
                dom = cluster_dom.get(int(cid))
                if dom is None:
                    continue
                frames = t.get("frames", [])
                scores = t.get("scores", [])
                n = sum(1 for fr, sc in zip(frames, scores)
                        if f0 <= int(fr) <= f1 and float(sc) >= SPEAK_TH)
                if n > 0:
                    spk_frames[dom] += n
            if not spk_frames:
                n_kept_offcam += 1   # off-camera — 보존
                continue
            best_spk, best_n = spk_frames.most_common(1)[0]
            if best_spk != spk and best_spk in majors and best_n >= MIN_FRAMES:
                s["audio_speaker_orig"] = spk
                s["speaker"] = best_spk
                s["from_asd_dominant_reassign"] = True
                n_reassign += 1
                print(f"    REASSIGN [{ss:.2f}-{se:.2f}] {spk} → {best_spk} "
                      f"(ASD speaking {best_n}fr; '{str(s.get('text',''))[:24]}')")

        # 인접 동일 화자 merge
        segs.sort(key=lambda x: float(x.get("group_start", x.get("start", 0))))
        merged = []
        for s in segs:
            if merged and merged[-1].get("speaker") == s.get("speaker"):
                pe = float(merged[-1].get("group_end", merged[-1].get("end", 0)))
                cs = float(s.get("group_start", s.get("start", 0)))
                if cs - pe <= 0.3:
                    merged[-1]["group_end"] = s.get("group_end", s.get("end"))
                    merged[-1]["text"] = (merged[-1].get("text", "") + " " + s.get("text", "")).strip()
                    continue
            merged.append(dict(s))
        data["groups"] = merged

        spk_set = {str(s.get("speaker", "")) for s in merged
                   if not str(s.get("speaker", "")).startswith("SPEAKER_BG")}
        out = meta / f"{chunk_name}{out_suffix}"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [✓] {chunk_name}: reassigned={n_reassign}, off-cam kept={n_kept_offcam}, "
              f"{len(segs)}→{len(merged)} segs, {len(spk_set)} main SPK")
        print(f"      saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out-suffix", default="_segments_asd_reassigned.json")
    args = ap.parse_args()
    main(args.run_dir, args.out_suffix)
