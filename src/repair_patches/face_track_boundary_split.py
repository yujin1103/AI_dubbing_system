"""Face-track boundary split (2026-06-02).

Problem: when a distinct on-screen speaker (e.g. test4 paramedic "that's where
we're going", 0.64s) is MERGED into a neighbouring main speaker's segment, the
downstream face_identity_split — which labels per WHOLE segment — cannot isolate
it. golden avoided this because the FULL orchestrator's build_dub_input ran a
Silero-VAD boundary refinement that kept the interjection as its own segment; the
fast DIARIZE_ONLY path skips that step, so the two utterances stay merged.

This patch re-creates that boundary split using signals we ALREADY have (no new
model): for each segment, if a DISTINCT face identity (ArcFace cos far from the
segment's dominant face) is ASD-actively speaking over a sub-range, cut that
sub-range out as its own segment, snapping the cut points to ASR word gaps
(silence). face_identity_split then mints it as its own FACE_ speaker.

Runs AFTER gap_fill (operates on the merged structure) and BEFORE
face_identity_split. Additive/conservative: only splits on a clearly distinct
ASD-active interloper face; on-camera only (off-camera segments untouched).

Input:  run_dir/meta/<chunk>_segments_gapfilled.json
        run_dir/meta/faces.json        (per-track {asd,emb,t0,t1,sz})
        run_dir/meta/<chunk>_words.json (word timings, for gap snapping)
Output: overwrites gapfilled in-place (FTB_NO_INPLACE=1 -> *_ftbsplit.json only).
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

SPEAK       = float(os.environ.get("FTB_SPEAK", "0.5"))       # min ASD to be "speaking"
INTRUDER_ASD = float(os.environ.get("FTB_INTRUDER_ASD", "1.5"))  # interloper must speak strongly
FACE_CLUS_COS = float(os.environ.get("FTB_FACE_CLUS_COS", "0.45"))  # same-person cluster
DISTINCT_COS = float(os.environ.get("FTB_DISTINCT_COS", "0.20"))   # interloper face distinct from owner
MIN_SUB     = float(os.environ.get("FTB_MIN_SUB", "0.30"))   # min sub-segment dur
WORD_GAP    = float(os.environ.get("FTB_WORD_GAP", "0.25"))  # min word gap to snap a cut to
DEBUG       = bool(os.environ.get("FTB_DEBUG"))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _cluster_tracks(tracks):
    """Greedy ArcFace clusters (largest face first), return list of cluster dicts."""
    for t in tracks:
        t["emb"] = np.asarray(t["emb"], dtype=float)
    spk = [t for t in tracks if t.get("asd", 0) >= SPEAK]
    cl = []
    for it in sorted(range(len(spk)), key=lambda i: -spk[i].get("sz", 0)):
        best = None
        for ci, c in enumerate(cl):
            m = np.mean([cos(spk[it]["emb"], spk[j]["emb"]) for j in c])
            if m >= FACE_CLUS_COS and (best is None or m > best[1]):
                best = (ci, m)
        cl[best[0]].append(it) if best else cl.append([it])
    clusters = []
    for i, c in enumerate(cl):
        clusters.append({
            "id": i,
            "emb": np.mean([spk[j]["emb"] for j in c], axis=0),
            "spans": [(spk[j]["t0"], spk[j]["t1"], spk[j].get("asd", 0)) for j in c],
        })
    return clusters


def _snap(t, words, lo, hi):
    """Snap cut time t to the nearest ASR word gap >= WORD_GAP within (lo,hi)."""
    gaps = []
    ws = sorted([w for w in words if lo <= w.get("start", 0) <= hi or lo <= w.get("end", 0) <= hi],
                key=lambda w: w.get("start", 0))
    for a, b in zip(ws, ws[1:]):
        g0, g1 = a.get("end", 0), b.get("start", 0)
        if g1 - g0 >= WORD_GAP:
            gaps.append((g0 + g1) / 2.0)
    if not gaps:
        return t
    return min(gaps, key=lambda g: abs(g - t))


def _text_in(words, s, e):
    return " ".join(w.get("word", w.get("text", "")).strip()
                    for w in words if w.get("start", 0) >= s - 0.05 and w.get("end", 0) <= e + 0.05).strip()


def split_groups(groups, clusters, words):
    out = []
    n_split = 0
    for g in groups:
        gs = float(g.get("group_start", g.get("start", 0)))
        ge = float(g.get("group_end", g.get("end", 0)))
        spk = g.get("speaker", "")
        if ge - gs < 2 * MIN_SUB or "BG" in spk:
            out.append(g); continue
        # dominant face cluster of this segment
        dom = None
        for c in clusters:
            o = sum(ov(gs, ge, a, b) for a, b, _ in c["spans"])
            if o > 0 and (dom is None or o > dom[0]):
                dom = (o, c)
        if dom is None:
            out.append(g); continue
        dom_emb = dom[1]["emb"]
        # find a distinct interloper cluster speaking strongly over a sub-range
        intr = None
        for c in clusters:
            if c["id"] == dom[1]["id"]:
                continue
            if cos(c["emb"], dom_emb) >= DISTINCT_COS:
                continue  # not a distinct person
            for a, b, asd in c["spans"]:
                s = max(gs, a); e = min(ge, b)
                if e - s >= MIN_SUB and asd >= INTRUDER_ASD:
                    if intr is None or (e - s) > (intr[1] - intr[0]):
                        intr = (s, e)
        if intr is None:
            out.append(g); continue
        # snap cut points to word gaps
        c0 = _snap(intr[0], words, gs, ge)
        c1 = _snap(intr[1], words, gs, ge)
        c0 = min(max(c0, gs), ge); c1 = min(max(c1, gs), ge)
        cuts = sorted(set(round(x, 3) for x in [gs, c0, c1, ge] if gs <= x <= ge))
        if len(cuts) < 3:
            out.append(g); continue
        pieces = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= MIN_SUB]
        if len(pieces) < 2:
            out.append(g); continue
        for i, (s, e) in enumerate(pieces):
            ng = dict(g)
            ng["group_start"] = s; ng["group_end"] = e
            ng["start"] = s; ng["end"] = e
            ng["text"] = _text_in(words, s, e)
            ng["speaker"] = spk  # keep owner; face_identity_split relabels the interloper piece
            out.append(ng)
        n_split += 1
        if DEBUG:
            print(f"    [ftb-split] {spk} [{gs:.2f}-{ge:.2f}] -> {len(pieces)} pieces "
                  f"(intruder {intr[0]:.2f}-{intr[1]:.2f} -> cut {c0:.2f},{c1:.2f})")
    return out, n_split


def main(run_dir):
    meta = os.path.join(run_dir, "meta")
    faces_p = os.path.join(meta, "faces.json")
    if not os.path.exists(faces_p):
        print("[skip] no meta/faces.json"); return
    tracks = json.load(open(faces_p))
    for gp in sorted(glob.glob(os.path.join(meta, "*_segments_gapfilled.json"))):
        chunk = os.path.basename(gp).replace("_segments_gapfilled.json", "")
        wp = os.path.join(meta, chunk + "_words.json")
        words = (json.load(open(wp)).get("words", []) if os.path.exists(wp) else [])
        data = json.load(open(gp)); groups = data["groups"]
        clusters = _cluster_tracks([dict(t) for t in tracks])
        new_groups, n = split_groups(groups, clusters, words)
        for i, g in enumerate(new_groups):
            g["group_idx"] = i
        print(f"  [{chunk}] face-track boundary split: {len(groups)} -> {len(new_groups)} groups ({n} segments split)")
        json.dump({"groups": new_groups}, open(os.path.join(meta, chunk + "_segments_ftbsplit.json"), "w"),
                  ensure_ascii=False, indent=2)
        if os.environ.get("FTB_NO_INPLACE", "").strip() not in ("1", "true", "True"):
            json.dump({"groups": new_groups}, open(gp, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); a = ap.parse_args()
    main(a.run_dir)
