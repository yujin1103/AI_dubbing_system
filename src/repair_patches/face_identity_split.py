"""Face-identity catch-all decomposition (port of verified strategy S2, 2026-05-31).

When voice clustering collapses two speakers who overlap in time into one
"catch-all" cluster (e.g. test5 mom+dad), their FACES stay distinct even though
their voices merge. This patch uses per-track ArcFace face identity (faces.json)
to (a) split off a strongly-ASD distinct speaking face as a new speaker, (b) merge
word-level over-split artifacts back to their base, and (c) return catch-all
boundary fragments to their true face owner. Additive; test4 6/6 preserved.

A face cluster is only minted as a NEW speaker if its face is distinct from the
local voice owner AND a voice-probe veto confirms the audio spoken during its
spans does NOT already belong to an existing CLEAN (non-catch-all) speaker
(reverse-shot guard — fixes test6 over-split where the answerer's own face
cluster would otherwise be minted).

Common thresholds (no per-video hardcoding). Input:
  run_dir/meta/<chunk>_segments_gapfilled.json   (voice diarization)
  run_dir/meta/faces.json                        (per-track {asd,emb,t0,t1,sz})
  run_dir/vocals/<chunk>*clean_vocals.wav        (for voice-probe veto)
Output: overwrites gapfilled in-place (FID_NO_INPLACE=1 -> *_faceid.json only).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from collections import defaultdict, Counter
import numpy as np

SPEAK         = float(os.environ.get("FID_SPEAK", "0.5"))
FACE_CLUS_COS = float(os.environ.get("FID_FACE_CLUS_COS", "0.45"))
V1_ASD        = float(os.environ.get("FID_V1_ASD", "1.5"))
SPLIT_TH      = float(os.environ.get("FID_SPLIT_TH", "0.35"))
RETURN_COS    = float(os.environ.get("FID_RETURN_COS", "0.50"))
FRAG_DUR      = float(os.environ.get("FID_FRAG_DUR", "1.2"))
CATCH_OTH     = float(os.environ.get("FID_CATCH_OTH", "0.50"))
STRONG_RETURN = float(os.environ.get("FID_STRONG_RETURN", "0.80"))
COHES         = float(os.environ.get("FID_COHES", "0.55"))      # min cluster cohesion to mint
VOICE_SAME    = float(os.environ.get("FID_VOICE_SAME", "0.60"))  # face's spoken voice >= this to an existing speaker -> reverse-shot, do NOT mint
# Strong-face override of the voice veto: an UNAMBIGUOUSLY different face (cos far
# below SPLIT_TH) speaking with strong ASD IS a new person, even if its short/noisy
# voice clip mis-probes as an existing clean speaker. Fixes the test4 paramedic
# ("that's where we're going", 0.64s, face cos 0.09 vs Sean 0.70) that the voice
# veto wrongly suppressed when the catch-all structure differs. Reverse-shots
# (same person, high face cos) never satisfy this, so test6 over-split stays vetoed.
STRONG_SPLIT  = float(os.environ.get("FID_STRONG_SPLIT_COS", "0.20"))
STRONG_ASD    = float(os.environ.get("FID_STRONG_OVERRIDE_ASD", "2.0"))
# strong-face override only for SHORT clusters: a short clip's voice probe is
# unreliable (paramedic 1.0s), so face evidence overrides. A LONG distinct-face
# cluster whose voice confidently matches a CLEAN speaker is that speaker's own
# reverse-shot (test4 man1 10.6s, voice 0.90→SPK00) — respect the veto, don't mint.
STRONG_MAX_DUR = float(os.environ.get("FID_STRONG_MAX_DUR", "2.5"))
DEBUG         = bool(os.environ.get("FID_DEBUG"))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _build_voice_probe(vocals_path, voice):
    """Return fn(spans)->(sim, speaker) over clean voice-speaker centroids, or None
    if audio/eres2 unavailable. Used to veto minting a face cluster whose spoken
    voice clearly IS an existing speaker (reverse-shot face fragment)."""
    if not vocals_path or not os.path.exists(vocals_path):
        return None
    # Force CPU: the GPU is held by the running daemons; a CUDA eres2 load there
    # returns garbage embeddings (silently disabling the veto). This patch runs as
    # its own subprocess, so hard-setting CVD is safe. CPU embedding is fine here.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import soundfile as sf
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "daemons"))
        from eres2netv2_helper import get_eres2netv2_model, extract_eres2netv2_emb
        get_eres2netv2_model()
        y, sr = sf.read(vocals_path)
        y = (y.mean(1) if getattr(y, "ndim", 1) > 1 else y).astype("float32")
    except Exception:
        if DEBUG:
            import traceback; traceback.print_exc()
        return None

    def vemb(s, e, pad=0.1):
        a = y[max(0, int((s - pad) * sr)):int((e + pad) * sr)]
        if len(a) < int(0.25 * sr):
            return None
        v = extract_eres2netv2_emb(a, sr=sr)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    ce = defaultdict(list)
    for g in voice:
        if "BG" in g["spk"] or g["e"] - g["s"] < 0.6:
            continue
        e = vemb(g["s"], g["e"])
        if e is not None:
            ce[g["spk"]].append(e)
    cent = {k: (lambda c: c / (np.linalg.norm(c) + 1e-9))(np.mean(np.stack(v), 0))
            for k, v in ce.items() if v}
    if not cent:
        return None

    def probe(spans):
        embs = [vemb(a, b) for a, b in spans]
        embs = [e for e in embs if e is not None]
        if not embs:
            return None
        m = np.mean(np.stack(embs), 0); m = m / (np.linalg.norm(m) + 1e-9)
        return max(((float(np.dot(m, c)), k) for k, c in cent.items()), default=(0.0, None))
    return probe


def decompose(voice, tracks, voice_probe=None):
    """voice: list of {s,e,spk}; tracks: list of {asd,emb,t0,t1,sz}. Sets v['lab'].
    voice_probe(spans)->(sim,speaker): vetoes minting a face whose voice IS an existing speaker."""
    for t in tracks:
        t["emb"] = np.asarray(t["emb"], dtype=float)
    tracks = [t for t in tracks if t["asd"] >= SPEAK]

    # face identity clusters (greedy, largest face first)
    cl = []
    for it in sorted(range(len(tracks)), key=lambda i: -tracks[i]["sz"]):
        best = None
        for ci, c in enumerate(cl):
            m = np.mean([cos(tracks[it]["emb"], tracks[j]["emb"]) for j in c])
            if m >= FACE_CLUS_COS and (best is None or m > best[1]):
                best = (ci, m)
        cl[best[0]].append(it) if best else cl.append([it])

    def _cohesion(members, cent):
        if len(members) < 2:
            return 1.0
        return float(np.mean([cos(m, cent) for m in members]))

    fclus = []
    for i, c in enumerate(cl):
        cent = np.mean([tracks[j]["emb"] for j in c], axis=0)
        fclus.append({"emb": cent,
                      "spans": [(tracks[j]["t0"], tracks[j]["t1"]) for j in c],
                      "asd": np.mean([tracks[j]["asd"] for j in c]),
                      "dur": sum(tracks[j]["t1"] - tracks[j]["t0"] for j in c),
                      "cohes": _cohesion([tracks[j]["emb"] for j in c], cent),
                      "id": i})

    def is_bg(s):
        return "BG" in s

    def base_label(s):
        return s.split("__")[0]

    def is_wordsplit(s):
        return "__" in s

    spks = set(v["spk"] for v in voice)
    spk_face = {}
    for spk in spks:
        segs = [v for v in voice if v["spk"] == spk]; best = None
        for c in fclus:
            o = sum(ov(a, b, v["s"], v["e"]) for a, b in c["spans"] for v in segs)
            if o > 0 and (best is None or o > best[0]):
                best = (o, c["emb"])
        spk_face[spk] = best[1] if best else None

    def seg_face_id(s, e):
        best = None
        for c in fclus:
            o = sum(ov(s, e, a, b) for a, b in c["spans"])
            if o > 0 and (best is None or o > best[0]):
                best = (o, c["id"])
        return best[1] if best else None

    # --- pre-compute catch-all (raw voice speaker whose speaking faces are mostly
    # owned by OTHERS) BEFORE minting, so the voice-veto can distinguish:
    #   * span-voice matches a CLEAN speaker  -> reverse-shot fragment, VETO mint
    #   * span-voice matches the CATCH-ALL    -> a real person hidden in the merge,
    #                                            ALLOW mint (test5 mom/dad)
    _face_owner = {}
    for c in fclus:
        best = None
        for spk in spks:
            o = sum(ov(a, b, v["s"], v["e"]) for a, b in c["spans"] for v in voice if v["spk"] == spk)
            if o > 0 and (best is None or o > best[0]):
                best = (o, spk)
        _face_owner[c["id"]] = best[1] if best else None

    def _oth_ratio(spk):
        selfd = otherd = noned = 0.0
        for v in voice:
            if v["spk"] != spk:
                continue
            dur = v["e"] - v["s"]; fid = seg_face_id(v["s"], v["e"])
            if fid is None:
                noned += dur
            elif _face_owner.get(fid) == spk:
                selfd += dur
            else:
                otherd += dur
        return otherd / max(selfd + otherd + noned, 1e-6), otherd, selfd

    catchall = None; _best_oth = CATCH_OTH
    for s in [x for x in spks if not is_bg(x) and "__" not in x]:
        r, od, sd = _oth_ratio(s)
        if r >= _best_oth and od > sd:
            _best_oth = r; catchall = s

    # v1 additive: strong-ASD distinct face -> new FACE_ speaker
    extra = []
    for c in fclus:
        if c["asd"] < V1_ASD or c["dur"] < 0.4:
            continue
        if c["cohes"] < COHES:   # conflated/noisy cluster -> not a single identity, skip
            continue
        vb = None
        for spk in spks:
            o = sum(ov(a, b, v["s"], v["e"]) for a, b in c["spans"] for v in voice if v["spk"] == spk)
            if o > 0 and (vb is None or o > vb[0]):
                vb = (o, spk)
        vf = spk_face.get(vb[1]) if vb else None
        if not (vf is None or cos(c["emb"], vf) < SPLIT_TH):
            continue
        # voice veto: if the audio spoken during this face's spans clearly matches a
        # CLEAN (non-catch-all) existing speaker, this is that speaker's own face
        # (reverse-shot / cluster fragmentation), NOT a new person -> veto.
        # If it matches the CATCH-ALL, it's a distinct person merged into that
        # catch-all by voice overlap -> allow mint (test5 mom/dad). The matched-
        # speaker identity, not the raw similarity, is what discriminates
        # (test6 사회자=0.88→SPEAKER_01 clean=veto;  test5 엄마=0.88→SPEAKER_02 catch-all=mint).
        vp = voice_probe(c["spans"]) if voice_probe is not None else None
        # strong distinct face + strong ASD overrides the voice veto (see STRONG_* note)
        _strong_face = (vf is not None and cos(c["emb"], vf) < STRONG_SPLIT and c["asd"] >= STRONG_ASD
                        and c["dur"] < STRONG_MAX_DUR)
        if DEBUG:
            print(f"    [mint-cand] cluster{c['id']} asd={c['asd']:.1f} dur={c['dur']:.1f} "
                  f"cohes={c['cohes']:.2f} facecos={cos(c['emb'], vf) if vf is not None else None} "
                  f"strong_face={_strong_face} voiceprobe={vp} catchall={catchall}")
        if vp is not None and vp[0] >= VOICE_SAME and vp[1] != catchall and not _strong_face:
            continue
        extra.append(c)

    # v3 additive (2026-06-03): recover the DOMINANT face of a multi-face voice catch-all.
    # gate-220 skips a cluster whose face == its voice-speaker's representative face (cos~1.0).
    # Correct for a CLEAN single speaker (man1: his face IS his own), but WRONG when the
    # voice-speaker is a CATCH-ALL holding several distinct people: the catch-all's repr face
    # is just its DOMINANT member (e.g. sean), so gate-220 wrongly skips that member while
    # minting the others (paramedic, etc). Signal vb is a catch-all = >=1 OTHER distinct face
    # already minted under the same vb. Then the skipped dominant face is also a distinct
    # person -> mint it. Clean speakers (no other face minted under their vb) are untouched.
    if os.environ.get("FID_CATCHALL_DOMFACE_MINT", "1") not in ("0", "false", "False"):
        def _vb_of(cc):
            best = None
            for spk in spks:
                o = sum(ov(a, b, v["s"], v["e"]) for a, b in cc["spans"] for v in voice if v["spk"] == spk)
                if o > 0 and (best is None or o > best[0]):
                    best = (o, spk)
            return best[1] if best else None
        _minted_vb = Counter(_vb_of(c) for c in extra)
        _extra_ids = {c["id"] for c in extra}
        for c in fclus:
            if c["id"] in _extra_ids or c["asd"] < V1_ASD or c["dur"] < 0.4 or c["cohes"] < COHES:
                continue
            vbc = _vb_of(c)
            vfc = spk_face.get(vbc)
            # require >=2 OTHER distinct faces already minted under vb = STRONG multi-person
            # catch-all (>=3 people: dominant + 2). minted_under=1 is ambiguous (reverse-shot
            # pair) → don't mint (avoids test6 GT-less over-split). test4 SPEAKER_03=2 → minted.
            _min_others = int(os.environ.get("FID_CATCHALL_MIN_OTHERS", "2"))
            if vfc is not None and cos(c["emb"], vfc) >= SPLIT_TH and _minted_vb.get(vbc, 0) >= _min_others:
                if DEBUG:
                    print("    [catchall-domface] mint c%d (vb=%s catch-all, minted_under=%d)" % (
                        c["id"], vbc, _minted_vb.get(vbc, 0)))
                extra.append(c)

    def dom_extra(s, e):
        best = None
        for c in extra:
            o = sum(ov(s, e, a, b) for a, b in c["spans"])
            if o > 0.3 * (e - s) and (best is None or o > best[0]):
                best = (o, c["id"])
        return best[1] if best else None

    def seg_face_emb(s, e):
        best = None
        for c in fclus:
            o = sum(ov(s, e, a, b) for a, b in c["spans"])
            if o > 0 and (best is None or o > best[0]):
                best = (o, c["emb"])
        return best[1] if best else None

    for v in voice:
        fe = dom_extra(v["s"], v["e"])
        v["lab"] = f"FACE_{fe}" if fe is not None else v["spk"]

    # (a) word-split merge
    for ws in [s for s in spks if is_wordsplit(s)]:
        base = base_label(ws); bemb = spk_face.get(base)
        for v in voice:
            if v["spk"] != ws or v["lab"].startswith("FACE_"):
                continue
            if bemb is None:
                v["lab"] = base; continue
            fe = seg_face_emb(v["s"], v["e"])
            if fe is None or cos(fe, bemb) >= FACE_CLUS_COS:
                v["lab"] = base

    # catch-all already computed before minting (uses raw voice labels). Mom/dad/
    # paramedic mints land under their own FACE_ labels; the catch-all speaker label
    # is unchanged by minting, so reuse `catchall` from above.
    cur_spks = set(v["lab"] for v in voice)

    return_targets = {}
    for s in cur_spks:
        if s == catchall or is_bg(s):
            continue
        if s.startswith("FACE_"):
            return_targets[s] = fclus[int(s.split("_")[1])]["emb"]
        elif spk_face.get(s) is not None:
            return_targets[s] = spk_face[s]

    # (b) boundary-fragment return
    if catchall is not None and return_targets:
        for v in voice:
            if v["lab"] != catchall:
                continue
            fe = seg_face_emb(v["s"], v["e"])
            if fe is None:
                continue
            bestc = None
            for s, emb in return_targets.items():
                c = cos(fe, emb)
                if bestc is None or c > bestc[0]:
                    bestc = (c, s)
            if bestc is None:
                continue
            cval, s = bestc
            short = (v["e"] - v["s"]) <= FRAG_DUR
            if cval >= STRONG_RETURN or (short and cval >= RETURN_COS):
                v["lab"] = s
    return catchall, sorted({v["lab"] for v in voice})


def main(run_dir):
    meta = os.path.join(run_dir, "meta")
    faces_p = os.path.join(meta, "faces.json")
    if not os.path.exists(faces_p):
        print("[skip] no meta/faces.json — face_clustering must emit it"); return
    tracks = json.load(open(faces_p))
    for gp in sorted(glob.glob(os.path.join(meta, "*_segments_gapfilled.json"))):
        data = json.load(open(gp)); groups = data["groups"]
        voice = [{"s": float(g.get("group_start", g.get("start", 0))),
                  "e": float(g.get("group_end", g.get("end", 0))),
                  "spk": g["speaker"], "_g": g} for g in groups]
        chunk = os.path.basename(gp).replace("_segments_gapfilled.json", "")
        vocals = (glob.glob(os.path.join(run_dir, "vocals", chunk + "*clean_vocals.wav"))
                  or glob.glob(os.path.join(run_dir, "vocals", "*clean_vocals.wav")))
        voice_probe = _build_voice_probe(vocals[0] if vocals else None, voice)
        if voice_probe is None:
            print("  [warn] voice-probe unavailable — minting without voice veto")
        catchall, finalspk = decompose(voice, [dict(t) for t in tracks], voice_probe=voice_probe)
        for v in voice:
            v["_g"]["speaker"] = v["lab"]
        print(f"  [{chunk}] catch-all={catchall} -> speakers: {finalspk}")
        json.dump({"groups": groups}, open(os.path.join(meta, chunk + "_segments_faceid.json"), "w"),
                  ensure_ascii=False, indent=2)
        if os.environ.get("FID_NO_INPLACE", "").strip() not in ("1", "true", "True"):
            json.dump({"groups": groups}, open(gp, "w"), ensure_ascii=False, indent=2)
            print(f"  [{chunk}] updated canonical -> {gp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); a = ap.parse_args()
    main(a.run_dir)
