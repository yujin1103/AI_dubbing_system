"""Cross-gender on-camera interjection detector (2026-05-31).

Recovers a distinct speaker who interjects in another speaker's segment when
voice-embedding clustering cannot (no clean reference / fully overlapped), using
the ROBUST F0(pitch)-gender signal + visual ASD face. test4-safe (0 false flags;
verified test4 6/6 unchanged). Catches test6 woman "Hey"(347Hz) inside the male
monologue; test5 female interjections in the doctor's segment.

Gate (common thresholds, NO per-video hardcoding):
  1. owner = confident single-gender speaker (robust F0, voiced>=OWNER_VOICED)
  2. word  = confidently OPPOSITE gender, pitch-STABLE (opp F0 range, std<=STD_MAX)
  3. a per-frame ASD-active face (score>=ASD_TH) at a position distinct from owner
Pitch-stability rejects pyin octave-doubling glitches from loud same-gender speech.

Input : run_dir (gapfilled segments + *_words.json + vocals + ASD cache pkl)
Output: meta/<chunk>_segments_genderintj.json  (additive: 0 flags => identical)
"""
from __future__ import annotations
import argparse, glob, json, os, pickle
from collections import defaultdict
import numpy as np, soundfile as sf, librosa

CACHE_DIR = "/workspace/media/cache/lightasd"
F_BOUND   = float(os.environ.get("GI_F_BOUND", "185"))
MALE_MAX  = float(os.environ.get("GI_MALE_MAX", "150"))
FEM_MIN   = float(os.environ.get("GI_FEM_MIN", "240"))
OWNER_VOICED = int(os.environ.get("GI_OWNER_VOICED", "80"))
WORD_VOICED  = int(os.environ.get("GI_WORD_VOICED", "8"))
STD_MAX   = float(os.environ.get("GI_STD_MAX", "14"))
OCT_FRAC  = float(os.environ.get("GI_OCT_FRAC", "0.15"))  # reject wf within ±15% of 2*ownerF0
F0_CEIL   = float(os.environ.get("GI_F0_CEIL", "420"))    # reject wf above this as falsetto/super-female (genuine female median tops ~380 measured, ~329 in clean speech; 420 leaves wide margin)
F0_CEIL_PYIN = float(os.environ.get("GI_F0_CEIL_PYIN", "520"))  # word-pass pyin fmax so a falsetto >400 is actually REPORTED (else clipped at 400 and F0_CEIL never fires)
ASD_TH    = float(os.environ.get("GI_ASD_TH", "0.3"))
CX_MARGIN = float(os.environ.get("GI_CX_MARGIN", "200"))
MAXWORD   = float(os.environ.get("GI_MAXWORD", "0.7"))
GROUP_GAP = float(os.environ.get("GI_GROUP_GAP", "5.0"))
PAD       = float(os.environ.get("GI_PAD", "0.15"))


def f0stats(y, sr, t0, t1, pad=0.03, fmax=400.0):
    # fmax default 400 for the OWNER pass (keeps octave noise out of owner gender).
    # The WORD pass passes fmax=F0_CEIL_PYIN so pyin can actually REPORT a falsetto
    # F0 above 400 — otherwise pyin clips at 400 and the F0_CEIL veto never fires.
    a = y[max(0, int((t0 - pad) * sr)):int((t1 + pad) * sr)]
    if len(a) < int(0.05 * sr):
        return None
    f0, _, _ = librosa.pyin(a, fmin=70, fmax=fmax, sr=sr, frame_length=1024, hop_length=256)
    v = f0[~np.isnan(f0)]
    return (float(np.median(v)), len(v), float(np.std(v))) if len(v) else (None, 0, 0)


def match_cache(vocals_path):
    y, sr = sf.read(vocals_path); y = y.mean(1) if y.ndim > 1 else y
    dur = len(y) / sr; best = None
    for p in glob.glob(os.path.join(CACHE_DIR, "*.pkl")):
        try:
            d = pickle.load(open(p, "rb"))
            diff = abs(d.get("n_frames", 0) / d.get("fps", 25.0) - dur)
            if best is None or diff < best[1]:
                best = (p, diff)
        except Exception:
            continue
    return best[0] if best and best[1] <= 3.0 else None


def frame_index(tracks, fps):
    idx = defaultdict(list)
    for t in tracks:
        fr = np.array(t["frames"], float); sc = np.array(t["scores"], float); bb = np.array(t["bboxes"], float)
        n = min(len(fr), len(sc), len(bb))
        for i in range(n):
            idx[int(fr[i])].append(((bb[i, 0] + bb[i, 2]) / 2, sc[i]))
    return idx


def gender(f0):
    return "F" if f0 >= F_BOUND else "M"


def main(run_dir):
    rd = run_dir; meta = os.path.join(rd, "meta")
    gaps = sorted(glob.glob(os.path.join(meta, "*_segments_gapfilled.json")))
    if not gaps:
        raise SystemExit("no gapfilled segments")
    for gp in gaps:
        chunk = os.path.basename(gp).replace("_segments_gapfilled.json", "")
        words_p = os.path.join(meta, chunk + "_words.json")
        voc = glob.glob(os.path.join(rd, "vocals", chunk + "*clean_vocals.wav"))
        if not (os.path.exists(words_p) and voc):
            print("[skip] " + chunk + ": missing words/vocals"); continue
        cache = match_cache(voc[0])
        if cache is None:
            print("[skip] " + chunk + ": no ASD cache"); continue
        y, sr = sf.read(voc[0]); y = (y.mean(1) if y.ndim > 1 else y).astype(np.float32)
        seg = json.load(open(gp))["groups"]
        d = pickle.load(open(cache, "rb")); fps = d["fps"]; fidx = frame_index(d["tracks"], fps)
        acc = defaultdict(list)
        for g in seg:
            if g["group_end"] - g["group_start"] < 0.6:
                continue
            r = f0stats(y, sr, g["group_start"], g["group_end"])
            if r and r[0]:
                acc[g["speaker"]].append(r)
        spk_f0 = {k: float(np.median([m for m, _, _ in l])) for k, l in acc.items()}
        spk_nv = {k: sum(n for _, n, _ in l) for k, l in acc.items()}

        def owner_at(t):
            for g in seg:
                if g["group_start"] <= t <= g["group_end"]:
                    return g["speaker"]
            return None

        def owner_cx(X):
            xs = [cx for g in seg if g["speaker"] == X
                  for f in range(int(g["group_start"] * fps), int(g["group_end"] * fps) + 1)
                  for (cx, sc) in fidx.get(f, []) if sc >= ASD_TH]
            return float(np.median(xs)) if xs else None

        W = json.load(open(words_p)); W = W.get("words", W)
        flags = []
        for w in W:
            t0, t1, tok = w.get("start"), w.get("end"), w.get("word", "")
            if t0 is None or t1 is None or t1 - t0 <= 0 or t1 - t0 > MAXWORD:
                continue
            X = owner_at((t0 + t1) / 2); of0 = spk_f0.get(X)
            if of0 is None or spk_nv.get(X, 0) < OWNER_VOICED:
                continue
            # confident-MALE owner only (test4-safe: avoids F0-misestimated owners
            # like a male whose median F0 reads high from octave/overlap noise).
            if of0 >= MALE_MAX:
                continue
            # word pass uses a raised fmax so a real falsetto (>400Hz) is reported,
            # not clipped — otherwise the F0_CEIL veto below can never see it.
            r = f0stats(y, sr, t0, t1, fmax=F0_CEIL_PYIN)
            if r is None or r[0] is None or r[1] < WORD_VOICED or r[2] > STD_MAX:
                continue
            wf = r[0]; wg = gender(wf)
            # word must be confidently FEMALE (opposite of the male owner)
            if wg != "F" or wf < FEM_MIN:
                continue
            # falsetto/super-female ceiling: a male PANIC SCREAM (falsetto) reads as
            # very high F0 (e.g. 496Hz). Genuine female speech medians top ~380Hz, so
            # wf > F0_CEIL(420) is almost certainly falsetto, not a real woman. This is
            # belt-and-suspenders: the real protection is the distinct-position face
            # gate below (a screamer shows their OWN face at their OWN position, so it
            # fails the |cx-owner_cx|>=CX_MARGIN test). Acoustics alone CANNOT separate
            # a female scream from a male falsetto scream (verified: both 360-490Hz).
            if wf > F0_CEIL:
                continue
            # reject pyin octave-doubling of a LOUD male voice (wf ~ 2*owner F0):
            # urgent/shouted male speech doubles to a stable female-looking pitch.
            # A genuine high female (e.g. "Hey!" 347 vs 2*132=264) is far from 2x.
            if abs(wf - 2.0 * of0) <= OCT_FRAC * 2.0 * of0:
                continue
            # gate3 (distinct-position face): the interjector's ASD-active face must be
            # at a position DIFFERENT from the owner's. A falsetto-screaming owner shows
            # their OWN face at their OWN position (cx ~ owner_cx), so no qualifying face
            # is found and fcx stays None -> rejected. This structurally vetoes self-
            # screams (measured: falsetto cx=1011 vs owner cx=972, dist 39 < CX_MARGIN).
            oc = owner_cx(X); fcx = None; fsc = -9
            for f in range(int(t0 * fps), int(t1 * fps) + 1):
                for (cx, sc) in fidx.get(f, []):
                    if sc >= ASD_TH and (oc is None or abs(cx - oc) >= CX_MARGIN) and sc > fsc:
                        fsc, fcx = sc, cx
            if fcx is None:
                continue
            flags.append({"s": t0, "e": t1, "tok": tok, "owner": X, "of0": of0,
                          "wf0": wf, "g": wg, "cx": fcx})
        flags.sort(key=lambda x: x["s"])
        groups = []
        for fl in flags:
            if groups and fl["g"] == groups[-1][-1]["g"] and fl["s"] - groups[-1][-1]["e"] <= GROUP_GAP:
                groups[-1].append(fl)
            else:
                groups.append([fl])
        existing = sorted({g["speaker"] for g in seg if g["speaker"].startswith("SPEAKER_")
                           and "BG" not in g["speaker"] and "__" not in g["speaker"]})
        nextn = max([int(s.split("_")[1]) for s in existing if s.split("_")[1].isdigit()] + [-1]) + 1
        out = [dict(g) for g in seg]
        for grp in groups:
            label = "SPEAKER_%02d" % nextn; nextn += 1
            # carve only the TIGHT flagged word intervals (merge adjacent gap<=0.5s),
            # leaving the owner's intervening words intact.
            ivs = []
            for fl in sorted(grp, key=lambda x: x["s"]):
                if ivs and fl["s"] - ivs[-1][1] <= 0.5:
                    ivs[-1][1] = fl["e"]; ivs[-1][2].append(fl["tok"])
                else:
                    ivs.append([fl["s"], fl["e"], [fl["tok"]]])
            for (iv0, iv1, toks) in ivs:
                rebuilt = []
                for s in out:
                    if s["speaker"] == grp[0]["owner"] and s["group_start"] <= iv0 and s["group_end"] >= iv1:
                        a0, a1 = s["group_start"], s["group_end"]
                        ws = max(a0, iv0 - PAD); we = min(a1, iv1 + PAD)
                        if ws - a0 >= 0.1:
                            s2 = dict(s); s2["group_end"] = ws; rebuilt.append(s2)
                        rebuilt.append({"group_start": ws, "group_end": we, "speaker": label,
                                        "text": " ".join(toks), "from_gender_interjection": True})
                        if a1 - we >= 0.1:
                            s3 = dict(s); s3["group_start"] = we; rebuilt.append(s3)
                    else:
                        rebuilt.append(s)
                out = rebuilt
            print("  [%s] interjection %s owner=%s(F0=%.0f) wF0=%.0f -> %s  intervals=%s" % (
                chunk, grp[0]["g"], grp[0]["owner"], grp[0]["of0"],
                float(np.median([f["wf0"] for f in grp])), label,
                [("%.2f-%.2f" % (a, b)) for a, b, _ in ivs]))
        out.sort(key=lambda x: x["group_start"])
        spks = sorted({s["speaker"] for s in out})
        print("  [%s] flags=%d groups=%d | speakers: %s" % (chunk, len(flags), len(groups), spks))
        # debug artifact
        outp = os.path.join(meta, chunk + "_segments_genderintj.json")
        json.dump({"groups": out}, open(outp, "w"), ensure_ascii=False, indent=2)
        # overwrite the canonical gapfilled in-place so downstream picks it up.
        # Additive + idempotent: 0 flags => byte-identical; re-runs find the new
        # speaker is no longer a confident-male owner, so they do not re-split.
        if os.environ.get("GI_NO_INPLACE", "").strip() not in ("1", "true", "True"):
            json.dump({"groups": out}, open(gp, "w"), ensure_ascii=False, indent=2)
            print("  updated canonical -> " + gp)
        else:
            print("  saved -> " + outp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); a = ap.parse_args()
    main(a.run_dir)
