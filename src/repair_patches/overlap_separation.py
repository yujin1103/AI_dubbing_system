"""Overlap separation recovery (2026-06-02) — opt-in post-step.

The base pipeline routes unrecoverable/overlapped speech to SPEAKER_BG (kept as
original audio). MOST such regions are genuinely unrecoverable (e.g. two people
shouting the SAME word "Adam" simultaneously — separation fails, stays BG). But
when an overlap hides TWO DIFFERENT utterances, MossFormer2 can separate them so
each can be dubbed as its own speaker — reducing reliance on BG.

This module is a TARGETED, CONDITIONAL post-step: it does NOT touch the existing
diarization/separation. It only revisits BG segments, runs MossFormer2, and:
  * keeps BG  if the separated streams are NOT distinct (same content / failed
    separation — the test5 "Adam" worst case), OR
  * recovers  if the streams are distinct (different ASR text AND distinct ERes2
    embeddings), reassigning each stream's span to its nearest MAIN speaker.

Validated separately (finding_mossformer2_overlap_separation): MossFormer2 cleanly
splits a 아빠+션 mix at both embedding and ASR level; the test5 Adam shout fails
(streams cos 0.73) and correctly stays BG.

Runs in venv_asr (modelscope MossFormer2 + funasr). Opt-in via OVL_ENABLE=1.
Input:  run_dir/meta/<chunk>_segments_gapfilled.json, run_dir/vocals/<chunk>*clean_vocals.wav
Output: overwrites gapfilled in-place (OVL_NO_INPLACE=1 -> *_ovlsep.json only).
"""
from __future__ import annotations
import argparse, glob, json, os, sys, tempfile, urllib.request
from collections import defaultdict
import numpy as np

SR8 = 8000
MIN_DUR   = float(os.environ.get("OVL_MIN_DUR", "0.8"))      # only segments >= this
PAD       = float(os.environ.get("OVL_PAD", "0.1"))
DISTINCT  = float(os.environ.get("OVL_DISTINCT_COS", "0.45")) # streams must be < this to be 2 speakers
ASSIGN_MIN = float(os.environ.get("OVL_ASSIGN_MIN", "0.40"))  # min cos to assign a stream to a main spk
MIN_WORDS = int(os.environ.get("OVL_MIN_WORDS", "1"))
ASR_URL   = os.environ.get("ASR_DAEMON_URL", "http://127.0.0.1:8902")
DEBUG     = bool(os.environ.get("OVL_DEBUG"))

_sep = None
_eres = None


def _load_models():
    global _sep, _eres
    if _sep is None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        _sep = pipeline(Tasks.speech_separation,
                        model="damo/speech_mossformer2_separation_temporal_8k")
    if _eres is None:
        sys.path.insert(0, "/workspace/src/daemons")
        from eres2netv2_helper import get_eres2netv2_model, extract_eres2netv2_emb
        get_eres2netv2_model()
        _eres = extract_eres2netv2_emb
    return _sep, _eres


def _emb16(y16):
    v = _eres(y16.astype("float32"), sr=16000)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _asr(y16):
    import soundfile as sf
    tmp = tempfile.mktemp(suffix=".wav"); sf.write(tmp, y16, 16000)
    try:
        data = json.dumps({"audio_path": tmp, "language": "English"}).encode()
        req = urllib.request.Request(f"{ASR_URL}/transcribe", data=data,
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        words = r.get("words", [])
        return " ".join(w.get("word", w.get("text", "")) for w in words).strip(), len(words)
    except Exception:
        return "", 0
    finally:
        try: os.remove(tmp)
        except Exception: pass


def _main_centroids(groups, y16, sr):
    """ERes2 centroid per non-BG main speaker (from its clean spans)."""
    import librosa
    ce = defaultdict(list)
    for g in groups:
        spk = g["speaker"]
        if "BG" in spk or "FACE" in spk:
            pass  # FACE_/main both fine as targets; skip only BG below
        if "BG" in spk:
            continue
        s = float(g.get("group_start", g.get("start", 0))); e = float(g.get("group_end", g.get("end", 0)))
        if e - s < 0.6:
            continue
        a = y16[int(s * sr):int(e * sr)]
        if len(a) < int(0.4 * sr):
            continue
        ce[spk].append(_emb16(a))
    return {k: (lambda c: c / (np.linalg.norm(c) + 1e-9))(np.mean(np.stack(v), 0))
            for k, v in ce.items() if v}


def process(groups, vocals_path):
    import soundfile as sf, librosa
    sep, _ = _load_models()
    y16, sr = librosa.load(vocals_path, sr=16000, mono=True)
    cent = _main_centroids(groups, y16, sr)
    if DEBUG:
        print(f"    [ovl] main centroids: {list(cent.keys())}")

    def cos(a, b):
        return float(np.dot(a, b))

    out = []; n_recovered = 0; n_kept = 0
    for g in groups:
        spk = g["speaker"]
        s = float(g.get("group_start", g.get("start", 0)))
        e = float(g.get("group_end", g.get("end", 0)))
        if "BG" not in spk or e - s < MIN_DUR:
            out.append(g); continue
        # extract window @8k, separate
        a8 = librosa.resample(y16[max(0, int((s - PAD) * sr)):int((e + PAD) * sr)],
                              orig_sr=sr, target_sr=SR8)
        tmp = tempfile.mktemp(suffix=".wav"); sf.write(tmp, a8, SR8)
        try:
            res = sep(tmp)
            streams = [np.frombuffer(p, dtype=np.int16).astype(np.float32) / 32768.0
                       for p in res["output_pcm_list"]]
        except Exception as ex:
            if DEBUG: print(f"    [ovl] sep 실패 {s:.1f}-{e:.1f}: {ex}")
            out.append(g); n_kept += 1; continue
        finally:
            try: os.remove(tmp)
            except Exception: pass
        # upsample streams to 16k, embed + ASR
        s16 = [librosa.resample(x, orig_sr=SR8, target_sr=16000) for x in streams]
        embs = [_emb16(x) for x in s16]
        txts = [_asr(x) for x in s16]
        distinct_emb = len(embs) >= 2 and cos(embs[0], embs[1]) < DISTINCT
        txtset = [t for t, n in txts if n >= MIN_WORDS]
        distinct_txt = len(set(t.lower() for t in txtset)) >= 2
        if DEBUG:
            print(f"    [ovl] BG [{s:.1f}-{e:.1f}] sepcos={cos(embs[0],embs[1]):.2f} "
                  f"txts={[t for t,_ in txts]} distinct={distinct_emb and distinct_txt}")
        if not (distinct_emb and distinct_txt):
            out.append(g); n_kept += 1; continue
        # recover: assign each stream to nearest main centroid
        recovered = []
        for i, (txt, nw) in enumerate(txts):
            if nw < MIN_WORDS:
                continue
            best = max(((cos(embs[i], c), k) for k, c in cent.items()), default=(0.0, None))
            tgt = best[1] if best[0] >= ASSIGN_MIN else spk  # fall back to BG label if no match
            ng = dict(g); ng["speaker"] = tgt; ng["text"] = txt
            ng["overlap_recovered"] = True
            recovered.append((best[0], tgt, ng))
        if len([r for r in recovered if r[1] != spk]) >= 1:
            for _, _, ng in recovered:
                out.append(ng)
            n_recovered += 1
            if DEBUG:
                print(f"    [ovl] RECOVERED -> {[(round(c,2),k) for c,k,_ in recovered]}")
        else:
            out.append(g); n_kept += 1
    return out, n_recovered, n_kept


def main(run_dir):
    if os.environ.get("OVL_ENABLE", "0") != "1":
        print("[ovl] disabled (set OVL_ENABLE=1 to run overlap separation recovery)"); return
    meta = os.path.join(run_dir, "meta")
    for gp in sorted(glob.glob(os.path.join(meta, "*_segments_gapfilled.json"))):
        chunk = os.path.basename(gp).replace("_segments_gapfilled.json", "")
        vocals = (glob.glob(os.path.join(run_dir, "vocals", chunk + "*clean_vocals.wav"))
                  or glob.glob(os.path.join(run_dir, "vocals", "*clean_vocals.wav")))
        if not vocals:
            print(f"  [{chunk}] no vocals — skip"); continue
        data = json.load(open(gp)); groups = data["groups"]
        new, nrec, nkept = process(groups, vocals[0])
        for i, g in enumerate(new):
            g["group_idx"] = i
        print(f"  [{chunk}] overlap-sep: {nrec} recovered, {nkept} kept-BG ({len(groups)}->{len(new)} groups)")
        json.dump({"groups": new}, open(os.path.join(meta, chunk + "_segments_ovlsep.json"), "w"),
                  ensure_ascii=False, indent=2)
        if os.environ.get("OVL_NO_INPLACE", "").strip() not in ("1", "true", "True"):
            json.dump({"groups": new}, open(gp, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); a = ap.parse_args()
    main(a.run_dir)
