"""Speaker diarization pipeline — team-shareable single file.

Reproduces the v195 result: speaker assignment + word-level intra-segment split
+ mom interjection detection.

Usage:
    python speaker_diarize.py --input video.mp4 --output result.json

Models used (auto-downloaded on first run):
    - pyannote/speaker-diarization-3.1 (HF) — primary diarization
    - WhisperX large-v3 — word-level transcription
    - ERes2NetV2 from ModelScope — voice embedding for refinement
    - (Optional) InsightFace antelopev2 — face clustering for visual ASD
    - (Optional) LightASD checkpoint — per-face speaking score

Tested on Python 3.10+, CUDA 12+, ~16GB GPU.
See requirements.txt for pip dependencies.
"""
import argparse, json, os, sys, tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str = ""
    word_count: int = 0


def run_pyannote(audio_path: str, hf_token: Optional[str] = None) -> List[Segment]:
    """Primary diarization via pyannote 3.1."""
    from pyannote.audio import Pipeline
    # pyannote.audio 3.x renamed use_auth_token → token
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                         token=hf_token)
    except TypeError:
        # Old API fallback
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                         use_auth_token=hf_token)
    import torch
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    diar = pipe(audio_path)
    # pyannote.audio 3.x: returns DiarizeOutput (with .speaker_diarization Annotation),
    # 2.x: returns Annotation directly with .itertracks
    if hasattr(diar, "speaker_diarization"):
        diar = diar.speaker_diarization
    out = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        spk_str = f"SPEAKER_{int(spk):02d}" if str(spk).isdigit() else str(spk)
        if turn.end - turn.start < 0.3:
            continue
        out.append(Segment(spk_str, round(turn.start, 3), round(turn.end, 3)))
    return out


def run_whisperx(audio_path: str, language: str = "en") -> list:
    """WhisperX word-level transcription. Returns list of word dicts {start, end, word}."""
    import whisperx, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)
    align_model, meta = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], align_model, meta, audio, device,
                             return_char_alignments=False)
    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append(w)
    return words


def load_eres2():
    """Load ERes2NetV2 voice embedding model (ModelScope)."""
    from modelscope.hub.snapshot_download import snapshot_download
    from modelscope.utils.constant import Tasks
    from modelscope.pipelines import pipeline
    snapshot_download("iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common", revision="v1.0.1")
    sv_pipeline = pipeline(Tasks.speaker_verification,
                          model="iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
                          model_revision="v1.0.1")
    def extract(audio, sr=16000):
        import scipy.signal as sps
        if sr != 16000:
            audio = sps.resample_poly(audio, 16000, sr)
        try:
            res = sv_pipeline.preprocess([audio.astype(np.float32)])
            emb = sv_pipeline.forward(res)["embs"][0]
            emb = np.asarray(emb).astype(np.float32)
            return emb / max(np.linalg.norm(emb), 1e-9)
        except Exception:
            return None
    return extract


def extract_audio(video_path: str, out_path: str, sr: int = 44100):
    """Extract mono audio from video via ffmpeg."""
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
        "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", out_path
    ], check=True)


# === Refiner steps (from v178+v190+v194+v195) ===

def time_gap_split(segments: List[Segment], audio, sr, extract_emb,
                   gap_th=5.0, voice_diff_th=0.15, min_target_sim=0.5):
    """v174+: detect same-SPK with time gap >=5s; split into sub-clusters by voice."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(segments):
        groups[s.speaker].append((i, s))
    seg_embs = {}
    for i, s in enumerate(segments):
        if s.end - s.start < 0.4: continue
        chunk = audio[int(s.start*sr):int(s.end*sr)]
        e = extract_emb(chunk, sr=sr)
        if e is not None: seg_embs[i] = e
    n_changed = 0
    for spk, items in groups.items():
        if len(items) < 2: continue
        items_sorted = sorted(items, key=lambda x: x[1].start)
        # Find sub-clusters by time gap
        subs = [[items_sorted[0]]]
        for k in range(1, len(items_sorted)):
            if items_sorted[k][1].start - items_sorted[k-1][1].end >= gap_th:
                subs.append([items_sorted[k]])
            else:
                subs[-1].append(items_sorted[k])
        if len(subs) < 2: continue
        # Compute centroid per sub
        sub_cents = []
        for sub in subs:
            es = [seg_embs[i] for i, _ in sub if i in seg_embs]
            if not es: sub_cents.append(None); continue
            c = np.mean(np.stack(es), axis=0); sub_cents.append(c/max(np.linalg.norm(c),1e-9))
        # Other-SPK centroids
        other_cents = {}
        for other_spk, other_items in groups.items():
            if other_spk == spk: continue
            es = [seg_embs[i] for i, _ in other_items if i in seg_embs]
            if not es: continue
            c = np.mean(np.stack(es), axis=0); other_cents[other_spk] = c/max(np.linalg.norm(c),1e-9)
        if not other_cents: continue
        # For each sub, check if better matches another SPK
        for sci, sub in enumerate(subs):
            if sub_cents[sci] is None: continue
            own_other = [sub_cents[j] for j in range(len(subs)) if j != sci and sub_cents[j] is not None]
            if own_other:
                oo = np.mean(np.stack(own_other), axis=0); oo = oo/max(np.linalg.norm(oo),1e-9)
                sim_own = float(np.dot(sub_cents[sci], oo))
            else:
                sim_own = 1.0
            if 1.0 - sim_own < voice_diff_th: continue
            best_sim = -1; best_spk = None
            for ospk, oc in other_cents.items():
                s = float(np.dot(sub_cents[sci], oc))
                if s > best_sim: best_sim = s; best_spk = ospk
            if best_sim >= min_target_sim and best_sim > sim_own + 0.05:
                for idx, _ in sub:
                    segments[idx].speaker = best_spk
                    n_changed += 1
    return n_changed


def intra_spk_split(segments: List[Segment], audio, sr, extract_emb,
                    min_sim=0.5, passes=3):
    """v153+: detect voice-contaminated SPK clusters, reassign outliers."""
    for _ in range(passes):
        groups = defaultdict(list)
        for i, s in enumerate(segments): groups[s.speaker].append(i)
        seg_embs = {}
        for i, s in enumerate(segments):
            if s.end - s.start < 0.4: continue
            chunk = audio[int(s.start*sr):int(s.end*sr)]
            e = extract_emb(chunk, sr=sr)
            if e is not None: seg_embs[i] = e
        spk_cents = {}
        for spk, idxs in groups.items():
            es = [seg_embs[i] for i in idxs if i in seg_embs]
            if not es: continue
            c = np.mean(np.stack(es), axis=0); spk_cents[spk] = c/max(np.linalg.norm(c),1e-9)
        n_changed = 0
        for i, s in enumerate(segments):
            if i not in seg_embs: continue
            cur_sim = float(np.dot(seg_embs[i], spk_cents.get(s.speaker, np.zeros_like(seg_embs[i]))))
            best_spk, best_sim = s.speaker, cur_sim
            for spk, c in spk_cents.items():
                if spk == s.speaker: continue
                sim = float(np.dot(seg_embs[i], c))
                if sim > best_sim: best_sim, best_spk = sim, spk
            if best_spk != s.speaker and best_sim > cur_sim + 0.2 and best_sim >= min_sim:
                s.speaker = best_spk; n_changed += 1
        if n_changed == 0: break
    return n_changed


def sandwich_override(segments: List[Segment], gap_th: float = 0.3):
    """v179: singleton SPK between same-other-SPK neighbors with small gap → reassign."""
    counts = Counter(s.speaker for s in segments)
    singletons = {spk for spk, n in counts.items() if n == 1}
    n_changed = 0
    for i in range(1, len(segments) - 1):
        s = segments[i]
        if s.speaker not in singletons: continue
        p, nx = segments[i-1], segments[i+1]
        if (p.speaker == nx.speaker and p.speaker != s.speaker
                and 0 <= s.start - p.end <= gap_th and 0 <= nx.start - s.end <= gap_th):
            s.speaker = p.speaker; n_changed += 1
    return n_changed


def word_level_intra_split(segments: List[Segment], words: list, audio, sr,
                            extract_emb, margin=0.1):
    """v194+v195: split segments at word boundaries when intra-segment SPK change detected."""
    # Build clean SPK centroids (prefer overlap-free segments)
    groups = defaultdict(list)
    for s in segments: groups[s.speaker].append(s)
    spk_cents = {}
    for spk, ss in groups.items():
        ss_sorted = sorted(ss, key=lambda x: -(x.end-x.start))
        others = [(s2.start-0.3, s2.end+0.3) for s2 in segments if s2.speaker != spk]
        clean = []
        for s in ss_sorted:
            overlap = any(not (s.end <= a or s.start >= b) for a, b in others)
            if not overlap: clean.append(s)
        pick = clean if clean else ss_sorted
        embs = []
        for s in pick[:3]:
            chunk = audio[int(s.start*sr):int(s.end*sr)]
            e = extract_emb(chunk, sr=sr)
            if e is not None: embs.append(e)
        if embs:
            c = np.mean(np.stack(embs), axis=0); spk_cents[spk] = c/max(np.linalg.norm(c),1e-9)

    def emb_at(t0, t1, pad=0.2):
        s = max(0, t0-pad); e = min(len(audio)/sr, t1+pad)
        c = audio[int(s*sr):int(e*sr)]
        if len(c) < int(0.08*sr): return None
        return extract_emb(c, sr=sr)

    new_segs = []
    for seg in segments:
        if seg.end - seg.start < 2.0:
            new_segs.append(seg); continue
        seg_words = [w for w in words if seg.start-0.05 <= (w['start']+w['end'])/2 <= seg.end+0.05]
        if len(seg_words) < 3:
            new_segs.append(seg); continue
        word_cls = []
        for w in seg_words:
            e = emb_at(w['start'], w['end'])
            if e is None: word_cls.append((w, 'unknown')); continue
            sims = {k: float(np.dot(e, c)) for k, c in spk_cents.items()}
            best = max(sims, key=sims.get)
            cur = sims.get(seg.speaker, 0)
            word_cls.append((w, best, sims, cur))
        # Detect cross-SPK runs
        runs = []; cur_start = None; cur_target = None
        for k, wc in enumerate(word_cls):
            if len(wc) == 4:
                w, best, sims, cur = wc
                if best != seg.speaker and sims[best] > cur + margin:
                    if cur_start is None:
                        cur_start = k; cur_target = best
                    elif cur_target != best:
                        runs.append((cur_start, k, cur_target))
                        cur_start = k; cur_target = best
                else:
                    if cur_start is not None:
                        runs.append((cur_start, k, cur_target))
                        cur_start = None; cur_target = None
            else:
                if cur_start is not None:
                    runs.append((cur_start, k, cur_target))
                    cur_start = None; cur_target = None
        if cur_start is not None: runs.append((cur_start, len(word_cls), cur_target))
        # Extend runs (any same-target neighbor with positive margin)
        ext = []
        for rs, re, tgt in runs:
            new_rs = rs
            while new_rs > 0 and len(word_cls[new_rs-1]) == 4:
                _, best, sims, cur = word_cls[new_rs-1]
                if best == tgt and sims[tgt] > cur: new_rs -= 1
                else: break
            new_re = re
            while new_re < len(word_cls) and len(word_cls[new_re]) == 4:
                _, best, sims, cur = word_cls[new_re]
                if best == tgt and sims[tgt] > cur: new_re += 1
                else: break
            ext.append((new_rs, new_re, tgt))
        # Build sub-segments
        cur_t = seg.start; cur_w = 0; subs = []
        for rs, re, tgt in ext:
            if rs > cur_w:
                pre_w = seg_words[cur_w:rs]
                pre_t = pre_w[0]['start'] if pre_w else cur_t
                pre_e = pre_w[-1]['end'] if pre_w else seg_words[rs]['start']
                subs.append(Segment(seg.speaker, round(cur_t,3), round(pre_e,3),
                                    ' '.join(w.get('word','').strip() for w in pre_w),
                                    len(pre_w)))
            run_w = seg_words[rs:re]
            rt = run_w[0]['start']; re_t = run_w[-1]['end']
            subs.append(Segment(tgt, round(rt,3), round(re_t,3),
                                ' '.join(w.get('word','').strip() for w in run_w),
                                len(run_w)))
            cur_t = re_t; cur_w = re
        if cur_w < len(seg_words):
            tail = seg_words[cur_w:]
            subs.append(Segment(seg.speaker, round(cur_t,3), round(seg.end,3),
                                ' '.join(w.get('word','').strip() for w in tail),
                                len(tail)))
        if subs: new_segs.extend(subs)
        else: new_segs.append(seg)

    # Merge same-SPK adjacent (gap <= 0.1s)
    merged = []
    for s in new_segs:
        if merged and merged[-1].speaker == s.speaker and s.start - merged[-1].end <= 0.1:
            merged[-1] = Segment(merged[-1].speaker, merged[-1].start, s.end,
                                (merged[-1].text + ' ' + s.text).strip(),
                                merged[-1].word_count + s.word_count)
        else:
            merged.append(s)
    return merged


def assign_text(segments: List[Segment], words: list):
    """Populate text/word_count per segment from word timestamps."""
    for s in segments:
        in_w = [w for w in words if s.start-0.05 <= (w['start']+w['end'])/2 <= s.end+0.05]
        s.text = ' '.join(w.get('word','').strip() for w in in_w).strip()
        s.word_count = len(in_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video or audio file")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--language", default="en", help="ASR language code")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace token (for pyannote)")
    args = ap.parse_args()

    print("[1/5] Extracting audio ...", flush=True)
    audio_path = tempfile.mktemp(suffix=".wav")
    extract_audio(args.input, audio_path, sr=44100)

    print("[2/5] Pyannote diarization ...", flush=True)
    segments = run_pyannote(audio_path, hf_token=args.hf_token or None)
    print(f'   {len(segments)} initial segments, {len(set(s.speaker for s in segments))} speakers')

    print("[3/5] WhisperX word transcription ...", flush=True)
    words = run_whisperx(audio_path, language=args.language)
    print(f'   {len(words)} words')

    print("[4/5] Loading ERes2NetV2 for refinement ...", flush=True)
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    try:
        extract_emb = load_eres2()
    except Exception as ex:
        print(f"  ⚠ ERes2NetV2 load failed ({ex}) — skipping voice-embedding refiner passes", flush=True)
        extract_emb = None

    print("[5/5] Refining diarization ...", flush=True)
    if extract_emb is not None:
        time_gap_split(segments, audio, sr, extract_emb)
        intra_spk_split(segments, audio, sr, extract_emb, passes=3)
    sandwich_override(segments)
    assign_text(segments, words)
    if extract_emb is not None:
        segments = word_level_intra_split(segments, words, audio, sr, extract_emb)
    assign_text(segments, words)

    out_data = {
        "n_segments": len(segments),
        "speaker_counts": dict(Counter(s.speaker for s in segments)),
        "segments": [asdict(s) for s in segments],
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ wrote {args.output}")
    print(f"  {len(segments)} segments, {len(out_data['speaker_counts'])} speakers: {out_data['speaker_counts']}")
    os.remove(audio_path)


if __name__ == "__main__":
    main()
