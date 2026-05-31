#!/usr/bin/env python3
"""기존 합성 클립을 재활용해 빠르게 re-mix — 배경음 bed(연속) + 합성(ducking) + BG 원본.
재합성 없이(~10초) 볼륨/배경 결합만 조정해 들어보기 위함."""
import argparse, json, subprocess, os
import numpy as np, soundfile as sf, scipy.signal as sps

SR = 44100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dub-dir", default="dub_bgtest")
    ap.add_argument("--dub-json", required=True)
    ap.add_argument("--bg-stem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg-gain", type=float, default=0.5)
    ap.add_argument("--synth-gain", type=float, default=0.75)
    ap.add_argument("--orig-gain", type=float, default=0.7)
    ap.add_argument("--duck", type=float, default=0.5)
    a = ap.parse_args()

    RD = a.run; dub = os.path.join(RD, a.dub_dir)
    video = os.path.join(RD, "chunks", os.path.basename(RD).split("_")[3] if False else "")
    # video 경로는 chunks 안 mp4
    import glob
    video = glob.glob(os.path.join(RD, "chunks", "*chunk_000.mp4"))[0]
    vdur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                                 "-of","default=nk=1:nw=1",video],capture_output=True,text=True).stdout)
    N = int(vdur * SR)

    def load(p):
        x, sr = sf.read(p)
        if x.ndim > 1: x = np.mean(x, axis=1)
        if sr != SR: x = sps.resample_poly(x, SR, sr)
        return x.astype(np.float32)

    # (1) 배경 bed
    bed = np.zeros(N, dtype=np.float32)
    b = load(a.bg_stem); m = min(N, len(b)); bed[:m] = b[:m] * a.bg_gain

    # (2) 합성 overlay (corrected_start 위치)
    tl = json.load(open(os.path.join(dub, "translations.json")))["translations"]
    synth = np.zeros(N, dtype=np.float32); mask = np.zeros(N, dtype=np.float32)
    placed = 0
    for t in tl:
        clip = os.path.join(dub, f"seg_{t['idx']:02d}_{t['speaker']}.wav")
        if not os.path.exists(clip): continue
        c = load(clip); s = int(float(t["corrected_start"]) * SR); e = min(s + len(c), N)
        synth[s:e] += c[:e-s] * a.synth_gain; mask[s:e] = 1.0; placed += 1

    # (3) ducking
    if a.duck < 1.0:
        w = max(1, int(0.05*SR)); sm = np.convolve(mask, np.ones(w)/w, mode="same")
        bed = bed * (1.0 - (1.0 - a.duck) * np.clip(sm, 0, 1))
    comp = bed + synth

    # (4) BG passthrough → 원본
    segs = json.load(open(a.dub_json))["segments"]
    bg = [(s["start"], s["end"]) for s in segs if s["speaker"].startswith("SPEAKER_BG")]
    if bg:
        orig_wav = os.path.join(dub, "_orig_full44.wav")
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",video,"-vn","-ac","1","-ar",str(SR),orig_wav],check=True)
        oa = load(orig_wav)  # 원본 full
        for bs, be in bg:
            s = int(bs*SR); e = min(int(be*SR), N, len(oa))
            if e > s: comp[s:e] = oa[s:e] * a.orig_gain

    pk = float(np.max(np.abs(comp))+1e-9)
    if pk > 0.95: comp *= 0.95/pk
    wav = os.path.join(dub, "_remix.wav"); sf.write(wav, comp, SR)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",video,"-i",wav,
                    "-c:v","copy","-c:a","aac","-b:a","192k","-map","0:v:0","-map","1:a:0","-shortest",a.out],check=True)
    print(f"[remix] {placed} 합성클립 + bed x{a.bg_gain} + duck {a.duck} + {len(bg)} BG x{a.orig_gain}")
    print(f"  → {a.out}")

if __name__ == "__main__":
    main()
