"""Stage 2 — Extract per-speaker clean reference clips for voice cloning.

Input:
  - Original video/audio
  - Diarization JSON from 1_diarize.py
Output:
  - Per-speaker WAV refs (5-10s each)
  - manifest.json with ref paths

Selection rules (avoid contamination):
  - Prefer longest single-speaker segment
  - Reject segments overlapping (within ±0.3s) other speaker's segment
  - RMS normalize per-clip to consistent loudness (-23 LUFS-ish target via peak norm)
  - Optionally loop-augment short refs (< 3s) for CosyVoice stability
"""
import argparse, json, os, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

TARGET_REF_MIN_S = 5.0   # prefer ≥5s
TARGET_REF_MAX_S = 12.0  # but ≤12s
GAP_GUARD_S = 0.3        # buffer between other-speaker segs
PEAK_NORM_DB = -3.0      # target peak -3dB

def extract_audio(video_path, out_wav, sr=24000):
    subprocess.run(['ffmpeg','-y','-i',str(video_path),'-vn','-ac','1','-ar',str(sr),
                   str(out_wav)], check=True, capture_output=True)

def is_clean(seg, all_segs):
    """seg has no other-speaker segment within ±GAP_GUARD_S window."""
    s, e = seg['start'], seg['end']
    for o in all_segs:
        if o is seg or o.get('speaker') == seg['speaker']: continue
        os_, oe = o['start'], o['end']
        if oe + GAP_GUARD_S < s or os_ - GAP_GUARD_S > e: continue
        return False
    return True

def pick_ref(segs_for_spk, all_segs):
    """Pick best ref slice: longest segment (clean 우선이되, 깨끗한 게 너무 짧으면 긴 것 채택).
    경계는 main 에서 트림하므로 긴 turn 을 쓰는 게 클론 품질에 유리."""
    def score(s):
        d = s['end'] - s['start']
        if d < TARGET_REF_MIN_S: return d           # 짧으면 길수록 좋음
        if d > TARGET_REF_MAX_S: return TARGET_REF_MAX_S - (d - TARGET_REF_MAX_S) * 0.01
        return d
    clean = [s for s in segs_for_spk if is_clean(s, all_segs)]
    best_clean = max(clean, key=score) if clean else None
    best_any = max(segs_for_spk, key=score)
    # 깨끗한 게 충분히 길면(>=3s) 그걸, 아니면 가장 긴 turn(트림으로 경계오염 완화)
    if best_clean is not None and (best_clean['end'] - best_clean['start']) >= 3.0:
        return best_clean
    return best_any

def normalize_peak(audio, target_db=PEAK_NORM_DB):
    peak = float(np.max(np.abs(audio)) + 1e-9)
    target_lin = 10 ** (target_db / 20)
    if peak > 0:
        audio = audio * (target_lin / peak)
    return audio.astype(np.float32)

def loop_augment(audio, sr, min_s=3.0):
    """If too short, loop with 0.1s gap until ≥ min_s."""
    if len(audio)/sr >= min_s: return audio
    gap = np.zeros(int(0.1*sr), dtype=audio.dtype)
    out = [audio]
    while sum(len(x) for x in out)/sr < min_s:
        out.append(gap); out.append(audio)
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='Input video/audio')
    ap.add_argument('--diarize-json', required=True, help='Output from 1_diarize.py')
    ap.add_argument('--out-dir', required=True, help='Where to write speaker refs')
    ap.add_argument('--sr', type=int, default=24000, help='Output sample rate (CosyVoice uses 24kHz)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.diarize_json) as f: data = json.load(f)
    # 'segments'(dub입력) 또는 'groups'(gapfilled 화자분리) 둘 다 지원.
    # ref 는 긴 화자 turn 이 좋으므로 gapfilled(groups) 직접 사용 권장.
    if isinstance(data, dict) and data.get('segments'):
        segs = data['segments']
    else:
        groups = data.get('groups', data if isinstance(data, list) else [])
        segs = [{'speaker': g.get('speaker', 'SPEAKER_00'),
                 'start': float(g.get('group_start', g.get('start', 0))),
                 'end': float(g.get('group_end', g.get('end', 0))),
                 'text': g.get('text', '')} for g in groups]

    # Extract audio at target SR
    audio_wav = out_dir / '_full_audio.wav'
    if not audio_wav.exists():
        extract_audio(args.video, audio_wav, sr=args.sr)
    audio, sr = sf.read(audio_wav)
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    print(f'Loaded {len(audio)/sr:.2f}s @ {sr}Hz')

    # Group by speaker
    by_spk = {}
    for s in segs:
        by_spk.setdefault(s['speaker'], []).append(s)

    refs = []
    for spk, ss in sorted(by_spk.items()):
        pick = pick_ref(ss, segs)
        s, e = pick['start'], pick['end']
        # 경계 오염(인접 화자 bleed) 완화: 1s 초과 segment 는 양끝 0.2s 트림
        if e - s > 1.0:
            s += 0.2; e -= 0.2
        # Trim to TARGET_REF_MAX_S if too long
        if e - s > TARGET_REF_MAX_S:
            e = s + TARGET_REF_MAX_S
        seg_audio = audio[int(s*sr):int(e*sr)].astype(np.float32)
        seg_audio = normalize_peak(seg_audio)
        seg_audio = loop_augment(seg_audio, sr)
        out_path = out_dir / f'{spk}.wav'
        sf.write(out_path, seg_audio, sr)
        refs.append({'speaker': spk, 'ref_path': str(out_path),
                    'src_start': s, 'src_end': e,
                    'duration': len(seg_audio)/sr,
                    'source_text': pick.get('text','')})
        print(f'  {spk}: {s:.2f}-{e:.2f}s ({(e-s):.2f}s) → {out_path.name}')

    manifest = out_dir / 'manifest.json'
    with open(manifest, 'w') as f:
        json.dump({'refs': refs, 'sr': sr}, f, indent=2, ensure_ascii=False)
    print(f'\n✓ {len(refs)} refs + {manifest}')

if __name__ == '__main__':
    main()
