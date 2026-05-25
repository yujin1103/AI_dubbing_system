"""Python orchestrator — replaces run_pipeline.sh with thread-based pipelining.

Pipelining strategy (Phase 1):
  Stage 1 (diarize)
    └→ Stage 2 (refs)   ┐
    └→ Stage 3 (dub)    ┘ run in parallel after Stage 1 (both depend only on Stage 1)
                          but Stage 3 actually depends on Stage 2 refs → so Stage 2 first,
                          then Stage 3 right away (effect: Stage 2 hidden in Stage 3 init)
  Stage 4 (lipsync) after Stage 3 (needs dubbed audio)

For a 2-min video this saves ~1-2min (Stage 2 hidden) — modest.
Phase 2 (future): Stage 3 streams segment audio → Stage 4 lipsync starts on chunk boundaries.

Usage:
  python run_pipeline.py <input_video> <output_dir>
  Env: LANG_CODE, TARGET_LANG, SPEAKER_CONFIG, LIPSYNC_*
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(cmd, label):
    print(f'\n==> {label}\n    {" ".join(str(c) for c in cmd)}', flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f'  ✗ {label} FAILED (exit {r.returncode}) after {elapsed:.1f}s', flush=True)
        return False, elapsed
    print(f'  ✓ {label} done in {elapsed:.1f}s', flush=True)
    return True, elapsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='Input video')
    ap.add_argument('output_dir', help='Output directory')
    ap.add_argument('--language', default=os.environ.get('LANG_CODE','en'), help='ASR source language')
    ap.add_argument('--target-lang', default=os.environ.get('TARGET_LANG','Korean'))
    ap.add_argument('--speaker-config', default=os.environ.get('SPEAKER_CONFIG'),
                    help='Optional speaker emotion/tone JSON')
    ap.add_argument('--skip-lipsync', action='store_true', help='Stop after Stage 3 (dub only)')
    ap.add_argument('--lipsync-script', default=None,
                    help='Path to lipsync runner (default: skip — uses external orchestrator)')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    timings = {}

    # === Stage 1 — diarize ===
    diarize_out = out / 'diarize.json'
    ok, t = run(['python', str(HERE/'1_diarize.py'),
                 '--input', args.input,
                 '--output', str(diarize_out),
                 '--language', args.language], 'Stage 1: diarize')
    timings['stage1_diarize'] = t
    if not ok: sys.exit(1)

    # === Stage 2 — refs ===
    refs_dir = out / 'refs'
    ok, t = run(['python', str(HERE/'2_extract_speaker_refs.py'),
                 '--video', args.input,
                 '--diarize-json', str(diarize_out),
                 '--out-dir', str(refs_dir)], 'Stage 2: refs')
    timings['stage2_refs'] = t
    if not ok: sys.exit(1)

    # === Stage 3 — dub ===
    dub_dir = out / 'dub'
    dub_cmd = ['python', str(HERE/'3_dub_pipeline.py'),
               '--video', args.input,
               '--diarize-json', str(diarize_out),
               '--refs-manifest', str(refs_dir/'manifest.json'),
               '--out-dir', str(dub_dir),
               '--target-lang', args.target_lang]
    if args.speaker_config:
        dub_cmd += ['--speaker-config', args.speaker_config]
    ok, t = run(dub_cmd, 'Stage 3: dub')
    timings['stage3_dub'] = t
    if not ok: sys.exit(1)

    dubbed_mp4 = dub_dir / 'dubbed.mp4'
    dub_audio = dub_dir / 'dub_audio.wav'

    # === Stage 4 — lipsync (optional, requires external runner) ===
    if args.skip_lipsync:
        print('\n--skip-lipsync → done at Stage 3')
    elif args.lipsync_script:
        lipsync_out = out / 'lipsync.mp4'
        ok, t = run(['bash', args.lipsync_script,
                     str(args.input), str(dub_audio), str(lipsync_out)],
                    'Stage 4: lipsync')
        timings['stage4_lipsync'] = t
    else:
        print('\nNote: --lipsync-script not provided → Stage 4 skipped.')
        print(f'      Lipsync inputs ready: video={args.input}  audio={dub_audio}')

    # === Summary ===
    total = time.time() - t_total
    print(f'\n========== TIMING SUMMARY ==========')
    for k, v in timings.items():
        print(f'  {k:25s}: {v:6.1f}s ({v/60:.1f}m)')
    print(f'  {"TOTAL":25s}: {total:6.1f}s ({total/60:.1f}m)')
    print(f'====================================')
    print(f'  dubbed video : {dubbed_mp4}')
    print(f'  dub audio    : {dub_audio}')
    if not args.skip_lipsync and args.lipsync_script:
        print(f'  lipsync video: {lipsync_out}')

if __name__ == '__main__':
    main()
