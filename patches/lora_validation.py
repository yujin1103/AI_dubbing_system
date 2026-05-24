"""LoRA validation on AIHub VS11 dataset — smart sample selection.

For each selected video:
  1. Read JSON label → get sentence-level timestamps
  2. Cut 8-10 second chunk from a sentence (audio + video aligned)
  3. Run lipsync with multiple LoRA scales (base, 0.5, 0.7, 1.0)
  4. Apply mouth_only_enhance v3 (TRT)
  5. Save side-by-side comparison + metrics

Usage (inside dubbing_pipeline container):
  python /workspace/patches/lora_validation_v2.py [--n-samples 5] [--n-sentences 2]

Output:
  /workspace/media/aihub_validation/results/
    <video_name>/
      sentence_<id>/
        original.mp4              (original 10s clip with mouth + audio)
        audio.wav                 (extracted audio)
        base.mp4                  (LoRA off, mouth_enhance v3)
        lora_0.5.mp4
        lora_0.7.mp4
        lora_1.0.mp4
        comparison.mp4            (4-up grid for visual comparison)
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


VIDEO_DIR = "/workspace/media/aihub_validation/videos/VS11"
LABEL_DIR = "/workspace/media/aihub_validation/labels/VL11"
RESULT_DIR = "/workspace/media/aihub_validation/results"

LATENTSYNC_CONFIG = "/opt/LatentSync/configs/unet/stage2_512_nf16.yaml"
BASE_CKPT = "/opt/LatentSync/checkpoints/latentsync_unet.pt"

LORA_CHECKPOINT_DIR = "/workspace/media/training_outputs/lora_nf4_train/train-2026_05_13-01:59:54/checkpoints"
TRT_ENGINE = "/workspace/trt_work/engines/unet_fp16.trt"

# LoRA scales to test
LORA_SCALES = {
    "base": None,        # no LoRA
    "lora_0.5": 0.5,
    "lora_0.7": 0.7,
    "lora_1.0": 1.0,
}


def _log(msg):
    print(f"[validate] {msg}", flush=True)


def find_label_for_video(video_path):
    """Find matching JSON label for a video file."""
    name = Path(video_path).stem
    # Search recursively in label dir
    for root, _, files in os.walk(LABEL_DIR):
        for f in files:
            if f == f"{name}.json":
                return os.path.join(root, f)
    return None


def pick_sentences(label_path, n=2, min_duration=6.0, max_duration=12.0):
    """From a video's label, pick N sentences with duration in [min,max]."""
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    sentences = data.get("Sentence_info", [])
    candidates = []
    for s in sentences:
        dur = s.get("end_time", 0) - s.get("start_time", 0)
        if min_duration <= dur <= max_duration:
            candidates.append(s)
    if len(candidates) == 0:
        # Fallback to first 2 sentences regardless of duration
        candidates = sentences[:n]
    random.seed(42)
    return random.sample(candidates, min(n, len(candidates)))


def cut_chunk(video_path, start_sec, end_sec, output_path):
    """Cut a chunk from video using ffmpeg (re-encode for accurate cut)."""
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec), "-i", video_path,
        "-t", str(duration),
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    return subprocess.run(cmd, check=True)


def extract_audio(video_path, audio_out, sample_rate=16000):
    """Extract mono 16kHz wav for LipSync input."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
        audio_out,
    ]
    return subprocess.run(cmd, check=True)


def run_lipsync(video_path, audio_path, output_path, lora_scale=None,
                lora_ckpt=None):
    """Run LatentSync inference with optional LoRA."""
    env = os.environ.copy()
    env["LATENTSYNC_USE_TRT"] = "1"
    env["LATENTSYNC_TRT_ENGINE"] = TRT_ENGINE
    env["LATENTSYNC_SCHEDULER"] = "dpm"
    env["LATENTSYNC_TEACACHE"] = "0.1"
    env["LATENTSYNC_CHUNK_SECONDS"] = "10"
    env["LATENTSYNC_VAE_CHUNK"] = "4"
    env["LATENTSYNC_USE_NVENC"] = "1"

    if lora_scale is not None and lora_ckpt:
        env["LATENTSYNC_LORA_CHECKPOINT"] = lora_ckpt
        env["LATENTSYNC_LORA_SCALE"] = str(lora_scale)

    cmd = [
        "/opt/venv_lipsync/bin/python", "-m", "scripts.inference",
        "--unet_config_path", LATENTSYNC_CONFIG,
        "--inference_ckpt_path", BASE_CKPT,
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--inference_steps", "10",
        "--guidance_scale", "1.5",
        "--seed", "1247",
    ]
    return subprocess.run(cmd, cwd="/opt/LatentSync", env=env,
                          capture_output=True, text=True)


def run_mouth_enhance(lipsync_path, original_path, output_path):
    """Apply mouth_only_enhance v3 (TRT)."""
    cmd = [
        "/opt/venv_gfpgan/bin/python",
        "/workspace/patches/mouth_only_enhance.py",
        "--lipsync", lipsync_path,
        "--original", original_path,
        "--output", output_path,
        "--mux-audio-from", original_path,
        "--color-match",
        "--temporal-smooth", "5",
        "--face-diag-min-ratio", "0.10",
        "--mask-erode-px", "2",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def make_comparison(sentence_dir):
    """Create 2x2 grid comparison video (base | lora_0.5 / lora_0.7 | lora_1.0)."""
    parts = [
        os.path.join(sentence_dir, f"{name}_enhanced.mp4")
        for name in ["base", "lora_0.5", "lora_0.7", "lora_1.0"]
    ]
    if not all(os.path.isfile(p) for p in parts):
        _log(f"skip comparison — missing files in {sentence_dir}")
        return None

    out = os.path.join(sentence_dir, "comparison.mp4")
    # 2x2 grid via ffmpeg complex filter
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", parts[0], "-i", parts[1], "-i", parts[2], "-i", parts[3],
        "-filter_complex",
        "[0:v]scale=960:540,drawtext=text='base':x=10:y=10:fontcolor=white:fontsize=24[v0];"
        "[1:v]scale=960:540,drawtext=text='lora 0.5':x=10:y=10:fontcolor=white:fontsize=24[v1];"
        "[2:v]scale=960:540,drawtext=text='lora 0.7':x=10:y=10:fontcolor=white:fontsize=24[v2];"
        "[3:v]scale=960:540,drawtext=text='lora 1.0':x=10:y=10:fontcolor=white:fontsize=24[v3];"
        "[v0][v1]hstack=inputs=2[top];"
        "[v2][v3]hstack=inputs=2[bot];"
        "[top][bot]vstack=inputs=2[grid]",
        "-map", "[grid]", "-map", "0:a",
        "-c:v", "h264_nvenc", "-preset", "p4",
        "-c:a", "aac", "-b:a", "192k",
        out,
    ]
    subprocess.run(cmd, check=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of videos to validate")
    parser.add_argument("--n-sentences", type=int, default=2,
                        help="Sentences per video")
    parser.add_argument("--lora-step", type=int, default=50000,
                        help="LoRA checkpoint step to use")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip mouth_enhance (use raw lipsync output)")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip 2x2 grid comparison rendering")
    args = parser.parse_args()

    # Resolve LoRA checkpoint
    lora_ckpt = os.path.join(LORA_CHECKPOINT_DIR, f"checkpoint-{args.lora_step}.pt")
    if not os.path.isfile(lora_ckpt):
        _log(f"WARN: lora ckpt not found: {lora_ckpt}")
        _log("Available checkpoints:")
        for f in sorted(os.listdir(LORA_CHECKPOINT_DIR)):
            _log(f"  {f}")
        _log(f"Continuing without LoRA validation (base only)")
        # Use latest available
        available = sorted([f for f in os.listdir(LORA_CHECKPOINT_DIR)
                           if f.startswith("checkpoint-") and f.endswith(".pt")])
        if available:
            lora_ckpt = os.path.join(LORA_CHECKPOINT_DIR, available[-1])
            _log(f"Falling back to {available[-1]}")
        else:
            lora_ckpt = None

    # Find sample videos
    videos = sorted(Path(VIDEO_DIR).rglob("*.mp4"))
    _log(f"Found {len(videos)} videos in {VIDEO_DIR}")
    random.seed(42)
    samples = random.sample(videos, min(args.n_samples, len(videos)))
    _log(f"Selected {len(samples)} for validation")
    for s in samples:
        _log(f"  • {s.name}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    total_start = time.time()

    for video_path in samples:
        name = video_path.stem
        _log(f"\n{'='*60}")
        _log(f"Video: {name}")
        _log(f"{'='*60}")

        label_path = find_label_for_video(str(video_path))
        if not label_path:
            _log(f"  no label — skip")
            continue

        try:
            sentences = pick_sentences(label_path, n=args.n_sentences)
        except Exception as e:
            _log(f"  label parse failed: {e}")
            continue

        video_out = Path(RESULT_DIR) / name
        video_out.mkdir(exist_ok=True)

        for sentence in sentences:
            sid = sentence["ID"]
            sentence_dir = video_out / f"sentence_{sid:03d}"
            sentence_dir.mkdir(exist_ok=True)
            start = max(0, sentence["start_time"] - 0.2)
            end = sentence["end_time"] + 0.2
            text = sentence.get("sentence_text", "")
            _log(f"\n  Sentence {sid}: [{start:.1f}-{end:.1f}s] \"{text[:50]}...\"")

            # 1. Cut chunk
            chunk_path = str(sentence_dir / "original.mp4")
            try:
                cut_chunk(str(video_path), start, end, chunk_path)
            except Exception as e:
                _log(f"    cut failed: {e} — skip")
                continue

            # 2. Extract audio
            audio_path = str(sentence_dir / "audio.wav")
            try:
                extract_audio(chunk_path, audio_path)
            except Exception as e:
                _log(f"    audio extract failed: {e} — skip")
                continue

            # 3. Run lipsync for each scale
            for variant, scale in LORA_SCALES.items():
                output_raw = str(sentence_dir / f"{variant}_raw.mp4")
                output_final = str(sentence_dir / f"{variant}_enhanced.mp4")
                if os.path.isfile(output_final):
                    _log(f"    {variant}: cached ✓")
                    continue

                t0 = time.time()
                _log(f"    {variant}: lipsync ...")
                use_ckpt = lora_ckpt if (scale is not None and lora_ckpt) else None
                r = run_lipsync(chunk_path, audio_path, output_raw,
                                lora_scale=scale, lora_ckpt=use_ckpt)
                if r.returncode != 0 or not os.path.isfile(output_raw):
                    _log(f"      FAIL: {r.stderr[-200:] if r.stderr else 'no output'}")
                    continue
                _log(f"      lipsync done in {time.time()-t0:.1f}s")

                # 4. Mouth enhance
                if not args.no_enhance:
                    t0 = time.time()
                    r2 = run_mouth_enhance(output_raw, chunk_path, output_final)
                    if r2.returncode == 0 and os.path.isfile(output_final):
                        _log(f"      enhance done in {time.time()-t0:.1f}s")
                    else:
                        _log(f"      enhance FAIL")
                else:
                    os.rename(output_raw, output_final)

            # 5. Build comparison grid
            if not args.no_comparison:
                try:
                    make_comparison(sentence_dir)
                    _log(f"    comparison grid: {sentence_dir}/comparison.mp4")
                except Exception as e:
                    _log(f"    comparison failed: {e}")

    elapsed = time.time() - total_start
    _log(f"\n{'='*60}")
    _log(f"DONE in {elapsed/60:.1f} min")
    _log(f"Results: {RESULT_DIR}")
    _log(f"{'='*60}")


if __name__ == "__main__":
    main()
