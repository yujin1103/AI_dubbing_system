"""LoRA + Pipeline validation — auto report generation.

For each test video:
  1. Run pipeline with given phase config (env vars)
  2. Measure timing per stage
  3. Generate visual outputs (lipsync + enhance)
  4. Compute SyncNet score
  5. Write markdown report with thumbnails + side-by-side
  6. Aggregate into summary

Phases:
  Phase 0: Current baseline (no new patches)
  Phase 1: + Track-level person ID locking
  Phase 2: + SyncNet post-validation (auto-reject)
  Phase 3: + Overlap detection + skip
  Phase 4: + Multi-frame voting for profile

Output:
  /workspace/media/validation_reports/
    phase_<N>_<timestamp>/
      <video_name>/
        report.md            # per-video report
        timing.json          # stage timings
        metrics.json         # SyncNet, MOS
        output.mp4           # processed result
        thumbnail_*.jpg      # screenshot frames
      summary.md             # all-videos summary
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPORT_ROOT = "/workspace/media/validation_reports"

# Test video catalog — to be populated with actual paths
TEST_CATEGORIES = {
    "A_clean_korean": [
        # AIHub VS11 samples (sentence-level clips)
        # Will be auto-picked from /workspace/media/aihub_validation/results/
    ],
    "B_drama_frontal": [
        # Existing test4_chunk_000_final.mp4 etc.
    ],
    "C_drama_multi": [
        # Multi-speaker drama
    ],
    "D_profile_motion": [
        # Profile angles, fast motion
    ],
}

# Phase configurations — env vars per phase
PHASE_CONFIGS = {
    0: {
        "name": "baseline",
        "description": "현재 v3 + LoRA + 모든 strict env",
        "env": {
            "LATENTSYNC_USE_TRT": "1",
            "LATENTSYNC_USE_NVENC": "1",
            "LATENTSYNC_PROFILE_THRESHOLD": "0.30",
            "LATENTSYNC_PROFILE_MATCH_THRESHOLD": "0.55",
            "LATENTSYNC_PROFILE_STRICT_NONE": "1",
            "LATENTSYNC_FACE_STRICT": "1",
            "LATENTSYNC_FACE_DIAG_MIN_RATIO": "0.10",
        }
    },
    1: {
        "name": "track_locking",
        "description": "+ Track-level person ID locking",
        "env": {
            # All phase 0 envs PLUS:
            "LATENTSYNC_TRACK_LOCKING": "1",
        }
    },
    2: {
        "name": "syncnet_validation",
        "description": "+ SyncNet post-validation (auto-reject low score)",
        "env": {
            "LATENTSYNC_TRACK_LOCKING": "1",
            "LATENTSYNC_SYNCNET_VALIDATE": "1",
            "LATENTSYNC_SYNCNET_THRESHOLD": "3.0",
        }
    },
    3: {
        "name": "overlap_skip",
        "description": "+ Overlap detection + skip",
        "env": {
            "LATENTSYNC_TRACK_LOCKING": "1",
            "LATENTSYNC_SYNCNET_VALIDATE": "1",
            "LATENTSYNC_OVERLAP_SKIP": "1",
        }
    },
    4: {
        "name": "multi_frame_voting",
        "description": "+ Multi-frame voting for profile match (N=5)",
        "env": {
            "LATENTSYNC_TRACK_LOCKING": "1",
            "LATENTSYNC_SYNCNET_VALIDATE": "1",
            "LATENTSYNC_OVERLAP_SKIP": "1",
            "LATENTSYNC_PROFILE_VOTE_FRAMES": "5",
        }
    },
}


def _log(msg):
    print(f"[ValidReport] {msg}", flush=True)


def get_video_info(path):
    """Get duration, fps, resolution."""
    r = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json", path
    ], capture_output=True, text=True)
    data = json.loads(r.stdout)["streams"][0]
    fps_str = data["r_frame_rate"]
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) > 0 else 25.0
    return {
        "width": int(data["width"]),
        "height": int(data["height"]),
        "fps": fps,
        "duration": float(data.get("duration", 0)),
    }


def extract_thumbnails(video_path, out_dir, n=4):
    """Extract N evenly-spaced thumbnails."""
    os.makedirs(out_dir, exist_ok=True)
    info = get_video_info(video_path)
    dur = info["duration"]
    thumbs = []
    for i in range(n):
        t = (dur * (i + 0.5)) / n
        out = os.path.join(out_dir, f"thumb_{i+1:02d}.jpg")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(t), "-i", video_path,
            "-vframes", "1", "-q:v", "3",
            out
        ], check=True)
        thumbs.append(out)
    return thumbs


def compute_syncnet_score(video_path):
    """Compute SyncNet confidence score (placeholder)."""
    # TODO: Wire up to actual SyncNet eval script
    # /opt/LatentSync/stable_syncnet.pt available
    return None


def run_pipeline_with_phase(input_video, output_dir, phase_id, dubbed_audio=None):
    """Run pipeline for one video with one phase config."""
    config = PHASE_CONFIGS[phase_id]
    _log(f"running phase {phase_id} ({config['name']}) on {os.path.basename(input_video)}")

    env = os.environ.copy()
    env.update(config["env"])

    output_path = os.path.join(output_dir, "output.mp4")
    log_path = os.path.join(output_dir, "pipeline.log")

    # If dubbed_audio provided, use parallel_lipsync_orchestrator
    if dubbed_audio:
        cmd = [
            "/opt/venv_lipsync/bin/python",
            "/workspace/patches/parallel_lipsync_orchestrator.py",
            "--input", input_video,
            "--audio", dubbed_audio,
            "--output", output_path,
            "--chunk-seconds", "10",
        ]
    else:
        # Just run lipsync with the video's own audio
        # Extract audio first
        audio_path = os.path.join(output_dir, "audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_video, "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", audio_path
        ], check=True)
        cmd = [
            "/opt/venv_lipsync/bin/python",
            "/workspace/patches/parallel_lipsync_orchestrator.py",
            "--input", input_video,
            "--audio", audio_path,
            "--output", output_path,
            "--chunk-seconds", "10",
        ]

    t0 = time.time()
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                          text=True, timeout=3600)
    elapsed = time.time() - t0

    success = r.returncode == 0 and os.path.isfile(output_path)
    return {
        "success": success,
        "elapsed_seconds": elapsed,
        "exit_code": r.returncode,
        "output_path": output_path if success else None,
    }


def generate_video_report(video_path, phase_id, output_dir):
    """Generate per-video markdown report."""
    name = Path(video_path).stem
    info = get_video_info(video_path)

    # Run pipeline
    result = run_pipeline_with_phase(video_path, output_dir, phase_id)

    # Extract thumbnails from input + output
    in_thumbs = extract_thumbnails(video_path, os.path.join(output_dir, "in_thumbs"), n=4)
    out_thumbs = []
    if result["success"]:
        out_thumbs = extract_thumbnails(result["output_path"],
                                        os.path.join(output_dir, "out_thumbs"), n=4)
        sync_score = compute_syncnet_score(result["output_path"])
    else:
        sync_score = None

    # Write report.md
    report_path = os.path.join(output_dir, "report.md")
    config = PHASE_CONFIGS[phase_id]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {name} — Phase {phase_id} ({config['name']})\n\n")
        f.write(f"**Description**: {config['description']}\n\n")
        f.write(f"## Video Info\n\n")
        f.write(f"- Resolution: {info['width']}×{info['height']}\n")
        f.write(f"- FPS: {info['fps']:.2f}\n")
        f.write(f"- Duration: {info['duration']:.1f}s\n\n")
        f.write(f"## Pipeline Result\n\n")
        f.write(f"- **Success**: {'✅' if result['success'] else '❌'}\n")
        f.write(f"- **Wall time**: {result['elapsed_seconds']:.1f}s "
                f"({result['elapsed_seconds']/60:.2f} min)\n")
        f.write(f"- **Time per second of video**: "
                f"{result['elapsed_seconds']/max(info['duration'],1):.1f}× realtime\n")
        if sync_score is not None:
            f.write(f"- **SyncNet score**: {sync_score:.2f}\n")
        f.write(f"\n## Env vars applied\n\n```\n")
        for k, v in config["env"].items():
            f.write(f"{k}={v}\n")
        f.write("```\n\n")
        f.write(f"## Visual snapshots\n\n### Input\n\n")
        for t in in_thumbs:
            f.write(f"![{os.path.basename(t)}]({os.path.basename(t)})\n")
        if out_thumbs:
            f.write(f"\n### Output\n\n")
            for t in out_thumbs:
                f.write(f"![{os.path.basename(t)}]({os.path.basename(t)})\n")

    # Save raw metrics
    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump({
            "phase": phase_id,
            "phase_name": config["name"],
            "video": str(video_path),
            "video_duration_s": info["duration"],
            "wall_time_s": result["elapsed_seconds"],
            "realtime_factor": result["elapsed_seconds"] / max(info["duration"], 1),
            "success": result["success"],
        }, f, indent=2, ensure_ascii=False)

    if sync_score is not None:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({"syncnet_score": sync_score}, f)

    return {
        "video": name,
        "report_path": report_path,
        "success": result["success"],
        "elapsed": result["elapsed_seconds"],
        "duration": info["duration"],
        "sync_score": sync_score,
    }


def generate_summary_report(phase_id, all_results, summary_path):
    """Aggregate all-video summary for a phase."""
    config = PHASE_CONFIGS[phase_id]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase {phase_id} Summary — {config['name']}\n\n")
        f.write(f"**Description**: {config['description']}\n\n")

        f.write(f"## Results Table\n\n")
        f.write("| Video | Duration | Wall time | RT factor | SyncNet | Status |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in all_results:
            sync = f"{r['sync_score']:.2f}" if r['sync_score'] is not None else "—"
            status = "✅" if r["success"] else "❌"
            rt = r["elapsed"] / max(r["duration"], 1)
            f.write(f"| {r['video']} | {r['duration']:.1f}s | "
                    f"{r['elapsed']:.1f}s ({r['elapsed']/60:.1f}min) | "
                    f"{rt:.1f}× | {sync} | {status} |\n")

        # Stats
        successful = [r for r in all_results if r["success"]]
        if successful:
            total_dur = sum(r["duration"] for r in successful)
            total_wall = sum(r["elapsed"] for r in successful)
            f.write(f"\n## Statistics ({len(successful)} successful)\n\n")
            f.write(f"- Total input duration: {total_dur:.1f}s "
                    f"({total_dur/60:.1f} min)\n")
            f.write(f"- Total wall time: {total_wall:.1f}s "
                    f"({total_wall/60:.1f} min)\n")
            f.write(f"- Avg realtime factor: {total_wall/total_dur:.2f}×\n")
            if any(r['sync_score'] for r in successful):
                scores = [r['sync_score'] for r in successful if r['sync_score'] is not None]
                f.write(f"- Avg SyncNet score: {sum(scores)/len(scores):.2f}\n")

        f.write(f"\n## Env vars\n\n```\n")
        for k, v in config["env"].items():
            f.write(f"{k}={v}\n")
        f.write("```\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True,
                    choices=list(PHASE_CONFIGS.keys()))
    ap.add_argument("--videos", nargs="+", required=True,
                    help="Test video paths")
    ap.add_argument("--output-root", default=REPORT_ROOT)
    args = ap.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    phase_dir = os.path.join(args.output_root,
                             f"phase_{args.phase}_{PHASE_CONFIGS[args.phase]['name']}_{timestamp}")
    os.makedirs(phase_dir, exist_ok=True)
    _log(f"phase output dir: {phase_dir}")

    all_results = []
    for v in args.videos:
        if not os.path.isfile(v):
            _log(f"skip (not found): {v}")
            continue
        video_out = os.path.join(phase_dir, Path(v).stem)
        os.makedirs(video_out, exist_ok=True)
        try:
            result = generate_video_report(v, args.phase, video_out)
            all_results.append(result)
        except Exception as e:
            _log(f"FAIL {v}: {e}")

    # Summary
    summary_path = os.path.join(phase_dir, "summary.md")
    generate_summary_report(args.phase, all_results, summary_path)
    _log(f"summary: {summary_path}")
    _log(f"DONE — {len(all_results)} videos processed")


if __name__ == "__main__":
    main()
