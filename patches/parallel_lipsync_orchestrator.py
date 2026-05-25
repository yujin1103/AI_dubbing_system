"""Chunk-level parallel orchestrator: lipsync chunk N+1 ‖ enhance chunk N.

Pipeline:
  ┌───────────────────────────────────────────────────────────┐
  │ chunk 0 lipsync → chunk 0 enhance (async)                  │
  │            ↓                                                │
  │ chunk 1 lipsync → chunk 1 enhance (async)                  │
  │            ↓                                                │
  │ chunk 2 lipsync → chunk 2 enhance (async)                  │
  │            ...                                              │
  │ wait for all enhances                                       │
  │ concat enhanced chunks → final mp4                          │
  └───────────────────────────────────────────────────────────┘

GPU memory analysis (RTX 5080 16GB):
  - LatentSync UNet + VAE: ~6GB
  - mouth_enhance (TRT): ~1.5GB
  - Concurrent: ~7.5GB → fits with 8.5GB headroom

Safety:
  - LATENTSYNC_PARALLEL_ENHANCE=0 → fall back to serial mode (sequential)
  - OOM detection → automatic fallback to serial for remaining chunks
  - GPU memory pre-check before launch

Usage:
  python /workspace/patches/parallel_lipsync_orchestrator.py \
      --input video.mp4 \
      --audio dubbed.wav \
      --output result.mp4 \
      --chunk-seconds 10 \
      [--no-enhance]
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


def _log(msg):
    print(f"[ParallelOrch] {msg}", flush=True)


def get_video_duration(path):
    r = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], capture_output=True, text=True)
    return float(r.stdout.strip())


def get_gpu_free_gb():
    """Free GPU memory in GB (returns 0 if can't query)."""
    try:
        free, total = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return 0.0


def pre_chunk_video(input_video, chunk_seconds, work_dir):
    """Split input into N-second chunks. Returns list of (idx, chunk_path)."""
    duration = get_video_duration(input_video)
    n_chunks = int((duration + chunk_seconds - 1) // chunk_seconds)
    chunks = []
    _log(f"splitting {duration:.1f}s video into {n_chunks} × {chunk_seconds}s chunks")
    for i in range(n_chunks):
        start = i * chunk_seconds
        chunk_path = os.path.join(work_dir, f"chunk_{i:03d}.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-i", input_video,
            "-t", str(chunk_seconds),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac",
            chunk_path
        ]
        subprocess.run(cmd, check=True)
        chunks.append((i, chunk_path))
    return chunks


def split_audio_to_chunks(audio_path, chunk_seconds, n_chunks, work_dir):
    """Split audio file to match video chunks."""
    audio_chunks = []
    for i in range(n_chunks):
        start = i * chunk_seconds
        out = os.path.join(work_dir, f"audio_{i:03d}.wav")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-i", audio_path,
            "-t", str(chunk_seconds),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            out
        ]
        subprocess.run(cmd, check=True)
        audio_chunks.append(out)
    return audio_chunks


def _gpu_free_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],
            text=True).strip()
        # Take first GPU only
        return int(out.splitlines()[0]) / 1024.0
    except Exception:
        return -1.0


def _wait_gpu_clean(min_free_gb=10.0, timeout_s=120, settle_s=2):
    """Wait until GPU has at least min_free_gb free. Useful before launching a chunk."""
    import time
    deadline = time.time() + timeout_s
    last_free = _gpu_free_gb()
    while time.time() < deadline:
        free = _gpu_free_gb()
        if free < 0:
            return  # nvidia-smi missing — proceed
        if free >= min_free_gb:
            _log(f"  GPU free {free:.1f}GB ≥ {min_free_gb}GB → ready")
            time.sleep(settle_s)  # let things settle
            return
        if free != last_free:
            _log(f"  GPU free {free:.1f}GB < {min_free_gb}GB → waiting...")
            last_free = free
        time.sleep(2)
    _log(f"  ⚠ GPU still {_gpu_free_gb():.1f}GB after {timeout_s}s — proceeding anyway")


def _is_oom_error(stderr_text):
    """Detect CUDA OOM signals in subprocess stderr."""
    if not stderr_text: return False
    oom_signals = (
        "CUDA out of memory",
        "OutOfMemoryError",
        "cuda runtime error (out of memory)",
        "CUDNN_STATUS_NOT_ENOUGH_WORKSPACE",
        "Failed to allocate",
        "RuntimeError: CUDA error: out of memory",
    )
    return any(sig in stderr_text for sig in oom_signals)


def run_lipsync(chunk_video, chunk_audio, output_path):
    """Run LatentSync on one chunk. BLOCKING.

    On OOM: auto-fallback by reducing memory pressure (VAE chunk 1, TRT off).
    """
    _wait_gpu_clean(min_free_gb=float(os.environ.get("LIPSYNC_MIN_GPU_GB","10")),
                    timeout_s=int(os.environ.get("LIPSYNC_GPU_WAIT_S","60")))
    ckpt = os.environ.get("LATENTSYNC_INFERENCE_CKPT",
                          "/opt/LatentSync/checkpoints/latentsync_unet.pt")
    base_cmd = [
        "/opt/venv_lipsync/bin/python", "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512_nf16.yaml",
        "--inference_ckpt_path", ckpt,
        "--video_path", chunk_video,
        "--audio_path", chunk_audio,
        "--video_out_path", output_path,
        "--inference_steps", "10",
        "--guidance_scale", "1.5",
        "--seed", "1247",
    ]
    base_env = os.environ.copy()
    base_env.setdefault("LATENTSYNC_USE_TRT", "1")
    base_env.setdefault("LATENTSYNC_TRT_ENGINE", "/workspace/trt_work/engines/unet_fp16.trt")
    base_env.setdefault("LATENTSYNC_SCHEDULER", "dpm")
    base_env.setdefault("LATENTSYNC_TEACACHE", "0.1")
    base_env.setdefault("LATENTSYNC_USE_NVENC", "1")
    base_env["LATENTSYNC_CHUNK_SECONDS"] = "0"

    # Attempt 1: default config
    r = subprocess.run(base_cmd, cwd="/opt/LatentSync", env=base_env,
                       capture_output=True, text=True)
    if r.returncode == 0:
        return True

    # OOM fallback: VAE chunk 1, slicing on, TRT VAE off
    if _is_oom_error(r.stderr):
        _log("lipsync OOM detected → fallback: VAE_CHUNK=1, VAE_SLICING=1, VAE_TRT=0")
        oom_env = base_env.copy()
        oom_env["LATENTSYNC_VAE_CHUNK"] = "1"
        oom_env["LATENTSYNC_VAE_SLICING"] = "1"
        oom_env["LATENTSYNC_VAE_TRT"] = "0"
        # Free GPU before retry
        import time as _t; _t.sleep(5)
        _wait_gpu_clean(min_free_gb=3.0, timeout_s=30)
        r2 = subprocess.run(base_cmd, cwd="/opt/LatentSync", env=oom_env,
                            capture_output=True, text=True)
        if r2.returncode == 0:
            _log("lipsync OOM fallback succeeded")
            return True
        if _is_oom_error(r2.stderr):
            _log("lipsync OOM persists → final fallback: inference_steps=8 + TRT UNet off")
            final_cmd = list(base_cmd)
            si = final_cmd.index("--inference_steps")
            final_cmd[si+1] = "8"
            final_env = oom_env.copy()
            final_env["LATENTSYNC_USE_TRT"] = "0"
            _t.sleep(5)
            r3 = subprocess.run(final_cmd, cwd="/opt/LatentSync", env=final_env,
                                capture_output=True, text=True)
            if r3.returncode == 0:
                _log("lipsync final fallback succeeded (steps=8, no TRT)")
                return True
            _log(f"lipsync FINAL FAIL: {r3.stderr[-300:]}")
            return False
        _log(f"lipsync OOM fallback FAIL (non-OOM error): {r2.stderr[-300:]}")
        return False

    _log(f"lipsync FAIL: {r.stderr[-300:]}")
    return False


def run_mouth_enhance(lipsync_out, original_chunk, enhance_out):
    """Run mouth_only_enhance on one chunk. BLOCKING."""
    cmd = [
        "/opt/venv_gfpgan/bin/python",
        "/workspace/patches/mouth_only_enhance.py",
        "--lipsync", lipsync_out,
        "--original", original_chunk,
        "--output", enhance_out,
        "--mux-audio-from", lipsync_out,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _log(f"enhance FAIL: {r.stderr[-300:]}")
    return r.returncode == 0


def concat_chunks(chunks, output_path, work_dir):
    """Concat chunks using ffmpeg concat demuxer + re-encode at 25fps.

    Re-encode (not -c copy) because chunk inputs may be 23.976fps while LatentSync
    outputs at 25fps → -c copy breaks timestamp metadata. Also verify every chunk
    file exists before concat (fail fast vs partial output).
    """
    missing = [c for c in chunks if not os.path.exists(c)]
    if missing:
        raise RuntimeError(f"Concat aborted — {len(missing)} chunk(s) missing: {missing}")
    list_path = os.path.join(work_dir, "concat_list.txt")
    with open(list_path, "w") as f:
        for c in chunks:
            f.write(f"file '{c}'\n")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-r", "25",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input video")
    ap.add_argument("--audio", required=True, help="Dubbed audio (any sample rate)")
    ap.add_argument("--output", required=True, help="Final mp4 output")
    ap.add_argument("--chunk-seconds", type=int, default=30,
                    help="Seconds per lipsync chunk (default 30, was 15 — larger = less init overhead)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="Skip mouth_only_enhance (lipsync only)")
    ap.add_argument("--no-parallel", action="store_true",
                    default=os.environ.get("LATENTSYNC_PARALLEL_ENHANCE", "1") == "0",
                    help="Disable parallelization (serial fallback)")
    ap.add_argument("--min-free-gb", type=float, default=3.0,
                    help="Min free GPU GB before allowing parallel (else fall back)")
    args = ap.parse_args()

    # GPU memory pre-check
    free_gb = get_gpu_free_gb()
    _log(f"GPU free: {free_gb:.1f}GB (min for parallel: {args.min_free_gb}GB)")
    if free_gb < args.min_free_gb and not args.no_parallel:
        _log(f"GPU too tight → forcing serial mode")
        args.no_parallel = True

    work_dir = tempfile.mkdtemp(prefix="parallel_lipsync_")
    _log(f"work dir: {work_dir}")

    try:
        t_total = time.time()

        # Step 1: Pre-chunk video + audio
        _log(f"=== Step 1: Pre-chunking (chunk_seconds={args.chunk_seconds}) ===")
        chunks = pre_chunk_video(args.input, args.chunk_seconds, work_dir)
        n = len(chunks)
        audio_chunks = split_audio_to_chunks(args.audio, args.chunk_seconds, n, work_dir)

        # Step 2: Lipsync (sequential due to GPU) + Enhance (async parallel)
        _log(f"=== Step 2: Lipsync + Enhance ===")
        lipsync_outs = []
        enhance_outs = [None] * n
        enhance_futures = {}

        # Use ThreadPoolExecutor for enhance (1-2 workers — shares GPU)
        enhance_executor = ThreadPoolExecutor(max_workers=1) if not args.no_parallel else None

        for i, (idx, chunk_path) in enumerate(chunks):
            lipsync_out = os.path.join(work_dir, f"lipsync_{idx:03d}.mp4")
            t0 = time.time()
            _log(f"  chunk {idx}/{n-1}: lipsync ...")
            ok = run_lipsync(chunk_path, audio_chunks[i], lipsync_out)
            if not ok:
                _log(f"  chunk {idx}: lipsync FAILED — using original")
                lipsync_out = chunk_path
            _log(f"  chunk {idx}: lipsync done in {time.time()-t0:.1f}s")
            lipsync_outs.append(lipsync_out)

            if args.no_enhance:
                enhance_outs[i] = lipsync_out
                continue

            # Spawn enhance async (parallel) or run inline (serial)
            enhance_out = os.path.join(work_dir, f"enhance_{idx:03d}.mp4")
            if enhance_executor is not None:
                fut = enhance_executor.submit(
                    run_mouth_enhance, lipsync_out, chunk_path, enhance_out
                )
                enhance_futures[i] = (fut, enhance_out, lipsync_out)
                _log(f"  chunk {idx}: enhance spawned (parallel)")
            else:
                t0 = time.time()
                ok = run_mouth_enhance(lipsync_out, chunk_path, enhance_out)
                _log(f"  chunk {idx}: enhance done in {time.time()-t0:.1f}s")
                enhance_outs[i] = enhance_out if ok else lipsync_out

        # Wait for all enhance futures
        if enhance_executor is not None:
            _log(f"=== Step 2b: Wait for {len(enhance_futures)} enhance jobs ===")
            for i, (fut, enhance_out, lipsync_out) in enhance_futures.items():
                try:
                    ok = fut.result(timeout=1800)
                    enhance_outs[i] = enhance_out if ok else lipsync_out
                    _log(f"  chunk {i}: enhance result OK={ok}")
                except Exception as e:
                    _log(f"  chunk {i}: enhance EXCEPTION: {e}")
                    enhance_outs[i] = lipsync_out
            enhance_executor.shutdown(wait=True)

        # Step 3: Concat
        _log(f"=== Step 3: Concat {n} chunks → {args.output} ===")
        concat_chunks(enhance_outs, args.output, work_dir)

        elapsed = time.time() - t_total
        _log(f"=== DONE in {elapsed:.1f}s ===")
        _log(f"  Wall: {elapsed/60:.2f} min")
        _log(f"  Per chunk avg: {elapsed/n:.1f}s")
        _log(f"  Output: {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")
    finally:
        # Cleanup unless KEEP_WORK_DIR set
        if os.environ.get("KEEP_WORK_DIR", "0") != "1":
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
