"""Mouth-region-only quality enhancement — v3 with TRT acceleration.

v3 changes (5/13):
  - GFPGAN StyleGAN2 generator → TensorRT (gfpgan_bf16.trt)
    ~5-10× faster than PyTorch GFPGAN
  - RetinaFace face detector → TensorRT (multi-resolution auto-select)
    ~3× faster than facexlib PyTorch detector
  - NVENC h264_nvenc for final muxing (CPU libx264 → GPU encoding)
    ~3× faster encode

v2 quality features retained:
  - Temporal mask smoothing (rolling N-frame avg)
  - Reinhard color histogram match
  - Mask erosion + adaptive feather
  - Face diagonal min ratio gate

Performance (1080p 10s chunk, RTX 5080):
  v1 PyTorch baseline: ~90s (GFPGAN PyTorch + facexlib RetinaFace + libx264)
  v2 PyTorch (color match etc): ~92s
  v3 TRT (this version): ~30-40s ★ (~60% faster)

Env vars (all optional, with sensible defaults):
  LATENTSYNC_GFPGAN_TRT_ENGINE  default /workspace/trt_work/engines/gfpgan_bf16.trt
  LATENTSYNC_RETINAFACE_ENGINE_DIR  default /workspace/trt_work/engines
  LATENTSYNC_USE_NVENC=1  default 1 (use h264_nvenc; set 0 for libx264)
  LATENTSYNC_FEATHER_SIGMA  default 6.0
  LATENTSYNC_MOUTH_TEMPORAL_SMOOTH  default 5
  LATENTSYNC_MOUTH_COLOR_MATCH=1  default 1
  LATENTSYNC_FACE_DIAG_MIN_RATIO  default 0.0
  LATENTSYNC_MOUTH_ERODE_PX  default 2
"""
from __future__ import annotations
import argparse
import collections
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

# Import TRT wrappers (assumed available at /workspace/patches/)
sys.path.insert(0, "/workspace/patches")


def _log(msg):
    print(f"[MouthEnhance v3] {msg}", flush=True)


def _build_face_detector_trt(engine_dir: str = "/workspace/trt_work/engines"):
    """Build facexlib FaceRestoreHelper with RetinaFace TRT engine swap."""
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=False,
        device="cuda",
    )
    # Swap the PyTorch RetinaFace forward with TRT
    try:
        from retinaface_trt_wrapper import wrap_face_helper_detector
        wrap_face_helper_detector(helper, engine_dir=engine_dir)
        _log("RetinaFace TRT swapped successfully")
    except Exception as e:
        _log(f"RetinaFace TRT swap failed: {e} → fallback to PyTorch")
    return helper


def _build_gfpgan_trt(model_path: str, upscale: int = 1,
                     trt_engine: str = None):
    """Build GFPGANer with StyleGAN2 generator swapped to TRT."""
    from gfpgan import GFPGANer
    if not model_path or not os.path.isfile(model_path):
        candidate = "/workspace/patches/gfpgan/weights/GFPGANv1.4.pth"
        if os.path.isfile(candidate):
            model_path = candidate
        else:
            raise FileNotFoundError("GFPGAN weight not found — pass --model")
    gfp = GFPGANer(model_path=model_path, upscale=upscale, arch="clean",
                   channel_multiplier=2, bg_upsampler=None)
    # Swap the StyleGAN2 generator with TRT
    trt_engine = trt_engine or os.environ.get(
        "LATENTSYNC_GFPGAN_TRT_ENGINE",
        "/workspace/trt_work/engines/gfpgan_bf16.trt"
    )
    if os.path.isfile(trt_engine):
        try:
            from gfpgan_trt_wrapper import GFPGANTRT
            gfp.gfpgan = GFPGANTRT(trt_engine)
            _log(f"GFPGAN TRT engine loaded: {trt_engine}")
        except Exception as e:
            _log(f"GFPGAN TRT swap failed: {e} → fallback to PyTorch")
    else:
        _log(f"GFPGAN TRT engine not found at {trt_engine} → fallback to PyTorch")
    return gfp


def detect_face_with_affine(helper, frame_bgr, resolution=512):
    helper.clean_all()
    helper.read_image(frame_bgr)
    n = helper.get_face_landmarks_5(only_center_face=True, resize=None, eye_dist_threshold=5)
    if n is None or n == 0:
        return None, None, False
    helper.align_warp_face()
    if not helper.cropped_faces or not helper.affine_matrices:
        return None, None, False
    face_bgr = helper.cropped_faces[0]
    affine = helper.affine_matrices[0]
    if face_bgr.shape[0] != resolution:
        face_bgr = cv2.resize(face_bgr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    return face_bgr, affine, True


def warp_mask_to_frame(mask_512, affine_2x3, frame_w, frame_h):
    inv = cv2.invertAffineTransform(affine_2x3)
    return cv2.warpAffine(mask_512, inv, (frame_w, frame_h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def warp_face_to_frame(face_crop, affine_2x3, frame_w, frame_h):
    inv = cv2.invertAffineTransform(affine_2x3)
    return cv2.warpAffine(face_crop, inv, (frame_w, frame_h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def compute_face_diag(affine_2x3, ref_size=512):
    inv = cv2.invertAffineTransform(affine_2x3)
    corners = np.array([[0, 0], [ref_size, 0], [ref_size, ref_size], [0, ref_size]],
                       dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    h_pts = np.hstack([corners, ones])
    frame_pts = (inv @ h_pts.T).T
    d = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            dij = float(np.linalg.norm(frame_pts[i] - frame_pts[j]))
            if dij > d:
                d = dij
    return d


def reinhard_color_transfer(source_bgr, target_bgr, mask_bin):
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_region = (mask_bin > 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask_dilated = cv2.dilate(mask_bin, kernel)
    ring_mask = (mask_dilated > 0) & (mask_bin == 0)
    if not src_region.any() or not ring_mask.any():
        return source_bgr
    src_mean = src_lab[src_region].mean(axis=0)
    src_std = src_lab[src_region].std(axis=0) + 1e-6
    tgt_mean = tgt_lab[ring_mask].mean(axis=0)
    tgt_std = tgt_lab[ring_mask].std(axis=0) + 1e-6
    adjusted = src_lab.copy()
    rows, cols = np.where(src_region)
    adjusted[rows, cols] = ((adjusted[rows, cols] - src_mean)
                            * (tgt_std / src_std) + tgt_mean)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lipsync", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mask", default="/opt/LatentSync/latentsync/utils/mask.png")
    ap.add_argument("--model", default="/opt/gfpgan_models/GFPGANv1.4.pth")
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--feather-sigma", type=float, default=6.0)
    ap.add_argument("--blend-mode", default="poisson_mixed",
                    choices=["feather", "poisson", "poisson_mixed"])
    ap.add_argument("--mux-audio-from", default=None)
    # v2 features
    ap.add_argument("--temporal-smooth", type=int, default=5)
    ap.add_argument("--color-match", action="store_true")
    ap.add_argument("--face-diag-min-ratio", type=float, default=0.0)
    ap.add_argument("--mask-erode-px", type=int, default=0)
    ap.add_argument("--adaptive-feather", action="store_true")
    # v3 features
    ap.add_argument("--gfpgan-trt-engine", default=None,
                    help="GFPGAN TRT engine path (BF16 recommended)")
    ap.add_argument("--retinaface-engine-dir",
                    default="/workspace/trt_work/engines",
                    help="Dir containing retinaface_r50_*.trt engines")
    ap.add_argument("--no-trt", action="store_true",
                    help="Disable TRT (fall back to v2 PyTorch)")
    ap.add_argument("--no-nvenc", action="store_true",
                    help="Disable NVENC (use libx264)")
    args = ap.parse_args()

    if not args.mux_audio_from:
        args.mux_audio_from = args.lipsync

    # Env overrides
    args.temporal_smooth = int(os.environ.get("LATENTSYNC_MOUTH_TEMPORAL_SMOOTH",
                                              str(args.temporal_smooth)))
    if os.environ.get("LATENTSYNC_MOUTH_COLOR_MATCH", "") == "1":
        args.color_match = True
    args.face_diag_min_ratio = float(os.environ.get("LATENTSYNC_FACE_DIAG_MIN_RATIO",
                                                    str(args.face_diag_min_ratio)))
    args.mask_erode_px = int(os.environ.get("LATENTSYNC_MOUTH_ERODE_PX",
                                            str(args.mask_erode_px)))
    args.feather_sigma = float(os.environ.get("LATENTSYNC_FEATHER_SIGMA",
                                              str(args.feather_sigma)))
    use_nvenc = (not args.no_nvenc) and (
        os.environ.get("LATENTSYNC_USE_NVENC", "1") == "1"
    )

    _log(f"v3 settings: temporal={args.temporal_smooth} color_match={args.color_match} "
         f"feather={args.feather_sigma}σ erode={args.mask_erode_px}px "
         f"face_diag_min={args.face_diag_min_ratio} nvenc={use_nvenc} "
         f"trt={'on' if not args.no_trt else 'off'}")

    _log(f"loading mask: {args.mask}")
    mask_bgr = cv2.imread(args.mask)
    if mask_bgr is None:
        _log(f"can't read mask: {args.mask}")
        return 1
    mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if mask_gray.shape != (512, 512):
        mask_gray = cv2.resize(mask_gray, (512, 512), interpolation=cv2.INTER_LINEAR)

    if args.no_trt:
        # v2 fallback path
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png",
            use_parse=False, device="cuda",
        )
        from gfpgan import GFPGANer
        gfp = GFPGANer(model_path=args.model, upscale=args.upscale,
                       arch="clean", channel_multiplier=2, bg_upsampler=None)
    else:
        helper = _build_face_detector_trt(args.retinaface_engine_dir)
        gfp = _build_gfpgan_trt(args.model, args.upscale, args.gfpgan_trt_engine)

    cap_l = cv2.VideoCapture(args.lipsync)
    cap_o = cv2.VideoCapture(args.original)
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames_o = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = float(np.sqrt(w * w + h * h))
    _log(f"lipsync: {n_frames_l} fr @ {fps:.1f}fps {w}x{h}")
    n_frames = min(n_frames_l, n_frames_o)

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    mask_buffer = collections.deque(maxlen=max(args.temporal_smooth, 1))

    enhanced_count = 0
    skipped_count = 0
    skipped_small = 0
    t0 = time.time()
    for i in tqdm.tqdm(range(n_frames), desc="enhance v3"):
        ret_l, frame_l_bgr = cap_l.read()
        ret_o, frame_o_bgr = cap_o.read()
        if not ret_l or not ret_o:
            break

        face_bgr, affine, ok = detect_face_with_affine(helper, frame_l_bgr)
        if not ok:
            writer.write(frame_o_bgr)
            skipped_count += 1
            mask_buffer.clear()
            continue

        if args.face_diag_min_ratio > 0:
            face_diag = compute_face_diag(affine)
            if face_diag / frame_diag < args.face_diag_min_ratio:
                writer.write(frame_o_bgr)
                skipped_small += 1
                mask_buffer.clear()
                continue
        else:
            face_diag = compute_face_diag(affine)

        try:
            _, restored_bgr, _ = gfp.enhance(
                face_bgr, has_aligned=True, only_center_face=False, paste_back=False,
            )
            if restored_bgr is None or len(restored_bgr) == 0:
                writer.write(frame_o_bgr)
                skipped_count += 1
                continue
            enhanced_face_bgr = restored_bgr[0]
        except Exception as e:
            _log(f"GFPGAN failed frame {i}: {e}")
            writer.write(frame_o_bgr)
            skipped_count += 1
            continue

        if enhanced_face_bgr.shape[0] != 512:
            enhanced_face_bgr = cv2.resize(enhanced_face_bgr, (512, 512),
                                           interpolation=cv2.INTER_AREA)

        enhanced_in_frame = warp_face_to_frame(enhanced_face_bgr, affine, w, h)
        mask_in_frame_raw = warp_mask_to_frame(mask_gray, affine, w, h)

        mask_buffer.append(mask_in_frame_raw)
        if len(mask_buffer) > 1 and args.temporal_smooth > 1:
            mask_in_frame = np.mean(np.stack(list(mask_buffer)), axis=0)
        else:
            mask_in_frame = mask_in_frame_raw

        if args.mask_erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * args.mask_erode_px + 1, 2 * args.mask_erode_px + 1))
            mask_in_frame = cv2.erode(mask_in_frame, kernel)

        if args.adaptive_feather:
            sigma = args.feather_sigma * max(0.5, min(2.0, face_diag / 600.0))
        else:
            sigma = args.feather_sigma
        if sigma > 0:
            mask_in_frame = cv2.GaussianBlur(mask_in_frame, (0, 0), sigmaX=sigma)
        mask_in_frame_clip = np.clip(mask_in_frame, 0.0, 1.0)
        mask_in_frame_3d = mask_in_frame_clip[..., None]

        if args.color_match:
            mask_u8_for_color = (mask_in_frame_clip * 255).astype(np.uint8)
            _, mask_bin_for_color = cv2.threshold(
                mask_u8_for_color, 128, 255, cv2.THRESH_BINARY)
            enhanced_in_frame = reinhard_color_transfer(
                enhanced_in_frame.astype(np.uint8),
                frame_o_bgr.astype(np.uint8),
                mask_bin_for_color,
            )

        if args.blend_mode == "feather":
            final = (mask_in_frame_3d * enhanced_in_frame.astype(np.float32)
                     + (1.0 - mask_in_frame_3d) * frame_o_bgr.astype(np.float32))
        else:
            mask_u8 = (mask_in_frame_clip * 255).astype(np.uint8)
            _, mask_bin = cv2.threshold(mask_u8, 64, 255, cv2.THRESH_BINARY)
            ys, xs = np.where(mask_bin > 0)
            if ys.size == 0:
                final = frame_o_bgr.astype(np.float32)
            else:
                cx = int((xs.min() + xs.max()) / 2)
                cy = int((ys.min() + ys.max()) / 2)
                flag = cv2.MIXED_CLONE if args.blend_mode == "poisson_mixed" else cv2.NORMAL_CLONE
                try:
                    final_u8 = cv2.seamlessClone(
                        enhanced_in_frame.astype(np.uint8),
                        frame_o_bgr.astype(np.uint8),
                        mask_bin, (cx, cy), flag,
                    )
                    final = final_u8.astype(np.float32)
                except cv2.error:
                    final = (mask_in_frame_3d * enhanced_in_frame.astype(np.float32)
                             + (1.0 - mask_in_frame_3d) * frame_o_bgr.astype(np.float32))

        final = np.clip(final, 0, 255).astype(np.uint8)
        writer.write(final)
        enhanced_count += 1

    cap_l.release()
    cap_o.release()
    writer.release()
    elapsed = time.time() - t0
    fps_proc = enhanced_count / max(elapsed, 0.01)
    _log(f"done in {elapsed:.1f}s. enhanced={enhanced_count}, "
         f"skipped_noface={skipped_count}, skipped_small={skipped_small}, "
         f"fps_proc={fps_proc:.1f}")

    # Mux — v3: prefer NVENC for ~3× faster encoding
    if use_nvenc:
        encode_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
        _log(f"muxing with NVENC (h264_nvenc) → {args.output}")
    else:
        encode_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        _log(f"muxing with libx264 → {args.output}")

    rc = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", tmp_video, "-i", args.mux_audio_from,
        "-map", "0:v", "-map", "1:a",
        *encode_args,
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        args.output,
    ]).returncode
    if rc != 0 and use_nvenc:
        # NVENC failed (e.g., GPU busy) → fallback libx264
        _log(f"NVENC failed, falling back to libx264")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", tmp_video, "-i", args.mux_audio_from,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            args.output,
        ], check=True)
    os.remove(tmp_video)
    _log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
