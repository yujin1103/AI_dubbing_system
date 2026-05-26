"""ERes2NetV2 embedding helper for short speaker segments.

Research finding (DiariZen / 3D-Speaker survey, 2026-05):
  ERes2NetV2 is "tailored to more effectively capture features from short-duration
  utterances" — directly addresses the 0.8-2.2s segment errors where ECAPA fails.

Usage (singleton pattern, lazy load):
  from patches.eres2netv2_helper import get_eres2netv2_model, extract_eres2netv2_emb
  model = get_eres2netv2_model()
  emb = extract_eres2netv2_emb(audio_array, sr=16000)
"""
from __future__ import annotations
import os
import sys
from typing import Optional
import numpy as np

_model = None
_feature_extractor = None
_device = None

MODEL_ID = os.environ.get(
    "ERES2NETV2_MODEL_ID",
    "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
)
LOCAL_CACHE = os.environ.get(
    "ERES2NETV2_CACHE_DIR",
    "/workspace/media/model_cache/eres2netv2"
)


def _get_model_config(model_id: str):
    """Return (model_config_dict, model_pt_filename, revision)."""
    if "w24s4ep4" in model_id:
        return {
            'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
            'args': {
                'feat_dim': 80, 'embedding_size': 192,
                'baseWidth': 24, 'scale': 4, 'expansion': 4,
            },
        }, "pretrained_eres2netv2w24s4ep4.ckpt", "v1.0.1"
    else:
        return {
            'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
            'args': {
                'feat_dim': 80, 'embedding_size': 192,
                'baseWidth': 26, 'scale': 2, 'expansion': 2,
            },
        }, "pretrained_eres2netv2.ckpt", "v1.0.1"


def get_eres2netv2_model():
    global _model, _feature_extractor, _device
    if _model is not None:
        return _model
    import torch
    # Ensure 3D-Speaker is on sys.path
    if "/opt/3D-Speaker" not in sys.path:
        sys.path.insert(0, "/opt/3D-Speaker")
    from speakerlab.process.processor import FBank
    from speakerlab.utils.builder import dynamic_import
    from modelscope.hub.snapshot_download import snapshot_download

    cfg, model_pt, revision = _get_model_config(MODEL_ID)
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    cache_dir = os.path.join(LOCAL_CACHE, MODEL_ID.replace("/", "_"))
    if not os.path.isfile(os.path.join(cache_dir, model_pt)):
        print(f"[ERes2NetV2] downloading {MODEL_ID} (revision {revision}) ...", flush=True)
        snapshot_download(MODEL_ID, revision=revision, cache_dir=LOCAL_CACHE)
        downloaded = os.path.join(LOCAL_CACHE, MODEL_ID)
        if os.path.isdir(downloaded):
            cache_dir = downloaded
    pt_path = os.path.join(cache_dir, model_pt)
    if not os.path.isfile(pt_path):
        # search recursively
        for root, dirs, files in os.walk(LOCAL_CACHE):
            if model_pt in files:
                pt_path = os.path.join(root, model_pt)
                break
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"ERes2NetV2 weights not found: {model_pt}")

    Model = dynamic_import(cfg["obj"])
    _model = Model(**cfg["args"])
    state = torch.load(pt_path, map_location="cpu")
    _model.load_state_dict(state, strict=False)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(_device).eval()
    _feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    print(f"[ERes2NetV2] loaded {MODEL_ID} on {_device}", flush=True)
    return _model


def extract_eres2netv2_emb(audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """Extract ERes2NetV2 192-dim L2-normalized embedding from an audio array.

    Args:
        audio: 1D float numpy array (any sr; will be cast to 16k mono)
        sr: input sample rate
    Returns:
        np.ndarray shape (192,) L2-normalized, or None on failure.
    """
    import torch
    try:
        global _feature_extractor, _device
        if _model is None:
            get_eres2netv2_model()
        if audio is None or len(audio) == 0:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        # resample to 16k if needed
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception:
                pass
        # require ≥ 0.4s
        if len(audio) < int(0.4 * sr):
            return None
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, n)
        feat = _feature_extractor(wav)  # FBank
        feat = feat.unsqueeze(0).to(_device)  # (1, 1, T, F)? 확인 필요
        # ERes2NetV2 expects (B, T, F) for fbank
        if feat.dim() == 4:
            feat = feat.squeeze(1)
        with torch.no_grad():
            emb = _model(feat)  # (1, 192)
        emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
        n = float(np.linalg.norm(emb_np))
        if n > 1e-8:
            emb_np = emb_np / n
        return emb_np
    except Exception as e:
        print(f"[ERes2NetV2] extract failed: {e}", flush=True)
        return None
