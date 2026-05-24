"""CAM++ embedding helper for speaker diarization (3rd voice signal).

공급망 검증 (2026-05-21):
  - 모델: iic/speech_campplus_sv_zh-cn_16k-common (Alibaba DAMO)
  - ONNX 이미 CosyVoice3 패키지에 포함됨 (/workspace/media/model_cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/campplus.onnx)
  - 3D-Speaker repo: github.com/modelscope/3D-Speaker (Apache 2.0, Alibaba)

CAM++는 ECAPA-TDNN과 다른 architecture (Context-Aware Masking Plus) — orthogonal 신호 제공.
"""
from __future__ import annotations
import os
import sys
from typing import Optional
import numpy as np

_session = None  # ONNX runtime session
_feature_extractor = None

CAMPPLUS_ONNX_PATH = os.environ.get(
    "CAMPPLUS_ONNX_PATH",
    "/workspace/media/model_cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/campplus.onnx"
)


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        if "/opt/3D-Speaker" not in sys.path:
            sys.path.insert(0, "/opt/3D-Speaker")
        from speakerlab.process.processor import FBank
        _feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    return _feature_extractor


def get_campplus_session():
    global _session
    if _session is not None:
        return _session
    import onnxruntime as ort
    if not os.path.isfile(CAMPPLUS_ONNX_PATH):
        raise FileNotFoundError(f"CAM++ ONNX not found: {CAMPPLUS_ONNX_PATH}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _session = ort.InferenceSession(CAMPPLUS_ONNX_PATH, providers=providers)
    print(f"[CAM++] loaded {CAMPPLUS_ONNX_PATH} (providers={_session.get_providers()})", flush=True)
    return _session


def extract_campplus_emb(audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """Extract CAM++ 192-dim L2-normalized embedding."""
    import torch
    try:
        sess = get_campplus_session()
        fe = _get_feature_extractor()
        if audio is None or len(audio) == 0:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception:
                pass
        if len(audio) < int(0.4 * sr):
            return None
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, n)
        feat = fe(wav)  # FBank: (1, T, 80) or (T, 80)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        if feat.dim() == 4:
            feat = feat.squeeze(1)
        feat_np = feat.numpy().astype(np.float32)
        # ONNX inference
        input_name = sess.get_inputs()[0].name
        emb = sess.run(None, {input_name: feat_np})[0]
        emb_np = emb.squeeze().astype(np.float32)
        n = float(np.linalg.norm(emb_np))
        if n > 1e-8:
            emb_np = emb_np / n
        return emb_np
    except Exception as e:
        print(f"[CAM++] extract failed: {e}", flush=True)
        return None
