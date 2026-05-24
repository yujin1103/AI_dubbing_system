"""CosyVoice3 Triton TRT-LLM client adapter — drop-in replacement for the
existing PyTorch _cosy_model.inference_cross_lingual() interface.

The Triton server's cosyvoice3 model expects:
  - reference_wav: FP32 audio @16kHz
  - reference_wav_len: INT32 length
  - reference_text: transcription of the reference audio (required by Triton)
  - target_text: text to synthesize
  - Output: FP32 audio @24kHz

For cross_lingual usage (where reference_text isn't needed), we pass an empty
string. CosyVoice3's LLM should handle this gracefully.

Server URL configuration:
  - HTTP: http://localhost:18000 (default)
  - gRPC: localhost:18001
  - Triton runs in cosyvoice-trt Docker container with --net=host
"""
from __future__ import annotations
import os
import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio


COSYVOICE_TRITON_URL = os.environ.get(
    "COSYVOICE_TRITON_URL", "http://localhost:18000"
)
MODEL_NAME = "cosyvoice3"
INFER_PATH = f"/v2/models/{MODEL_NAME}/infer"


def _load_reference_audio(prompt_wav) -> np.ndarray:
    """Convert prompt_wav (file path or torch tensor) to 16kHz mono FP32."""
    if isinstance(prompt_wav, str):
        waveform, sr = sf.read(prompt_wav)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != 16000:
            # Use torchaudio for resampling
            w = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
            w = torchaudio.transforms.Resample(sr, 16000)(w)
            waveform = w.squeeze(0).numpy()
    elif isinstance(prompt_wav, torch.Tensor):
        waveform = prompt_wav.detach().cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
    elif isinstance(prompt_wav, np.ndarray):
        waveform = prompt_wav
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
    else:
        raise TypeError(f"Unsupported prompt_wav type: {type(prompt_wav)}")
    return waveform.astype(np.float32)


def inference_cross_lingual(tts_text: str, prompt_wav, stream: bool = False,
                            speed: float = 1.0,
                            reference_text: str = ""):
    """Drop-in replacement yielding the same dict format as CosyVoice3.

    Returns generator yielding {'tts_speech': torch.Tensor (1, N) @24kHz}.
    """
    waveform = _load_reference_audio(prompt_wav)
    samples = waveform.reshape(1, -1).astype(np.float32)
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    payload = {
        "inputs": [
            {"name": "reference_wav", "shape": list(samples.shape),
             "datatype": "FP32", "data": samples.flatten().tolist()},
            {"name": "reference_wav_len", "shape": list(lengths.shape),
             "datatype": "INT32", "data": lengths.flatten().tolist()},
            {"name": "reference_text", "shape": [1, 1],
             "datatype": "BYTES", "data": [reference_text]},
            {"name": "target_text", "shape": [1, 1],
             "datatype": "BYTES", "data": [tts_text]},
        ]
    }

    rsp = requests.post(
        f"{COSYVOICE_TRITON_URL}{INFER_PATH}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if rsp.status_code != 200:
        raise RuntimeError(
            f"[CosyVoice Triton] HTTP {rsp.status_code}: {rsp.text[:200]}"
        )
    result = rsp.json()
    audio = np.array(result["outputs"][0]["data"], dtype=np.float32)

    # Apply speed (simple time-stretch via resample if needed)
    if abs(speed - 1.0) > 1e-3:
        n_out = int(len(audio) / speed)
        # Simple linear interpolation resample
        idx = np.linspace(0, len(audio) - 1, n_out)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)

    # Return as torch tensor (1, N) at 24kHz to match CosyVoice3 output format
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    yield {"tts_speech": audio_tensor}


def is_triton_alive(timeout: float = 2.0) -> bool:
    """Check if Triton server is responsive."""
    try:
        rsp = requests.get(f"{COSYVOICE_TRITON_URL}/v2/health/ready",
                          timeout=timeout)
        return rsp.status_code == 200
    except Exception:
        return False


class CosyVoice3TritonAdapter:
    """Adapter that mimics CosyVoice3 PyTorch API but routes to Triton server."""

    def __init__(self):
        self.sample_rate = 24000

    def inference_cross_lingual(self, tts_text, prompt_wav, stream=False,
                                speed=1.0, **kwargs):
        return inference_cross_lingual(tts_text, prompt_wav, stream, speed)

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav,
                            stream=False, speed=1.0, **kwargs):
        return inference_cross_lingual(tts_text, prompt_wav, stream, speed,
                                       reference_text=prompt_text)
