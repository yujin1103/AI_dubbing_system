"""CosyVoice3 데몬 (FastAPI server).

매 inference마다 60-90초 모델 로딩 절약.
한 번 로드하고 메모리 상주, HTTP 요청으로 합성.

사용:
    /opt/venv_cosy/bin/python cosyvoice_daemon.py --port 8901

client (orchestrator.py에서):
    requests.post("http://localhost:8901/synthesize", json={
        "text": "...", "ref_audio_path": "...", "speed": 1.0,
        "tone": "...", "emotion": "Neutral",
    })
    → {"audio_b64": "..."}  (base64 encoded WAV)
"""
import argparse
import base64
import io
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import soundfile as sf

app = FastAPI()

# Global model (loaded once)
_cosy_model = None
TTS_SAMPLE_RATE = 24000  # CosyVoice3 native output rate, orchestrator.py와 일치 필수


class SynthesizeRequest(BaseModel):
    text: str
    ref_audio_path: str
    speed: float = 1.0
    tone: str = ""
    emotion: str = "Neutral"


class SynthesizeResponse(BaseModel):
    audio_b64: str
    sample_rate: int
    duration: float
    success: bool
    error: Optional[str] = None


def _build_prefix(text: str, tone: str, emotion: str) -> str:
    """v13: 카테고리 fallback 강한 묘사 (drama 격렬한 감정 대응).
    Sad/Angry/Happy 등이 평이하게 합성되던 v12 보완 — 학습 분포 안에서 더 강한 표현.
    """
    if emotion in {"Sad", "Angry", "Happy", "Surprised", "Scared"}:
        cat_imperative = {
            "Sad":       "with deep, anguished sadness, slow pacing, restrained voice",
            "Angry":     "with sharp, raised, confrontational tone, intense urgency",
            "Happy":     "in a bright, energetic tone with lively, exuberant pacing",
            "Surprised": "with sudden, sharp surprise, raised intonation",
            "Scared":    "with raw, tense fear, shaky breathy voice",
        }[emotion]
        return f'You are a helpful assistant. Please say this sentence {cat_imperative}.<|endofprompt|>'
    return 'You are a helpful assistant.<|endofprompt|>'


@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델 로드 (orchestrator load_cosy 패턴 동일)."""
    global _cosy_model
    print(f"[CosyDaemon] loading CosyVoice3...", flush=True)
    t0 = time.time()
    sys.path.insert(0, "/opt/CosyVoice")
    sys.path.insert(0, "/opt/CosyVoice/third_party/Matcha-TTS")
    from cosyvoice.cli.cosyvoice import CosyVoice3  # 정확한 클래스
    import inspect

    init_params = inspect.signature(CosyVoice3.__init__).parameters
    kwargs = {}
    if "load_jit" in init_params:
        kwargs["load_jit"] = False
    if "load_onnx" in init_params:
        kwargs["load_onnx"] = False
    if "load_trt" in init_params:
        kwargs["load_trt"] = False

    # 모델 경로 — orchestrator와 동일
    cache_dir = os.environ.get("MODELSCOPE_CACHE", "/root/.cache/modelscope")
    local_model_dir = os.path.join(
        cache_dir, "hub", "FunAudioLLM", "Fun-CosyVoice3-0.5B-2512"
    )
    # workspace cache 우선
    workspace_cache = "/workspace/media/model_cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    if os.path.isdir(workspace_cache):
        local_model_dir = workspace_cache

    if os.path.isdir(local_model_dir):
        print(f"[CosyDaemon] local cache: {local_model_dir}", flush=True)
        _cosy_model = CosyVoice3(local_model_dir, **kwargs)
    else:
        print(f"[CosyDaemon] downloading from modelscope...", flush=True)
        _cosy_model = CosyVoice3("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", **kwargs)
    print(f"[CosyDaemon] CosyVoice3 loaded ({time.time()-t0:.1f}s)", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _cosy_model is not None}


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    if _cosy_model is None:
        return SynthesizeResponse(audio_b64="", sample_rate=0, duration=0,
                                  success=False, error="model not loaded")
    if not os.path.exists(req.ref_audio_path):
        return SynthesizeResponse(audio_b64="", sample_rate=0, duration=0,
                                  success=False, error=f"ref not found: {req.ref_audio_path}")
    try:
        prefix = _build_prefix(req.text, req.tone, req.emotion)
        # ref 16kHz mono cast
        import tempfile, subprocess
        ref_16k = os.path.join(tempfile.gettempdir(), f"ref_16k_{os.getpid()}_{time.time_ns()}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", req.ref_audio_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", ref_16k,
        ], check=True, capture_output=True)
        outputs = []
        for result in _cosy_model.inference_cross_lingual(
            tts_text=f"{prefix}{req.text}",
            prompt_wav=ref_16k,
            stream=False,
            speed=req.speed,
        ):
            outputs.append(result["tts_speech"].squeeze().numpy())
        os.unlink(ref_16k)
        if not outputs:
            return SynthesizeResponse(audio_b64="", sample_rate=TTS_SAMPLE_RATE, duration=0,
                                      success=False, error="no output")
        wav = np.concatenate(outputs, axis=0).astype(np.float32)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav * (0.9 / peak)
        # CosyVoice3 출력 24000Hz → 16000Hz resample
        if _cosy_model.sample_rate != TTS_SAMPLE_RATE:
            import librosa
            wav = librosa.resample(wav, orig_sr=_cosy_model.sample_rate, target_sr=TTS_SAMPLE_RATE)
        # to bytes
        buf = io.BytesIO()
        sf.write(buf, wav, TTS_SAMPLE_RATE, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        return SynthesizeResponse(
            audio_b64=audio_b64,
            sample_rate=TTS_SAMPLE_RATE,
            duration=len(wav) / TTS_SAMPLE_RATE,
            success=True,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return SynthesizeResponse(audio_b64="", sample_rate=0, duration=0,
                                  success=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8901)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"[CosyDaemon] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
