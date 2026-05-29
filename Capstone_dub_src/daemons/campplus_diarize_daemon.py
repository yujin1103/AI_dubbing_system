"""Alibaba CAM++ (3D-Speaker) Speaker Diarization daemon — port 8903.

modelscope damo/speech_campplus_speaker-diarization_common 사용.
CAM++ embedding + VAD + clustering (umap/hdbscan).

Note: torchaudio.sox_effects 없는 신버전 호환을 위해 16kHz로 ffmpeg 사전 변환.

사용:
  /opt/venv_diarizen/bin/python campplus_diarize_daemon.py --port 8903
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

os.environ.setdefault("MODELSCOPE_CACHE", "/workspace/media/model_cache/modelscope")
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
os.environ["CUDNN_FRONTEND_DISABLE_GRAPH"] = "1"
os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
_pipe = None


class DiarizeRequest(BaseModel):
    vocals_wav: str
    num_speakers: Optional[int] = None
    min_duration: float = 0.3


class DiarizeResponse(BaseModel):
    segments: List[dict]
    n_speakers: int
    success: bool
    error: Optional[str] = None


@app.on_event("startup")
async def load_model():
    global _pipe
    model_id = os.environ.get(
        "CAMPPLUS_MODEL", "damo/speech_campplus_speaker-diarization_common"
    )
    print(f"[CAMPP] loading {model_id}...", flush=True)
    t0 = time.time()
    try:
        from modelscope.pipelines import pipeline
        _pipe = pipeline("speaker-diarization", model=model_id)
        print(f"[CAMPP] loaded ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[CAMPP] load failed: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


def _ensure_16k_mono(path_in: str) -> str:
    """ffmpeg로 16kHz mono로 변환. modelscope sox_effects 회피용."""
    out = tempfile.NamedTemporaryFile(suffix="_16k.wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path_in,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out],
        check=True, capture_output=True,
    )
    return out


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(req: DiarizeRequest):
    if _pipe is None:
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error="CAM++ not loaded")
    if not os.path.exists(req.vocals_wav):
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error=f"file not found: {req.vocals_wav}")
    converted = None
    try:
        converted = _ensure_16k_mono(req.vocals_wav)
        result = _pipe(audio=converted)
        # CAM++ output format: {"text": [{"start": float, "stop": float, "spk": int}, ...]}
        # 또는 {"text": "..."}, {"segments": [...]} 다양. 표준화.
        segments = []
        speakers = set()
        # 다양한 key 시도
        items = None
        if isinstance(result, dict):
            for k in ("text", "segments", "result"):
                if k in result and isinstance(result[k], list):
                    items = result[k]
                    break
        if items is None:
            return DiarizeResponse(
                segments=[], n_speakers=0, success=False,
                error=f"unknown result format: {type(result).__name__} keys={list(result.keys()) if isinstance(result, dict) else None}",
            )
        for it in items:
            # CAM++ format: [start, stop, "spk_id"]
            if isinstance(it, (list, tuple)) and len(it) >= 3:
                s, e, spk = it[0], it[1], it[2]
            elif isinstance(it, dict):
                s = it.get("start")
                e = it.get("stop") if "stop" in it else it.get("end")
                spk = it.get("spk") if "spk" in it else it.get("speaker")
            else:
                continue
            if s is None or e is None or spk is None:
                continue
            if (e - s) < req.min_duration:
                continue
            spk_label = f"SPEAKER_{int(spk):02d}" if isinstance(spk, (int,)) or str(spk).isdigit() else str(spk)
            speakers.add(spk_label)
            segments.append({
                "start": round(float(s), 3),
                "end": round(float(e), 3),
                "speaker": spk_label,
            })
        return DiarizeResponse(
            segments=segments, n_speakers=len(speakers), success=True,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error=str(e))
    finally:
        if converted and os.path.exists(converted):
            try:
                os.unlink(converted)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8903)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"[CAMPP] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
