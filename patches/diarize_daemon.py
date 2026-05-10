"""Diarize (DiariZen) 데몬 — 20-30초 모델 로딩 절감.

사용:
    /opt/venv_diarizen/bin/python diarize_daemon.py --port 8903

client:
    POST /diarize {"vocals_wav": "...", "num_speakers": null}
       → {"segments": [...], "n_speakers": ...}
"""
import argparse
import os
import sys
import time
from typing import Optional, List

# === sm_120 호환 환경변수 (DiariZen worker와 동일) ===
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
os.environ["CUDNN_FRONTEND_DISABLE_GRAPH"] = "1"
os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

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
    print(f"[DiarizeDaemon] loading DiariZen...", flush=True)
    t0 = time.time()
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
        _pipe = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
        print(f"[DiarizeDaemon] loaded ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[DiarizeDaemon] load failed: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(req: DiarizeRequest):
    if _pipe is None:
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error="model not loaded")
    if not os.path.exists(req.vocals_wav):
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error=f"file not found: {req.vocals_wav}")
    try:
        diar = _pipe(req.vocals_wav)
        segments = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            spk_str = f"SPEAKER_{int(speaker):02d}" if str(speaker).isdigit() else str(speaker)
            if turn.end - turn.start < req.min_duration:
                continue
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": spk_str,
            })
        return DiarizeResponse(
            segments=segments,
            n_speakers=len(set(s["speaker"] for s in segments)),
            success=True,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8903)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"[DiarizeDaemon] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
