"""pyannote 3.1 Speaker Diarization daemon — port 8903 동일 인터페이스.

사용:
  /opt/venv_diarizen/bin/python pyannote_diarize_daemon.py --port 8903

환경변수:
  PYANNOTE_MODEL (default "pyannote/speaker-diarization-3.1")
                 alt: "pyannote/speaker-diarization-community-1"
  PYANNOTE_MIN_SPEAKERS / PYANNOTE_MAX_SPEAKERS (optional auto-detect bounds)
  HF_HUB_OFFLINE=1 권장 (모델 이미 cache됨)
"""
import argparse
import os
import sys
import time
from typing import List, Optional

# HF_HUB_OFFLINE 활성 시 metadata head 요청도 막혀서 cache가 있어도 안 됨.
# 대신 HF_HUB_DISABLE_TELEMETRY + LOCAL_FILES_ONLY 다른 방식 사용.
# 또는 명시적 local snapshot path 사용 (PYANNOTE_LOCAL_PATH env).
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
os.environ["CUDNN_FRONTEND_DISABLE_GRAPH"] = "1"
os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# v51 (venv_pyann, pyannote 4.x): cudnn 안 끄고 default 사용 가능
torch.backends.cudnn.benchmark = False
# PyTorch 2.6 weights_only=True default 회피
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
    model_id = os.environ.get(
        "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"
    )
    local_path = os.environ.get("PYANNOTE_LOCAL_PATH", "")
    print(f"[PyAnnote] loading {model_id} (local_path={local_path})...", flush=True)
    t0 = time.time()
    try:
        from pyannote.audio import Pipeline
        if local_path and os.path.exists(local_path):
            # local snapshot path 직접 사용
            print(f"[PyAnnote] using local snapshot: {local_path}", flush=True)
            _pipe = Pipeline.from_pretrained(local_path)
        else:
            _pipe = Pipeline.from_pretrained(model_id)
        if torch.cuda.is_available():
            _pipe = _pipe.to(torch.device("cuda"))
        print(f"[PyAnnote] loaded ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[PyAnnote] load failed: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(req: DiarizeRequest):
    if _pipe is None:
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error="pyannote not loaded")
    if not os.path.exists(req.vocals_wav):
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error=f"file not found: {req.vocals_wav}")
    try:
        # Determinism: re-seed RNG before each inference
        import random as _r, numpy as _np, torch as _t
        _r.seed(42); _np.random.seed(42); _t.manual_seed(42)
        if _t.cuda.is_available():
            _t.cuda.manual_seed_all(42)
        # pyannote는 16kHz mono wav를 직접 로드 (ffmpeg 변환 불필요, 자동 처리)
        kwargs = {}
        if req.num_speakers:
            kwargs["num_speakers"] = req.num_speakers
        # auto-detect bounds
        mn = os.environ.get("PYANNOTE_MIN_SPEAKERS")
        mx = os.environ.get("PYANNOTE_MAX_SPEAKERS")
        if mn:
            kwargs["min_speakers"] = int(mn)
        if mx:
            kwargs["max_speakers"] = int(mx)
        diarization = _pipe(req.vocals_wav, **kwargs)

        # pyannote 3.x → Annotation (.itertracks)
        # pyannote 4.x (community-1) → DiarizeOutput (.speaker_diarization Annotation)
        if hasattr(diarization, "itertracks"):
            iter_obj = diarization
        elif hasattr(diarization, "speaker_diarization"):
            iter_obj = diarization.speaker_diarization
        elif hasattr(diarization, "diarization"):
            iter_obj = diarization.diarization
        else:
            return DiarizeResponse(
                segments=[], n_speakers=0, success=False,
                error=f"unknown output type: {type(diarization).__name__}, attrs: {[a for a in dir(diarization) if not a.startswith('_')]}",
            )

        segments = []
        speakers = set()
        for turn, _, spk in iter_obj.itertracks(yield_label=True):
            if (turn.end - turn.start) < req.min_duration:
                continue
            # spk 형식: "SPEAKER_00" 또는 "0" 등 — 표준화
            if str(spk).isdigit():
                spk_label = f"SPEAKER_{int(spk):02d}"
            elif str(spk).startswith("SPEAKER_"):
                spk_label = str(spk)
            else:
                spk_label = f"SPEAKER_{spk}"
            speakers.add(spk_label)
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": spk_label,
            })
        return DiarizeResponse(
            segments=segments,
            n_speakers=len(speakers),
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
    print(f"[PyAnnote] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
