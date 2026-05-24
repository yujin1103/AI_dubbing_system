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
    # env: DIARIZEN_MODEL — model_id override (default v2)
    model_id = os.environ.get(
        "DIARIZEN_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    )
    print(f"[DiarizeDaemon] loading DiariZen ({model_id})...", flush=True)
    t0 = time.time()
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
        _pipe = DiariZenPipeline.from_pretrained(model_id)
        # v38b (5/15): load 후 attribute 직접 override (config_parse 인자 API 없음)
        # 환경변수가 있으면 적용 시도; 실패해도 정상 load 유지.
        ahc_thr = os.environ.get("DIARIZEN_AHC_THRESHOLD")
        if ahc_thr:
            try:
                ahc_thr_f = float(ahc_thr)
                # DiariZenPipeline에 cluster_pipeline 또는 비슷한 attribute가 있을 가능성
                applied = False
                for attr_name in ["cluster_pipeline", "clustering", "_cluster"]:
                    if hasattr(_pipe, attr_name):
                        obj = getattr(_pipe, attr_name)
                        if hasattr(obj, "ahc_threshold"):
                            setattr(obj, "ahc_threshold", ahc_thr_f)
                            print(f"[DiarizeDaemon] {attr_name}.ahc_threshold = {ahc_thr_f}", flush=True)
                            applied = True
                            break
                if not applied:
                    print(f"[DiarizeDaemon] ahc_threshold attribute 찾지 못함 (default 0.6 유지)", flush=True)
                    # debug: 사용 가능한 attribute 출력
                    pipe_attrs = [a for a in dir(_pipe) if not a.startswith('_')]
                    print(f"[DiarizeDaemon] available attrs: {pipe_attrs[:20]}", flush=True)
            except Exception as _se:
                print(f"[DiarizeDaemon] ahc_threshold override 실패: {_se}", flush=True)
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
        # Determinism: re-seed RNG before each inference (AHC/k-means rely on it)
        import random as _r, numpy as _np
        _r.seed(42); _np.random.seed(42); torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
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
