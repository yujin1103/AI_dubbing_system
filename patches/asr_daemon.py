"""ASR (Qwen3) 데몬 — 30-45초 모델 로딩 절감.

asr_worker.py의 main() 코드를 그대로 reuse. 모델은 1회 로드 후 메모리 상주.
"""
import argparse
import os
import sys
import time
import gc
from typing import Optional, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
_model = None


class TranscribeRequest(BaseModel):
    audio_path: str
    language: Optional[str] = None  # "English", "Korean", None=auto


class TranscribeResponse(BaseModel):
    words: List[dict]
    detected_language: str
    text: str
    success: bool
    error: Optional[str] = None


@app.on_event("startup")
async def load_model():
    global _model
    print(f"[AsrDaemon] loading Qwen3-ASR-1.7B...", flush=True)
    t0 = time.time()
    try:
        from qwen_asr import Qwen3ASRModel
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            max_inference_batch_size=1,
            max_new_tokens=512,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device,
            ),
        )
        print(f"[AsrDaemon] loaded ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[AsrDaemon] load failed: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    if _model is None:
        return TranscribeResponse(words=[], detected_language="", text="",
                                  success=False, error="model not loaded")
    if not os.path.exists(req.audio_path):
        return TranscribeResponse(words=[], detected_language="", text="",
                                  success=False, error=f"audio not found: {req.audio_path}")
    try:
        # "auto"는 None으로 변환 (Qwen3-ASR auto-detect)
        lang = req.language
        if lang and lang.lower() == "auto":
            lang = None
        results = _model.transcribe(
            audio=req.audio_path,
            language=lang,
            return_time_stamps=True,
        )
        if not results:
            return TranscribeResponse(words=[], detected_language="", text="",
                                      success=True, error="no result")
        r = results[0]
        words = []
        timestamps = getattr(r, 'time_stamps', None) or []
        for ts in timestamps:
            if hasattr(ts, 'text') and hasattr(ts, 'start_time'):
                words.append({
                    "word": ts.text,
                    "start": float(ts.start_time),
                    "end": float(ts.end_time),
                })
            elif isinstance(ts, (list, tuple)) and len(ts) >= 3:
                words.append({
                    "word": ts[0],
                    "start": float(ts[1]),
                    "end": float(ts[2]),
                })
            elif hasattr(ts, 'word'):
                words.append({
                    "word": ts.word,
                    "start": float(ts.start),
                    "end": float(ts.end),
                })
        return TranscribeResponse(
            words=words,
            detected_language=r.language if hasattr(r, 'language') else "",
            text=r.text if hasattr(r, 'text') else "",
            success=True,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TranscribeResponse(words=[], detected_language="", text="",
                                  success=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8902)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"[AsrDaemon] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
