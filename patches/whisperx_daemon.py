"""WhisperX ASR daemon — asr_daemon과 같은 인터페이스 (/transcribe), port 8902 교체용.

Whisper Large-v3 + wav2vec2 word-level forced alignment.
출력 형식: words=[{word, start, end}], detected_language, text

ENV:
  WHISPERX_MODEL (default "large-v3")
  WHISPERX_BATCH (default 16)
  WHISPERX_COMPUTE (default "float16")
  WHISPERX_DEVICE (default "cuda")
"""
import argparse
import os
import time
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


app = FastAPI()
_model = None
_align_model = None
_align_metadata = None
_align_lang = None


class TranscribeRequest(BaseModel):
    audio_path: str
    language: Optional[str] = "auto"


class TranscribeResponse(BaseModel):
    words: List[dict]
    detected_language: str
    text: str
    success: bool
    error: Optional[str] = None


@app.on_event("startup")
async def load_model():
    global _model, _align_model, _align_metadata, _align_lang
    model_name = os.environ.get("WHISPERX_MODEL", "large-v3")
    compute_type = os.environ.get("WHISPERX_COMPUTE", "float16")
    device = os.environ.get("WHISPERX_DEVICE", "cuda")
    print(f"[WhisperX] loading {model_name} ({compute_type}) on {device}...", flush=True)
    t0 = time.time()
    try:
        import whisperx
        _model = whisperx.load_model(model_name, device, compute_type=compute_type)
        print(f"[WhisperX] loaded ({time.time()-t0:.1f}s)", flush=True)
        # default align model (english)
        try:
            _align_model, _align_metadata = whisperx.load_align_model(language_code="en", device=device)
            _align_lang = "en"
            print(f"[WhisperX] align model (en) loaded", flush=True)
        except Exception as ae:
            print(f"[WhisperX] align model load failed: {ae}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WhisperX] load failed: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    global _align_model, _align_metadata, _align_lang
    if _model is None:
        return TranscribeResponse(words=[], detected_language="", text="",
                                  success=False, error="model not loaded")
    if not os.path.exists(req.audio_path):
        return TranscribeResponse(words=[], detected_language="", text="",
                                  success=False, error=f"audio not found: {req.audio_path}")
    try:
        import whisperx
        lang = req.language if req.language and req.language.lower() != "auto" else None
        audio = whisperx.load_audio(req.audio_path)
        batch_size = int(os.environ.get("WHISPERX_BATCH", "16"))
        if lang:
            result = _model.transcribe(audio, batch_size=batch_size, language=lang)
        else:
            result = _model.transcribe(audio, batch_size=batch_size)
        detected_lang = result.get("language", "") or lang or ""

        # align model (language-specific)
        if detected_lang and detected_lang != _align_lang:
            try:
                _align_model, _align_metadata = whisperx.load_align_model(
                    language_code=detected_lang, device="cuda"
                )
                _align_lang = detected_lang
                print(f"[WhisperX] reloaded align model for {detected_lang}", flush=True)
            except Exception as ae:
                print(f"[WhisperX] align load failed ({detected_lang}): {ae}", flush=True)

        words_out = []
        all_text = []
        if _align_model and result.get("segments"):
            try:
                aligned = whisperx.align(
                    result["segments"], _align_model, _align_metadata, audio, "cuda",
                    return_char_alignments=False,
                )
                for seg in aligned.get("segments", []):
                    text = seg.get("text", "").strip()
                    if text:
                        all_text.append(text)
                    for w in seg.get("words", []):
                        if "start" in w and "end" in w:
                            words_out.append({
                                "word": w.get("word", "").strip(),
                                "start": float(w["start"]),
                                "end": float(w["end"]),
                            })
            except Exception as e:
                print(f"[WhisperX] align failed: {e} — fallback segment-level", flush=True)
                # fallback: segment level
                for seg in result.get("segments", []):
                    words_out.append({
                        "word": seg.get("text", "").strip(),
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                    })
        else:
            for seg in result.get("segments", []):
                text = seg.get("text", "").strip()
                if text:
                    all_text.append(text)
                words_out.append({
                    "word": text,
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                })

        return TranscribeResponse(
            words=words_out,
            detected_language=detected_lang,
            text=" ".join(all_text),
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
    print(f"[WhisperX] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
