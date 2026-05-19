"""NeMo Speaker Diarization daemon — DiariZen 대체 (port 8903 동일 인터페이스).

NVIDIA NeMo ClusteringDiarizer 사용:
  - VAD: nemo_msdd_telephonic_v1 (또는 silero VAD 외부)
  - Speaker embedding: TitaNet-L (nvidia/speakerverification_en_titanet_large)
  - Clustering: AHC (Agglomerative Hierarchical Clustering)
  - Optional: MSDD (Multi-Scale Diarization Decoder, neural refinement)

DiariZen daemon과 동일한 HTTP API (orchestrator는 그대로 사용 가능):
  POST /diarize {"vocals_wav": "...", "num_speakers": null}
   → {"segments": [{"start": ..., "end": ..., "speaker": "SPEAKER_00"}, ...], "n_speakers": ...}

사용:
  /opt/venv_diarizen/bin/python nemo_diarize_daemon.py --port 8903

환경변수:
  NEMO_USE_MSDD (default 0) — 1이면 MSDD 적용 (느림, 정확도 ↑)
  NEMO_AHC_THRESHOLD (default 0.7) — AHC clustering 임계값
  NEMO_MAX_SPEAKERS (default 8) — 최대 화자 수
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# sm_120 호환 (DiariZen daemon과 동일)
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
_diarizer = None
_config_path = None  # NeMo ClusteringDiarizer config


class DiarizeRequest(BaseModel):
    vocals_wav: str
    num_speakers: Optional[int] = None
    min_duration: float = 0.3


class DiarizeResponse(BaseModel):
    segments: List[dict]
    n_speakers: int
    success: bool
    error: Optional[str] = None


def _build_nemo_config(use_msdd: bool = False) -> str:
    """ClusteringDiarizer config YAML 생성 후 path 반환.

    Reference: NeMo tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
    """
    import yaml
    ahc_threshold = float(os.environ.get("NEMO_AHC_THRESHOLD", "0.7"))
    max_speakers = int(os.environ.get("NEMO_MAX_SPEAKERS", "8"))
    vad_min_dur_off = float(os.environ.get("NEMO_VAD_MIN_DUR_OFF", "0.6"))
    vad_min_dur_on = float(os.environ.get("NEMO_VAD_MIN_DUR_ON", "0.0"))
    # NeMo ClusteringDiarizer parameters
    max_rp_threshold = float(os.environ.get("NEMO_MAX_RP_THRESHOLD", "0.25"))
    sparse_search_volume = int(os.environ.get("NEMO_SPARSE_SEARCH_VOL", "30"))
    enhanced_count_thres = int(os.environ.get("NEMO_ENHANCED_COUNT_THRES", "80"))
    sigmoid_threshold_str = os.environ.get("NEMO_SIGMOID_THRESHOLD", "0.7,1.0")
    sigmoid_threshold = [float(x) for x in sigmoid_threshold_str.split(",")]
    vad_onset = float(os.environ.get("NEMO_VAD_ONSET", "0.5"))
    vad_offset = float(os.environ.get("NEMO_VAD_OFFSET", "0.3"))

    cfg = {
        "name": "ClusterDiarizer",
        "num_workers": 1,
        "sample_rate": 16000,
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": True,
        "diarizer": {
            "manifest_filepath": None,  # 호출 시 set
            "out_dir": None,            # 호출 시 set
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.63,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.5,
                    "onset": vad_onset,
                    "offset": vad_offset,
                    "pad_onset": 0.0,
                    "pad_offset": 0.0,
                    "min_duration_on": vad_min_dur_on,
                    "min_duration_off": vad_min_dur_off,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                    "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": max_speakers,
                    "enhanced_count_thres": enhanced_count_thres,
                    "max_rp_threshold": max_rp_threshold,
                    "sparse_search_volume": sparse_search_volume,
                    "maj_vote_spk_count": False,
                },
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic" if use_msdd else None,
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    "infer_batch_size": 25,
                    "sigmoid_threshold": sigmoid_threshold,
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "diar_window_length": 50,
                    "overlap_infer_spk_limit": 5,
                },
            },
            "asr": {
                "model_path": None,
                "parameters": {
                    "asr_based_vad": False,
                },
            },
        },
    }

    if not use_msdd:
        cfg["diarizer"].pop("msdd_model", None)

    tmp_yaml = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp_yaml)
    tmp_yaml.close()
    return tmp_yaml.name


@app.on_event("startup")
async def load_model():
    """NeMo ClusteringDiarizer는 manifest/wav 별로 새 instance 생성 권장.
    여기서는 model_path를 미리 caching만."""
    global _diarizer, _config_path
    print(f"[NemoDiarize] preparing NeMo config...", flush=True)
    t0 = time.time()
    try:
        use_msdd = os.environ.get("NEMO_USE_MSDD", "0") == "1"
        _config_path = _build_nemo_config(use_msdd=use_msdd)
        print(f"[NemoDiarize] config: {_config_path} (msdd={use_msdd})", flush=True)
        # Pre-import to warm up
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_config_path)
        print(f"[NemoDiarize] config loaded ({time.time()-t0:.1f}s)", flush=True)
        _diarizer = "ready"  # actual instance created per-request
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[NemoDiarize] init failed: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _diarizer is not None}


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(req: DiarizeRequest):
    if _diarizer is None or _config_path is None:
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error="NeMo not initialized")
    if not os.path.exists(req.vocals_wav):
        return DiarizeResponse(segments=[], n_speakers=0,
                               success=False, error=f"file not found: {req.vocals_wav}")

    try:
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf

        # per-request output dir
        out_dir = tempfile.mkdtemp(prefix="nemo_diar_")
        manifest_path = os.path.join(out_dir, "input_manifest.json")

        # 16kHz mono로 변환 (NeMo 요구)
        prepared_wav = os.path.join(out_dir, "input_16k.wav")
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", req.vocals_wav,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", prepared_wav,
        ], check=True, capture_output=True)

        # NeMo manifest 작성
        manifest_entry = {
            "audio_filepath": prepared_wav,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": req.num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        cfg = OmegaConf.load(_config_path)
        cfg.diarizer.manifest_filepath = manifest_path
        cfg.diarizer.out_dir = out_dir
        if req.num_speakers:
            cfg.diarizer.clustering.parameters.oracle_num_speakers = True

        diarizer_inst = ClusteringDiarizer(cfg=cfg)
        diarizer_inst.diarize()

        # RTTM 파일 파싱
        rttm_dir = os.path.join(out_dir, "pred_rttms")
        stem = Path(prepared_wav).stem
        rttm_path = os.path.join(rttm_dir, f"{stem}.rttm")
        segments = []
        speakers_found = set()
        if os.path.exists(rttm_path):
            with open(rttm_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8 or parts[0] != "SPEAKER":
                        continue
                    start = float(parts[3])
                    dur = float(parts[4])
                    spk = parts[7]
                    if dur < req.min_duration:
                        continue
                    spk_label = f"SPEAKER_{int(spk):02d}" if spk.isdigit() else str(spk)
                    speakers_found.add(spk_label)
                    segments.append({
                        "start": round(start, 3),
                        "end": round(start + dur, 3),
                        "speaker": spk_label,
                    })

        # cleanup
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass

        return DiarizeResponse(
            segments=segments,
            n_speakers=len(speakers_found),
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
    print(f"[NemoDiarize] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
