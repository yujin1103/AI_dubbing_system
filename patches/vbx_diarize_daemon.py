"""VBx (BUT-FIT) diarization daemon — port 8953.

ResNet101 ONNX x-vector + AHC clustering (cosine distance threshold).
PLDA + VBHMM은 생략 (단순화).

ENV:
  VBX_PATH (default /tmp/VBx)
  VBX_ONNX (default models/ResNet101_16kHz/nnet/final.onnx)
  VBX_SEG_LEN (default 144 frames = 1.44s)
  VBX_SEG_JUMP (default 24 frames = 0.24s)
  VBX_AHC_THRESHOLD (default 0.65) — cosine distance threshold for AHC
"""
import argparse
import os
import sys
import time
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
_session = None
_input_name = None
_output_name = None
_features_mod = None
_plda_mu = None
_plda_tr = None
_plda_psi = None
_use_plda = False
_lda_mean1 = None
_lda_mean2 = None
_lda_mat = None


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
    global _session, _input_name, _output_name, _features_mod
    global _plda_mu, _plda_tr, _plda_psi, _use_plda
    vbx_path = os.environ.get("VBX_PATH", "/tmp/VBx")
    sys.path.insert(0, os.path.join(vbx_path, "VBx"))
    onnx_rel = os.environ.get("VBX_ONNX", "models/ResNet101_16kHz/nnet/final.onnx")
    onnx_path = os.path.join(vbx_path, "VBx", onnx_rel)
    print(f"[VBx] loading {onnx_path}", flush=True)
    t0 = time.time()
    try:
        import onnxruntime as ort
        import features as vbx_features
        _features_mod = vbx_features
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        _session = ort.InferenceSession(onnx_path, providers=providers)
        _input_name = _session.get_inputs()[0].name
        _output_name = _session.get_outputs()[0].name
        print(f"[VBx] loaded ({time.time()-t0:.1f}s) — input={_input_name} output={_output_name}", flush=True)
        # v113+: PLDA + LDA transform 로드 옵션
        _use_plda = os.environ.get("VBX_USE_PLDA", "0") == "1"
        if _use_plda:
            model_dir = os.path.join(vbx_path, "VBx", os.path.dirname(onnx_rel).replace("/nnet", ""))
            plda_path = os.path.join(model_dir, "plda")
            transform_path = os.path.join(model_dir, "transform.h5")
            from kaldi_utils import read_plda
            _plda_mu, _plda_tr, _plda_psi = read_plda(plda_path)
            print(f"[VBx] PLDA loaded: mu={_plda_mu.shape}, tr={_plda_tr.shape}, psi={_plda_psi.shape}", flush=True)
            # LDA transform (256-dim → 128-dim)
            import h5py
            import numpy as _np
            with h5py.File(transform_path, "r") as f:
                _lda_mean1 = _np.array(f["mean1"])
                _lda_mean2 = _np.array(f["mean2"])
                _lda_mat = _np.array(f["lda"])
            print(f"[VBx] LDA loaded: mean1={_lda_mean1.shape}, mean2={_lda_mean2.shape}, lda={_lda_mat.shape}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[VBx] load failed: {e}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _session is not None}


def _extract_features(audio, sr=16000):
    """VBx style mel features (predict.py 순서 그대로)."""
    import numpy as np
    window_ms = 25
    shift_ms = 10
    winlen = int(sr * window_ms / 1000)  # 400
    shift = int(sr * shift_ms / 1000)    # 160
    noverlap = winlen - shift            # 240
    fbank_mx = _features_mod.mel_fbank_mx(winlen, sr, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7700, htk_bug=False)
    # int16 변환 + dither
    signal_int = (audio * (2 ** 15)).astype(int)
    signal_dithered = _features_mod.add_dither(signal_int, level=8)
    signal_f = signal_dithered.astype(np.float32)
    feats = _features_mod.fbank_htk(signal_f, winlen, noverlap, fbank_mx,
                                    USEPOWER=True, ZMEANSOURCE=True)
    feats = _features_mod.cmvn_floating_kaldi(feats, LC=150, RC=149, norm_vars=False).astype(np.float32)
    return feats


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(req: DiarizeRequest):
    if _session is None:
        return DiarizeResponse(segments=[], n_speakers=0, success=False, error="vbx not loaded")
    if not os.path.exists(req.vocals_wav):
        return DiarizeResponse(segments=[], n_speakers=0, success=False,
                               error=f"file not found: {req.vocals_wav}")
    try:
        import numpy as np
        import soundfile as sf

        wav, sr = sf.read(req.vocals_wav)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        if sr != 16000:
            import scipy.signal as ss
            wav = ss.resample_poly(wav, 16000, sr)
            sr = 16000
        feats = _extract_features(wav, sr)

        seg_len = int(os.environ.get("VBX_SEG_LEN", "144"))   # frames (1.44s @ 10ms shift)
        seg_jump = int(os.environ.get("VBX_SEG_JUMP", "24"))  # frames (0.24s)

        xvectors = []
        seg_times = []
        slen = feats.shape[0]
        for start in range(0, slen - seg_len, seg_jump):
            data = feats[start:start + seg_len]  # (144, 64)
            # ONNX 입력 (1, 64, 144) — predict.py 참고
            inp = data.astype(np.float32).transpose()[np.newaxis, :, :]
            out = _session.run([_output_name], {_input_name: inp})[0]
            xvec = out.squeeze()
            if xvec.ndim > 1:
                xvec = xvec.flatten()
            xvectors.append(xvec)
            # 시간 변환 (frame_idx → sec): frame_idx * 10ms
            seg_times.append((start * 0.01, (start + seg_len) * 0.01))

        if len(xvectors) < 2:
            return DiarizeResponse(segments=[], n_speakers=0, success=False, error="too few segments")

        xvec_arr = np.stack(xvectors)
        # L2 normalize
        norms = np.linalg.norm(xvec_arr, axis=1, keepdims=True)
        xvec_arr = xvec_arr / np.maximum(norms, 1e-9)

        # v113+: LDA + PLDA transform (사용 시)
        if _use_plda and _plda_mu is not None and _lda_mat is not None:
            # LDA transform (256 → 128)
            from sklearn.preprocessing import normalize as _l2n
            x = _l2n(xvec_arr - _lda_mean1, axis=1, norm='l2')
            x = _lda_mat.T.dot(x.transpose()).transpose() - _lda_mean2
            x = _l2n(x, axis=1, norm='l2')
            # PLDA centering + transform
            x = (x - _plda_mu) @ _plda_tr.T
            x = _l2n(x, axis=1, norm='l2')
            xvec_arr = x
            print(f"[VBx] LDA+PLDA-transformed: shape={xvec_arr.shape}", flush=True)

        # AHC clustering (cosine distance threshold)
        dist_thr = float(os.environ.get("VBX_AHC_THRESHOLD", "0.65"))
        from sklearn.cluster import AgglomerativeClustering
        if req.num_speakers and req.num_speakers > 0:
            clusterer = AgglomerativeClustering(
                n_clusters=req.num_speakers, metric="cosine", linkage="average",
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None, distance_threshold=dist_thr,
                metric="cosine", linkage="average",
            )
        labels = clusterer.fit_predict(xvec_arr)
        n_clusters = len(set(labels))
        print(f"[VBx] {len(xvectors)} x-vectors → {n_clusters} clusters (dist_thr={dist_thr})", flush=True)

        # window → segments (label 변화 시 새 segment)
        segs = []
        cur_label = None
        cur_start = None
        for i, lbl in enumerate(labels):
            t_start, t_end = seg_times[i]
            if cur_label is None:
                cur_label = lbl
                cur_start = t_start
                cur_end_running = t_end
                continue
            if lbl == cur_label:
                cur_end_running = t_end
                continue
            # label changed
            if cur_end_running - cur_start >= req.min_duration:
                segs.append({
                    "start": round(cur_start, 3),
                    "end": round(cur_end_running, 3),
                    "speaker": f"SPEAKER_{int(cur_label):02d}",
                })
            cur_label = lbl
            cur_start = t_start
            cur_end_running = t_end
        # 마지막
        if cur_label is not None and cur_end_running - cur_start >= req.min_duration:
            segs.append({
                "start": round(cur_start, 3),
                "end": round(cur_end_running, 3),
                "speaker": f"SPEAKER_{int(cur_label):02d}",
            })

        speakers = set(s["speaker"] for s in segs)
        return DiarizeResponse(segments=segs, n_speakers=len(speakers), success=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DiarizeResponse(segments=[], n_speakers=0, success=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8953)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"[VBx] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
