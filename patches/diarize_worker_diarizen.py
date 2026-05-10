"""DiariZen 화자 분리 worker (별도 venv_diarizen에서 subprocess 호출).

Usage:
    /opt/venv_diarizen/bin/python diarize_worker_diarizen.py /path/to/vocals.wav

Output (stdout JSON):
    [
        {"start": 0.5, "end": 5.2, "speaker": "SPEAKER_00"},
        ...
    ]
"""
import argparse
import json
import os
import sys

# === sm_120 호환 환경변수 ===
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
os.environ["CUDNN_FRONTEND_DISABLE_GRAPH"] = "1"
os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
# === Tier 1A: cuDNN 비활성화 (sm_120 conv1d 호환 fix) ===
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# === PyTorch 2.6+ weights_only 호환 ===
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _safe_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vocals_wav")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="강제 화자 수 (모르면 자동)")
    parser.add_argument("--min-duration", type=float, default=0.3,
                        help="이 미만 turn은 인접 화자에 흡수")
    args = parser.parse_args()

    if not os.path.exists(args.vocals_wav):
        print(json.dumps({"error": f"file not found: {args.vocals_wav}"}))
        return 1

    try:
        from diarizen.pipelines.inference import DiariZenPipeline
    except ImportError as e:
        print(json.dumps({"error": f"diarizen import: {e}"}))
        return 1

    try:
        # 모델 로드
        pipe = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    except Exception as e:
        print(json.dumps({"error": f"load model: {e}"}))
        return 1

    try:
        diar = pipe(args.vocals_wav)
    except Exception as e:
        print(json.dumps({"error": f"inference: {e}"}))
        return 1

    # JSON 결과
    segments = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        # SPEAKER_X 통일 형식 (DiariZen은 number만 반환)
        spk_str = f"SPEAKER_{int(speaker):02d}" if str(speaker).isdigit() else str(speaker)
        if turn.end - turn.start < args.min_duration:
            continue
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": spk_str,
        })

    print(json.dumps({"segments": segments, "n_speakers": len(set(s["speaker"] for s in segments))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
