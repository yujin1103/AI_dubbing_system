"""DiariZen pipeline 단독 테스트 (test3 audio)."""
import os
import sys
import time

import os
# Tier 1C: sm_120 + cuDNN 호환 환경변수
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
os.environ["CUDNN_FRONTEND_DISABLE_GRAPH"] = "1"
os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# PyTorch 2.6+ weights_only 호환 patch
import torch
# Tier 1A: cuDNN 비활성화 (가장 가능성 높은 sm_120 fix)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
try:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    # 추가: pyannote/lightning이 사용할 만한 클래스
    import collections
    torch.serialization.add_safe_globals([collections.OrderedDict])
except Exception:
    pass
# torch.load weights_only=False monkey-patch
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False  # force override
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

print("=== DiariZen 테스트 시작 ===")

try:
    from diarizen.pipelines.inference import DiariZenPipeline
    print("[OK] diarizen import")
except Exception as e:
    print(f"[FAIL] diarizen import: {e}")
    sys.exit(1)

# pre-trained model 로드
print("\n[Load] BUT-FIT/diarizen-wavlm-large-s80-md-v2 ...")
t0 = time.time()
try:
    pipe = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    print(f"[Load] OK ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"[Load] FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# test3 audio (vocals만)
audio_path = "/workspace/media/runs/20260506_011047_test3v20_eeafe3/vocals/test3v20_chunk_000_clean_vocals.wav"
if not os.path.exists(audio_path):
    audio_path = "/workspace/media/runs/20260506_011047_test3v20_eeafe3/vocals/test3v20_chunk_000_vocals.wav"

print(f"\n[Inference] {audio_path}")
t0 = time.time()
try:
    diar = pipe(audio_path)
    print(f"[Inference] OK ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"[Inference] FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 결과 출력
print("\n=== 화자 분리 결과 ===")
speakers = set()
for turn, _, speaker in diar.itertracks(yield_label=True):
    speakers.add(speaker)
    print(f"  [{turn.start:.2f}~{turn.end:.2f}s] SPEAKER_{speaker}")

print(f"\n총 화자 수: {len(speakers)} ({sorted(speakers)})")
