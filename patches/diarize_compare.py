"""htdemucs vs BS-Roformer vocals → pyannote diarization 비교."""
import os
import sys
import time
from pathlib import Path

HTDEMUCS_VOCALS = "/tmp/htdemucs_test/htdemucs/audio_trimmed/vocals.wav"
BS_VOCALS = "/tmp/bsroformer_test/audio_trimmed_(Vocals)_model_bs_roformer_ep_317_sdr_12.wav"

# 환경변수로 token 가져오기
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    # .env 파일에서 직접 읽기
    env_path = "/workspace/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_HUB_TOKEN="):
                    HF_TOKEN = line.strip().split("=", 1)[1].strip('"').strip("'")
                    break
if not HF_TOKEN:
    print("[FAIL] HF_TOKEN not found in env or .env")
    sys.exit(1)
print(f"[Init] HF_TOKEN found (len={len(HF_TOKEN)})")

from pyannote.audio import Pipeline
print("[Init] pyannote imported")

# 모델 로드 (한 번만)
print("[Load] pyannote/speaker-diarization-3.1...")
t0 = time.time()
pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN,
)
import torch
pipe.to(torch.device("cuda"))
print(f"[Load] OK ({time.time()-t0:.1f}s)")

# 두 vocals 모두 diarize
for label, vocals_path in [("htdemucs", HTDEMUCS_VOCALS), ("BS-Roformer", BS_VOCALS)]:
    print(f"\n{'='*60}")
    print(f"{label}: {vocals_path}")
    print('='*60)
    t0 = time.time()
    diarization = pipe(vocals_path)
    print(f"  diarize 시간: {time.time()-t0:.1f}s")

    # pyannote 4.x: DiarizeOutput에서 annotation 꺼내야 함
    annot = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization

    # 화자 통계
    speakers = set()
    segments = []
    for turn, _, speaker in annot.itertracks(yield_label=True):
        speakers.add(speaker)
        segments.append((turn.start, turn.end, speaker))

    print(f"  화자 수: {len(speakers)} ({sorted(speakers)})")
    print(f"  segment 수: {len(segments)}")

    # 화자별 발화 시간
    spk_durations = {}
    for s, e, spk in segments:
        spk_durations[spk] = spk_durations.get(spk, 0) + (e - s)
    print(f"  화자별 발화 시간:")
    for spk, dur in sorted(spk_durations.items(), key=lambda x: -x[1]):
        print(f"    {spk}: {dur:.1f}s")

    # 1초 이상 segment만
    long_segs = [s for s in segments if s[1] - s[0] >= 1.0]
    print(f"  1초 이상 segment: {len(long_segs)}")

    # 처음 5개 segment
    print(f"  처음 5개 segment:")
    for s, e, spk in segments[:5]:
        print(f"    [{s:.2f}~{e:.2f}] {spk}")

print("\n[DONE]")
