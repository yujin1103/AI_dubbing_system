"""pyannote-audio 3.1.1의 AudioMetaData import를 fallback로 patch."""
from pathlib import Path

p = Path('/opt/DiariZen/pyannote-audio/pyannote/audio/tasks/segmentation/mixins.py')
src = p.read_text()

old = "from torchaudio import AudioMetaData"
new = """try:
    from torchaudio import AudioMetaData
except ImportError:
    from collections import namedtuple
    AudioMetaData = namedtuple("AudioMetaData", ["sample_rate", "num_frames", "num_channels"])"""

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: AudioMetaData fallback 적용")
else:
    print("NOT FOUND")
