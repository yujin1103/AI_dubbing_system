"""4-way fusion diarization client (preserved).

Calls the preserved fusion daemon (src/daemons/fusion_diarize_daemon.py) over HTTP.

The fusion daemon aggregates 4 diarization engines:
  - DiariZen WavLM-large-s80-md-v2 (port 8913)
  - NeMo TitaNet-Large (port 8923)
  - pyannote/speaker-diarization-community-1 (port 8933)
  - pyannote/speaker-diarization-3.1 (port 8943)

All daemons must be running (use src/daemons/start_daemons.sh inside the
dubbing_pipeline container).

Output: a pyannote-style RTTM file at `output_rttm`.

Validated runs (boost 3.0x / win 4s / hop 3s):
  - test4 v305f: 6 SPK, gap_fill (mm=0.99 bm=0.30 sm=0.10) → score 0.9976
  - test5 v305f: 4 SPK + 1 BG, gap_fill (mm=0.40 bm=0.30 sm=0.45) → score 1.1667
"""
from __future__ import annotations

import os
from pathlib import Path

from common import get_logger, resolve_project_path

logger = get_logger("preserved_fusion")

DEFAULT_FUSION_URL = os.environ.get("FUSION_URL", "http://127.0.0.1:8903")


def diarize_with_4way_fusion(
    input_audio: str | Path,
    output_rttm: str | Path,
    *,
    num_speakers: int | None = None,
    min_duration: float = 0.3,
    timeout: int = 600,
    fusion_url: str | None = None,
    **_ignored,
) -> None:
    """Call the preserved fusion daemon and write its segments as RTTM."""
    import requests

    url = (fusion_url or DEFAULT_FUSION_URL).rstrip("/")
    audio_path = str(resolve_project_path(input_audio))
    rttm_path = resolve_project_path(output_rttm)
    rttm_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "vocals_wav": audio_path,
        "num_speakers": num_speakers,
        "min_duration": min_duration,
    }
    logger.info("preserved fusion: POST %s/diarize  audio=%s n_speakers=%s",
                url, audio_path, num_speakers)
    response = requests.post(f"{url}/diarize", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise RuntimeError(f"fusion daemon error: {data.get('error')}")
    segments = data.get("segments", [])
    logger.info("preserved fusion: %s segments, %s speakers", len(segments), data.get("n_speakers"))

    uri = Path(audio_path).stem
    lines = []
    for seg in segments:
        start = float(seg.get("start", seg.get("group_start", 0.0)))
        end = float(seg.get("end", seg.get("group_end", 0.0)))
        dur = max(0.0, end - start)
        spk = seg.get("speaker", "SPEAKER_00")
        lines.append(f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>")
    rttm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("preserved fusion: wrote %s", rttm_path)
