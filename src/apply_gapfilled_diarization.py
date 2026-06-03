"""Bridge: 검증된 gapfilled 화자분리를 청크 단계 입력(diarization_json)으로 export.

문제: apply_repair_patches 는 검증 결과를 run_dir/meta/{chunk}_segments_gapfilled.json
(groups 형식)에 쓰지만, merge_speaker_chunks 는 paths.diarization_json(repair 전 RTTM 결과)
을 읽는다 → 검증된 gapfilled(BG 화자·gap 회수 포함)가 청크로 흘러가지 않음.

해결: 이 모듈이 gapfilled groups 를 diarization segment 리스트로 변환해 diarization_json
(+ stabilized_json) 에 덮어쓴다. 이후 merge_speaker_chunks 가 검증된 화자 turn 을 청크로 묶는다.

사용:
    python apply_gapfilled_diarization.py <run_dir> <output_diarization_json> [--stabilized <json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import get_logger, save_json

logger = get_logger("apply_gapfilled_diarization")


def _load_gapfilled_segments(run_dir: str | Path) -> list[dict[str, Any]]:
    meta = Path(run_dir) / "meta"
    gf_files = sorted(meta.glob("*_segments_gapfilled.json"))
    if not gf_files:
        raise FileNotFoundError(f"no *_segments_gapfilled.json under {meta}")
    segments: list[dict[str, Any]] = []
    for gf in gf_files:
        data = json.load(open(gf, encoding="utf-8"))
        groups = data.get("groups", data) if isinstance(data, dict) else data
        for g in groups:
            if not isinstance(g, dict):
                continue
            start = float(g.get("group_start", g.get("start", 0.0)))
            end = float(g.get("group_end", g.get("end", 0.0)))
            spk = str(g.get("speaker", "SPEAKER_00"))
            if end > start:
                segments.append({"speaker": spk, "start": start, "end": end})
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments


def apply_gapfilled_to_diarization(
    run_dir: str | Path,
    output_json: str | Path,
    *,
    stabilized_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    segments = _load_gapfilled_segments(run_dir)
    save_json(segments, output_json)
    n_spk = len({s["speaker"] for s in segments})
    logger.info(
        "Exported gapfilled → %s : %d segments, %d speakers", output_json, len(segments), n_spk
    )
    if stabilized_json:
        save_json(segments, stabilized_json)
        logger.info("Also wrote stabilized copy → %s", stabilized_json)
    return segments


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export gapfilled diarization to diarization_json for chunking.")
    ap.add_argument("run_dir")
    ap.add_argument("output_json")
    ap.add_argument("--stabilized", default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_gapfilled_to_diarization(args.run_dir, args.output_json, stabilized_json=args.stabilized)


if __name__ == "__main__":
    main()
