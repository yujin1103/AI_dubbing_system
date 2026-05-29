from __future__ import annotations

import argparse
from pathlib import Path

from common import get_logger, resolve_project_path, save_json
from stabilize_diarization import stabilize_diarization_file

logger = get_logger("rttm_to_json")


def parse_rttm_line(line: str) -> dict[str, float | str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split()
    if len(parts) < 8 or parts[0] != "SPEAKER":
        return None
    start = float(parts[3])
    duration = float(parts[4])
    speaker = parts[7]
    end = start + duration
    return {"speaker": speaker, "start": start, "end": end, "duration": duration}


def convert_rttm_to_json(input_rttm: str | Path, output_json: str | Path) -> list[dict[str, float | str]]:
    input_path = resolve_project_path(input_rttm)
    if not input_path.exists():
        raise FileNotFoundError(f"RTTM file not found: {input_path}")

    records: list[dict[str, float | str]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        record = parse_rttm_line(line)
        if record is not None:
            records.append(record)
    records.sort(key=lambda item: float(item["start"]))
    save_json(records, output_json)
    logger.info("Converted %s RTTM rows into %s", len(records), resolve_project_path(output_json))
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert RTTM diarization output into JSON.")
    parser.add_argument("input_rttm")
    parser.add_argument("output_json")
    parser.add_argument("--stabilized-output-json")
    parser.add_argument("--same-speaker-gap-sec", type=float, default=0.1)
    parser.add_argument("--bridge-max-sec", type=float, default=0.4)
    parser.add_argument("--bridge-gap-sec", type=float, default=0.25)
    parser.add_argument("--tiny-segment-sec", type=float, default=0.12)
    parser.add_argument("--absorb-gap-sec", type=float, default=0.25)
    parser.add_argument("--max-passes", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    convert_rttm_to_json(args.input_rttm, args.output_json)
    if args.stabilized_output_json:
        stabilize_diarization_file(
            args.output_json,
            args.stabilized_output_json,
            same_speaker_gap_sec=args.same_speaker_gap_sec,
            bridge_max_sec=args.bridge_max_sec,
            bridge_gap_sec=args.bridge_gap_sec,
            tiny_segment_sec=args.tiny_segment_sec,
            absorb_gap_sec=args.absorb_gap_sec,
            max_passes=args.max_passes,
        )


if __name__ == "__main__":
    main()
