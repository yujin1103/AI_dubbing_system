from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import format_chunk_id, get_logger, load_json, save_json

logger = get_logger("merge_speaker_chunks")


def merge_speaker_chunks(
    input_json: str | Path,
    output_json: str | Path,
    *,
    gap_threshold: float = 0.5,
    min_chunk_sec: float = 0.0,
    max_chunk_sec: float = 0.0,
) -> list[dict[str, Any]]:
    segments = load_json(input_json)
    ordered = sorted(segments, key=lambda item: float(item["start"]))

    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    source_count = 0
    max_chunk_sec = max(0.0, float(max_chunk_sec or 0.0))

    def finalize(item: dict[str, Any], count: int) -> None:
        duration = float(item["end"]) - float(item["start"])
        if duration < min_chunk_sec:
            logger.warning(
                "Dropping short chunk for %s (%.3fs < %.3fs)",
                item["speaker"],
                duration,
                min_chunk_sec,
            )
            return
        item["duration"] = round(duration, 3)
        item["source_segment_count"] = count
        merged.append(item)

    def split_segment(segment: dict[str, Any]) -> list[dict[str, Any]]:
        speaker = str(segment["speaker"])
        start = float(segment["start"])
        end = float(segment["end"])
        if max_chunk_sec <= 0 or end - start <= max_chunk_sec:
            return [{"speaker": speaker, "start": start, "end": end}]

        pieces: list[dict[str, Any]] = []
        cursor = start
        while cursor < end:
            piece_end = min(end, cursor + max_chunk_sec)
            pieces.append(
                {
                    "speaker": speaker,
                    "start": cursor,
                    "end": piece_end,
                    "chunking_split_reason": "max_chunk_sec",
                }
            )
            cursor = piece_end
        return pieces

    for segment in ordered:
        for piece in split_segment(segment):
            speaker = str(piece["speaker"])
            start = float(piece["start"])
            end = float(piece["end"])
            if current is None:
                current = dict(piece)
                source_count = 1
                continue

            gap = start - float(current["end"])
            merged_duration = max(float(current["end"]), end) - float(current["start"])
            can_merge_duration = max_chunk_sec <= 0 or merged_duration <= max_chunk_sec
            if speaker == current["speaker"] and gap <= gap_threshold and can_merge_duration:
                current["end"] = max(float(current["end"]), end)
                source_count += 1
                continue

            finalize(current, source_count)
            current = dict(piece)
            source_count = 1

    if current is not None:
        finalize(current, source_count)

    for index, item in enumerate(merged, start=1):
        item["chunk_id"] = format_chunk_id(index)

    save_json(merged, output_json)
    logger.info("Merged %s diarization rows into %s chunks", len(ordered), len(merged))
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge nearby same-speaker diarization segments into chunk spans.")
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    parser.add_argument("--gap-threshold", type=float, default=0.5)
    parser.add_argument("--min-chunk-sec", type=float, default=0.0)
    parser.add_argument("--max-chunk-sec", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    merge_speaker_chunks(
        args.input_json,
        args.output_json,
        gap_threshold=args.gap_threshold,
        min_chunk_sec=args.min_chunk_sec,
        max_chunk_sec=args.max_chunk_sec,
    )


if __name__ == "__main__":
    main()
