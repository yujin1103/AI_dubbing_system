# 화자분리 결과의 짧은 라벨 튐을 보수적으로 안정화한다.
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import get_logger, load_json, save_json

logger = get_logger("stabilize_diarization")


def _duration(segment: dict[str, Any]) -> float:
    return float(segment["end"]) - float(segment["start"])


def _gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(0.0, float(right["start"]) - float(left["end"]))


def _normalize_segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: float(item.get("start", 0.0))):
        start = float(row.get("start", 0.0))
        end = float(row.get("end", start))
        speaker = str(row.get("speaker", "") or "").strip()
        if not speaker or end <= start:
            continue
        normalized.append(
            {
                **row,
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
            }
        )
    return normalized


def _finalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        finalized.append(
            {
                **segment,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            }
        )
    return finalized


def _merge_same_speaker(
    segments: list[dict[str, Any]],
    *,
    same_speaker_gap_sec: float,
) -> tuple[list[dict[str, Any]], int]:
    merged: list[dict[str, Any]] = []
    merge_count = 0
    for segment in segments:
        current = dict(segment)
        if merged and current["speaker"] == merged[-1]["speaker"] and _gap(merged[-1], current) <= same_speaker_gap_sec:
            merged[-1]["end"] = max(float(merged[-1]["end"]), float(current["end"]))
            merged[-1]["duration"] = round(_duration(merged[-1]), 3)
            merge_count += 1
            continue
        merged.append(current)
    return merged, merge_count


def _choose_neighbor_speaker(
    segments: list[dict[str, Any]],
    index: int,
    *,
    absorb_gap_sec: float,
) -> str | None:
    previous = segments[index - 1] if index > 0 else None
    current = segments[index]
    next_segment = segments[index + 1] if index + 1 < len(segments) else None
    candidates: list[tuple[float, float, str]] = []

    if previous is not None:
        previous_gap = _gap(previous, current)
        if previous_gap <= absorb_gap_sec:
            candidates.append((previous_gap, -_duration(previous), str(previous["speaker"])))
    if next_segment is not None:
        next_gap = _gap(current, next_segment)
        if next_gap <= absorb_gap_sec:
            candidates.append((next_gap, -_duration(next_segment), str(next_segment["speaker"])))

    if previous is not None and next_segment is not None:
        previous_gap = _gap(previous, current)
        next_gap = _gap(current, next_segment)
        if previous["speaker"] == next_segment["speaker"] and previous_gap <= absorb_gap_sec and next_gap <= absorb_gap_sec:
            return str(previous["speaker"])

    if not candidates:
        return None
    candidates.sort()
    speaker = candidates[0][2]
    return None if speaker == current["speaker"] else speaker


def _absorb_tiny_segments(
    segments: list[dict[str, Any]],
    *,
    tiny_segment_sec: float,
    absorb_gap_sec: float,
) -> int:
    changed = 0
    for index, segment in enumerate(segments):
        if _duration(segment) > tiny_segment_sec:
            continue
        speaker = _choose_neighbor_speaker(segments, index, absorb_gap_sec=absorb_gap_sec)
        if speaker is None:
            continue
        segment["speaker"] = speaker
        changed += 1
    return changed


def _bridge_aba_segments(
    segments: list[dict[str, Any]],
    *,
    bridge_max_sec: float,
    bridge_gap_sec: float,
) -> int:
    changed = 0
    for index in range(1, len(segments) - 1):
        previous = segments[index - 1]
        current = segments[index]
        next_segment = segments[index + 1]
        if previous["speaker"] != next_segment["speaker"] or current["speaker"] == previous["speaker"]:
            continue
        if _duration(current) > bridge_max_sec:
            continue
        if _gap(previous, current) > bridge_gap_sec or _gap(current, next_segment) > bridge_gap_sec:
            continue
        if _duration(previous) + _duration(next_segment) < _duration(current):
            continue
        current["speaker"] = str(previous["speaker"])
        changed += 1
    return changed


def stabilize_diarization_records(
    records: list[dict[str, Any]],
    *,
    same_speaker_gap_sec: float = 0.1,
    bridge_max_sec: float = 0.4,
    bridge_gap_sec: float = 0.25,
    tiny_segment_sec: float = 0.12,
    absorb_gap_sec: float = 0.25,
    max_passes: int = 5,
) -> list[dict[str, Any]]:
    segments = _normalize_segments(records)
    total_tiny_relabeled = 0
    total_bridge_relabeled = 0
    total_merged = 0

    for _ in range(max(1, int(max_passes))):
        tiny_relabeled = _absorb_tiny_segments(
            segments,
            tiny_segment_sec=tiny_segment_sec,
            absorb_gap_sec=absorb_gap_sec,
        )
        bridge_relabeled = _bridge_aba_segments(
            segments,
            bridge_max_sec=bridge_max_sec,
            bridge_gap_sec=bridge_gap_sec,
        )
        segments, merged = _merge_same_speaker(
            segments,
            same_speaker_gap_sec=same_speaker_gap_sec,
        )
        total_tiny_relabeled += tiny_relabeled
        total_bridge_relabeled += bridge_relabeled
        total_merged += merged
        if tiny_relabeled == 0 and bridge_relabeled == 0 and merged == 0:
            break

    stabilized = _finalize_segments(segments)
    logger.info(
        "Stabilized diarization: input=%d output=%d tiny_relabeled=%d bridge_relabeled=%d merged=%d",
        len(records),
        len(stabilized),
        total_tiny_relabeled,
        total_bridge_relabeled,
        total_merged,
    )
    return stabilized


def stabilize_diarization_file(
    input_json: str | Path,
    output_json: str | Path,
    *,
    same_speaker_gap_sec: float = 0.1,
    bridge_max_sec: float = 0.4,
    bridge_gap_sec: float = 0.25,
    tiny_segment_sec: float = 0.12,
    absorb_gap_sec: float = 0.25,
    max_passes: int = 5,
) -> list[dict[str, Any]]:
    records = load_json(input_json)
    stabilized = stabilize_diarization_records(
        records,
        same_speaker_gap_sec=same_speaker_gap_sec,
        bridge_max_sec=bridge_max_sec,
        bridge_gap_sec=bridge_gap_sec,
        tiny_segment_sec=tiny_segment_sec,
        absorb_gap_sec=absorb_gap_sec,
        max_passes=max_passes,
    )
    save_json(stabilized, output_json)
    return stabilized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stabilize short speaker-label jitter in diarization JSON.")
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    parser.add_argument("--same-speaker-gap-sec", type=float, default=0.1)
    parser.add_argument("--bridge-max-sec", type=float, default=0.4)
    parser.add_argument("--bridge-gap-sec", type=float, default=0.25)
    parser.add_argument("--tiny-segment-sec", type=float, default=0.12)
    parser.add_argument("--absorb-gap-sec", type=float, default=0.25)
    parser.add_argument("--max-passes", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stabilize_diarization_file(
        args.input_json,
        args.output_json,
        same_speaker_gap_sec=args.same_speaker_gap_sec,
        bridge_max_sec=args.bridge_max_sec,
        bridge_gap_sec=args.bridge_gap_sec,
        tiny_segment_sec=args.tiny_segment_sec,
        absorb_gap_sec=args.absorb_gap_sec,
        max_passes=args.max_passes,
    )


if __name__ == "__main__":
    main()
