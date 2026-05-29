from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from common import get_logger, load_json, load_json_if_exists, resolve_project_path, save_json

logger = get_logger("audio_features")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype("float32", copy=False)
    return audio.mean(axis=1).astype("float32", copy=False)


def _dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value)


def _frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((0, frame_size), dtype="float32")
    if signal.size < frame_size:
        padded = np.pad(signal, (0, frame_size - signal.size))
        return padded.reshape(1, frame_size)

    frame_count = 1 + max(0, (signal.size - frame_size) // hop_size)
    frames = np.zeros((frame_count, frame_size), dtype="float32")
    for index in range(frame_count):
        start = index * hop_size
        end = start + frame_size
        frames[index] = signal[start:end]
    return frames


def analyze_audio_path(
    audio_path: str | Path,
    *,
    silence_threshold_db: float = -42.0,
    min_pause_sec: float = 0.18,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, Any]:
    resolved = resolve_project_path(audio_path)
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "clipping_ratio": 0.0,
            "silence_ratio": 1.0,
            "voiced_ratio": 0.0,
            "pause_count": 0,
            "pause_map": [],
        }

    data, sample_rate = sf.read(str(resolved), always_2d=True, dtype="float32")
    mono = _to_mono(data)
    duration = float(len(mono)) / float(sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono), dtype="float64"))) if mono.size else 0.0
    clipping_ratio = float(np.mean(np.abs(mono) >= 0.995)) if mono.size else 0.0

    frame_size = max(1, int(round(sample_rate * (frame_ms / 1000.0))))
    hop_size = max(1, int(round(sample_rate * (hop_ms / 1000.0))))
    frames = _frame_signal(mono, frame_size=frame_size, hop_size=hop_size)
    if frames.size:
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype="float64"))
        frame_db = np.array([_dbfs(float(item)) for item in frame_rms], dtype="float32")
        silence_mask = frame_db <= silence_threshold_db
        silence_ratio = float(np.mean(silence_mask))
        voiced_ratio = 1.0 - silence_ratio
    else:
        silence_mask = np.zeros((0,), dtype=bool)
        silence_ratio = 1.0
        voiced_ratio = 0.0

    pause_map: list[dict[str, float]] = []
    if silence_mask.size:
        current_start: int | None = None
        for index, is_silent in enumerate(silence_mask):
            if is_silent and current_start is None:
                current_start = index
                continue
            if (not is_silent) and current_start is not None:
                pause_start = current_start * hop_size / float(sample_rate)
                pause_end = min(duration, (index * hop_size + frame_size) / float(sample_rate))
                pause_duration = pause_end - pause_start
                if pause_duration >= min_pause_sec:
                    pause_map.append(
                        {
                            "start": round(pause_start, 3),
                            "end": round(pause_end, 3),
                            "duration": round(pause_duration, 3),
                        }
                    )
                current_start = None
        if current_start is not None:
            pause_start = current_start * hop_size / float(sample_rate)
            pause_end = duration
            pause_duration = pause_end - pause_start
            if pause_duration >= min_pause_sec:
                pause_map.append(
                    {
                        "start": round(pause_start, 3),
                        "end": round(pause_end, 3),
                        "duration": round(pause_duration, 3),
                    }
                )

    return {
        "path": str(resolved),
        "exists": True,
        "duration": round(duration, 3),
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "rms_dbfs": round(_dbfs(rms), 3),
        "peak_dbfs": round(_dbfs(peak), 3),
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": round(silence_ratio, 6),
        "voiced_ratio": round(voiced_ratio, 6),
        "pause_count": len(pause_map),
        "pause_map": pause_map,
    }


def compact_feature_summary(feature: dict[str, Any] | None) -> dict[str, Any]:
    if not feature:
        return {}
    return {
        "duration": feature.get("duration"),
        "sample_rate": feature.get("sample_rate"),
        "rms_dbfs": feature.get("rms_dbfs"),
        "peak_dbfs": feature.get("peak_dbfs"),
        "clipping_ratio": feature.get("clipping_ratio"),
        "silence_ratio": feature.get("silence_ratio"),
        "voiced_ratio": feature.get("voiced_ratio"),
        "pause_count": feature.get("pause_count"),
    }


def _round_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _build_feature_input_signature(row: dict[str, Any]) -> str:
    wav_value = row.get("wav")
    wav_path = resolve_project_path(wav_value) if wav_value else None
    stat = wav_path.stat() if wav_path and wav_path.exists() else None
    payload = {
        "chunk_id": str(row.get("chunk_id", "") or ""),
        "wav": str(wav_value or ""),
        "speaker": str(row.get("speaker", "") or ""),
        "start": _round_optional_float(row.get("start")),
        "end": _round_optional_float(row.get("end")),
        "timeline_duration": _round_optional_float(row.get("duration")),
        "file_size": int(stat.st_size) if stat is not None else None,
        "file_mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def default_chunk_features_path(reference_path: str | Path) -> Path:
    return resolve_project_path(reference_path).with_name("chunk_features.json")


def analyze_chunk_records(
    chunk_records: list[dict[str, Any]],
    *,
    output_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for row in chunk_records:
        chunk_id = str(row.get("chunk_id", "")).strip()
        wav_value = row.get("wav")
        if not chunk_id or not wav_value:
            continue
        feature = analyze_audio_path(wav_value)
        feature["chunk_id"] = chunk_id
        feature["wav"] = row.get("wav")
        feature["speaker"] = row.get("speaker")
        feature["start"] = row.get("start")
        feature["end"] = row.get("end")
        if row.get("duration") is not None:
            feature["timeline_duration"] = row.get("duration")
        feature["feature_input_signature"] = _build_feature_input_signature(row)
        features.append(feature)

    if output_json:
        save_json(features, output_json)
        logger.info("Saved chunk audio features for %s rows to %s", len(features), resolve_project_path(output_json))
    return features


def ensure_chunk_features(
    chunk_records: list[dict[str, Any]],
    *,
    reference_path: str | Path,
) -> list[dict[str, Any]]:
    features_path = default_chunk_features_path(reference_path)
    existing = load_json_if_exists(features_path, default=[])
    existing_map = {str(item.get("chunk_id", "")).strip(): item for item in existing if item.get("chunk_id")}
    expected_signatures = {
        str(row.get("chunk_id", "")).strip(): _build_feature_input_signature(row)
        for row in chunk_records
        if row.get("chunk_id")
    }

    if existing and all(
        chunk_id in existing_map and str(existing_map[chunk_id].get("feature_input_signature", "") or "") == signature
        for chunk_id, signature in expected_signatures.items()
    ):
        return existing

    return analyze_chunk_records(chunk_records, output_json=features_path)


def load_chunk_feature_map(reference_path: str | Path) -> dict[str, dict[str, Any]]:
    features_path = default_chunk_features_path(reference_path)
    features = load_json_if_exists(features_path, default=[])
    return {str(item.get("chunk_id", "")).strip(): item for item in features if item.get("chunk_id")}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze chunk wav files and save audio feature summaries.")
    parser.add_argument("chunk_json")
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = load_json(args.chunk_json)
    analyze_chunk_records(records, output_json=args.output_json or default_chunk_features_path(args.chunk_json))


if __name__ == "__main__":
    main()
