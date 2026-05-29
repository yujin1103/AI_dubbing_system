from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from audio_features import analyze_audio_path
from common import get_logger, load_json, resolve_project_path

logger = get_logger("compose_audio")


def _match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError(f"channels must be positive: {channels}")
    if data.shape[1] == channels:
        return data.astype("float32", copy=False)
    if channels == 1:
        return data.mean(axis=1, keepdims=True).astype("float32", copy=False)
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1).astype("float32", copy=False)
    if data.shape[1] > channels:
        return data[:, :channels].astype("float32", copy=False)
    padding = np.repeat(data[:, -1:], channels - data.shape[1], axis=1)
    return np.concatenate([data, padding], axis=1).astype("float32", copy=False)


def _read_audio(
    path: str | Path,
    *,
    target_sample_rate: int,
    target_channels: int,
) -> tuple[np.ndarray, int]:
    data, frame_rate = sf.read(str(resolve_project_path(path)), always_2d=True, dtype="float32")
    if frame_rate != target_sample_rate:
        tensor = torch.from_numpy(data.T.copy()).float()
        tensor = torchaudio.transforms.Resample(orig_freq=frame_rate, new_freq=target_sample_rate)(tensor)
        data = tensor.T.cpu().numpy().astype("float32", copy=False)
        frame_rate = target_sample_rate
    return _match_channels(data, target_channels), frame_rate


def _target_peak_linear(target_peak_dbfs: float | None) -> float | None:
    if target_peak_dbfs is None:
        return None
    return 10.0 ** (float(target_peak_dbfs) / 20.0)


def _limit_peak(data: np.ndarray, *, target_peak_dbfs: float | None) -> np.ndarray:
    target_peak = _target_peak_linear(target_peak_dbfs)
    if target_peak is None or target_peak <= 0:
        return data
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > target_peak:
        data = data * (target_peak / peak)
    return np.clip(data, -1.0, 1.0).astype("float32", copy=False)


def compose_audio(
    master_timeline_json: str | Path,
    output_wav: str | Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    background_wav: str | Path | None = None,
    background_gain: float = 1.0,
    dub_gain: float = 1.0,
    target_peak_dbfs: float | None = -1.0,
) -> Path:
    timeline = load_json(master_timeline_json)
    sample_rate = int(sample_rate)
    channels = int(channels)

    background_audio = np.zeros((0, channels), dtype="float32")
    if background_wav and resolve_project_path(background_wav).exists():
        background_audio, background_rate = _read_audio(
            background_wav,
            target_sample_rate=sample_rate,
            target_channels=channels,
        )
        if background_rate != sample_rate:
            raise ValueError(
                f"Background wav sample rate mismatch: {background_rate} != {sample_rate}"
            )
        background_audio = background_audio * float(background_gain)

    prepared: list[tuple[int, np.ndarray]] = []
    total_samples = len(background_audio)

    for row in timeline:
        if row.get("dub_stale"):
            logger.warning("Dub row is stale, skipping: %s", row["chunk_id"])
            continue
        if row.get("dub_error"):
            logger.warning("Dub row has error recorded, skipping: %s | %s", row["chunk_id"], row["dub_error"])
            continue
        dub_path = resolve_project_path(row["dub_wav"])
        if not dub_path.exists():
            logger.warning("Dub wav missing, skipping: %s", dub_path)
            continue
        audio, rate = _read_audio(dub_path, target_sample_rate=sample_rate, target_channels=channels)
        if rate != sample_rate:
            raise ValueError(f"Dub wav sample rate mismatch: {dub_path} -> {rate}")
        audio = audio * float(dub_gain)
        start_index = int(round(float(row["start"]) * sample_rate))
        prepared.append((start_index, audio))
        total_samples = max(total_samples, start_index + len(audio))

    if total_samples == 0 and timeline:
        total_samples = int(round(max(float(item["end"]) for item in timeline) * sample_rate))

    mix = np.zeros((total_samples, channels), dtype="float32")
    if len(background_audio):
        mix[: len(background_audio)] += background_audio[:total_samples]
    for start_index, audio in prepared:
        end_index = min(total_samples, start_index + len(audio))
        if end_index > start_index:
            mix[start_index:end_index] += audio[: end_index - start_index]

    mix = _limit_peak(mix, target_peak_dbfs=target_peak_dbfs)

    output_path = resolve_project_path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mix, sample_rate, subtype="PCM_16")

    feature = analyze_audio_path(output_path)
    logger.info(
        "Composed final dub wav at %s | sample_rate=%s channels=%s rms=%s peak=%s clipping=%s",
        output_path,
        feature.get("sample_rate"),
        feature.get("channels"),
        feature.get("rms_dbfs"),
        feature.get("peak_dbfs"),
        feature.get("clipping_ratio"),
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose dub chunk wav files into a final timeline-aligned wav.")
    parser.add_argument("master_timeline_json")
    parser.add_argument("output_wav")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--background-wav")
    parser.add_argument("--background-gain", type=float, default=1.0)
    parser.add_argument("--dub-gain", type=float, default=1.0)
    parser.add_argument("--target-peak-dbfs", type=float, default=-1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    compose_audio(
        args.master_timeline_json,
        args.output_wav,
        sample_rate=args.sample_rate,
        channels=args.channels,
        background_wav=args.background_wav,
        background_gain=args.background_gain,
        dub_gain=args.dub_gain,
        target_peak_dbfs=args.target_peak_dbfs,
    )


if __name__ == "__main__":
    main()
