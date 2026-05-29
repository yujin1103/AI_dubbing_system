# Silero VAD 로 dialogue.wav 의 비발화 구간을 bgm.wav 로 재배치 (합산 보존 유지).
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio

from common import ensure_parent, get_logger, resolve_project_path

logger = get_logger("redirect_nonspeech")

VAD_SAMPLE_RATE = 16000
SUM_TO_ORIGINAL_TOLERANCE = 1e-4


def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(resolve_project_path(path)), always_2d=True, dtype="float32")
    return data, int(sample_rate)


def _save_audio(path: str | Path, data: np.ndarray, sample_rate: int) -> Path:
    output_path = ensure_parent(path)
    clipped = np.clip(data, -1.0, 1.0)
    sf.write(str(output_path), clipped, sample_rate, subtype="PCM_16")
    return output_path


def _to_vad_waveform(stereo_44k: np.ndarray, sample_rate: int) -> torch.Tensor:
    mono = stereo_44k.mean(axis=1).astype(np.float32, copy=False)
    tensor = torch.from_numpy(mono).unsqueeze(0)
    if sample_rate != VAD_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SAMPLE_RATE)
        tensor = resampler(tensor)
    return tensor.squeeze(0)


def _build_speech_mask(
    speech_segments: Iterable[dict],
    *,
    total_samples: int,
    sample_rate: int,
    pad_ms: int,
    ramp_ms: int,
) -> np.ndarray:
    pad_samples = int(round(pad_ms * sample_rate / 1000.0))
    ramp_samples = max(1, int(round(ramp_ms * sample_rate / 1000.0)))
    mask = np.zeros(total_samples, dtype=np.float32)

    for seg in speech_segments:
        start = max(0, int(seg["start"]) - pad_samples)
        end = min(total_samples, int(seg["end"]) + pad_samples)
        if end <= start:
            continue
        mask[start:end] = 1.0

    if ramp_samples > 1:
        ramp = np.linspace(0.0, 1.0, ramp_samples, dtype=np.float32)
        in_speech = False
        for index in range(total_samples):
            if mask[index] > 0 and not in_speech:
                # 발화 시작 — 직전 ramp_samples 를 0→1 로 페이드인
                fade_start = max(0, index - ramp_samples)
                ramp_len = index - fade_start
                if ramp_len > 0:
                    mask[fade_start:index] = np.maximum(mask[fade_start:index], ramp[-ramp_len:])
                in_speech = True
            elif mask[index] == 0 and in_speech:
                # 발화 종료 — 다음 ramp_samples 를 1→0 으로 페이드아웃
                fade_end = min(total_samples, index + ramp_samples)
                ramp_len = fade_end - index
                if ramp_len > 0:
                    mask[index:fade_end] = np.maximum(mask[index:fade_end], ramp[::-1][:ramp_len])
                in_speech = False

    return mask


def redirect_nonspeech_to_bgm(
    dialogue_wav: str | Path,
    bgm_wav: str | Path,
    *,
    model_path: str | Path,
    threshold: float = 0.4,
    min_speech_ms: int = 250,
    pad_ms: int = 100,
    ramp_ms: int = 15,
) -> tuple[Path, Path]:
    try:
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError(
            "silero-vad is not installed in the active environment."
        ) from exc

    dialogue_path = resolve_project_path(dialogue_wav)
    bgm_path = resolve_project_path(bgm_wav)
    resolved_model_path = resolve_project_path(model_path)
    if not dialogue_path.exists():
        raise FileNotFoundError(f"dialogue audio not found: {dialogue_path}")
    if not bgm_path.exists():
        raise FileNotFoundError(f"bgm audio not found: {bgm_path}")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"silero-vad model not found: {resolved_model_path}")

    dialogue, dialogue_sr = _load_audio(dialogue_path)
    bgm, bgm_sr = _load_audio(bgm_path)
    if dialogue_sr != bgm_sr:
        raise ValueError(
            f"dialogue ({dialogue_sr} Hz) and bgm ({bgm_sr} Hz) sample rates differ."
        )
    if dialogue.shape != bgm.shape:
        raise ValueError(
            f"dialogue shape {dialogue.shape} != bgm shape {bgm.shape}"
        )

    vad_waveform = _to_vad_waveform(dialogue, dialogue_sr)
    model = torch.jit.load(str(resolved_model_path), map_location="cpu")
    model.eval()

    speech_segments_16k = get_speech_timestamps(
        vad_waveform,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
    )
    logger.info(
        "Silero VAD detected %d speech segments (threshold=%.2f, min_speech_ms=%d)",
        len(speech_segments_16k),
        threshold,
        min_speech_ms,
    )

    scale = dialogue_sr / VAD_SAMPLE_RATE
    speech_segments_native = [
        {"start": int(round(seg["start"] * scale)), "end": int(round(seg["end"] * scale))}
        for seg in speech_segments_16k
    ]
    mask_1d = _build_speech_mask(
        speech_segments_native,
        total_samples=dialogue.shape[0],
        sample_rate=dialogue_sr,
        pad_ms=pad_ms,
        ramp_ms=ramp_ms,
    )
    mask_2d = mask_1d[:, np.newaxis]

    speech_ratio = float(mask_1d.mean())
    logger.info(
        "Speech mask coverage: %.2f%% of duration (%.2fs / %.2fs)",
        speech_ratio * 100.0,
        speech_ratio * dialogue.shape[0] / dialogue_sr,
        dialogue.shape[0] / dialogue_sr,
    )

    redirected = dialogue * (1.0 - mask_2d)
    new_dialogue = dialogue * mask_2d
    new_bgm = bgm + redirected

    diff = float(np.max(np.abs((new_dialogue + new_bgm) - (dialogue + bgm))))
    if diff > SUM_TO_ORIGINAL_TOLERANCE:
        raise RuntimeError(
            f"Sum-to-original violated after redirect: max diff={diff:.3e} (tolerance={SUM_TO_ORIGINAL_TOLERANCE})"
        )
    logger.info("Sum-to-original check passed: max diff=%.3e", diff)

    _save_audio(dialogue_path, new_dialogue, dialogue_sr)
    _save_audio(bgm_path, new_bgm, bgm_sr)
    logger.info("Redirected non-speech to bgm; updated %s and %s", dialogue_path, bgm_path)
    return dialogue_path, bgm_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redirect non-speech regions of dialogue.wav into bgm.wav using Silero VAD."
    )
    parser.add_argument("dialogue_wav")
    parser.add_argument("bgm_wav")
    parser.add_argument("--model-path", default="models/vad/silero_vad.jit")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--pad-ms", type=int, default=100)
    parser.add_argument("--ramp-ms", type=int, default=15)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    redirect_nonspeech_to_bgm(
        args.dialogue_wav,
        args.bgm_wav,
        model_path=args.model_path,
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        pad_ms=args.pad_ms,
        ramp_ms=args.ramp_ms,
    )


if __name__ == "__main__":
    main()
