# BS-RoFormer + MDX23C-InstVoc_HQ subtractive ensemble 으로 dialogue/bgm 을 분리.
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
import torchaudio

from common import copy_file, ensure_parent, get_logger, probe_duration_seconds, resolve_project_path, run_command

logger = get_logger("separate_audio")

DEFAULT_ENSEMBLE_MODELS: tuple[str, ...] = (
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "MDX23C-8KFFT-InstVoc_HQ.ckpt",
)
DEFAULT_ENSEMBLE_WEIGHTS: tuple[float, ...] = (0.5, 0.5)
DEFAULT_ENSEMBLE_MODEL_DIR = "models/separation"
SUM_TO_ORIGINAL_TOLERANCE = 1e-4


def create_silent_wav(
    output_wav: str | Path,
    *,
    duration_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    output_path = resolve_project_path(output_wav)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout={'mono' if channels == 1 else 'stereo'}:sample_rate={sample_rate}",
            "-t",
            f"{duration_seconds:.3f}",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    return output_path


def _load_audio_tensor(path: str | Path) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(str(resolve_project_path(path)), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T).float()
    return waveform, int(sample_rate)


def _convert_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    source_channels = int(waveform.shape[0])
    if source_channels == channels:
        return waveform
    if channels == 1:
        return waveform.mean(dim=0, keepdim=True)
    if source_channels == 1:
        return waveform.repeat(channels, 1)
    if source_channels >= channels:
        return waveform[:channels, :]
    raise ValueError(
        f"Input audio has {source_channels} channels, cannot convert to {channels} channels."
    )


def _resample_waveform(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    if sample_rate == target_sample_rate:
        return waveform, sample_rate
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=target_sample_rate,
    )
    return resampler(waveform), target_sample_rate


def _write_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> Path:
    output_path = ensure_parent(path)
    audio = waveform.detach().cpu().transpose(0, 1).clamp(-1.0, 1.0).numpy()
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")
    return output_path


def _probe_audio_format(path: str | Path) -> tuple[int, int]:
    resolved = resolve_project_path(path)
    with sf.SoundFile(str(resolved)) as handle:
        return int(handle.samplerate), int(handle.channels)


def _find_vocals_output(output_files: Sequence[str | Path], *, base_dir: Path) -> Path:
    paths: list[Path] = []
    for item in output_files:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        paths.append(candidate)
    # audio-separator 의 출력 파일 stem 표기는 항상 "(Vocals)" / "(Instrumental)" 괄호 패턴.
    # 모델 파일명 자체에 "InstVoc" 같은 문자열이 들어가는 경우 (예: MDX23C-8KFFT-InstVoc_HQ)
    # 단순 substring "voc" 검색은 Instrumental 파일을 오인할 수 있으므로 괄호 마커로 식별한다.
    for candidate in paths:
        if "(vocals)" in candidate.name.lower():
            return candidate
    non_inst = [p for p in paths if "(instrumental)" not in p.name.lower()]
    if len(non_inst) == 1:
        return non_inst[0]
    if len(paths) == 1:
        return paths[0]
    raise RuntimeError(f"Cannot identify vocals output among separator results: {paths}")


def _align_to_reference(waveform: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    aligned = _convert_channels(waveform, int(reference.shape[0]))
    target_length = int(reference.shape[1])
    current_length = int(aligned.shape[1])
    if current_length == target_length:
        return aligned
    if current_length > target_length:
        return aligned[:, :target_length]
    pad = torch.zeros(
        (aligned.shape[0], target_length - current_length),
        dtype=aligned.dtype,
        device=aligned.device,
    )
    return torch.cat([aligned, pad], dim=1)


def _separate_with_subtractive_ensemble(
    input_wav: str | Path,
    *,
    models: Sequence[str],
    weights: Sequence[float],
    model_dir: str | Path,
    device: str | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not models:
        raise ValueError("Subtractive ensemble requires at least one separator model.")
    if len(models) != len(weights):
        raise ValueError(
            f"Ensemble model count ({len(models)}) does not match weight count ({len(weights)})."
        )

    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "audio-separator is not installed in the active environment."
        ) from exc

    input_path = resolve_project_path(input_wav)
    model_dir_path = resolve_project_path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    original, sample_rate = _load_audio_tensor(input_path)
    weight_tensor = torch.tensor(list(weights), dtype=torch.float32)
    if not torch.isfinite(weight_tensor).all() or float(weight_tensor.sum()) <= 0:
        raise ValueError(f"Invalid ensemble weights: {weights}")
    weight_tensor = weight_tensor / float(weight_tensor.sum())

    use_autocast = bool(device) and str(device).startswith("cuda") and torch.cuda.is_available()

    accumulated_vocals = torch.zeros_like(original)
    with tempfile.TemporaryDirectory(prefix="separator-", dir=None) as tmpdir:
        tmpdir_path = Path(tmpdir)
        separator = Separator(
            output_dir=str(tmpdir_path),
            output_format="WAV",
            normalization_threshold=1.0,
            sample_rate=sample_rate,
            use_soundfile=True,
            use_autocast=use_autocast,
            model_file_dir=str(model_dir_path),
        )
        for index, (model_name, weight) in enumerate(zip(models, weight_tensor.tolist())):
            logger.info(
                "Loading separator model %s/%s: %s (weight=%.3f)",
                index + 1,
                len(models),
                model_name,
                weight,
            )
            separator.load_model(model_filename=model_name)
            output_files = separator.separate(str(input_path))
            vocals_path = _find_vocals_output(output_files, base_dir=tmpdir_path)
            vocals_tensor, vocals_sr = _load_audio_tensor(vocals_path)
            if vocals_sr != sample_rate:
                vocals_tensor, _ = _resample_waveform(
                    vocals_tensor,
                    sample_rate=vocals_sr,
                    target_sample_rate=sample_rate,
                )
            vocals_tensor = _align_to_reference(vocals_tensor, original)
            accumulated_vocals = accumulated_vocals + weight * vocals_tensor

    bgm = original - accumulated_vocals
    diff = (accumulated_vocals + bgm - original).abs().max().item()
    if diff > SUM_TO_ORIGINAL_TOLERANCE:
        raise RuntimeError(
            f"Sum-to-original violated: max diff={diff:.3e} (tolerance={SUM_TO_ORIGINAL_TOLERANCE})"
        )
    logger.info("Sum-to-original check passed: max diff=%.3e", diff)

    return accumulated_vocals, bgm, sample_rate


def separate_audio(
    input_wav: str | Path,
    dialogue_wav: str | Path,
    *,
    use_separator: bool = True,
    bgm_wav: str | Path | None = None,
    ensemble_models: Sequence[str] = DEFAULT_ENSEMBLE_MODELS,
    ensemble_weights: Sequence[float] = DEFAULT_ENSEMBLE_WEIGHTS,
    ensemble_model_dir: str | Path = DEFAULT_ENSEMBLE_MODEL_DIR,
    device: str | None = None,
) -> tuple[Path, Path | None]:
    input_path = resolve_project_path(input_wav)
    dialogue_path = resolve_project_path(dialogue_wav)
    bgm_path = resolve_project_path(bgm_wav) if bgm_wav else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input wav not found: {input_path}")

    if not use_separator:
        copy_file(input_path, dialogue_path)
        if bgm_path is not None:
            create_silent_wav(bgm_path, duration_seconds=probe_duration_seconds(input_path))
        logger.info("Separator disabled; copied %s to %s", input_path, dialogue_path)
        return dialogue_path, bgm_path

    sample_rate, channels = _probe_audio_format(input_path)
    if sample_rate < 44100 or channels < 2:
        logger.warning(
            "Subtractive ensemble input is only %s Hz / %s ch. "
            "Quality is best at 44.1 kHz stereo; rerun extract_audio with high-quality settings.",
            sample_rate,
            channels,
        )

    vocals, bgm, output_sample_rate = _separate_with_subtractive_ensemble(
        input_path,
        models=ensemble_models,
        weights=ensemble_weights,
        model_dir=ensemble_model_dir,
        device=device,
    )
    _write_wav(dialogue_path, vocals, output_sample_rate)
    if bgm_path is not None:
        _write_wav(bgm_path, bgm, output_sample_rate)
    logger.info("Separated dialogue to %s", dialogue_path)
    return dialogue_path, bgm_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate dialogue and background using subtractive ensemble (BS-RoFormer + MDX23C)."
    )
    parser.add_argument("input_wav")
    parser.add_argument("dialogue_wav")
    parser.add_argument("--bgm-wav")
    parser.add_argument("--no-separator", action="store_true", help="Disable separation; copy input through.")
    parser.add_argument(
        "--ensemble-model",
        action="append",
        dest="ensemble_models",
        help="Repeat to add models. Defaults to BS-RoFormer + MDX23C-InstVoc_HQ.",
    )
    parser.add_argument(
        "--ensemble-weight",
        action="append",
        dest="ensemble_weights",
        type=float,
        help="One weight per --ensemble-model in matching order.",
    )
    parser.add_argument("--ensemble-model-dir", default=DEFAULT_ENSEMBLE_MODEL_DIR)
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    models = tuple(args.ensemble_models) if args.ensemble_models else DEFAULT_ENSEMBLE_MODELS
    weights = (
        tuple(args.ensemble_weights)
        if args.ensemble_weights
        else DEFAULT_ENSEMBLE_WEIGHTS[: len(models)]
    )
    separate_audio(
        args.input_wav,
        args.dialogue_wav,
        use_separator=not args.no_separator,
        bgm_wav=args.bgm_wav,
        ensemble_models=models,
        ensemble_weights=weights,
        ensemble_model_dir=args.ensemble_model_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
