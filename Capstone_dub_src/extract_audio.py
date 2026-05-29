from __future__ import annotations

import argparse
from pathlib import Path

from common import get_logger, resolve_project_path, run_command

logger = get_logger("extract_audio")


def extract_audio(
    input_video: str | Path,
    output_wav: str | Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    input_path = resolve_project_path(input_video)
    output_path = resolve_project_path(output_wav)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    logger.info("Extracted audio to %s", output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract mono WAV audio from a video file.")
    parser.add_argument("input_video")
    parser.add_argument("output_wav")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extract_audio(
        args.input_video,
        args.output_wav,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )


if __name__ == "__main__":
    main()
