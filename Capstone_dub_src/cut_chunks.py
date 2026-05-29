from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import get_logger, load_json, project_relative, resolve_project_path, run_command, save_json

logger = get_logger("cut_chunks")


def cut_chunks(
    input_audio: str | Path,
    chunk_json: str | Path,
    *,
    chunks_dir: str | Path,
    output_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    audio_path = resolve_project_path(input_audio)
    chunks_path = resolve_project_path(chunks_dir)
    chunks_path.mkdir(parents=True, exist_ok=True)

    records = load_json(chunk_json)
    for item in records:
        chunk_id = str(item["chunk_id"])
        start = float(item["start"])
        end = float(item["end"])
        duration = max(0.0, end - start)
        output_wav = chunks_path / f"{chunk_id}.wav"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                output_wav,
            ]
        )
        item["wav"] = project_relative(output_wav)

    save_json(records, output_json or chunk_json)
    logger.info("Generated %s chunk wav files in %s", len(records), chunks_path)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cut speaker chunks into individual wav files.")
    parser.add_argument("input_audio")
    parser.add_argument("chunk_json")
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cut_chunks(
        args.input_audio,
        args.chunk_json,
        chunks_dir=args.chunks_dir,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
