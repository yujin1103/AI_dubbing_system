from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import get_logger, load_json, save_json
from tts_runtime import (
    TtsRuntimeConfig,
    _compact_timeline_rows,
    initialize_tts_session,
    passthrough_source_chunks,
    process_chunk,
)

logger = get_logger("run_tts")


def synthesize_dub_chunks(
    master_timeline_json: str | Path,
    *,
    model_dir: str | Path,
    cosyvoice_repo: str | Path | None = None,
    system_prompt: str = "",
    stream: bool = False,
    skip_existing: bool = True,
    min_prompt_sec: float = 1.2,
    passthrough_source_audio: bool = False,
    use_cross_lingual: bool = False,
    target_language: str = "",
    fit_to_duration: bool = False,
    output_json: str | Path | None = None,
    engine: str = "cosyvoice",
    device: str = "cuda:0",
    speed: float = 1.0,
    reference_mode: str = "auto",
    compact_timeline: bool = False,
    duration_fit_min_tempo: float = 0.85,
    duration_fit_max_tempo: float = 1.2,
    duration_fit_trim_overlong: bool = False,
    trim_silence: bool = False,
    silence_trim_threshold_dbfs: float = -45.0,
    max_leading_silence_sec: float = 0.1,
    max_trailing_silence_sec: float = 0.2,
    cap_risky_self_reference: bool = True,
    prompt_cap_max_sec: float = 4.5,
    style_priority: str = "instruction",
) -> list[dict[str, Any]]:
    rows = load_json(master_timeline_json)
    config = TtsRuntimeConfig.from_kwargs(
        model_dir=model_dir,
        system_prompt=system_prompt,
        stream=stream,
        skip_existing=skip_existing,
        min_prompt_sec=min_prompt_sec,
        passthrough_source_audio=passthrough_source_audio,
        use_cross_lingual=use_cross_lingual,
        target_language=target_language,
        fit_to_duration=fit_to_duration,
        engine=engine,
        device=device,
        speed=speed,
        reference_mode=reference_mode,
        compact_timeline=compact_timeline,
        duration_fit_min_tempo=duration_fit_min_tempo,
        duration_fit_max_tempo=duration_fit_max_tempo,
        duration_fit_trim_overlong=duration_fit_trim_overlong,
        trim_silence=trim_silence,
        silence_trim_threshold_dbfs=silence_trim_threshold_dbfs,
        max_leading_silence_sec=max_leading_silence_sec,
        max_trailing_silence_sec=max_trailing_silence_sec,
        cap_risky_self_reference=cap_risky_self_reference,
        prompt_cap_max_sec=prompt_cap_max_sec,
        style_priority=style_priority,
    )
    output_target = output_json or master_timeline_json

    if config.passthrough_source_audio:
        rows = passthrough_source_chunks(
            rows,
            runtime_settings=config.runtime_settings,
            skip_existing=config.skip_existing,
        )
        save_json(_compact_timeline_rows(rows) if config.compact_timeline else rows, output_target)
        return rows

    session = initialize_tts_session(
        model_dir=model_dir,
        cosyvoice_repo=cosyvoice_repo,
        rows=rows,
        normalized_reference_mode=config.normalized_reference_mode,
        min_prompt_sec=config.min_prompt_sec,
        output_target_path=output_target,
    )
    logger.info(
        "Running CosyVoice TTS for %s rows | device=%s | mode=%s | reference_mode=%s | skip_existing=%s | min_prompt_sec=%.2f | fit_to_duration=%s",
        len(rows),
        config.device,
        "cross_lingual" if config.use_cross_lingual else "zero_shot",
        config.normalized_reference_mode,
        config.skip_existing,
        config.min_prompt_sec,
        config.fit_to_duration,
    )

    for row in rows:
        process_chunk(row, config=config, session=session)

    save_json(_compact_timeline_rows(rows) if config.compact_timeline else rows, output_target)
    logger.info("Generated dub wav files for %s chunks with CosyVoice", len(rows))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CosyVoice TTS for each translated chunk.")
    parser.add_argument("master_timeline_json")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cosyvoice-repo", required=True)
    parser.add_argument("--engine", default="cosyvoice", choices=["cosyvoice"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--min-prompt-sec", type=float, default=1.2)
    parser.add_argument("--passthrough-source-audio", action="store_true")
    parser.add_argument("--fit-to-duration", action="store_true")
    parser.add_argument("--use-cross-lingual", action="store_true")
    parser.add_argument("--target-language", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--reference-mode", default="auto", choices=["auto", "self"])
    parser.add_argument("--compact-timeline", action="store_true")
    parser.add_argument("--duration-fit-min-tempo", type=float, default=0.85)
    parser.add_argument("--duration-fit-max-tempo", type=float, default=1.2)
    parser.add_argument("--duration-fit-trim-overlong", action="store_true")
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--silence-trim-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--max-leading-silence-sec", type=float, default=0.1)
    parser.add_argument("--max-trailing-silence-sec", type=float, default=0.2)
    parser.add_argument("--no-cap-risky-self-reference", action="store_true")
    parser.add_argument("--prompt-cap-max-sec", type=float, default=4.5)
    parser.add_argument("--style-priority", default="instruction", choices=["instruction", "balanced", "voice"])
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    synthesize_dub_chunks(
        args.master_timeline_json,
        model_dir=args.model_dir,
        cosyvoice_repo=args.cosyvoice_repo,
        system_prompt=args.system_prompt,
        stream=args.stream,
        skip_existing=not args.no_skip_existing,
        min_prompt_sec=args.min_prompt_sec,
        passthrough_source_audio=args.passthrough_source_audio,
        use_cross_lingual=args.use_cross_lingual,
        target_language=args.target_language,
        fit_to_duration=args.fit_to_duration,
        output_json=args.output_json,
        engine=args.engine,
        device=args.device,
        speed=args.speed,
        reference_mode=args.reference_mode,
        compact_timeline=args.compact_timeline,
        duration_fit_min_tempo=args.duration_fit_min_tempo,
        duration_fit_max_tempo=args.duration_fit_max_tempo,
        duration_fit_trim_overlong=args.duration_fit_trim_overlong,
        trim_silence=args.trim_silence,
        silence_trim_threshold_dbfs=args.silence_trim_threshold_dbfs,
        max_leading_silence_sec=args.max_leading_silence_sec,
        max_trailing_silence_sec=args.max_trailing_silence_sec,
        cap_risky_self_reference=not args.no_cap_risky_self_reference,
        prompt_cap_max_sec=args.prompt_cap_max_sec,
        style_priority=args.style_priority,
    )


if __name__ == "__main__":
    main()
