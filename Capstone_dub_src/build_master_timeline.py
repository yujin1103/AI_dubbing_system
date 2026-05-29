from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import build_dub_input_signature, build_dub_runtime_settings, get_logger, load_json, load_json_if_exists, project_relative, resolve_project_path, save_json

logger = get_logger("build_master_timeline")


def _merge_quality_gates(*rows: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for row in rows:
        gates = row.get("quality_gates")
        if isinstance(gates, dict):
            merged.update(gates)
    return merged


def _default_chunk_overrides_path(output_json: str | Path) -> Path:
    return resolve_project_path(output_json).with_name("chunk_overrides.json")


def _load_chunk_overrides(path_value: str | Path | None, output_json: str | Path) -> dict[str, dict[str, Any]]:
    path = resolve_project_path(path_value) if path_value else _default_chunk_overrides_path(output_json)
    if not path.exists():
        return {}
    data = load_json_if_exists(path, default={})
    if not isinstance(data, dict):
        logger.warning("Ignoring chunk overrides because it is not an object: %s", path)
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _normalize_reference_override(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"self", "speaker_bank", "manual"}:
        return normalized
    if normalized in {"speaker_best", "bank", "best"}:
        return "speaker_bank"
    return ""


def _apply_chunk_override(row: dict[str, Any], override: dict[str, Any]) -> None:
    speaker = str(override.get("speaker", "") or "").strip()
    if speaker and speaker != str(row.get("speaker", "") or ""):
        row.setdefault("speaker_original", row.get("speaker"))
        row["speaker"] = speaker
        row["speaker_override"] = True

    reference_mode = _normalize_reference_override(override.get("reference_mode"))
    reference_chunk_id = str(override.get("reference_chunk_id", "") or "").strip()
    if reference_mode:
        row["reference_mode_override"] = reference_mode
        if reference_chunk_id:
            row["reference_chunk_id_override"] = reference_chunk_id
        else:
            row.pop("reference_chunk_id_override", None)


def build_master_timeline(
    speaker_chunks_json: str | Path,
    asr_json: str | Path,
    translated_json: str | Path,
    output_json: str | Path,
    *,
    dub_dir: str | Path,
    emotion_json: str | Path | None = None,
    dub_runtime: dict[str, Any] | None = None,
    chunk_overrides_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    chunks = load_json(speaker_chunks_json)
    asr_rows = {row["chunk_id"]: row for row in load_json(asr_json)}
    translated_rows = {row["chunk_id"]: row for row in load_json(translated_json)}
    emotion_rows = (
        {row["chunk_id"]: row for row in load_json_if_exists(emotion_json, default=[])}
        if emotion_json
        else {}
    )
    existing_timeline_rows = {row["chunk_id"]: row for row in load_json_if_exists(output_json, default=[])}
    chunk_overrides = _load_chunk_overrides(chunk_overrides_json, output_json)

    dub_root = resolve_project_path(dub_dir)
    timeline: list[dict[str, Any]] = []
    stale_count = 0
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        asr_row = asr_rows.get(chunk_id, {})
        translated_row = translated_rows.get(chunk_id, {})
        emotion_row = emotion_rows.get(chunk_id, {})
        dub_wav = dub_root / f"{chunk_id}_dub.wav"
        timeline_row = {
            "chunk_id": chunk_id,
            "speaker": chunk["speaker"],
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
            "duration": round(float(chunk["end"]) - float(chunk["start"]), 3),
            "wav": chunk.get("wav"),
            "text_src": asr_row.get("text_src", translated_row.get("text_src", "")),
            "text_translated": translated_row.get("text_translated", ""),
            "dub_wav": project_relative(dub_wav),
        }
        if asr_row.get("source_audio_summary"):
            timeline_row["source_audio_summary"] = asr_row["source_audio_summary"]
        for key in (
            "source_emotion",
            "tts_emotion_hint",
            "emotion_model",
            "emotion_backend",
            "emotion_error",
        ):
            if key in emotion_row:
                timeline_row[key] = emotion_row[key]
        if translated_row.get("text_tts"):
            timeline_row["text_tts"] = translated_row["text_tts"]
        for key in (
            "text_translated_initial",
            "text_translated_context",
            "text_translated_budgeted",
            "translation_budget",
            "translation_context_refined",
            "translation_revision_count",
            "translation_style_hint",
            "translation_blocked",
            "translation_blocked_reason",
        ):
            if key in translated_row:
                timeline_row[key] = translated_row[key]
        merged_quality_gates = _merge_quality_gates(asr_row, translated_row)
        if merged_quality_gates:
            timeline_row["quality_gates"] = merged_quality_gates

        override = chunk_overrides.get(str(chunk_id))
        if override:
            _apply_chunk_override(timeline_row, override)

        existing_row = existing_timeline_rows.get(chunk_id, {})
        existing_text_src = str(existing_row.get("text_src", "") or "").strip()
        existing_text_translated = str(existing_row.get("text_translated", "") or "").strip()
        new_text_src = str(timeline_row.get("text_src", "") or "").strip()
        new_text_translated = str(timeline_row.get("text_translated", "") or "").strip()
        text_carryover_safe = (
            existing_text_src == new_text_src
            and existing_text_translated == new_text_translated
        )
        if text_carryover_safe:
            for key in (
                "tts_instruct_text",
                "tts_instruct_source",
                "tts_instruct_emotion_label",
            ):
                if key in existing_row:
                    timeline_row[key] = existing_row[key]
        elif existing_row:
            logger.info(
                "Discarding stale tts_instruct_* for %s because text content changed since last build.",
                chunk_id,
            )
        runtime_settings: dict[str, Any] = {}
        selected_runtime = dub_runtime
        if selected_runtime is None and isinstance(existing_row.get("dub_runtime"), dict):
            selected_runtime = existing_row["dub_runtime"]
        if isinstance(selected_runtime, dict) and selected_runtime:
            runtime_settings = build_dub_runtime_settings(**selected_runtime)
            timeline_row["dub_runtime"] = runtime_settings

        current_signature = build_dub_input_signature(timeline_row, runtime=runtime_settings)
        existing_signature = str(existing_row.get("dub_input_signature", "") or "")
        if existing_signature:
            timeline_row["dub_input_signature"] = existing_signature
        if dub_wav.exists() and existing_signature != current_signature:
            timeline_row["dub_stale"] = True
            stale_count += 1

        timeline.append(timeline_row)

    save_json(timeline, output_json)
    if stale_count:
        logger.warning("Marked %s existing dub rows as stale because their TTS input signature changed.", stale_count)
    logger.info("Built master timeline with %s rows", len(timeline))
    return timeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join chunk metadata, ASR, and translated text into master_timeline.json.")
    parser.add_argument("speaker_chunks_json")
    parser.add_argument("asr_json")
    parser.add_argument("translated_json")
    parser.add_argument("output_json")
    parser.add_argument("--dub-dir", required=True)
    parser.add_argument("--emotion-json")
    parser.add_argument("--chunk-overrides-json")
    parser.add_argument("--dub-runtime-json")
    parser.add_argument("--dub-runtime-json-file")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dub_runtime = None
    if args.dub_runtime_json_file:
        dub_runtime = load_json(args.dub_runtime_json_file)
    elif args.dub_runtime_json:
        dub_runtime = json.loads(args.dub_runtime_json)
    build_master_timeline(
        args.speaker_chunks_json,
        args.asr_json,
        args.translated_json,
        args.output_json,
        dub_dir=args.dub_dir,
        emotion_json=args.emotion_json,
        chunk_overrides_json=args.chunk_overrides_json,
        dub_runtime=dub_runtime,
    )


if __name__ == "__main__":
    main()
