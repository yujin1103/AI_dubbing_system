from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import get_logger, load_json, project_relative, resolve_project_path, save_json
from quality_gate import assess_tts_transcript, attach_stage_quality
from run_asr import transcribe_chunks

logger = get_logger("validate_tts_output")


def _select_target_text(row: dict[str, Any]) -> str:
    for key in ("text_tts", "text_translated", "text_src"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _build_dub_asr_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        dub_wav = row.get("dub_wav")
        if not dub_wav:
            continue
        records.append(
            {
                "chunk_id": row.get("chunk_id"),
                "speaker": row.get("speaker"),
                "wav": dub_wav,
                "start": row.get("start"),
                "end": row.get("end"),
                "duration": row.get("duration"),
            }
        )
    return records


def validate_tts_output(
    master_timeline_json: str | Path,
    *,
    output_json: str | Path,
    asr_input_json: str | Path | None = None,
    model_dir: str | Path,
    device: str = "cuda:0",
    dtype: str = "float16",
    language: str | None = "Korean",
    mark_stale_on_fail: bool = True,
    fail_on_error: bool = False,
) -> list[dict[str, Any]]:
    rows = load_json(master_timeline_json)
    records = _build_dub_asr_records(rows)
    if not records:
        raise RuntimeError(f"No dub_wav records found in {master_timeline_json}")

    resolved_output = resolve_project_path(output_json)
    resolved_input = (
        resolve_project_path(asr_input_json)
        if asr_input_json
        else resolved_output.with_name(f"{resolved_output.stem}_input.json")
    )
    save_json(records, resolved_input)

    asr_rows = transcribe_chunks(
        resolved_input,
        resolved_output,
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        language=language,
        features_json=resolved_output.with_name(f"{resolved_output.stem}_features.json"),
    )
    asr_by_chunk = {str(row.get("chunk_id", "")): row for row in asr_rows}

    failures: list[str] = []
    for row in rows:
        chunk_id = str(row.get("chunk_id", ""))
        asr_row = asr_by_chunk.get(chunk_id)
        if not asr_row:
            continue
        transcript = str(asr_row.get("text_src", "") or "").strip()
        row["tts_asr_text"] = transcript
        assessment = assess_tts_transcript(row, transcript=transcript, target_text=_select_target_text(row))
        attach_stage_quality(row, "tts_asr", assessment)
        if assessment["accepted"]:
            row.pop("tts_validation_error", None)
            if mark_stale_on_fail:
                row.pop("dub_stale", None)
        else:
            failures.append(chunk_id)
            row["tts_validation_error"] = ",".join(assessment.get("critical_flags", []))
            if mark_stale_on_fail:
                row["dub_stale"] = True

    save_json(rows, master_timeline_json)
    logger.info(
        "Validated %s dub chunks with ASR. failures=%s output=%s",
        len(asr_rows),
        len(failures),
        project_relative(resolved_output),
    )
    if failures and fail_on_error:
        raise RuntimeError(f"TTS ASR validation failed for {len(failures)} chunks: {', '.join(failures)}")
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASR-validate generated dub wav files and flag repeated or mismatched TTS.")
    parser.add_argument("master_timeline_json")
    parser.add_argument("output_json")
    parser.add_argument("--asr-input-json")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--no-mark-stale-on-fail", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_tts_output(
        args.master_timeline_json,
        output_json=args.output_json,
        asr_input_json=args.asr_input_json,
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        language=args.language,
        mark_stale_on_fail=not args.no_mark_stale_on_fail,
        fail_on_error=args.fail_on_error,
    )


if __name__ == "__main__":
    main()
