from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_features import ensure_chunk_features
from common import (
    build_dub_input_signature,
    build_dub_runtime_settings,
    copy_file,
    get_logger,
    project_relative,
    resolve_project_path,
)
from quality_gate import assess_reference_candidate, assess_tts_output, attach_stage_quality
from reference_policy import build_speaker_reference_bank, choose_reference_for_row

logger = get_logger("run_tts")

COMPACT_TIMELINE_KEYS = (
    "chunk_id",
    "speaker",
    "start",
    "end",
    "duration",
    "wav",
    "text_src",
    "text_translated",
    "text_tts",
    "source_emotion",
    "tts_instruct_text",
    "tts_instruct_source",
    "tts_instruct_emotion_label",
    "tts_instruct_applied",
    "tts_preflight",
    "tts_asr_text",
    "tts_validation_error",
    "dub_wav",
    "dub_error",
    "dub_stale",
    "dub_reference_chunk_id",
    "dub_reference_wav",
    "dub_reference_mode",
    "dub_reference_preprocess",
    "speaker_original",
    "speaker_override",
    "reference_mode_override",
    "reference_chunk_id_override",
)


@dataclass(frozen=True)
class TtsRuntimeConfig:
    engine: str
    normalized_reference_mode: str
    runtime_settings: dict[str, Any]
    target_language: str
    system_prompt: str
    fit_to_duration: bool
    duration_fit_min_tempo: float
    duration_fit_max_tempo: float
    duration_fit_trim_overlong: bool
    trim_silence: bool
    silence_trim_threshold_dbfs: float
    max_leading_silence_sec: float
    max_trailing_silence_sec: float
    cap_risky_self_reference: bool
    prompt_cap_max_sec: float
    skip_existing: bool
    stream: bool
    speed: float
    min_prompt_sec: float
    use_cross_lingual: bool
    compact_timeline: bool
    passthrough_source_audio: bool
    device: str
    style_priority: str

    @classmethod
    def from_kwargs(
        cls,
        *,
        model_dir: str | Path,
        system_prompt: str = "",
        stream: bool = False,
        skip_existing: bool = True,
        min_prompt_sec: float = 1.2,
        passthrough_source_audio: bool = False,
        use_cross_lingual: bool = False,
        target_language: str = "",
        fit_to_duration: bool = False,
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
    ) -> TtsRuntimeConfig:
        normalized_engine = (engine or "cosyvoice").strip().lower()
        if normalized_engine != "cosyvoice":
            raise ValueError(f"Unsupported TTS engine after cleanup: {engine}")

        normalized_style_priority = (style_priority or "instruction").strip().lower()
        if normalized_style_priority not in {"instruction", "balanced", "voice"}:
            raise ValueError(f"Unsupported style_priority: {style_priority}")
        normalized_reference_mode = _normalize_reference_mode(reference_mode)
        runtime_settings = build_dub_runtime_settings(
            engine="cosyvoice",
            model_dir=str(model_dir),
            target_language=target_language,
            system_prompt=system_prompt,
            reference_mode=normalized_reference_mode,
            min_prompt_sec=min_prompt_sec,
            fit_to_duration=fit_to_duration,
            passthrough_source_audio=passthrough_source_audio,
            use_cross_lingual=use_cross_lingual,
            speed=speed,
            cap_risky_self_reference=cap_risky_self_reference,
            prompt_cap_max_sec=prompt_cap_max_sec,
            duration_fit_trim_overlong=duration_fit_trim_overlong,
            trim_silence=trim_silence,
            silence_trim_threshold_dbfs=silence_trim_threshold_dbfs,
            max_leading_silence_sec=max_leading_silence_sec,
            max_trailing_silence_sec=max_trailing_silence_sec,
            style_priority=normalized_style_priority,
        )
        return cls(
            engine=normalized_engine,
            normalized_reference_mode=normalized_reference_mode,
            runtime_settings=runtime_settings,
            target_language=target_language,
            system_prompt=system_prompt,
            fit_to_duration=fit_to_duration,
            duration_fit_min_tempo=duration_fit_min_tempo,
            duration_fit_max_tempo=duration_fit_max_tempo,
            duration_fit_trim_overlong=duration_fit_trim_overlong,
            trim_silence=trim_silence,
            silence_trim_threshold_dbfs=silence_trim_threshold_dbfs,
            max_leading_silence_sec=max_leading_silence_sec,
            max_trailing_silence_sec=max_trailing_silence_sec,
            cap_risky_self_reference=cap_risky_self_reference,
            prompt_cap_max_sec=prompt_cap_max_sec,
            skip_existing=skip_existing,
            stream=stream,
            speed=speed,
            min_prompt_sec=min_prompt_sec,
            use_cross_lingual=use_cross_lingual,
            compact_timeline=compact_timeline,
            passthrough_source_audio=passthrough_source_audio,
            device=device,
            style_priority=normalized_style_priority,
        )


@dataclass
class TtsSession:
    cosyvoice: Any
    sf: Any
    feature_map: dict[str, dict[str, Any]]
    speaker_best_map: dict[str, Any]
    row_by_chunk_id: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ChunkReference:
    prompt_audio: Path
    prompt_text_src: str
    reference_chunk_id: str
    resolved_reference_mode: str
    preflight: dict[str, Any]
    reference_duration: float


@dataclass(frozen=True)
class InferenceInputs:
    translated_text: str
    prompt_text: str
    cross_lingual_text: str
    instruct_text: str
    prompt_audio_for_tts: Path
    temp_prompt_audio: Path | None
    mode_label: str


def _prepare_cosyvoice_imports(repo_dir: str | Path) -> None:
    repo_path = resolve_project_path(repo_dir)
    matcha_path = repo_path / "third_party" / "Matcha-TTS"
    for candidate in (repo_path, matcha_path):
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def _validate_whisper_package() -> None:
    spec = importlib.util.find_spec("whisper")
    if spec is None:
        return
    origin = str(spec.origin or "")
    if origin.endswith(r"site-packages\whisper.py"):
        raise RuntimeError(
            "Wrong `whisper` package is installed. Uninstall `whisper` and install `openai-whisper`. "
            f"Current module path: {origin}"
        )


def _normalize_reference_mode(value: str) -> str:
    normalized = (value or "auto").strip().lower()
    if normalized in {"", "auto", "hybrid"}:
        return "auto"
    if normalized in {"self", "self_only", "force_self"}:
        return "self"
    raise ValueError(f"Unsupported reference mode: {value}")


def _build_prompt_text(text_src: str, system_prompt: str) -> str:
    text_src = (text_src or "").strip()
    system_prompt = (system_prompt or "").strip()
    if not system_prompt:
        return f"<|endofprompt|>{text_src}"
    if "<|endofprompt|>" in system_prompt:
        return f"{system_prompt}{text_src}"
    return f"{system_prompt}<|endofprompt|>{text_src}"


def _language_token_for_target_language(target_language: str) -> str:
    normalized = (target_language or "").strip().lower()
    if normalized in {"ko", "korean"} or "korea" in normalized:
        return "<|ko|>"
    if normalized in {"ja", "jp", "japanese"} or "japan" in normalized:
        return "<|ja|>"
    if normalized in {"en", "english"}:
        return "<|en|>"
    if normalized in {"zh", "chinese"} or "china" in normalized:
        return "<|zh|>"
    if normalized in {"yue", "cantonese"}:
        return "<|yue|>"
    return ""


def _build_cross_lingual_text(
    tts_text: str,
    system_prompt: str,
    target_language: str = "",
) -> str:
    tts_text = (tts_text or "").strip()
    prompt = (system_prompt or "").strip()
    language_token = _language_token_for_target_language(target_language)
    if "<|endofprompt|>" in tts_text:
        return tts_text if not language_token or language_token in tts_text else f"{language_token}{tts_text}"
    prefix = language_token
    if not prompt:
        body = f"<|endofprompt|>{tts_text}"
        return f"{prefix}{body}" if prefix else body
    if "<|endofprompt|>" in prompt:
        body = f"{prompt}{tts_text}"
    else:
        body = f"{prompt}<|endofprompt|>{tts_text}"
    return f"{language_token}{body}" if language_token and language_token not in body else body


def _resolve_tts_text(row: dict[str, Any]) -> tuple[str, str]:
    text_tts = (row.get("text_tts") or "").strip()
    if text_tts:
        return text_tts, "text_tts"
    text_translated = (row.get("text_translated") or "").strip()
    if text_translated:
        return text_translated, "text_translated"
    text_src = (row.get("text_src") or "").strip()
    if text_src:
        return text_src, "text_src"
    return "", ""


def _reference_override(row: dict[str, Any]) -> tuple[str, str]:
    mode = str(row.get("reference_mode_override", "") or "").strip().lower()
    if mode in {"speaker_best", "bank", "best"}:
        mode = "speaker_bank"
    if mode not in {"self", "speaker_bank", "manual"}:
        return "", ""
    return mode, str(row.get("reference_chunk_id_override", "") or "").strip()


def _candidate_from_row(
    candidate_row: dict[str, Any],
    *,
    mode: str,
    feature_map: dict[str, dict[str, Any]],
    min_prompt_sec: float,
) -> dict[str, Any] | None:
    chunk_id = str(candidate_row.get("chunk_id", "") or "").strip()
    wav_value = candidate_row.get("wav")
    if not chunk_id or not wav_value:
        return None
    feature = feature_map.get(chunk_id, {})
    duration = float(candidate_row.get("duration") or 0.0)
    candidate = {
        "wav": resolve_project_path(wav_value),
        "chunk_id": chunk_id,
        "text_src": str(candidate_row.get("text_src", "") or ""),
        "duration": duration,
        "feature": feature,
    }
    assessment = assess_reference_candidate(
        {
            "wav": str(candidate["wav"]),
            "text_src": candidate["text_src"],
            "duration": duration,
        },
        chunk_feature=feature,
        min_prompt_sec=min_prompt_sec,
    )
    return {**candidate, "assessment": assessment, "resolved_mode": mode}


def _resolve_override_reference(
    row: dict[str, Any],
    *,
    mode: str,
    reference_chunk_id: str,
    config: TtsRuntimeConfig,
    session: TtsSession,
) -> dict[str, Any] | None:
    if mode == "self":
        selected = _candidate_from_row(
            row,
            mode="self",
            feature_map=session.feature_map,
            min_prompt_sec=config.min_prompt_sec,
        )
    elif reference_chunk_id:
        selected_row = session.row_by_chunk_id.get(reference_chunk_id)
        if selected_row is None:
            row["dub_error"] = f"Reference chunk not found: {reference_chunk_id}"
            logger.warning("%s for %s", row["dub_error"], row.get("chunk_id"))
            return None
        if mode == "speaker_bank" and str(selected_row.get("speaker", "") or "") != str(row.get("speaker", "") or ""):
            row["dub_error"] = (
                f"Reference chunk {reference_chunk_id} is not in speaker {row.get('speaker')}"
            )
            logger.warning("%s", row["dub_error"])
            return None
        selected = _candidate_from_row(
            selected_row,
            mode=mode,
            feature_map=session.feature_map,
            min_prompt_sec=config.min_prompt_sec,
        )
    else:
        speaker_key = str(row.get("speaker") or "")
        selected = session.speaker_best_map.get(speaker_key)
        if selected:
            selected = {**selected, "resolved_mode": "speaker_bank"}

    if not selected:
        row["dub_error"] = f"No reference candidate available for override mode: {mode}"
        logger.warning("%s (%s)", row["dub_error"], row.get("chunk_id"))
        return None

    assessment = selected.get("assessment") or {}
    if not bool(assessment.get("accepted", False)):
        row["dub_error"] = (
            f"Reference candidate rejected: {', '.join(assessment.get('critical_flags') or [])}"
        )
        logger.warning("%s (%s)", row["dub_error"], row.get("chunk_id"))
        return None

    mode_label = str(selected.get("resolved_mode") or mode)
    return {
        "prompt_audio": selected["wav"],
        "prompt_text_src": selected.get("text_src", "") or str(row.get("text_src", "") or ""),
        "reference_chunk_id": selected["chunk_id"],
        "reference_mode": mode_label,
        "assessment": assessment,
        "decision": "override",
        "candidate_summaries": [
            {
                "mode": mode_label,
                "chunk_id": selected.get("chunk_id"),
                "assessment": assessment,
            }
        ],
        "rejection_reasons": [],
    }


def _compact_timeline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for row in rows:
        compacted_row: dict[str, Any] = {}
        for key in COMPACT_TIMELINE_KEYS:
            if key in row:
                compacted_row[key] = row[key]
        compacted.append(compacted_row)
    return compacted


def _set_dub_reference_fields(
    row: dict[str, Any],
    *,
    prompt_audio: Path,
    reference_chunk_id: str,
    reference_mode: str,
) -> None:
    row["dub_reference_chunk_id"] = reference_chunk_id
    row["dub_reference_wav"] = project_relative(prompt_audio)
    row["dub_reference_mode"] = reference_mode


def _spoken_character_count(text: str) -> int:
    return len(re.sub(r"\s+", "", (text or "").strip()))


def _estimate_spoken_duration_seconds(text: str, target_language: str) -> float:
    chars = _spoken_character_count(text)
    if chars <= 0:
        return 0.0
    normalized = (target_language or "").strip().lower()
    chars_per_second = 6.2 if normalized in {"ko", "korean"} or "korea" in normalized else 12.0
    return max(0.6, float(chars) / chars_per_second)


def _build_tts_preflight(
    row: dict[str, Any],
    *,
    tts_text: str,
    target_language: str,
    reference_duration: float,
    reference_mode: str,
) -> dict[str, Any]:
    target_duration = float(row.get("duration") or 0.0)
    estimated_speech_duration = _estimate_spoken_duration_seconds(tts_text, target_language)
    reference_to_estimated_ratio = (
        reference_duration / estimated_speech_duration if estimated_speech_duration > 0 else 1.0
    )
    target_to_estimated_ratio = target_duration / estimated_speech_duration if estimated_speech_duration > 0 else 1.0
    warnings: list[str] = []
    if reference_mode.startswith("self") and estimated_speech_duration > 0:
        if reference_to_estimated_ratio > 1.75:
            warnings.append("self_reference_much_longer_than_target_text")
        elif reference_to_estimated_ratio > 1.35:
            warnings.append("self_reference_longer_than_target_text")
        if target_to_estimated_ratio > 1.75:
            warnings.append("timeline_slot_much_longer_than_target_text")
    return {
        "estimated_speech_duration": round(estimated_speech_duration, 3),
        "target_duration": round(target_duration, 3),
        "reference_duration": round(reference_duration, 3),
        "reference_to_estimated_ratio": round(reference_to_estimated_ratio, 3),
        "target_to_estimated_ratio": round(target_to_estimated_ratio, 3),
        "warnings": warnings,
    }


def _make_capped_prompt_audio(
    prompt_audio: Path,
    *,
    output_wav: Path,
    sf_module: Any,
    max_seconds: float,
) -> Path:
    if max_seconds <= 0:
        return prompt_audio
    data, sample_rate = sf_module.read(str(prompt_audio), always_2d=True)
    if data.size == 0:
        return prompt_audio
    max_frames = int(max_seconds * sample_rate)
    if max_frames <= 0 or len(data) <= max_frames:
        return prompt_audio
    capped_path = output_wav.with_suffix(".prompt.tmp.wav")
    sf_module.write(str(capped_path), data[:max_frames], sample_rate, subtype="PCM_16")
    return capped_path


def _prepare_prompt_audio_for_tts(
    row: dict[str, Any],
    *,
    prompt_audio: Path,
    output_wav: Path,
    preflight: dict[str, Any],
    reference_mode: str,
    sf_module: Any,
    enable_prompt_cap: bool,
    prompt_cap_max_sec: float,
) -> tuple[Path, Path | None]:
    row.pop("dub_reference_preprocess", None)
    if not enable_prompt_cap or not reference_mode.startswith("self"):
        return prompt_audio, None
    warnings = set(preflight.get("warnings") or [])
    if "self_reference_much_longer_than_target_text" not in warnings:
        return prompt_audio, None
    estimated = float(preflight.get("estimated_speech_duration") or 0.0)
    cap_seconds = min(prompt_cap_max_sec, max(1.4, estimated * 1.4))
    capped_path = _make_capped_prompt_audio(
        prompt_audio,
        output_wav=output_wav,
        sf_module=sf_module,
        max_seconds=cap_seconds,
    )
    if capped_path == prompt_audio:
        return prompt_audio, None
    row["dub_reference_preprocess"] = {
        "type": "cap_prompt_audio",
        "reason": "self_reference_much_longer_than_target_text",
        "max_seconds": round(cap_seconds, 3),
        "source_wav": project_relative(prompt_audio),
    }
    logger.info(
        "Capped self reference for %s to %.3fs to reduce repeated TTS generation risk",
        row.get("chunk_id"),
        cap_seconds,
    )
    return capped_path, capped_path


def passthrough_source_chunks(
    rows: list[dict[str, Any]],
    *,
    runtime_settings: dict[str, Any],
    skip_existing: bool,
) -> list[dict[str, Any]]:
    copied = 0
    for row in rows:
        wav_value = row.get("wav")
        if not wav_value:
            logger.warning("Source wav is missing for %s, skipping passthrough", row["chunk_id"])
            continue
        source_wav = resolve_project_path(wav_value)
        output_wav = resolve_project_path(row["dub_wav"])
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        row["dub_runtime"] = runtime_settings
        if not source_wav.exists():
            logger.warning("Source wav not found for %s: %s", row["chunk_id"], source_wav)
            continue
        if skip_existing and output_wav.exists() and not row.get("dub_stale"):
            logger.info("Skipping existing dub wav: %s", output_wav)
            row.pop("dub_error", None)
            row.pop("dub_stale", None)
            continue
        copy_file(source_wav, output_wav)
        row["dub_wav"] = project_relative(output_wav)
        row["dub_input_signature"] = build_dub_input_signature(row, runtime=runtime_settings)
        _set_dub_reference_fields(
            row,
            prompt_audio=source_wav,
            reference_chunk_id=row["chunk_id"],
            reference_mode="passthrough_source_audio",
        )
        row["reference_decision"] = "passthrough"
        row["reference_rejection_reasons"] = []
        row["reference_candidates"] = []
        attach_stage_quality(
            row,
            "reference",
            assess_reference_candidate(
                {
                    "wav": row.get("wav"),
                    "text_src": row.get("text_src", ""),
                    "duration": row.get("duration", 0.0),
                },
                min_prompt_sec=0.0,
            ),
        )
        attach_stage_quality(
            row,
            "tts",
            assess_tts_output(output_wav, expected_duration=float(row.get("duration") or 0.0)),
        )
        row.pop("dub_error", None)
        row.pop("dub_stale", None)
        copied += 1
    logger.info("Passthrough copied %s source chunk wav files into dub directory", copied)
    return rows


def _measure_duration_seconds(path: Path, sf_module: Any) -> float:
    data, sample_rate = sf_module.read(str(path), always_2d=True)
    return float(len(data)) / float(sample_rate)


def _trim_excess_edge_silence(
    path: Path,
    *,
    sf_module: Any,
    threshold_dbfs: float,
    max_leading_silence_sec: float,
    max_trailing_silence_sec: float,
) -> None:
    if max_leading_silence_sec < 0 or max_trailing_silence_sec < 0:
        raise ValueError("Silence trim allowances must be non-negative")
    data, sample_rate = sf_module.read(str(path), always_2d=True, dtype="float32")
    if len(data) == 0 or sample_rate <= 0:
        return
    threshold = 10.0 ** (float(threshold_dbfs) / 20.0)
    mono_abs = np.max(np.abs(data), axis=1)
    voiced = np.flatnonzero(mono_abs > threshold)
    if voiced.size == 0:
        return
    first_voiced = int(voiced[0])
    last_voiced = int(voiced[-1])
    leading_allowance = int(round(max_leading_silence_sec * sample_rate))
    trailing_allowance = int(round(max_trailing_silence_sec * sample_rate))
    start = max(0, first_voiced - leading_allowance)
    end = min(len(data), last_voiced + 1 + trailing_allowance)
    if start == 0 and end == len(data):
        return
    sf_module.write(str(path), data[start:end], sample_rate, subtype="PCM_16")
    logger.info(
        "Trimmed edge silence for %s (start %.3fs -> %.3fs, end %.3fs -> %.3fs)",
        path.name,
        first_voiced / float(sample_rate),
        start / float(sample_rate),
        last_voiced / float(sample_rate),
        end / float(sample_rate),
    )


def _build_atempo_chain(speed_factor: float) -> str:
    factors: list[float] = []
    remaining = float(speed_factor)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.01:
        factors.append(remaining)
    if not factors:
        return "anull"
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def _run_ffmpeg(command: list[str | Path]) -> None:
    from common import run_command

    run_command([str(item) for item in command])


def _fit_audio_to_duration(
    output_wav: Path,
    *,
    target_duration: float,
    sf_module: Any,
    min_tempo: float = 0.85,
    max_tempo: float = 1.2,
    trim_overlong: bool = False,
) -> None:
    if target_duration <= 0 or not output_wav.exists():
        return
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg is not available in this runtime; skipping duration fit for %s", output_wav.name)
        return
    actual_duration = _measure_duration_seconds(output_wav, sf_module)
    if actual_duration <= 0:
        return
    ratio = actual_duration / target_duration
    tempo = min(max(ratio, min_tempo), max_tempo)
    overlong = actual_duration > target_duration
    tmp_path = output_wav.with_suffix(".fit.tmp.wav")
    filter_chain = _build_atempo_chain(tempo)
    if overlong and not trim_overlong:
        if filter_chain == "anull":
            logger.info(
                "Leaving overlong TTS untrimmed for %s (target %.3fs, actual %.3fs)",
                output_wav.name,
                target_duration,
                actual_duration,
            )
            return
        filter_spec = filter_chain
    elif filter_chain == "anull":
        filter_spec = f"apad=pad_dur={target_duration:.3f},atrim=0:{target_duration:.3f}"
    else:
        filter_spec = f"{filter_chain},apad=pad_dur={target_duration:.3f},atrim=0:{target_duration:.3f}"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            output_wav,
            "-filter:a",
            filter_spec,
            "-c:a",
            "pcm_s16le",
            tmp_path,
        ]
    )
    tmp_path.replace(output_wav)
    logger.info(
        "Fitted %s to target duration %.3fs (original %.3fs, ratio %.2f, tempo %.2f, trim_overlong=%s)",
        output_wav.name,
        target_duration,
        actual_duration,
        ratio,
        tempo,
        trim_overlong,
    )


def initialize_tts_session(
    *,
    model_dir: str | Path,
    cosyvoice_repo: str | Path | None,
    rows: list[dict[str, Any]],
    normalized_reference_mode: str,
    min_prompt_sec: float,
    output_target_path: str | Path,
) -> TtsSession:
    if not cosyvoice_repo:
        raise RuntimeError("CosyVoice repo path is required")

    _prepare_cosyvoice_imports(cosyvoice_repo)
    _validate_whisper_package()

    try:
        import soundfile as sf
        from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice3
    except ImportError as exc:
        raise RuntimeError(
            f"CosyVoice import failed: {exc}. "
            "The repo path is visible, but a required dependency is missing in the active environment."
        ) from exc

    feature_rows = ensure_chunk_features(rows, reference_path=output_target_path)
    feature_map = {str(item.get("chunk_id", "")).strip(): item for item in feature_rows if item.get("chunk_id")}
    needs_speaker_bank = normalized_reference_mode != "self" or any(
        _reference_override(row)[0] == "speaker_bank" for row in rows
    )
    speaker_best_map = (
        build_speaker_reference_bank(rows, feature_map=feature_map, min_prompt_sec=min_prompt_sec)
        if needs_speaker_bank
        else {}
    )
    row_by_chunk_id = {
        str(row.get("chunk_id", "") or "").strip(): row
        for row in rows
        if row.get("chunk_id")
    }
    # AutoModel 은 cosyvoice2.yaml 을 먼저 감지해 CosyVoice2 로 오판(이 모델은 v2/v3 yaml 둘 다 보유).
    # cosyvoice_daemon 과 동일하게 cosyvoice3.yaml 있으면 CosyVoice3 강제(instruct2 감정 경로 보존).
    import os as _os
    _md = str(resolve_project_path(model_dir))
    if _os.path.exists(_os.path.join(_md, "cosyvoice3.yaml")):
        cosyvoice = CosyVoice3(_md)
    else:
        cosyvoice = AutoModel(model_dir=_md)
    return TtsSession(
        cosyvoice=cosyvoice,
        sf=sf,
        feature_map=feature_map,
        speaker_best_map=speaker_best_map,
        row_by_chunk_id=row_by_chunk_id,
    )


def process_chunk(row: dict[str, Any], *, config: TtsRuntimeConfig, session: TtsSession) -> None:
    translated_text, _ = _resolve_tts_text(row)
    output_wav = resolve_project_path(row["dub_wav"])
    temp_output_wav = output_wav.with_suffix(".gen.tmp.wav")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    row["dub_runtime"] = config.runtime_settings

    if _should_skip_existing(row, output_wav, skip_existing=config.skip_existing):
        return
    if row.get("dub_stale") and output_wav.exists():
        logger.info("Regenerating stale dub wav for %s: %s", row["chunk_id"], output_wav)
    if temp_output_wav.exists():
        temp_output_wav.unlink()

    if not translated_text:
        logger.warning("Both text_translated and text_src are empty for %s, skipping TTS", row["chunk_id"])
        row["dub_error"] = "No text available for TTS"
        return

    reference = _resolve_chunk_reference(row, config=config, session=session)
    if reference is None:
        return

    inputs = _prepare_inference_inputs(row, reference, config=config, session=session)
    if not _run_cosyvoice_inference(row, inputs, config=config, session=session, temp_output_wav=temp_output_wav):
        return

    _finalize_chunk_success(row, output_wav, runtime_settings=config.runtime_settings)


def _should_skip_existing(row: dict[str, Any], output_wav: Path, *, skip_existing: bool) -> bool:
    if skip_existing and output_wav.exists() and not row.get("dub_stale"):
        logger.info("Skipping existing dub wav: %s", output_wav)
        row.pop("dub_error", None)
        row.pop("dub_stale", None)
        return True
    return False


def _resolve_chunk_reference(
    row: dict[str, Any],
    *,
    config: TtsRuntimeConfig,
    session: TtsSession,
) -> ChunkReference | None:
    translated_text, _ = _resolve_tts_text(row)
    mode, reference_chunk_id = _reference_override(row)
    if mode:
        reference = _resolve_override_reference(
            row,
            mode=mode,
            reference_chunk_id=reference_chunk_id,
            config=config,
            session=session,
        )
        if reference is None:
            return None
    else:
        try:
            reference = choose_reference_for_row(
                row,
                feature_map=session.feature_map,
                min_prompt_sec=config.min_prompt_sec,
                reference_mode=config.normalized_reference_mode,
                speaker_best_map=session.speaker_best_map,
            )
        except Exception as exc:
            row["dub_error"] = str(exc)
            logger.warning("Failed to select reference for %s: %s", row["chunk_id"], exc)
            return None

    prompt_audio = Path(reference["prompt_audio"])
    prompt_text_src = str(reference.get("prompt_text_src", "") or "")
    reference_chunk_id = str(reference["reference_chunk_id"])
    resolved_reference_mode = str(reference["reference_mode"])
    reference_duration = float(
        (reference.get("assessment") or {}).get("metrics", {}).get("duration") or row.get("duration") or 0.0
    )
    preflight = _build_tts_preflight(
        row,
        tts_text=translated_text,
        target_language=config.target_language,
        reference_duration=reference_duration,
        reference_mode=resolved_reference_mode,
    )
    row["tts_preflight"] = preflight
    row["reference_decision"] = reference.get("decision")
    row["reference_candidates"] = reference.get("candidate_summaries", [])
    row["reference_rejection_reasons"] = reference.get("rejection_reasons", [])
    attach_stage_quality(row, "reference", reference["assessment"])
    if reference_chunk_id != row["chunk_id"]:
        logger.info("Using speaker reference %s for short chunk %s", reference_chunk_id, row["chunk_id"])
    _set_dub_reference_fields(
        row,
        prompt_audio=prompt_audio,
        reference_chunk_id=reference_chunk_id,
        reference_mode=resolved_reference_mode,
    )
    if not prompt_audio.exists():
        row["dub_error"] = f"Prompt audio not found: {prompt_audio}"
        logger.warning("Prompt audio missing for %s: %s", row["chunk_id"], prompt_audio)
        return None
    needs_prompt_text = not config.use_cross_lingual
    if needs_prompt_text and not prompt_text_src.strip():
        row["dub_error"] = "Reference transcript is missing for zero-shot TTS"
        logger.warning("Reference transcript is missing for %s", row["chunk_id"])
        return None
    return ChunkReference(
        prompt_audio=prompt_audio,
        prompt_text_src=prompt_text_src,
        reference_chunk_id=reference_chunk_id,
        resolved_reference_mode=resolved_reference_mode,
        preflight=preflight,
        reference_duration=reference_duration,
    )


def _prepare_inference_inputs(
    row: dict[str, Any],
    ref: ChunkReference,
    *,
    config: TtsRuntimeConfig,
    session: TtsSession,
) -> InferenceInputs:
    translated_text, text_field = _resolve_tts_text(row)
    instruct_text = str(row.get("tts_instruct_text", "") or "").strip()
    if config.style_priority == "voice":
        instruct_text = ""
        mode_label = "cross_lingual" if config.use_cross_lingual else "zero_shot"
    else:
        mode_label = "instruct2" if instruct_text else ("cross_lingual" if config.use_cross_lingual else "zero_shot")
    logger.info(
        "Generating dub for %s using %s from %s with reference %s (%s)",
        row["chunk_id"],
        mode_label,
        text_field or "text_src",
        ref.reference_chunk_id,
        ref.resolved_reference_mode,
    )
    prompt_text = _build_prompt_text(ref.prompt_text_src, config.system_prompt)
    row.pop("tts_emotion_token", None)
    row.pop("tts_emotion_token_applied", None)
    row.pop("tts_emotion_prompt_applied", None)
    cross_lingual_text = _build_cross_lingual_text(
        translated_text,
        config.system_prompt,
        target_language=config.target_language,
    )
    output_wav = resolve_project_path(row["dub_wav"])
    prompt_audio_for_tts, temp_prompt_audio = _prepare_prompt_audio_for_tts(
        row,
        prompt_audio=ref.prompt_audio,
        output_wav=output_wav,
        preflight=ref.preflight,
        reference_mode=ref.resolved_reference_mode,
        sf_module=session.sf,
        enable_prompt_cap=config.cap_risky_self_reference,
        prompt_cap_max_sec=config.prompt_cap_max_sec,
    )
    return InferenceInputs(
        translated_text=translated_text,
        prompt_text=prompt_text,
        cross_lingual_text=cross_lingual_text,
        instruct_text=instruct_text,
        prompt_audio_for_tts=prompt_audio_for_tts,
        temp_prompt_audio=temp_prompt_audio,
        mode_label=mode_label,
    )


def _run_cosyvoice_inference(
    row: dict[str, Any],
    inputs: InferenceInputs,
    *,
    config: TtsRuntimeConfig,
    session: TtsSession,
    temp_output_wav: Path,
) -> bool:
    output_wav = resolve_project_path(row["dub_wav"])
    target_duration = float(row.get("duration") or 0.0)
    saved = False
    try:
        if config.style_priority == "voice":
            if config.use_cross_lingual:
                result_iterator = session.cosyvoice.inference_cross_lingual(
                    inputs.cross_lingual_text,
                    str(inputs.prompt_audio_for_tts),
                    stream=config.stream,
                    speed=config.speed,
                )
            else:
                result_iterator = session.cosyvoice.inference_zero_shot(
                    inputs.translated_text,
                    inputs.prompt_text,
                    str(inputs.prompt_audio_for_tts),
                    stream=config.stream,
                    speed=config.speed,
                )
            row.pop("tts_instruct_applied", None)
        elif inputs.instruct_text:
            result_iterator = session.cosyvoice.inference_instruct2(
                inputs.translated_text,
                inputs.instruct_text,
                str(inputs.prompt_audio_for_tts),
                stream=config.stream,
                speed=config.speed,
            )
            row["tts_instruct_applied"] = True
        elif config.use_cross_lingual:
            result_iterator = session.cosyvoice.inference_cross_lingual(
                inputs.cross_lingual_text,
                str(inputs.prompt_audio_for_tts),
                stream=config.stream,
                speed=config.speed,
            )
            row.pop("tts_instruct_applied", None)
        else:
            result_iterator = session.cosyvoice.inference_zero_shot(
                inputs.translated_text,
                inputs.prompt_text,
                str(inputs.prompt_audio_for_tts),
                stream=config.stream,
                speed=config.speed,
            )
            row.pop("tts_instruct_applied", None)
        for result in result_iterator:
            audio = result["tts_speech"].detach().cpu().float().numpy()
            if audio.ndim == 1:
                session.sf.write(str(temp_output_wav), audio, session.cosyvoice.sample_rate, subtype="PCM_16")
            else:
                session.sf.write(str(temp_output_wav), audio.T, session.cosyvoice.sample_rate, subtype="PCM_16")
            if config.trim_silence:
                _trim_excess_edge_silence(
                    temp_output_wav,
                    sf_module=session.sf,
                    threshold_dbfs=config.silence_trim_threshold_dbfs,
                    max_leading_silence_sec=config.max_leading_silence_sec,
                    max_trailing_silence_sec=config.max_trailing_silence_sec,
                )
            tts_assessment = assess_tts_output(temp_output_wav, expected_duration=target_duration)
            severe_duration_issue = any(
                flag in {"tts_under_duration_severe", "tts_over_duration_severe"}
                for flag in tts_assessment.get("critical_flags", [])
            )
            if config.fit_to_duration or (target_duration > 0 and severe_duration_issue):
                _fit_audio_to_duration(
                    temp_output_wav,
                    target_duration=target_duration,
                    sf_module=session.sf,
                    min_tempo=config.duration_fit_min_tempo,
                    max_tempo=config.duration_fit_max_tempo,
                    trim_overlong=config.duration_fit_trim_overlong,
                )
                tts_assessment = assess_tts_output(temp_output_wav, expected_duration=target_duration)
            temp_output_wav.replace(output_wav)
            attach_stage_quality(row, "tts", tts_assessment)
            saved = True
            break
    except Exception as exc:
        if temp_output_wav.exists():
            temp_output_wav.unlink()
        row["dub_error"] = str(exc)
        logger.warning("CosyVoice TTS failed for %s: %s", row["chunk_id"], exc)
        return False
    finally:
        if inputs.temp_prompt_audio is not None and inputs.temp_prompt_audio.exists():
            inputs.temp_prompt_audio.unlink()

    if not saved:
        if temp_output_wav.exists():
            temp_output_wav.unlink()
        row["dub_error"] = "CosyVoice did not return audio"
        logger.warning("CosyVoice returned no audio for %s", row["chunk_id"])
        return False
    return True


def _finalize_chunk_success(row: dict[str, Any], output_wav: Path, *, runtime_settings: dict[str, Any]) -> None:
    row["dub_input_signature"] = build_dub_input_signature(row, runtime=runtime_settings)
    row.pop("dub_error", None)
    row.pop("dub_stale", None)
    row["dub_wav"] = project_relative(output_wav)
