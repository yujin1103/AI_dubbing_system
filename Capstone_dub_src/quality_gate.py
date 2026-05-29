from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# audio_features 는 numpy/librosa 의존 — webapp-backend 같은 가벼운 컨테이너에서도 quality_gate 를 import 할 수 있도록 lazy.
from common import resolve_project_path


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _spoken_character_count(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    return len(re.sub(r"\s+", "", normalized))


def _word_count(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    return len(re.findall(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]+", normalized))


def _repetition_ratio(text: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]+", _normalize_text(text).lower())
    if len(tokens) < 3:
        return 0.0
    return 1.0 - _safe_ratio(float(len(set(tokens))), float(len(tokens)))


def _content_text(value: str) -> str:
    return "".join(re.findall(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]+", _normalize_text(value).lower()))


def _first_content_token(value: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]+", _normalize_text(value).lower())
    return tokens[0] if tokens else ""


def _text_similarity(left: str, right: str) -> float:
    left_text = _content_text(left)
    right_text = _content_text(right)
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def detect_register(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return "unknown"

    polite_patterns = [
        r"\uC694[.!?]?$",
        r"\uB2C8\uB2E4[.!?]?$",
        r"\uC2B5\uB2C8\uB2E4[.!?]?$",
        r"\uC138\uC694[.!?]?$",
        r"\uC8E0[.!?]?$",
        r"\uC2E4\uB840\uD569\uB2C8\uB2E4[.!?]?$",
        r"\uBD80\uD0C1\uB4DC\uB9BD\uB2C8\uB2E4[.!?]?$",
    ]
    casual_patterns = [
        r"\uC57C[.!?]?$",
        r"\uC5B4[.!?]?$",
        r"\uC544[.!?]?$",
        r"\uD574[.!?]?$",
        r"\uAC70\uC57C[.!?]?$",
        r"\uC8FC[.!?]?$",
        r"\uC904\uB798[.!?]?$",
        r"\uD558\uC9C0 \uB9C8[.!?]?$",
    ]

    polite_score = sum(1 for pattern in polite_patterns if re.search(pattern, normalized))
    casual_score = sum(1 for pattern in casual_patterns if re.search(pattern, normalized))

    if polite_score and casual_score:
        return "mixed"
    if polite_score:
        return "polite"
    if casual_score:
        return "casual"
    return "unknown"


def _finalize_assessment(
    *,
    stage: str,
    flags: list[str],
    warnings: list[str],
    metrics: dict[str, Any] | None = None,
    accepted: bool | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    if accepted is None:
        accepted = not flags

    score = 1.0
    score -= min(0.75, 0.25 * len(flags))
    score -= min(0.2, 0.05 * len(warnings))
    score = max(0.0, round(score, 3))

    return {
        "stage": stage,
        "accepted": bool(accepted),
        "critical_flags": flags,
        "warnings": warnings,
        "score": score,
        "metrics": metrics,
    }


def attach_stage_quality(row: dict[str, Any], stage: str, assessment: dict[str, Any]) -> dict[str, Any]:
    gates = row.get("quality_gates")
    if not isinstance(gates, dict):
        gates = {}
        row["quality_gates"] = gates
    gates[stage] = assessment
    return row


def assess_asr_row(row: dict[str, Any], *, chunk_feature: dict[str, Any] | None = None) -> dict[str, Any]:
    text = _normalize_text(str(row.get("text_src", "")))
    duration = float(row.get("duration") or 0.0)
    flags: list[str] = []
    warnings: list[str] = []

    metrics = {
        "duration": round(duration, 3),
        "char_count": _spoken_character_count(text),
        "word_count": _word_count(text),
    }

    if duration > 0.8 and not text:
        flags.append("blank_transcript")
    if text:
        cps = _safe_ratio(float(metrics["char_count"]), duration)
        repetition = _repetition_ratio(text)
        metrics["chars_per_second"] = round(cps, 3)
        metrics["repetition_ratio"] = round(repetition, 3)
        if duration >= 2.0 and cps < 0.8:
            warnings.append("suspiciously_sparse_transcript")
        if cps > 18.0:
            warnings.append("suspiciously_dense_transcript")
        if repetition > 0.6:
            warnings.append("repetitive_transcript")
    if chunk_feature:
        metrics["silence_ratio"] = chunk_feature.get("silence_ratio")
        metrics["voiced_ratio"] = chunk_feature.get("voiced_ratio")
        if float(chunk_feature.get("voiced_ratio", 1.0) or 1.0) < 0.1:
            warnings.append("low_voiced_ratio")
        if float(chunk_feature.get("silence_ratio", 0.0) or 0.0) > 0.8:
            warnings.append("mostly_silent_chunk")

    return _finalize_assessment(stage="asr", flags=flags, warnings=warnings, metrics=metrics)


def assess_reference_candidate(
    row: dict[str, Any],
    *,
    chunk_feature: dict[str, Any] | None = None,
    min_prompt_sec: float = 1.2,
) -> dict[str, Any]:
    text = _normalize_text(str(row.get("text_src", "")))
    duration = float(row.get("duration") or 0.0)
    wav_value = row.get("wav")
    flags: list[str] = []
    warnings: list[str] = []

    metrics = {
        "duration": round(duration, 3),
        "has_text": bool(text),
        "wav": wav_value,
    }

    if not wav_value or not resolve_project_path(wav_value).exists():
        flags.append("missing_reference_audio")
    if not text:
        flags.append("missing_reference_transcript")
    if duration <= 0:
        flags.append("invalid_reference_duration")
    elif duration < 0.55:
        flags.append("reference_too_short_critical")
    elif duration < min_prompt_sec:
        warnings.append("reference_short")

    if chunk_feature:
        silence_ratio = float(chunk_feature.get("silence_ratio", 0.0) or 0.0)
        voiced_ratio = float(chunk_feature.get("voiced_ratio", 1.0) or 1.0)
        clipping_ratio = float(chunk_feature.get("clipping_ratio", 0.0) or 0.0)
        metrics["silence_ratio"] = silence_ratio
        metrics["voiced_ratio"] = voiced_ratio
        metrics["clipping_ratio"] = clipping_ratio
        if silence_ratio > 0.85:
            flags.append("reference_mostly_silent")
        elif silence_ratio > 0.6:
            warnings.append("reference_high_silence")
        if voiced_ratio < 0.08:
            flags.append("reference_low_voiced_ratio")
        elif voiced_ratio < 0.25:
            warnings.append("reference_sparse_voicing")
        if clipping_ratio > 0.05:
            flags.append("reference_clipping_severe")
        elif clipping_ratio > 0.01:
            warnings.append("reference_clipping")

    return _finalize_assessment(stage="reference", flags=flags, warnings=warnings, metrics=metrics)


def assess_translation_row(
    row: dict[str, Any],
    *,
    target_language: str,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_src = _normalize_text(str(row.get("text_src", "")))
    text_translated = _normalize_text(str(row.get("text_translated", "")))
    flags: list[str] = []
    warnings: list[str] = []

    metrics = {
        "source_char_count": _spoken_character_count(text_src),
        "target_char_count": _spoken_character_count(text_translated),
        "target_register": detect_register(text_translated),
        "target_language": target_language,
    }

    if text_src and not text_translated:
        flags.append("blank_translation")
    if text_translated and _repetition_ratio(text_translated) > 0.55:
        warnings.append("repetitive_translation")
    if metrics["target_register"] == "mixed":
        warnings.append("mixed_register")

    if budget:
        actual_units = int(budget.get("actual_units", 0) or 0)
        min_units = int(budget.get("min_units", 0) or 0)
        max_units = int(budget.get("max_units", 0) or 0)
        metrics["budget"] = {
            "actual_units": actual_units,
            "min_units": min_units,
            "max_units": max_units,
        }
        if actual_units and max_units and actual_units > max_units:
            warnings.append("translation_over_budget")
        elif actual_units and min_units and actual_units < min_units:
            warnings.append("translation_under_budget")

    return _finalize_assessment(stage="translation", flags=flags, warnings=warnings, metrics=metrics)


def assess_tts_output(
    output_wav: str | Path,
    *,
    expected_duration: float,
) -> dict[str, Any]:
    resolved = resolve_project_path(output_wav)
    if not resolved.exists():
        return _finalize_assessment(
            stage="tts",
            flags=["missing_tts_output"],
            warnings=[],
            metrics={"path": str(resolved)},
        )

    from audio_features import analyze_audio_path  # numpy 의존 — 호출 시점에만 끌어옴
    feature = analyze_audio_path(resolved)
    actual_duration = float(feature.get("duration", 0.0) or 0.0)
    duration_ratio = _safe_ratio(actual_duration, expected_duration) if expected_duration > 0 else 1.0
    flags: list[str] = []
    warnings: list[str] = []

    if actual_duration <= 0:
        flags.append("empty_tts_output")
    if expected_duration > 0:
        if duration_ratio < 0.55:
            flags.append("tts_under_duration_severe")
        elif duration_ratio < 0.78:
            warnings.append("tts_under_duration")
        if duration_ratio > 1.45:
            flags.append("tts_over_duration_severe")
        elif duration_ratio > 1.22:
            warnings.append("tts_over_duration")
    if float(feature.get("silence_ratio", 0.0) or 0.0) > 0.7:
        flags.append("tts_mostly_silent")
    elif float(feature.get("silence_ratio", 0.0) or 0.0) > 0.45:
        warnings.append("tts_high_silence")
    if float(feature.get("clipping_ratio", 0.0) or 0.0) > 0.05:
        flags.append("tts_clipping_severe")
    elif float(feature.get("clipping_ratio", 0.0) or 0.0) > 0.01:
        warnings.append("tts_clipping")

    metrics = {
        "expected_duration": round(expected_duration, 3),
        "actual_duration": round(actual_duration, 3),
        "duration_ratio": round(duration_ratio, 3),
        "rms_dbfs": feature.get("rms_dbfs"),
        "peak_dbfs": feature.get("peak_dbfs"),
        "silence_ratio": feature.get("silence_ratio"),
        "voiced_ratio": feature.get("voiced_ratio"),
        "clipping_ratio": feature.get("clipping_ratio"),
    }
    return _finalize_assessment(stage="tts", flags=flags, warnings=warnings, metrics=metrics)


def assess_tts_transcript(
    row: dict[str, Any],
    *,
    transcript: str,
    target_text: str,
) -> dict[str, Any]:
    normalized_transcript = _normalize_text(transcript)
    normalized_target = _normalize_text(target_text)
    duration = float(row.get("duration") or 0.0)
    flags: list[str] = []
    warnings: list[str] = []

    transcript_chars = _spoken_character_count(normalized_transcript)
    target_chars = _spoken_character_count(normalized_target)
    transcript_words = _word_count(normalized_transcript)
    repetition = _repetition_ratio(normalized_transcript)
    char_ratio = _safe_ratio(float(transcript_chars), float(target_chars))
    similarity = _text_similarity(normalized_target, normalized_transcript)

    metrics = {
        "duration": round(duration, 3),
        "target_char_count": target_chars,
        "transcript_char_count": transcript_chars,
        "transcript_word_count": transcript_words,
        "char_ratio": round(char_ratio, 3),
        "similarity": round(similarity, 3),
        "repetition_ratio": round(repetition, 3),
    }

    if duration > 0.8 and not normalized_transcript:
        flags.append("blank_tts_transcript")
    if transcript_words >= 5 and repetition > 0.3:
        flags.append("repetitive_tts_transcript")
    elif transcript_words >= 5 and repetition > 0.2:
        warnings.append("possibly_repetitive_tts_transcript")

    if normalized_target and normalized_transcript:
        if char_ratio < 0.45:
            flags.append("tts_transcript_too_short")
        elif char_ratio < 0.65:
            warnings.append("tts_transcript_short")
        if char_ratio > 1.8:
            flags.append("tts_transcript_too_long")
        elif char_ratio > 1.35:
            warnings.append("tts_transcript_long")
        if similarity < 0.28:
            flags.append("tts_transcript_mismatch")
        elif similarity < 0.42:
            warnings.append("tts_transcript_low_similarity")

        first_target_token = _first_content_token(normalized_target)
        first_transcript_token = _first_content_token(normalized_transcript)
        if first_target_token and first_transcript_token and first_target_token != first_transcript_token:
            if len(first_target_token) <= 4 and first_target_token not in normalized_transcript:
                warnings.append("tts_transcript_missing_opening_token")

    return _finalize_assessment(stage="tts_asr", flags=flags, warnings=warnings, metrics=metrics)
