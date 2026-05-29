from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_features import compact_feature_summary
from common import resolve_project_path
from quality_gate import assess_reference_candidate


def _build_candidate(
    *,
    row: dict[str, Any],
    mode: str,
    wav: str | Path,
    chunk_id: str,
    text_src: str | None = None,
    feature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "wav": resolve_project_path(wav),
        "chunk_id": chunk_id,
        "text_src": text_src if text_src is not None else str(row.get("text_src", "") or ""),
        "duration": float(row.get("duration") or 0.0),
        "feature": feature or {},
    }


def build_speaker_reference_bank(
    rows: list[dict[str, Any]],
    *,
    feature_map: dict[str, dict[str, Any]] | None = None,
    min_prompt_sec: float,
) -> dict[str, dict[str, Any]]:
    feature_map = feature_map or {}
    refs: dict[str, dict[str, Any]] = {}
    for row in rows:
        speaker = row.get("speaker")
        wav_value = row.get("wav")
        chunk_id = str(row.get("chunk_id", "")).strip()
        if not speaker or not wav_value or not chunk_id:
            continue
        candidate = _build_candidate(
            row=row,
            mode="speaker_best",
            wav=wav_value,
            chunk_id=chunk_id,
            feature=feature_map.get(chunk_id),
        )
        assessment = assess_reference_candidate(
            {
                "wav": str(candidate["wav"]),
                "text_src": candidate["text_src"],
                "duration": candidate["duration"],
            },
            chunk_feature=candidate["feature"],
            min_prompt_sec=min_prompt_sec,
        )
        if not assessment["accepted"]:
            continue
        score = (
            assessment["score"],
            float(candidate["duration"] or 0.0),
            float(candidate["feature"].get("voiced_ratio", 0.0) or 0.0),
        )
        current = refs.get(str(speaker))
        if current is None or score > current["selection_score"]:
            refs[str(speaker)] = {
                **candidate,
                "assessment": assessment,
                "selection_score": score,
            }
    return refs


def choose_reference_for_row(
    row: dict[str, Any],
    *,
    feature_map: dict[str, dict[str, Any]] | None = None,
    min_prompt_sec: float,
    reference_mode: str = "self",
    speaker_best_map: dict[str, dict[str, Any]] | None = None,
    speaker_memory_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    feature_map = feature_map or {}
    speaker_best_map = speaker_best_map or {}
    speaker_memory_map = speaker_memory_map or {}

    normalized_reference_mode = (reference_mode or "self").strip().lower()
    if normalized_reference_mode not in {"self", "auto"}:
        normalized_reference_mode = "self"

    chunk_id = str(row.get("chunk_id", "")).strip()
    own_feature = feature_map.get(chunk_id, {})
    own_candidate: dict[str, Any] | None = None
    own_assessment: dict[str, Any] | None = None
    if row.get("wav"):
        own_candidate = _build_candidate(
            row=row,
            mode="self",
            wav=row["wav"],
            chunk_id=chunk_id,
            feature=own_feature,
        )
        own_assessment = assess_reference_candidate(
            {
                "wav": str(own_candidate["wav"]),
                "text_src": own_candidate["text_src"],
                "duration": own_candidate["duration"],
            },
            chunk_feature=own_feature,
            min_prompt_sec=min_prompt_sec,
        )

    candidate_summaries: list[dict[str, Any]] = []
    if own_candidate is not None and own_assessment is not None:
        candidate_summaries.append(
            {
                "mode": "self",
                "chunk_id": chunk_id,
                "assessment": own_assessment,
                "feature": compact_feature_summary(own_feature),
            }
        )
        if own_assessment["accepted"]:
            return {
                "prompt_audio": own_candidate["wav"],
                "prompt_text_src": own_candidate["text_src"],
                "reference_chunk_id": own_candidate["chunk_id"],
                "reference_mode": "self",
                "assessment": own_assessment,
                "decision": "self",
                "candidate_summaries": candidate_summaries,
                "rejection_reasons": [],
            }

    alternatives: list[dict[str, Any]] = []
    speaker_key = str(row.get("speaker") or "")
    for label, mapping in (("speaker_best", speaker_best_map), ("speaker_memory", speaker_memory_map)):
        candidate = mapping.get(speaker_key)
        if not candidate:
            continue
        if str(candidate.get("chunk_id", "")) == chunk_id:
            continue
        assessment = candidate.get("assessment")
        if not isinstance(assessment, dict):
            assessment = assess_reference_candidate(
                {
                    "wav": str(candidate["wav"]),
                    "text_src": candidate.get("text_src", ""),
                    "duration": candidate.get("duration", 0.0),
                },
                chunk_feature=candidate.get("feature"),
                min_prompt_sec=min_prompt_sec,
            )
        candidate_summaries.append(
            {
                "mode": label,
                "chunk_id": candidate.get("chunk_id"),
                "assessment": assessment,
                "feature": compact_feature_summary(candidate.get("feature")),
            }
        )
        if assessment["accepted"]:
            alternatives.append({**candidate, "assessment": assessment, "resolved_mode": label})

    if alternatives:
        alternatives = sorted(
            alternatives,
            key=lambda item: (
                float(item["assessment"]["score"]),
                float(item.get("duration", 0.0) or 0.0),
                float((item.get("feature") or {}).get("voiced_ratio", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected = alternatives[0]
        return {
            "prompt_audio": selected["wav"],
            "prompt_text_src": selected.get("text_src", "") or str(row.get("text_src", "") or ""),
            "reference_chunk_id": selected["chunk_id"],
            "reference_mode": selected["resolved_mode"],
            "assessment": selected["assessment"],
            "decision": "fallback" if own_candidate is not None else "speaker_only",
            "candidate_summaries": candidate_summaries,
            "rejection_reasons": [] if own_assessment is None else own_assessment["critical_flags"],
        }

    if own_candidate is not None and own_assessment is not None:
        force_mode = "self_force_last_resort" if normalized_reference_mode == "self" else "self_unverified"
        return {
            "prompt_audio": own_candidate["wav"],
            "prompt_text_src": own_candidate["text_src"],
            "reference_chunk_id": own_candidate["chunk_id"],
            "reference_mode": force_mode,
            "assessment": own_assessment,
            "decision": "self_last_resort",
            "candidate_summaries": candidate_summaries,
            "rejection_reasons": own_assessment["critical_flags"],
        }

    raise FileNotFoundError(f"No usable reference candidate found for {chunk_id}")
