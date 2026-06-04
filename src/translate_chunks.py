from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from itertools import islice
from pathlib import Path
from typing import Any

# audio_features 는 numpy 의존 — 모듈 레벨에서 끌어오면 webapp-backend 같은 가벼운 컨테이너가 깨진다.
# 풀 파이프라인 컨테이너에서만 실제로 사용되니, 사용 함수 안에서 lazy import 한다.
from common import get_logger, load_env_file, load_json, load_json_if_exists, save_json
from quality_gate import assess_translation_row, attach_stage_quality, detect_register

logger = get_logger("translate_chunks")


class TranslationBlocked(Exception):
    """LLM 응답이 content_filter 등으로 차단되어 본문이 비어 돌아온 경우. 호출 측이 청크 단위 fallback 처리 가능하도록 분리된 예외."""

    def __init__(self, reason: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.payload = payload or {}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _normalize_language(value: str) -> str:
    return _normalize_text(value).lower()


def _translation_target_language(row: dict[str, Any]) -> str:
    quality_gates = row.get("quality_gates")
    if not isinstance(quality_gates, dict):
        return ""
    translation_gate = quality_gates.get("translation")
    if not isinstance(translation_gate, dict):
        return ""
    metrics = translation_gate.get("metrics")
    if not isinstance(metrics, dict):
        return ""
    return str(metrics.get("target_language", "") or "")


def _can_preserve_existing_translation(existing: dict[str, Any], *, current_src: str, target_language: str) -> bool:
    if not existing.get("text_translated"):
        return False
    if _normalize_text(existing.get("text_src", "")) != current_src:
        return False
    existing_target_language = _translation_target_language(existing)
    return bool(existing_target_language) and _normalize_language(existing_target_language) == _normalize_language(target_language)


def _is_japanese_target(target_language: str) -> bool:
    normalized = (target_language or "").strip().lower()
    return "japan" in normalized or normalized in {"ja", "jp", "japanese"}


def _is_korean_target(target_language: str) -> bool:
    normalized = (target_language or "").strip().lower()
    return "korea" in normalized or normalized in {"ko", "kr", "korean"}


def _count_spoken_units(text: str, target_language: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    if _is_korean_target(target_language) or _is_japanese_target(target_language):
        tokens = re.findall(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3\u3040-\u30FF\u4E00-\u9FFF]", normalized)
        return len(tokens)
    return len(re.findall(r"[A-Za-z0-9]+", normalized))


def _compute_duration_budget(
    row: dict[str, Any],
    *,
    target_language: str,
    chunk_feature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration_sec = float(row.get("duration") or 0.0)
    source_text = str(row.get("text_src", "") or "")
    source_units = _count_spoken_units(source_text, target_language=target_language)
    pause_count = int((chunk_feature or {}).get("pause_count", 0) or 0)

    if _is_korean_target(target_language):
        min_rate, target_rate, max_rate = 3.6, 4.8, 6.2
    elif _is_japanese_target(target_language):
        min_rate, target_rate, max_rate = 4.2, 5.8, 7.4
    else:
        min_rate, target_rate, max_rate = 2.0, 3.0, 4.2

    min_units = max(1, int(round(duration_sec * min_rate))) if duration_sec > 0 else 0
    target_units = max(1, int(round(duration_sec * target_rate))) if duration_sec > 0 else source_units
    max_units = max(min_units, int(round(duration_sec * max_rate))) if duration_sec > 0 else source_units

    if source_units:
        max_units = max(min_units, min(max_units, max(source_units * 2, min_units)))
        target_units = min(max_units, max(min_units, target_units))

    if pause_count >= 2:
        max_units += min(6, pause_count)

    return {
        "duration_sec": round(duration_sec, 3),
        "source_units": source_units,
        "pause_count": pause_count,
        "min_units": int(min_units),
        "target_units": int(target_units),
        "max_units": int(max_units),
    }


def _measure_budget_fit(text: str, *, target_language: str, budget: dict[str, Any]) -> dict[str, Any]:
    actual_units = _count_spoken_units(text, target_language=target_language)
    min_units = int(budget.get("min_units", 0) or 0)
    max_units = int(budget.get("max_units", 0) or 0)
    target_units = int(budget.get("target_units", 0) or 0)
    within_budget = (actual_units >= min_units if min_units else True) and (actual_units <= max_units if max_units else True)
    return {
        **budget,
        "actual_units": actual_units,
        "within_budget": within_budget,
        "over_budget": bool(max_units and actual_units > max_units),
        "under_budget": bool(min_units and actual_units < min_units and actual_units > 0),
        "distance_from_target": abs(actual_units - target_units) if target_units else 0,
    }


# 다국어 무관 프롬프트 보강 — config translation.scene_context / register 로 build_translation_entries 가 런타임 1회 설정.
_SCENE_CONTEXT: str = ""
_REGISTER_OVERRIDE: str = ""


def _extra_prompt_rules(target_label: str) -> str:
    """모든 번역 프롬프트에 붙는 공통 규칙 — (1) 항상 타겟언어 완역(영어 잔류 차단) (2) 씬/등장인물 컨텍스트."""
    block = (
        f"\nIMPORTANT: The output MUST be written entirely in {target_label}. "
        f"Translate EVERYTHING into {target_label} — including country names, place names, numbers, "
        "statistics, lists, and fast rapid speech. Never leave any words in the source language.\n"
    )
    scene = (_SCENE_CONTEXT or "").strip()
    if scene:
        block += (
            "Scene/character context (use this to fix tone, register, pronouns, who speaks to whom, "
            f"and continuity across lines): {scene}\n"
        )
    return block


def _build_budget_rewrite_messages(
    row: dict[str, Any],
    *,
    current_translation: str,
    source_language: str,
    target_language: str,
    budget: dict[str, Any],
    register_hint: str,
) -> tuple[str, str]:
    source_label = source_language or "source language"
    target_label = target_language or "target language"
    japanese_target = _is_japanese_target(target_language)
    min_units = int(budget.get("min_units", 0) or 0)
    target_units = int(budget.get("target_units", 0) or 0)
    max_units = int(budget.get("max_units", 0) or 0)
    register_rule = ""
    normalized_register = (register_hint or "").strip().lower()
    if normalized_register in {"polite", "casual"}:
        register_rule = f"Keep the output in consistent {normalized_register} register unless the source clearly forces a change.\n"

    if japanese_target:
        system_prompt = (
            "You are a dubbing translation finisher.\n"
            f"Rewrite one translated line into natural {target_label} so it fits dubbing timing.\n"
            "Return one minified JSON object with exactly two string keys: text_translated and text_tts.\n"
            "Preserve the original meaning, tone, emotion, and scene intent.\n"
            "Do not add plot facts, explanations, stage directions, or markdown.\n"
            "Treat the output as a performance script: every number, date, unit, price, and abbreviation must be written exactly the way a voice actor would pronounce it aloud, not as Arabic digits or written shorthand.\n"
            f"Fit the spoken line into roughly {min_units}-{max_units} speech units, preferably near {target_units}.\n"
            + register_rule
        )
        user_prompt = (
            f"Source dialogue ({source_label}):\n{row.get('text_src', '')}\n\n"
            f"Current translation ({target_label}):\n{current_translation}\n\n"
            "Rewrite it to fit timing better while staying natural for dubbing."
        )
        return system_prompt, user_prompt

    system_prompt = (
        "You are a dubbing translation finisher.\n"
        f"Rewrite one translated line into natural spoken {target_label} so it fits dubbing timing.\n"
        "Return only the revised final line.\n"
        "Preserve meaning, tone, subtext, and character voice.\n"
        "Do not add explanations, stage directions, or markdown.\n"
        "Treat the output as a performance script: every number, date, unit, price, and abbreviation must be written exactly the way a voice actor would pronounce it aloud, not as Arabic digits or written shorthand.\n"
        f"Fit the spoken line into roughly {min_units}-{max_units} speech units, preferably near {target_units}.\n"
        "Prefer a line an actor could say naturally in one take.\n"
        + register_rule
    )
    user_prompt = (
        f"Source dialogue ({source_label}):\n{row.get('text_src', '')}\n\n"
        f"Current translation ({target_label}):\n{current_translation}\n\n"
        "Rewrite it to fit timing better while staying natural for dubbing."
    )
    return system_prompt, user_prompt


def _build_translation_messages(
    text: str,
    *,
    source_language: str,
    target_language: str,
    target_duration_sec: float | None = None,
    duration_budget: dict[str, Any] | None = None,
    register_hint: str = "",
) -> tuple[str, str]:
    source_label = source_language or "source language"
    target_label = target_language or "target language"
    japanese_target = _is_japanese_target(target_language)
    duration_rule = ""
    if target_duration_sec and target_duration_sec > 0:
        duration_rule = (
            f"9. The target spoken duration is about {target_duration_sec:.2f} seconds. "
            "Choose wording that can be spoken naturally within that time. If needed, compress aggressively while keeping the core meaning and tone.\n"
        )
    budget_rule = ""
    if duration_budget:
        min_units = int(duration_budget.get("min_units", 0) or 0)
        max_units = int(duration_budget.get("max_units", 0) or 0)
        target_units = int(duration_budget.get("target_units", 0) or 0)
        if min_units or max_units or target_units:
            budget_rule = (
                f"10. Keep the translated spoken length within roughly {min_units}-{max_units} speech units, "
                f"with {target_units} as the preferred target.\n"
            )
    register_rule = ""
    normalized_register = (register_hint or "").strip().lower()
    if normalized_register in {"polite", "casual"}:
        style = "formal/polite speech" if normalized_register == "polite" else "casual/banmal speech"
        register_rule = f"11. Keep the line in consistent {style} unless the source clearly forces a change.\n"
    if japanese_target:
        system_prompt = (
            "You are a professional AI dubbing translator for film and video.\n"
            f"Translate spoken dialogue from {source_label} into natural {target_label} for dubbing.\n"
            "Return one minified JSON object with exactly two string keys: text_translated and text_tts.\n"
            "Rules:\n"
            "1. text_translated must be natural contemporary Japanese in normal orthography.\n"
            "2. text_tts must be a Katakana-focused Japanese reading line optimized for TTS pronunciation.\n"
            "3. Translate for spoken dubbing, not for literal subtitles. Prefer idiomatic, performable lines.\n"
            "4. Preserve tone, emotion, intent, relationships, subtext, and character voice.\n"
            "5. Keep names and world-specific terms consistent.\n"
            "6. Keep both fields concise enough for dubbing and natural mouthfeel.\n"
            "7. Do not add explanations, stage directions, speaker labels, or extra wording not supported by the source.\n"
            "8. If the source is fragmentary or noisy, repair only when the meaning is strongly implied; otherwise keep it short and natural without inventing plot facts.\n"
            "9. Treat the output as a performance script: every number, date, unit, price, and abbreviation must be written exactly the way a voice actor would pronounce it aloud, not as Arabic digits or written shorthand.\n"
            "10. Output JSON only. No markdown, code fences, notes, or extra keys.\n"
            "11. If the input is empty, both fields must be empty strings.\n"
            + duration_rule +
            budget_rule +
            register_rule +
            "12. Example output: {\"text_translated\":\"...\",\"text_tts\":\"...\"}"
        )
        user_prompt = f"Source dialogue ({source_label}):\n{text}"
        return system_prompt, user_prompt

    system_prompt = (
        "You are a professional AI dubbing translator for film and video.\n"
        f"Translate spoken dialogue from {source_label} into natural {target_label} for dubbing.\n"
        "Rules:\n"
        "1. Return only the final translated dialogue.\n"
        "2. Do not include explanations, notes, numbering, quotes, speaker labels, markdown, or analysis.\n"
        "3. Translate for spoken dubbing, not literal subtitles. Prefer idiomatic lines that an actor could perform naturally.\n"
        "4. Make it sound natural when spoken aloud in a dubbed performance.\n"
        "5. Preserve tone, emotion, intent, relationships, subtext, and character voice.\n"
        "6. Keep names and world-specific terms consistent unless a natural localized form is clearly better.\n"
        "7. Keep it concise enough for dubbing and approximate line length.\n"
        "8. Prefer natural Korean phrasing over stiff or explanatory wording.\n"
        "9. Avoid random shifts between 존댓말 and 반말 unless the source clearly implies a change in relationship or register.\n"
        "10. If the input is a short exclamation, reaction, or name call, preserve that punch and tone naturally.\n"
        "11. If the source line is fragmentary or noisy, repair only when the most plausible meaning is obvious from the line itself; otherwise keep it short and natural without inventing plot facts.\n"
        "12. Treat the output as a performance script: every number, date, unit, price, and abbreviation must be written exactly the way a voice actor would pronounce it aloud, not as Arabic digits or written shorthand.\n"
        + duration_rule +
        budget_rule +
        register_rule +
        "13. If the input is empty, return an empty string."
        + _extra_prompt_rules(target_label)
    )
    user_prompt = f"Source dialogue ({source_label}):\n{text}"
    return system_prompt, user_prompt


def _build_full_transcript_context(rows: list[dict[str, Any]]) -> str:
    transcript_lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        chunk_id = str(row.get("chunk_id", "")).strip()
        speaker = str(row.get("speaker", "") or "UNKNOWN").strip()
        start = row.get("start")
        end = row.get("end")
        if start is not None and end is not None:
            timing = f"{float(start):.2f}-{float(end):.2f}s"
        else:
            duration = row.get("duration")
            timing = f"{float(duration):.2f}s" if duration is not None else "n/a"
        text_src = _normalize_text(str(row.get("text_src", "")))
        if not text_src:
            text_src = "[empty or unrecognized line]"
        transcript_lines.append(f"{index:03d}. [{chunk_id}] [{speaker}] [{timing}] {text_src}")
    return "\n".join(transcript_lines)


def _build_context_refinement_messages(
    batch_rows: list[dict[str, Any]],
    *,
    full_transcript: str,
    source_language: str,
    target_language: str,
) -> tuple[str, str]:
    source_label = source_language or "source language"
    target_label = target_language or "target language"
    japanese_target = _is_japanese_target(target_language)

    if japanese_target:
        system_prompt = (
            "You are a senior dubbing translation editor.\n"
            f"You are revising draft dialogue translated from {source_label} into {target_label}.\n"
            "You will receive the full source transcript for global context and a batch of draft line translations.\n"
            "Return one minified JSON object with exactly one top-level key: items.\n"
            "items must be an array of objects in the same order as the batch.\n"
            "Each item must contain exactly these string keys: chunk_id, text_translated, text_tts.\n"
            "Rules:\n"
            "1. Use the full transcript to resolve context, tone, continuity, pronouns, and speaker relationships.\n"
            "2. Rewrite each line into natural dubbing dialogue, not literal subtitle-style translation.\n"
            "3. Preserve emotion, intent, dramatic rhythm, humor, tension, and character voice.\n"
            "4. Keep each line concise and speakable for dubbing.\n"
            "5. Keep names and world-specific terms consistent across the scene.\n"
            "6. Avoid random shifts between formal and casual speech unless context clearly requires it.\n"
            "7. If ASR wording is noisy or fragmentary, repair only when the surrounding context strongly supports it. Do not invent new plot facts.\n"
            "8. Treat the output as a performance script: every number, date, unit, price, and abbreviation must be written exactly the way a voice actor would pronounce it aloud, not as Arabic digits or written shorthand.\n"
            "9. If a source line is empty, return empty strings for both text fields.\n"
            "10. Output JSON only. No markdown, explanations, or extra keys."
        )
    else:
        system_prompt = (
            "You are a senior dubbing translation editor.\n"
            f"You are revising draft dialogue translated from {source_label} into {target_label}.\n"
            "You will receive the full source transcript for global context and a batch of draft line translations.\n"
            "Return one minified JSON object with exactly one top-level key: items.\n"
            "items must be an array of objects in the same order as the batch.\n"
            "Each item must contain exactly these string keys: chunk_id and text_translated.\n"
            "Rules:\n"
            "1. Use the full transcript to resolve context, tone, continuity, pronouns, and speaker relationships.\n"
            "2. Rewrite each line into natural dubbing dialogue, not literal subtitle-style translation.\n"
            "3. Preserve emotion, intent, dramatic rhythm, humor, tension, and character voice.\n"
            "4. Keep each line concise, speakable, and suitable for dubbed performance in the target language.\n"
            "5. Prefer idiomatic spoken Korean over stiff or explanatory wording.\n"
            "6. Keep names and world-specific terms consistent across the scene.\n"
            "7. SPLIT SENTENCES: a source line is often ONE fragment of a sentence that continues in the previous or next line. Translate each fragment so it is a clean, deliverable Korean piece that connects naturally with its neighbours — never leave a dangling subject with no predicate, or a bare noun where the line needs a verb. Keep the SAME register and tense across all fragments of one sentence.\n"
            "8. Avoid random shifts between 존댓말 and 반말 unless context clearly requires it; a sentence continued across chunks must NOT change register mid-way.\n"
            "9. Choose the context-correct word sense — do not pick a wrong homonym (e.g. biological 'rebirth' after surviving a life stage is 거듭남/재탄생, NOT 환생/reincarnation).\n"
            "10. Keep essential arguments (objects such as 나를/날, 그를) when dropping them makes the line unclear or sound truncated.\n"
            "11. If ASR wording is noisy or fragmentary, repair only when the surrounding context strongly supports it. Do not invent new plot facts.\n"
            "12. Treat the output as a performance script: every number, date, unit, price, and abbreviation must be written exactly the way a voice actor would pronounce it aloud, not as Arabic digits or written shorthand.\n"
            "13. If a source line is empty, return an empty string for that line.\n"
            "14. Output JSON only. No markdown, explanations, or extra keys."
            + _extra_prompt_rules(target_label)
        )

    batch_payload: list[dict[str, Any]] = []
    for row in batch_rows:
        item = {
            "chunk_id": row["chunk_id"],
            "speaker": row.get("speaker"),
            "start": row.get("start"),
            "end": row.get("end"),
            "duration_sec": row.get("duration"),
            "text_src": row.get("text_src", ""),
            "draft_translation": row.get("text_translated", ""),
        }
        if row.get("translation_budget"):
            item["duration_budget"] = row["translation_budget"]
        if row.get("translation_style_hint"):
            item["speaker_style_hint"] = row["translation_style_hint"]
        if row.get("text_tts"):
            item["draft_tts"] = row["text_tts"]
        batch_payload.append(item)

    example = (
        '{"items":[{"chunk_id":"chunk_0001","text_translated":"..."},{"chunk_id":"chunk_0002","text_translated":"..."}]}'
        if not japanese_target
        else '{"items":[{"chunk_id":"chunk_0001","text_translated":"...","text_tts":"..."},{"chunk_id":"chunk_0002","text_translated":"...","text_tts":"..."}]}'
    )
    user_prompt = (
        f"Full source transcript ({source_label}):\n"
        f"{full_transcript}\n\n"
        f"Batch to revise ({source_label} -> {target_label}):\n"
        f"{json.dumps(batch_payload, ensure_ascii=False)}\n\n"
        f"Return JSON in this shape: {example}"
    )
    return system_prompt, user_prompt


def _iter_batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError(f"context batch size must be positive, got {batch_size}")
    iterator = iter(rows)
    batches: list[list[dict[str, Any]]] = []
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        batches.append(batch)
    return batches


def _join_url(base_url: str, endpoint: str) -> str:
    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


def _detect_content_filter_block(payload: dict[str, Any]) -> str | None:
    """OpenAI/Azure 응답에서 content_filter 차단을 식별해 사람이 읽을 수 있는 reason 반환. 차단 아니면 None."""
    choices = payload.get("choices", [])
    for choice in choices:
        if choice.get("finish_reason") == "content_filter":
            categories = choice.get("content_filter_results") or {}
            triggered = [
                f"{name}/{info.get('severity', 'unknown')}"
                for name, info in categories.items()
                if isinstance(info, dict) and info.get("filtered")
            ]
            return "content_filter: " + (", ".join(triggered) if triggered else "blocked")
    prompt_filter = payload.get("prompt_filter_results") or []
    for entry in prompt_filter:
        cats = (entry or {}).get("content_filter_results") or {}
        triggered = [
            f"prompt:{name}/{info.get('severity', 'unknown')}"
            for name, info in cats.items()
            if isinstance(info, dict) and info.get("filtered")
        ]
        if triggered:
            return "content_filter: " + ", ".join(triggered)
    return None


def _extract_generated_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return _normalize_text(content)
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if isinstance(text_value, str) and text_value.strip():
                        texts.append(text_value)
            if texts:
                return _normalize_text(" ".join(texts))

    candidates = payload.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
        if texts:
            return _normalize_text(" ".join(texts))

    if isinstance(payload.get("text"), str):
        return _normalize_text(payload["text"])

    blocked_reason = _detect_content_filter_block(payload)
    if blocked_reason:
        raise TranslationBlocked(blocked_reason, payload=payload)
    raise RuntimeError(f"No generated text found in API response: {payload}")


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: int,
    *,
    max_attempts: int = 3,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                body = response.read().decode("utf-8")
                return int(response.status), json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw_error": body}
            return int(exc.code), parsed
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_sec = min(6, attempt * 2)
            logger.warning(
                "Transient translation API error on attempt %s/%s: %s | retrying in %ss",
                attempt,
                max_attempts,
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    if last_error is not None:
        raise RuntimeError(f"Translation API request failed after {max_attempts} attempts: {last_error}") from last_error
    raise RuntimeError("Translation API request failed without a captured error")


def _build_payload(
    text: str,
    *,
    source_language: str,
    target_language: str,
    target_duration_sec: float | None,
    duration_budget: dict[str, Any] | None,
    register_hint: str,
    endpoint: str,
    model_name: str,
    response_format_json: bool = False,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
) -> dict[str, Any]:
    if system_prompt_override is not None or user_prompt_override is not None:
        if system_prompt_override is None or user_prompt_override is None:
            raise ValueError("Both system_prompt_override and user_prompt_override must be provided together")
        system_prompt = system_prompt_override
        user_prompt = user_prompt_override
    else:
        system_prompt, user_prompt = _build_translation_messages(
            text,
            source_language=source_language,
            target_language=target_language,
            target_duration_sec=target_duration_sec,
            duration_budget=duration_budget,
            register_hint=register_hint,
        )
    normalized_endpoint = endpoint.strip().lower()
    japanese_target = _is_japanese_target(target_language)
    if normalized_endpoint.endswith("/chat/completions") or "/chat/completions" in normalized_endpoint:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        if japanese_target or response_format_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    merged_prompt = f"{system_prompt}\n\n{user_prompt}"
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": merged_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.9,
            "maxOutputTokens": 512,
        },
    }


def _translate_with_vectorengine_api(
    text: str,
    *,
    source_language: str,
    target_language: str,
    target_duration_sec: float | None,
    duration_budget: dict[str, Any] | None,
    register_hint: str,
    api_key: str,
    base_url: str,
    endpoint: str,
    model_name: str,
    timeout_sec: int,
    response_format_json: bool = False,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
) -> str:
    if not text.strip() and not user_prompt_override:
        return ""

    payload_variants: list[tuple[str, dict[str, Any]]] = [
        (
            "json_object" if response_format_json else "default",
            _build_payload(
                text,
                source_language=source_language,
                target_language=target_language,
                target_duration_sec=target_duration_sec,
                duration_budget=duration_budget,
                register_hint=register_hint,
                endpoint=endpoint,
                model_name=model_name,
                response_format_json=response_format_json,
                system_prompt_override=system_prompt_override,
                user_prompt_override=user_prompt_override,
            ),
        )
    ]
    if response_format_json:
        payload_variants.append(
            (
                "prompt_only_json",
                _build_payload(
                    text,
                    source_language=source_language,
                    target_language=target_language,
                    target_duration_sec=target_duration_sec,
                    duration_budget=duration_budget,
                    register_hint=register_hint,
                    endpoint=endpoint,
                    model_name=model_name,
                    response_format_json=False,
                    system_prompt_override=system_prompt_override,
                    user_prompt_override=user_prompt_override,
                ),
            )
        )

    base_request_url = _join_url(base_url, endpoint)
    auth_attempts = [
        (base_request_url, {"Authorization": f"Bearer {api_key}"}, "Authorization: Bearer"),
        (base_request_url, {"x-api-key": api_key}, "x-api-key"),
        (base_request_url, {"x-goog-api-key": api_key}, "x-goog-api-key"),
        (
            base_request_url + ("&" if "?" in base_request_url else "?") + urllib.parse.urlencode({"key": api_key}),
            {},
            "query:key",
        ),
    ]

    errors: list[str] = []
    for url, headers, auth_label in auth_attempts:
        auth_failed = False
        for payload_label, payload in payload_variants:
            status, response_payload = _post_json(url, payload, headers, timeout_sec)
            if 200 <= status < 300:
                logger.info("Translation API accepted auth scheme: %s | payload=%s", auth_label, payload_label)
                return _extract_generated_text(response_payload)
            errors.append(f"{auth_label} | {payload_label} -> HTTP {status}: {response_payload}")
            if status in {401, 403}:
                auth_failed = True
                break
            if status not in {400, 404, 422}:
                break
        if not auth_failed:
            break

    raise RuntimeError("VectorEngine translation request failed. " + " | ".join(errors))


def _strip_json_wrappers(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_translation_response(raw_text: str, *, target_language: str) -> tuple[str, str | None]:
    normalized = _normalize_text(raw_text)
    if not _is_japanese_target(target_language):
        return normalized, None
    candidate = _strip_json_wrappers(raw_text)
    try:
        parsed = json.loads(candidate)
        text_translated = _normalize_text(str(parsed.get("text_translated", "")))
        text_tts = _normalize_text(str(parsed.get("text_tts", "")))
        if text_translated or text_tts:
            return text_translated or text_tts, text_tts or text_translated
    except json.JSONDecodeError:
        pass
    return normalized, normalized


def _parse_context_refinement_response(raw_text: str, *, target_language: str) -> list[dict[str, str]]:
    candidate = _strip_json_wrappers(raw_text)
    parsed = json.loads(candidate)
    items = parsed.get("items")
    if not isinstance(items, list):
        raise RuntimeError(f"Context refinement response is missing items array: {parsed}")

    japanese_target = _is_japanese_target(target_language)
    cleaned_items: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError(f"Context refinement item is not an object: {item}")
        chunk_id = _normalize_text(str(item.get("chunk_id", "")))
        text_translated = _normalize_text(str(item.get("text_translated", "")))
        if not chunk_id:
            raise RuntimeError(f"Context refinement item is missing chunk_id: {item}")
        cleaned: dict[str, str] = {
            "chunk_id": chunk_id,
            "text_translated": text_translated,
        }
        if japanese_target:
            cleaned["text_tts"] = _normalize_text(str(item.get("text_tts", text_translated)))
        cleaned_items.append(cleaned)
    return cleaned_items


def _refine_translations_with_context(
    translated_rows: list[dict[str, Any]],
    *,
    source_language: str,
    target_language: str,
    api_key: str,
    base_url: str,
    endpoint: str,
    model_name: str,
    timeout_sec: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not translated_rows:
        return translated_rows

    # blocked 청크는 refine 입력에서 제외 — LLM 이 또 차단당하거나 batch 응답 검증을 깨뜨릴 수 있음.
    refinable_rows = [row for row in translated_rows if not row.get("translation_blocked")]
    blocked_rows = [row for row in translated_rows if row.get("translation_blocked")]

    full_transcript = _build_full_transcript_context(refinable_rows or translated_rows)
    batches = _iter_batches(refinable_rows, batch_size)
    japanese_target = _is_japanese_target(target_language)
    logger.info(
        "Starting context refinement pass for %s rows (skipping %s blocked) in %s batches using global transcript context",
        len(refinable_rows),
        len(blocked_rows),
        len(batches),
    )

    refined_map: dict[str, dict[str, Any]] = {}

    # 무손실 속도: 문맥정제 배치 LLM 호출을 동시 실행(배치 독립·full_transcript 고정 입력 → 동일 출력).
    # 결과(_refine_pre)는 아래 순차 루프에서 조회해 파싱/검증/적용은 단일스레드로 수행.
    _refine_pre: dict[int, tuple[str, Any]] = {}
    def _refine_call(_ib):
        _bi, _batch = _ib
        _sp, _up = _build_context_refinement_messages(_batch, full_transcript=full_transcript, source_language=source_language, target_language=target_language)
        try:
            _resp = _translate_with_vectorengine_api(
                "", source_language=source_language, target_language=target_language,
                target_duration_sec=None, duration_budget=None, register_hint="",
                api_key=api_key, base_url=base_url, endpoint=endpoint, model_name=model_name,
                timeout_sec=timeout_sec, response_format_json=True,
                system_prompt_override=_sp, user_prompt_override=_up)
            return (_bi, ("ok", _resp))
        except TranslationBlocked as _exc:
            return (_bi, ("blocked", _exc.reason))
    if len(batches) > 1:
        import os as _os4
        from concurrent.futures import ThreadPoolExecutor as _TPE4
        _conc4 = max(1, int(_os4.environ.get("TRANSLATE_LLM_CONCURRENCY", "8")))
        if _conc4 > 1:
            with _TPE4(max_workers=min(_conc4, len(batches))) as _ex4:
                for _bi, _res in _ex4.map(_refine_call, list(enumerate(batches, start=1))):
                    _refine_pre[_bi] = _res

    for batch_index, batch in enumerate(batches, start=1):
        _pre = _refine_pre.get(batch_index)
        if _pre is not None and _pre[0] == "blocked":
            logger.warning("Context refinement batch %s blocked by translator: %s", batch_index, _pre[1])
            for row in batch:
                refined_map[row["chunk_id"]] = row
            continue
        if _pre is not None:
            raw_response = _pre[1]
        else:
            system_prompt, user_prompt = _build_context_refinement_messages(
                batch, full_transcript=full_transcript, source_language=source_language, target_language=target_language)
            try:
                raw_response = _translate_with_vectorengine_api(
                    "", source_language=source_language, target_language=target_language,
                    target_duration_sec=None, duration_budget=None, register_hint="",
                    api_key=api_key, base_url=base_url, endpoint=endpoint, model_name=model_name,
                    timeout_sec=timeout_sec, response_format_json=True,
                    system_prompt_override=system_prompt, user_prompt_override=user_prompt)
            except TranslationBlocked as exc:
                logger.warning("Context refinement batch %s blocked by translator: %s", batch_index, exc.reason)
                for row in batch:
                    refined_map[row["chunk_id"]] = row
                continue
        parsed_items = _parse_context_refinement_response(raw_response, target_language=target_language)
        parsed_map = {item["chunk_id"]: item for item in parsed_items}
        expected_ids = [row["chunk_id"] for row in batch]
        missing = [chunk_id for chunk_id in expected_ids if chunk_id not in parsed_map]
        extras = [chunk_id for chunk_id in parsed_map if chunk_id not in expected_ids]
        if missing or extras:
            raise RuntimeError(
                f"Context refinement batch {batch_index} returned mismatched chunk ids. Missing={missing} Extra={extras}"
            )

        for row in batch:
            previous_translation = row.get("text_translated", "")
            row["text_translated_initial"] = previous_translation
            item = parsed_map[row["chunk_id"]]
            refined_translation = item.get("text_translated", "")
            if not refined_translation and _normalize_text(str(row.get("text_src", ""))):
                logger.warning(
                    "Context refinement returned empty translation for %s; keeping first-pass translation",
                    row["chunk_id"],
                )
                refined_translation = previous_translation
            row["text_translated"] = refined_translation
            row["text_translated_context"] = refined_translation
            if japanese_target:
                row["text_tts"] = item.get("text_tts", row["text_translated"])
            elif "text_tts" in row:
                row.pop("text_tts", None)
            row["translation_context_refined"] = True
            refined_map[row["chunk_id"]] = row

        logger.info(
            "Completed context refinement batch %s/%s (%s rows)",
            batch_index,
            len(batches),
            len(batch),
        )

    # 입력 순서를 보존하면서 blocked row 도 그대로 끼워 반환.
    return [refined_map.get(row["chunk_id"], row) for row in translated_rows]


def _rewrite_translation_to_budget(
    row: dict[str, Any],
    *,
    source_language: str,
    target_language: str,
    budget: dict[str, Any],
    register_hint: str,
    api_key: str,
    base_url: str,
    endpoint: str,
    model_name: str,
    timeout_sec: int,
) -> tuple[str, str | None]:
    system_prompt, user_prompt = _build_budget_rewrite_messages(
        row,
        current_translation=str(row.get("text_translated", "") or ""),
        source_language=source_language,
        target_language=target_language,
        budget=budget,
        register_hint=register_hint,
    )
    raw_response = _translate_with_vectorengine_api(
        "",
        source_language=source_language,
        target_language=target_language,
        target_duration_sec=float(budget.get("duration_sec", 0.0) or 0.0),
        duration_budget=budget,
        register_hint=register_hint,
        api_key=api_key,
        base_url=base_url,
        endpoint=endpoint,
        model_name=model_name,
        timeout_sec=timeout_sec,
        response_format_json=_is_japanese_target(target_language),
        system_prompt_override=system_prompt,
        user_prompt_override=user_prompt,
    )
    return _parse_translation_response(raw_response, target_language=target_language)


def _apply_duration_budget_control(
    translated_rows: list[dict[str, Any]],
    *,
    source_language: str,
    target_language: str,
    enabled: bool,
    max_budget_rewrites: int,
    api_key: str,
    base_url: str,
    endpoint: str,
    model_name: str,
    timeout_sec: int,
    chunk_feature_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    # setup(예산/fit 계산)은 순차(LLM 없음·빠름). 과예산 행만 모아 rewrite LLM 체인을 병렬 실행.
    _over_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in translated_rows:
        chunk_id = str(row.get("chunk_id", "")).strip()
        chunk_feature = chunk_feature_map.get(chunk_id)
        budget = _compute_duration_budget(row, target_language=target_language, chunk_feature=chunk_feature)
        fit = _measure_budget_fit(str(row.get("text_translated", "") or ""), target_language=target_language, budget=budget)
        row["translation_budget"] = fit
        row["translation_style_hint"] = str(row.get("translation_style_hint") or detect_register(str(row.get("text_translated", "") or "")))
        row.setdefault("text_translated_context", row.get("text_translated", ""))
        row.setdefault("text_translated_initial", row.get("text_translated", ""))
        row["text_translated_budgeted"] = row.get("text_translated", "")
        row["translation_revision_count"] = int(row.get("translation_revision_count", 0) or 0)
        if enabled and not row.get("translation_blocked") and row.get("text_src") and row.get("text_translated") and not fit["within_budget"]:
            _over_rows.append((row, budget))

    def _rewrite_row(_rb: tuple[dict[str, Any], dict[str, Any]]) -> None:
        row, budget = _rb
        chunk_id = str(row.get("chunk_id", "")).strip()
        fit = row["translation_budget"]
        attempts = 0
        while attempts < max_budget_rewrites and not fit["within_budget"]:
            try:
                revised_translation, revised_tts = _rewrite_translation_to_budget(
                    row, source_language=source_language, target_language=target_language,
                    budget=fit, register_hint=str(row.get("translation_style_hint", "") or ""),
                    api_key=api_key, base_url=base_url, endpoint=endpoint,
                    model_name=model_name, timeout_sec=timeout_sec)
            except TranslationBlocked as exc:
                logger.warning("Budget rewrite for %s blocked by translator: %s", chunk_id, exc.reason)
                break
            attempts += 1
            if revised_translation:
                row["text_translated"] = revised_translation
                row["text_translated_budgeted"] = revised_translation
                if revised_tts:
                    row["text_tts"] = revised_tts
                row["translation_style_hint"] = detect_register(revised_translation)
                fit = _measure_budget_fit(str(row.get("text_translated", "") or ""), target_language=target_language, budget=budget)
                row["translation_budget"] = fit
            else:
                break
        row["translation_revision_count"] = int(row.get("translation_revision_count", 0) or 0) + attempts

    if _over_rows:
        import os as _os3
        from concurrent.futures import ThreadPoolExecutor as _TPE3
        _conc3 = max(1, int(_os3.environ.get("TRANSLATE_LLM_CONCURRENCY", "8")))
        if len(_over_rows) > 1 and _conc3 > 1:
            with _TPE3(max_workers=min(_conc3, len(_over_rows))) as _ex3:
                list(_ex3.map(_rewrite_row, _over_rows))
        else:
            for _rb in _over_rows:
                _rewrite_row(_rb)

        attach_stage_quality(
            row,
            "translation",
            assess_translation_row(
                row,
                target_language=target_language,
                budget=row.get("translation_budget"),
            ),
        )
    return translated_rows


_SCENE_CONTEXT_SYSTEM = (
    "You prepare a concise SCENE CONTEXT that will guide the dubbing translation of this "
    "video. From the dialogue transcript (speaker-tagged), write 2-4 sentences covering: "
    "(1) the setting/situation, (2) who the speakers are and their relationship, (3) the "
    "register/formality between them, (4) the overall tone. This guides translation tone, "
    "pronouns, honorifics and cross-line continuity. Output ONLY the scene context prose "
    "(no preamble, no bullet points), in English."
)


def generate_scene_context(
    asr_rows: list[dict[str, Any]],
    *,
    source_language: str,
    target_language: str,
    env_file: str | Path = ".env",
    timeout_sec: int = 60,
) -> str:
    """ASR 전사본 전체를 LLM 1회 요약 → scene_context 자동 생성(수동 힌트 대체).
    어떤 타겟언어든 무관(영어 컨텍스트가 톤/존댓말/대명사를 가이드). 실패 시 빈 문자열."""
    load_env_file(env_file)
    api_key = os.environ.get("VECTORENGINE_API_KEY", "").strip()
    if not api_key:
        logger.warning("auto scene_context: VECTORENGINE_API_KEY 없음 — skip")
        return ""
    base_url = os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai/").strip()
    model_name = os.environ.get("VECTORENGINE_MODEL", "gpt-5.4").strip()
    endpoint = os.environ.get("VECTORENGINE_ENDPOINT", "/v1/chat/completions").strip()
    transcript = "\n".join(
        f"[{r.get('speaker', '?')}] {(r.get('text_src') or '').strip()}"
        for r in asr_rows if (r.get('text_src') or '').strip()
    )[:6000]
    if not transcript.strip():
        return ""
    try:
        ctx = _translate_with_vectorengine_api(
            "", source_language=source_language, target_language=target_language,
            target_duration_sec=None, duration_budget=None, register_hint="",
            api_key=api_key, base_url=base_url, endpoint=endpoint, model_name=model_name,
            timeout_sec=timeout_sec, response_format_json=False,
            system_prompt_override=_SCENE_CONTEXT_SYSTEM,
            user_prompt_override="Transcript:\n" + transcript,
        )
        return (ctx or "").strip()
    except Exception as exc:
        logger.warning("auto scene_context 생성 실패: %s", exc)
        return ""


def build_translation_entries(
    asr_json: str | Path,
    output_json: str | Path,
    *,
    mode: str = "copy_source",
    source_language: str = "English",
    target_language: str = "Japanese",
    env_file: str | Path = ".env",
    timeout_sec: int = 60,
    context_refine: bool = True,
    context_batch_size: int = 12,
    duration_control: bool = True,
    max_budget_rewrites: int = 2,
    scene_context: str = "",
    register_override: str = "",
    auto_scene_context: bool = False,
) -> list[dict[str, Any]]:
    # 프롬프트 보강(씬 컨텍스트/격식)을 모듈 전역에 1회 설정 → 모든 번역/정제 프롬프트에 주입.
    global _SCENE_CONTEXT, _REGISTER_OVERRIDE
    _SCENE_CONTEXT = scene_context or ""
    _REGISTER_OVERRIDE = (register_override or "").strip().lower()
    asr_rows = load_json(asr_json)
    existing_rows = load_json_if_exists(output_json, default=[])
    existing_map = {row["chunk_id"]: row for row in existing_rows}
    from audio_features import load_chunk_feature_map  # numpy 의존 — 호출 시점에만 끌어옴
    chunk_feature_map = load_chunk_feature_map(asr_json)

    if mode == "copy_source":
        logger.warning(
            "Translation mode is copy_source; text_translated will be overwritten from text_src on every run."
        )

    api_key = ""
    base_url = ""
    endpoint = ""
    model_name = ""
    if mode == "vectorengine_gpt":
        load_env_file(env_file)
        api_key = os.environ.get("VECTORENGINE_API_KEY", "").strip()
        base_url = os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai/").strip()
        model_name = os.environ.get("VECTORENGINE_MODEL", "gpt-5.4").strip()
        endpoint = os.environ.get("VECTORENGINE_ENDPOINT", "/v1/chat/completions").strip()
        if not api_key:
            raise RuntimeError(f"VECTORENGINE_API_KEY is missing. Put it in {env_file}.")
        # 수동 scene_context 없고 auto 켜졌으면 ASR 전사본에서 자동 생성(완전 자동 문맥번역).
        if auto_scene_context and not _SCENE_CONTEXT:
            _auto_ctx = generate_scene_context(
                asr_rows, source_language=source_language, target_language=target_language,
                env_file=env_file, timeout_sec=timeout_sec,
            )
            if _auto_ctx:
                _SCENE_CONTEXT = _auto_ctx
                logger.info("auto scene_context 생성: %s", _auto_ctx[:200])

    translated_rows: list[dict[str, Any]] = []
    speaker_register_memory: dict[str, str] = {}
    reset_count = 0

    # 무손실 속도: Pass A 초기번역 LLM 호출을 동시 실행(청크 독립·temp=0.1·동일 프롬프트 → 동일 출력).
    # register_hint 의 speaker별 순차 누적만 제거(=_REGISTER_OVERRIDE 사용); 레지스터 일관성은
    # Pass B 문맥정제(full transcript)가 보장. _translate_with_vectorengine_api 는 stateless(호출별
    # urllib Request) → thread-safe. 동시성 env TRANSLATE_LLM_CONCURRENCY(기본 8); 동일 API 가
    # 3_dub_pipeline 에서 max_workers=10 으로 검증됨. 결과는 아래 순차 루프에서 _llm_pre 로 조회.
    _llm_pre: dict[str, tuple[str, Any]] = {}
    if mode == "vectorengine_gpt":
        def _pre_translate(_row):
            _existing = existing_map.get(_row["chunk_id"], {})
            _csrc = _normalize_text(_row.get("text_src", ""))
            if _can_preserve_existing_translation(_existing, current_src=_csrc, target_language=target_language):
                return None  # 기존 보존 — LLM 불필요
            _dur = None
            if _row.get("start") is not None and _row.get("end") is not None:
                _dur = max(0.0, float(_row["end"]) - float(_row["start"]))
            _bud = _compute_duration_budget(
                {"duration": _dur, "text_src": _row.get("text_src", "")},
                target_language=target_language, chunk_feature=chunk_feature_map.get(str(_row["chunk_id"])))
            try:
                _resp = _translate_with_vectorengine_api(
                    _row.get("text_src", ""), source_language=source_language, target_language=target_language,
                    target_duration_sec=_dur, duration_budget=_bud, register_hint=(_REGISTER_OVERRIDE or ""),
                    api_key=api_key, base_url=base_url, endpoint=endpoint, model_name=model_name, timeout_sec=timeout_sec)
                return (_row["chunk_id"], ("ok", _resp))
            except TranslationBlocked as _exc:
                return (_row["chunk_id"], ("blocked", _exc.reason))
        import os as _os
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _conc = max(1, int(_os.environ.get("TRANSLATE_LLM_CONCURRENCY", "8")))
        if len(asr_rows) > 1 and _conc > 1:
            with _TPE(max_workers=_conc) as _ex:
                for _r in _ex.map(_pre_translate, asr_rows):
                    if _r is not None:
                        _llm_pre[_r[0]] = _r[1]

    for row in asr_rows:
        existing = existing_map.get(row["chunk_id"], {})
        text_src = row.get("text_src", "")
        current_src = _normalize_text(text_src)
        existing_src = _normalize_text(existing.get("text_src", ""))
        duration = None
        if row.get("start") is not None and row.get("end") is not None:
            duration = max(0.0, float(row["end"]) - float(row["start"]))
        budget = _compute_duration_budget(
            {
                "duration": duration,
                "text_src": text_src,
            },
            target_language=target_language,
            chunk_feature=chunk_feature_map.get(str(row["chunk_id"])),
        )
        speaker = str(row.get("speaker") or "")
        register_hint = _REGISTER_OVERRIDE or speaker_register_memory.get(speaker, "")
        blocked_reason: str | None = None

        if mode == "copy_source":
            text_translated = text_src
            text_tts = None
            if existing.get("text_translated") and _normalize_text(existing.get("text_translated", "")) != current_src:
                reset_count += 1
        elif mode == "blank":
            preserve_existing = _can_preserve_existing_translation(
                existing,
                current_src=current_src,
                target_language=target_language,
            )
            if preserve_existing:
                text_translated = existing["text_translated"]
                text_tts = existing.get("text_tts")
            else:
                text_translated = ""
                text_tts = None
                if existing.get("text_translated") and existing_src != current_src:
                    reset_count += 1
        elif mode == "vectorengine_gpt":
            preserve_existing = _can_preserve_existing_translation(
                existing,
                current_src=current_src,
                target_language=target_language,
            )
            if preserve_existing:
                text_translated = existing["text_translated"]
                text_tts = existing.get("text_tts")
                blocked_reason = existing.get("translation_blocked_reason") if existing.get("translation_blocked") else None
            else:
                blocked_reason = None
                _pre = _llm_pre.get(row["chunk_id"])  # 병렬 pre-pass 결과(있으면 재호출 안 함)
                if _pre is not None and _pre[0] == "blocked":
                    logger.warning("Chunk %s blocked by translator: %s", row.get("chunk_id"), _pre[1])
                    text_translated = ""
                    text_tts = None
                    blocked_reason = _pre[1]
                else:
                    if _pre is not None:
                        raw_response = _pre[1]
                    else:
                        try:
                            raw_response = _translate_with_vectorengine_api(
                                text_src, source_language=source_language, target_language=target_language,
                                target_duration_sec=duration, duration_budget=budget, register_hint=register_hint,
                                api_key=api_key, base_url=base_url, endpoint=endpoint,
                                model_name=model_name, timeout_sec=timeout_sec,
                            )
                        except TranslationBlocked as exc:
                            logger.warning("Chunk %s blocked by translator: %s", row.get("chunk_id"), exc.reason)
                            text_translated = ""
                            text_tts = None
                            blocked_reason = exc.reason
                            raw_response = None
                    if blocked_reason is None:
                        text_translated, text_tts = _parse_translation_response(raw_response, target_language=target_language)
                        if existing.get("text_translated") and _normalize_text(existing.get("text_translated", "")) != _normalize_text(text_translated):
                            reset_count += 1
        else:
            raise ValueError(f"Unsupported translation mode: {mode}")

        detected_register = detect_register(text_translated)
        if speaker and detected_register in {"polite", "casual"}:
            speaker_register_memory[speaker] = detected_register

        translated_row = {
            "chunk_id": row["chunk_id"],
            "speaker": row.get("speaker"),
            "start": row.get("start"),
            "end": row.get("end"),
            "duration": duration,
            "text_src": text_src,
            "text_translated": text_translated,
            "text_translated_initial": text_translated,
            "text_translated_context": text_translated,
            "text_translated_budgeted": text_translated,
            "translation_budget": _measure_budget_fit(text_translated, target_language=target_language, budget=budget),
            "translation_context_refined": False,
            "translation_revision_count": 0,
            "translation_style_hint": speaker_register_memory.get(speaker, register_hint),
        }
        if blocked_reason:
            translated_row["translation_blocked"] = True
            translated_row["translation_blocked_reason"] = blocked_reason
        if text_tts:
            translated_row["text_tts"] = text_tts
        if row.get("source_audio_summary"):
            translated_row["source_audio_summary"] = row["source_audio_summary"]
        translated_row = attach_stage_quality(
            translated_row,
            "translation",
            assess_translation_row(
                translated_row,
                target_language=target_language,
                budget=translated_row.get("translation_budget"),
            ),
        )
        translated_rows.append(translated_row)
        if mode == "vectorengine_gpt":
            save_json(translated_rows, output_json)

    if mode == "vectorengine_gpt" and context_refine:
        translated_rows = _refine_translations_with_context(
            translated_rows,
            source_language=source_language,
            target_language=target_language,
            api_key=api_key,
            base_url=base_url,
            endpoint=endpoint,
            model_name=model_name,
            timeout_sec=timeout_sec,
            batch_size=context_batch_size,
        )
        for row in translated_rows:
            if "text_translated_context" not in row or not row["text_translated_context"]:
                row["text_translated_context"] = row.get("text_translated", "")

    translated_rows = _apply_duration_budget_control(
        translated_rows,
        source_language=source_language,
        target_language=target_language,
        enabled=(mode == "vectorengine_gpt" and duration_control),
        max_budget_rewrites=max(0, int(max_budget_rewrites)),
        api_key=api_key,
        base_url=base_url,
        endpoint=endpoint,
        model_name=model_name,
        timeout_sec=timeout_sec,
        chunk_feature_map=chunk_feature_map,
    )

    save_json(translated_rows, output_json)
    if reset_count:
        logger.warning(
            "Reset %s translated rows because current chunk text no longer matched the previous timeline.",
            reset_count,
        )
    logger.info("Prepared translation file with %s rows at %s", len(translated_rows), output_json)
    return translated_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or refresh translated.json from ASR output.")
    parser.add_argument("asr_json")
    parser.add_argument("output_json")
    parser.add_argument("--mode", default="copy_source", choices=["copy_source", "blank", "vectorengine_gpt"])
    parser.add_argument("--source-language", default="English")
    parser.add_argument("--target-language", default="Japanese")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--context-refine", dest="context_refine", action="store_true")
    parser.add_argument("--no-context-refine", dest="context_refine", action="store_false")
    parser.set_defaults(context_refine=True)
    parser.add_argument("--context-batch-size", type=int, default=12)
    parser.add_argument("--duration-control", dest="duration_control", action="store_true")
    parser.add_argument("--no-duration-control", dest="duration_control", action="store_false")
    parser.set_defaults(duration_control=True)
    parser.add_argument("--max-budget-rewrites", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_translation_entries(
        args.asr_json,
        args.output_json,
        mode=args.mode,
        source_language=args.source_language,
        target_language=args.target_language,
        env_file=args.env_file,
        timeout_sec=args.timeout_sec,
        context_refine=args.context_refine,
        context_batch_size=args.context_batch_size,
        duration_control=args.duration_control,
        max_budget_rewrites=args.max_budget_rewrites,
    )


if __name__ == "__main__":
    main()
