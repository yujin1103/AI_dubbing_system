from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import threading
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

# === CosyVoice 모델 캐시 + 프리워밍 ===========================================
# 모델 로드(~47s)는 run_tts 시작에서 직렬 수행되어 그 직전 네트워크 LLM
# (translate+instruct, GPU idle) 동안 GPU가 놀게 된다. prewarm_cosyvoice_model 을
# 백그라운드 스레드로 미리 호출하면 로드를 네트워크 LLM 시간 아래로 숨긴다.
# 캐시된 모델 객체를 재사용하므로 합성 결과는 byte-identical(무손실).
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_cosyvoice_model(model_dir: str | Path, cosyvoice_repo: str | Path | None):
    """CosyVoice 모델만 로드(캐시 미사용). prewarm 과 세션 초기화 양쪽에서 호출."""
    if not cosyvoice_repo:
        raise RuntimeError("CosyVoice repo path is required")
    _prepare_cosyvoice_imports(cosyvoice_repo)
    _validate_whisper_package()
    try:
        from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice3
    except ImportError as exc:
        raise RuntimeError(
            f"CosyVoice import failed: {exc}. "
            "The repo path is visible, but a required dependency is missing in the active environment."
        ) from exc
    import os as _os
    _md = str(resolve_project_path(model_dir))
    # AutoModel 은 cosyvoice2.yaml 을 먼저 감지해 CosyVoice2 로 오판(이 모델은 v2/v3 yaml 둘 다 보유).
    # cosyvoice_daemon 과 동일하게 cosyvoice3.yaml 있으면 CosyVoice3 강제(instruct2 감정 경로 보존).
    if _os.path.exists(_os.path.join(_md, "cosyvoice3.yaml")):
        return CosyVoice3(_md)
    return AutoModel(model_dir=_md)


def _get_cosyvoice_model(model_dir: str | Path, cosyvoice_repo: str | Path | None):
    """캐시된 CosyVoice 모델 반환(없으면 로드+캐시). thread-safe.

    prewarm 스레드가 로드 중이면 lock 에서 대기 → 중복 로드 없음.
    """
    key = str(resolve_project_path(model_dir))
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            import time as _t
            _t0 = _t.time()
            logger.info("Loading CosyVoice model (%s)...", key)
            model = _load_cosyvoice_model(model_dir, cosyvoice_repo)
            _MODEL_CACHE[key] = model
            logger.info("CosyVoice model loaded (%.1fs)", _t.time() - _t0)
        else:
            logger.info("Reusing prewarmed CosyVoice model (%s)", key)
        return model


def prewarm_cosyvoice_model(model_dir: str | Path, cosyvoice_repo: str | Path | None) -> None:
    """백그라운드 프리워밍 진입점 — translate/instruct(네트워크) 동안 GPU 로드 오버랩.

    실패해도 비치명적: run_tts 가 lazy 로드로 폴백. 결과 무손실.
    """
    try:
        _get_cosyvoice_model(model_dir, cosyvoice_repo)
        logger.info("CosyVoice prewarm complete")
    except Exception as exc:  # noqa: BLE001 — 프리워밍 실패는 lazy 로드로 폴백
        logger.warning("CosyVoice prewarm failed (non-fatal, lazy load will retry): %s", exc)


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
    f0_guard: bool
    f0_guard_attempts: int

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
        f0_guard: bool = False,
        f0_guard_attempts: int = 4,
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
            f0_guard=bool(f0_guard),
            f0_guard_attempts=int(f0_guard_attempts),
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
    """CosyVoice repo + Matcha-TTS 를 sys.path 에 추가. 설정 경로에 cosyvoice 패키지가 없으면
    알려진 위치(/opt/CosyVoice · 리포 루트 CosyVoice · third_party/CosyVoice)를 자동 탐색 —
    설정의 경로 하드코딩에 의존하지 않고 어느 환경에서도 cosyvoice 를 임포트 가능하게 한다."""
    def _resolve(p: str | Path) -> Path:
        ps = str(p)
        return Path(ps) if ps.startswith("/") else resolve_project_path(ps)

    candidates: list[Path] = []
    if repo_dir:
        candidates.append(_resolve(repo_dir))
    for extra in ("/opt/CosyVoice", "CosyVoice", "third_party/CosyVoice"):
        try:
            candidates.append(_resolve(extra))
        except Exception:
            continue
    # cosyvoice 패키지가 실제로 있는 첫 후보 선택(없으면 설정 경로 그대로 — 기존 동작 보존)
    repo_path = next((c for c in candidates if (c / "cosyvoice").is_dir()),
                     candidates[0] if candidates else None)
    if repo_path is None:
        return
    for candidate in (repo_path, repo_path / "third_party" / "Matcha-TTS"):
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


# 검증된 언어→중국어 출력지시 시드(테스트 완료, 정확성 보장). 미등록 언어는 LLM 으로 자동 생성·캐시.
# 这是 cache+fallback 이지 언어 화이트리스트가 아님 — 어떤 target_language 든 _llm_language_directive 로 자동 처리.
_LANG_DIRECTIVE_SEED: dict[str, str] = {
    "korean": "请用韩语说。", "ko": "请用韩语说。",
    "japanese": "请用日语说。", "ja": "请用日语说。", "jp": "请用日语说。",
    "english": "请用英语说。", "en": "请用英语说。",
    "spanish": "请用西班牙语说。", "es": "请用西班牙语说。",
    "french": "请用法语说。", "fr": "请用法语说。",
    "german": "请用德语说。", "de": "请用德语说。",
    "cantonese": "请用广东话说。", "yue": "请用广东话说。",
    "chinese": "", "zh": "", "": "",  # 중국어/미지정은 지시 없음(모델 기본)
}
_LANG_DIRECTIVE_CACHE: dict[str, str] = dict(_LANG_DIRECTIVE_SEED)
_LANG_DIRECTIVE_LOCK = threading.Lock()


def _llm_language_directive(target_language: str) -> str:
    """임의 target_language → CosyVoice3 중국어 출력지시(请用X语说。)를 LLM 으로 자동 생성.
    언어 하드코딩 테이블 없이 '모든 언어'를 지원하기 위함. 키 없음/실패 시 '' (모델 기본 언어)."""
    import os
    try:
        from common import load_env_file
        from translate_chunks import _translate_with_vectorengine_api  # lazy: 순환참조 회피
    except Exception:
        return ""
    try:
        load_env_file(os.environ.get("DUB_ENV_FILE", ".env"))
    except Exception:
        pass
    api_key = os.environ.get("VECTORENGINE_API_KEY", "").strip()
    if not api_key:
        return ""
    system_prompt = (
        "You map a language name to a Mandarin Chinese TTS directive. Output EXACTLY one line in the form "
        "请用X语说。 where X is the Chinese name of the target language "
        "(Japanese->请用日语说。, Italian->请用意大利语说。, Cantonese->请用广东话说。, Arabic->请用阿拉伯语说。). "
        "Output only that single line — no quotes, no explanation."
    )
    try:
        raw = _translate_with_vectorengine_api(
            "",
            source_language="English", target_language="Chinese",
            target_duration_sec=None, duration_budget=None, register_hint="",
            api_key=api_key,
            base_url=os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai/").strip(),
            endpoint=os.environ.get("VECTORENGINE_ENDPOINT", "/v1/chat/completions").strip(),
            model_name=os.environ.get("VECTORENGINE_MODEL", "gpt-5.4").strip(),
            timeout_sec=30, response_format_json=False,
            system_prompt_override=system_prompt,
            user_prompt_override=f"Target language: {target_language}",
        )
    except Exception:
        return ""
    line = raw.strip().splitlines()[0].strip() if (raw and raw.strip()) else ""
    # 안전 가드: 정확히 请用…说 형태만 수용(LLM 잡설/오류 방지)
    if line.startswith("请用") and "说" in line and len(line) <= 24:
        return line
    return ""


def _chinese_language_directive(target_language: str) -> str:
    """CosyVoice3 instruct2 출력 언어를 강제하는 중국어 자연어 지시(请用X语说。).

    CosyVoice3 의 언어/방언 제어는 중국어로 학습됨(README '请用广东话表达'). instruct 앞에
    이 지시를 끼우면 음성 LM 이 해당 언어 운율로 조건화된다(없으면 짧은 텍스트가 모델 기본
    언어로 샘). 검증 시드 캐시 우선, 미등록 언어는 LLM 으로 자동 확장 — 언어 하드코딩 없음.
    """
    normalized = (target_language or "").strip().lower()
    with _LANG_DIRECTIVE_LOCK:
        if normalized in _LANG_DIRECTIVE_CACHE:
            return _LANG_DIRECTIVE_CACHE[normalized]
    directive = _llm_language_directive(target_language)
    with _LANG_DIRECTIVE_LOCK:
        _LANG_DIRECTIVE_CACHE[normalized] = directive  # '' 도 캐시(런 내 반복 호출 방지)
    return directive


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
    except ImportError as exc:
        raise RuntimeError(
            f"soundfile import failed: {exc}. "
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
    # 모델 로드는 캐시 경유(프리워밍 스레드가 이미 로드했으면 즉시 재사용 → 무손실).
    cosyvoice = _get_cosyvoice_model(model_dir, cosyvoice_repo)
    return TtsSession(
        cosyvoice=cosyvoice,
        sf=sf,
        feature_map=feature_map,
        speaker_best_map=speaker_best_map,
        row_by_chunk_id=row_by_chunk_id,
    )


_HANGUL_SYL_RE = re.compile(r"[가-힣]")
# 다국어 감탄사 음절: 한글 음절 + 일본어 히라가나/가타카나 + 한자(CJK). 영어/숫자는 제외.
# (한국어 텍스트엔 kana/kanji 가 없으므로 한국어 동작은 기존과 byte-identical)
_INTERJECTION_SYL_RE = re.compile(r"[가-힣぀-ゟ゠-ヿ一-鿿]")
# 무발화 게이트 임계: 청크 voiced_ratio(=1-silence_ratio, 에너지기준)가 이 미만이면 무음 단편으로 보고
# 더빙 스킵. 실발화는 보통 voiced_ratio≥0.5, 무음 단편은 ≈0.0 이라 0.05 면 무음만 안전하게 제거.
_NONSPEECH_VOICED_MIN = 0.05


def _is_short_interjection(translated_text: str) -> bool:
    """단음절 감탄사("아"/"어", 일본어 "あ"/"え", 원문 "Oh"/"Ah") 판별. 음절 1개면 True.

    CosyVoice3 는 1음소(단모음) 텍스트를 안정적으로 합성 못해 중국어/영어로 샘
    (언어지시 请用韩语说/请用日语说 로도 해결 안 됨 — 물리적 한계). 이런 청크는 화자 원본
    오디오를 passthrough 하는 게 자연스럽다(감탄사는 언어 보편적). 2음절 이상은 TTS.
    다국어: 한글·kana·kanji 음절을 세며, 한국어 텍스트엔 kana/kanji 가 없어 무회귀.
    """
    return len(_INTERJECTION_SYL_RE.findall(translated_text or "")) == 1


# 반응 감탄사(filler) — 원문(text_src) 기준이라 타깃언어 무관. 무음이면 드롭 판정에 사용.
# (Yeah/Okay/No 등 '실제 응답'은 제외 → 합성 유지.) EN·KO·JA 흔한 filler.
_SOURCE_FILLER_WORDS = {
    "oh", "ohh", "ah", "ahh", "uh", "uhh", "um", "umm", "mm", "mmm", "hmm", "hm",
    "eh", "ehh", "ooh", "oof", "huh",
    "어", "아", "음", "오", "으", "에", "어어", "아아", "으음", "오오",
    "あ", "ああ", "あっ", "えっ", "うっ", "ええ", "おお", "ん",
}


def _is_source_filler(source_text: str) -> bool:
    """원문이 반응 감탄사 한 마디인지(언어 무관). 무발화 게이트의 '아/어 끝몰림' 판별에 사용."""
    t = re.sub(r"[^0-9A-Za-z가-힣ぁ-ヿ一-鿿]", "", (source_text or "").strip().lower())
    return bool(t) and t in _SOURCE_FILLER_WORDS


# F0-가이드 재합성: 교차언어 클로닝이 화자 피치를 못 지켜 음역이 바뀌는(예 남성→여성대) 문제를
# seed 다양화로 여러 번 합성 → 원본 화자 F0 에 가장 가까운 take 자동 선택. 언어 무관.
_F0_GUARD_SEEDS = (13, 42, 100, 7, 1, 2024, 77, 555)
_F0_DRIFT_OCTAVES = 0.5    # |log2(take/src)| 이 값 초과면 드리프트로 보고 재합성(트리거)
_F0_ACCEPT_OCTAVES = 0.3   # 이 값 이하 take 확보 시 조기종료(원본 음역에 충분히 근접)
_F0_SRC_MIN_HZ = 80.0      # 원본 F0 신뢰범위 — 밖이면 pyin 옥타브오류/도달불가로 보고 가드 스킵
_F0_SRC_MAX_HZ = 340.0


def _median_f0_hz(wav_path: str | Path) -> float | None:
    """voiced 프레임 median F0(Hz). 무성/짧음/실패면 None. (librosa.pyin)"""
    try:
        import numpy as _np
        import librosa as _lb
        y, _sr = _lb.load(str(wav_path), sr=16000, mono=True)
        if y is None or len(y) < int(16000 * 0.2):
            return None
        f0, _vf, _vp = _lb.pyin(y, fmin=65, fmax=500, sr=16000, frame_length=1024)
        v = f0[~_np.isnan(f0)]
        if len(v) < 3:
            return None
        return float(_np.median(v))
    except Exception:
        return None


def _synthesize_with_f0_guard(
    row: dict[str, Any],
    reference: "ChunkReference",
    *,
    config: TtsRuntimeConfig,
    session: TtsSession,
    temp_output_wav: Path,
) -> bool:
    """1차 합성 후 원본 화자 F0 대비 피치 드리프트를 측정, 크면 seed 를 바꿔 재합성하여
    원본 음역에 가장 가까운 take 를 채택. 비드리프트 청크는 1차 take 그대로(기존 동작 무변경)."""
    import math
    output_wav = resolve_project_path(row["dub_wav"])

    def _one_take() -> bool:
        # 매 take 마다 inputs 재생성(temp prompt audio 가 _run 의 finally 에서 소거되므로).
        inputs = _prepare_inference_inputs(row, reference, config=config, session=session)
        return _run_cosyvoice_inference(
            row, inputs, config=config, session=session, temp_output_wav=temp_output_wav
        )

    # take0: ambient RNG — guard 비대상(비드리프트) 청크는 이 결과 그대로라 기존과 동일.
    if not _one_take():
        return False

    src_ref = resolve_project_path(row.get("wav") or "")
    src_f0 = _median_f0_hz(src_ref) if (src_ref and src_ref.exists()) else None
    if src_f0 is None or not (_F0_SRC_MIN_HZ <= src_f0 <= _F0_SRC_MAX_HZ):
        return True  # 원본 무성/측정불가/극단(옥타브오류·도달불가) → 가드 스킵(take0 유지)
    take_f0 = _median_f0_hz(output_wav)
    if take_f0 is None:
        return True

    def _score(f: float | None) -> float:
        return abs(math.log2(f / src_f0)) if (f and f > 0) else 99.0

    best_score = _score(take_f0)
    if best_score <= _F0_DRIFT_OCTAVES:
        return True  # 이미 원본 음역에 충분히 가까움

    try:
        from cosyvoice.utils.common import set_all_random_seed
    except Exception:
        return True

    best_copy = output_wav.with_suffix(".f0best.wav")
    copy_file(output_wav, best_copy)
    logger.info(
        "F0 guard %s: src=%.0fHz take0=%.0fHz (drift %.2f oct) → 재합성 시도",
        row["chunk_id"], src_f0, take_f0, best_score,
    )
    attempts = max(1, min(int(config.f0_guard_attempts), len(_F0_GUARD_SEEDS)))
    for seed in _F0_GUARD_SEEDS[:attempts]:
        set_all_random_seed(int(seed))
        if not _one_take():
            continue
        sc = _score(_median_f0_hz(output_wav))
        if sc < best_score:
            best_score = sc
            copy_file(output_wav, best_copy)
        if best_score <= _F0_ACCEPT_OCTAVES:
            break
    copy_file(best_copy, output_wav)
    try:
        best_copy.unlink()
    except OSError:
        pass
    logger.info("F0 guard %s: 채택 take 거리 %.2f oct (src %.0fHz)", row["chunk_id"], best_score, src_f0)
    return True


def process_chunk(row: dict[str, Any], *, config: TtsRuntimeConfig, session: TtsSession) -> None:
    translated_text, _ = _resolve_tts_text(row)
    output_wav = resolve_project_path(row["dub_wav"])
    temp_output_wav = output_wav.with_suffix(".gen.tmp.wav")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    row["dub_runtime"] = config.runtime_settings

    if _should_skip_existing(row, output_wav, skip_existing=config.skip_existing):
        return
    source_text = (row.get("text_src") or "").strip()
    # ★무발화 게이트(전체): 분리 dialogue 가 사실상 무음(voiced_ratio<min)인 청크는 더빙을 만들지 않는다.
    # = 그 화자가 그 시점에 가청 발화를 안 함(화면상 입 닫힘). 합성하면 '입 닫았는데 말 나옴'(유령 발화),
    # passthrough 하면 faint 잔음('아/어 끝몰림')이 된다. 타깃언어·텍스트 무관(원본 오디오 기준) — ASR 이
    # 무음/잡음/분리잔여를 텍스트로 옮긴 모든 가짜 청크 제거. 실발화(voiced>=min)만 합성/passthrough.
    # ※energy 기반 silence 라 voiced=0 은 '가청 발화 없음'을 뜻함. test4 에서 실발화는 voiced≥0.7 로 명확히 갈림.
    _feat = session.feature_map.get(str(row.get("chunk_id", "") or "").strip(), {})
    _vr = _feat.get("voiced_ratio")
    if _vr is not None and _vr < _NONSPEECH_VOICED_MIN:
        if output_wav.exists():
            output_wav.unlink()
        row["dub_nonspeech_skip"] = True
        row.pop("dub_error", None)
        row.pop("dub_stale", None)
        logger.info("Non-speech gate: drop silent chunk %s (src=%r voiced=%.3f)", row["chunk_id"], source_text[:16], _vr)
        return
    # 실발화 1음절 감탄사("아"/"어"/"Oh")는 CosyVoice 합성 실패(중국어로 샘) → 화자 원본 passthrough.
    if _is_short_interjection(translated_text):
        src_wav = resolve_project_path(row.get("wav") or "")
        if src_wav.exists():
            if temp_output_wav.exists():
                temp_output_wav.unlink()
            copy_file(src_wav, output_wav)
            _finalize_chunk_success(row, output_wav, runtime_settings=config.runtime_settings)
            row["dub_passthrough_interjection"] = True
            logger.info("Passthrough source audio for short interjection %s (text=%r)", row["chunk_id"], translated_text)
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

    if config.f0_guard and config.use_cross_lingual:
        ok = _synthesize_with_f0_guard(row, reference, config=config, session=session, temp_output_wav=temp_output_wav)
    else:
        inputs = _prepare_inference_inputs(row, reference, config=config, session=session)
        ok = _run_cosyvoice_inference(row, inputs, config=config, session=session, temp_output_wav=temp_output_wav)
    if not ok:
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
            # cross-lingual(영어화자→한국어 등): instruct 에 중국어 언어지시('请用韩语说。')를
            # 프리픽스 뒤·감정절 앞에 끼워 출력 언어를 강제한다. 없으면 짧은 텍스트가
            # 모델 기본 언어(중국어/영어) 운율로 샘. (tts_text 한국어는 건드리지 않음)
            _instruct = inputs.instruct_text
            if config.use_cross_lingual:
                _lang_dir = _chinese_language_directive(config.target_language)
                if _lang_dir and _lang_dir not in _instruct:
                    _prefix = "You are a helpful assistant."
                    if _instruct.startswith(_prefix):
                        _instruct = f"{_prefix} {_lang_dir}{_instruct[len(_prefix):].lstrip()}"
                    else:
                        _instruct = f"{_lang_dir}{_instruct}"
            result_iterator = session.cosyvoice.inference_instruct2(
                inputs.translated_text,
                _instruct,
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
