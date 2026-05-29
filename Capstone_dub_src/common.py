from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')
_WHITESPACE_RE = re.compile(r"\s+")


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logging.getLogger(name)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path: str | Path) -> str:
    resolved = resolve_project_path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def ensure_dir(value: str | Path) -> Path:
    path = resolve_project_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(value: str | Path) -> Path:
    path = resolve_project_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: str | Path, dst: str | Path) -> None:
    ensure_parent(dst)
    shutil.copyfile(resolve_project_path(src), resolve_project_path(dst))


def load_json(path: str | Path) -> Any:
    with resolve_project_path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_json_if_exists(path: str | Path, default: Any = None) -> Any:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return default
    with resolved.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> Path | None:
    env_path = resolve_project_path(path)
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def deep_get(mapping: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def require_value(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    value = deep_get(mapping, keys)
    if value is None:
        dotted = ".".join(keys)
        raise KeyError(f"Missing required config value: {dotted}")
    return value


def sanitize_path_component(value: str, *, default: str = "item") -> str:
    text = str(value or "").strip()
    text = _INVALID_PATH_CHARS.sub("_", text)
    text = _WHITESPACE_RE.sub("_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip(" ._")
    return text or default


def _expand_config_string(value: str, placeholders: dict[str, str]) -> str:
    expanded = value
    for key, replacement in placeholders.items():
        expanded = expanded.replace(f"{{{key}}}", replacement)
    return expanded


def expand_config_templates(config: dict[str, Any]) -> dict[str, Any]:
    input_video = str(config.get("input_video", "") or "")
    input_path = Path(input_video)
    input_stem = sanitize_path_component(input_path.stem or "input", default="input")
    tts_engine = sanitize_path_component(str(deep_get(config, ("tts", "engine"), "tts") or "tts").lower(), default="tts")
    placeholders = {
        "input_stem": input_stem,
        "tts_engine": tts_engine,
    }

    input_value = config.get("input_video")
    if isinstance(input_value, str):
        config["input_video"] = _expand_config_string(input_value, placeholders)

    paths = config.get("paths")
    if isinstance(paths, dict):
        for key, value in list(paths.items()):
            if isinstance(value, str):
                paths[key] = _expand_config_string(value, placeholders)

    audio = config.get("audio")
    if isinstance(audio, dict):
        chunk_source = audio.get("chunk_source")
        if isinstance(chunk_source, str):
            audio["chunk_source"] = _expand_config_string(chunk_source, placeholders)

    return config


def load_config(config_path: str | Path | None = None, *, input_video: str | None = None) -> dict[str, Any]:
    target = resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    with target.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if input_video:
        data["input_video"] = input_video
    return expand_config_templates(data)


def normalize_signature_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_signature_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_normalize_signature_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_signature_value(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        return round(value, 6)
    return value


def build_dub_runtime_settings(
    *,
    engine: str,
    model_dir: str = "",
    model_id: str = "",
    config_path: str = "",
    codec_dir: str = "",
    tokenizer_dir: str = "",
    tokenizer_id: str = "",
    target_language: str = "",
    system_prompt: str = "",
    reference_mode: str = "",
    min_prompt_sec: float | None = None,
    fit_to_duration: bool | None = None,
    passthrough_source_audio: bool | None = None,
    use_cross_lingual: bool | None = None,
    speed: float | None = None,
    dtype: str = "",
    attn_implementation: str = "",
    heads_backend: str = "",
    max_new_tokens: int | None = None,
    low_memory: bool | None = None,
    num_step: int | None = None,
    guidance_scale: float | None = None,
    use_model_duration: bool | None = None,
    cap_risky_self_reference: bool | None = None,
    prompt_cap_max_sec: float | None = None,
    duration_fit_trim_overlong: bool | None = None,
    trim_silence: bool | None = None,
    silence_trim_threshold_dbfs: float | None = None,
    max_leading_silence_sec: float | None = None,
    max_trailing_silence_sec: float | None = None,
    style_priority: str = "",
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "engine": str(engine or "").strip().lower(),
    }
    optional_strings = {
        "model_dir": model_dir,
        "model_id": model_id,
        "config_path": config_path,
        "codec_dir": codec_dir,
        "tokenizer_dir": tokenizer_dir,
        "tokenizer_id": tokenizer_id,
        "target_language": target_language,
        "system_prompt": system_prompt,
        "reference_mode": reference_mode,
        "style_priority": style_priority,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "heads_backend": heads_backend,
    }
    for key, value in optional_strings.items():
        normalized = str(value or "").strip()
        if normalized:
            settings[key] = normalized
    if min_prompt_sec is not None:
        settings["min_prompt_sec"] = round(float(min_prompt_sec), 3)
    if fit_to_duration is not None:
        settings["fit_to_duration"] = bool(fit_to_duration)
    if passthrough_source_audio is not None:
        settings["passthrough_source_audio"] = bool(passthrough_source_audio)
    if use_cross_lingual is not None:
        settings["use_cross_lingual"] = bool(use_cross_lingual)
    if speed is not None:
        settings["speed"] = round(float(speed), 3)
    if max_new_tokens is not None:
        settings["max_new_tokens"] = int(max_new_tokens)
    if low_memory is not None:
        settings["low_memory"] = bool(low_memory)
    if num_step is not None:
        settings["num_step"] = int(num_step)
    if guidance_scale is not None:
        settings["guidance_scale"] = round(float(guidance_scale), 3)
    if use_model_duration is not None:
        settings["use_model_duration"] = bool(use_model_duration)
    if cap_risky_self_reference is not None:
        settings["cap_risky_self_reference"] = bool(cap_risky_self_reference)
    if prompt_cap_max_sec is not None:
        settings["prompt_cap_max_sec"] = round(float(prompt_cap_max_sec), 3)
    if duration_fit_trim_overlong is not None:
        settings["duration_fit_trim_overlong"] = bool(duration_fit_trim_overlong)
    if trim_silence is not None:
        settings["trim_silence"] = bool(trim_silence)
    if silence_trim_threshold_dbfs is not None:
        settings["silence_trim_threshold_dbfs"] = round(float(silence_trim_threshold_dbfs), 3)
    if max_leading_silence_sec is not None:
        settings["max_leading_silence_sec"] = round(float(max_leading_silence_sec), 3)
    if max_trailing_silence_sec is not None:
        settings["max_trailing_silence_sec"] = round(float(max_trailing_silence_sec), 3)
    return settings


def build_dub_input_signature(row: dict[str, Any], *, runtime: dict[str, Any] | None = None) -> str:
    payload = {
        "speaker": str(row.get("speaker", "")),
        "reference_mode_override": str(row.get("reference_mode_override", "")),
        "reference_chunk_id_override": str(row.get("reference_chunk_id_override", "")),
        "text_src": normalize_signature_text(str(row.get("text_src", ""))),
        "text_translated": normalize_signature_text(str(row.get("text_translated", ""))),
        "text_tts": normalize_signature_text(str(row.get("text_tts", ""))),
        "source_emotion": _normalize_signature_value(row.get("source_emotion", {}) or {}),
        "tts_instruct_text": normalize_signature_text(str(row.get("tts_instruct_text", ""))),
        "wav": str(row.get("wav", "")),
        "duration": round(float(row.get("duration", 0.0) or 0.0), 3),
        "runtime": _normalize_signature_value(runtime if runtime is not None else row.get("dub_runtime", {}) or {}),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: str | Path | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    logger = get_logger("subprocess")
    rendered = [str(part) for part in command]
    logger.info("Running command: %s", subprocess.list2cmdline(rendered))
    result = subprocess.run(
        rendered,
        cwd=str(resolve_project_path(cwd)) if cwd else None,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {rendered}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return result


def probe_duration_seconds(audio_path: str | Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            resolve_project_path(audio_path),
        ],
        capture_output=True,
    )
    return float(result.stdout.strip())


def format_chunk_id(index: int) -> str:
    return f"chunk_{index:04d}"


def ensure_project_layout(config: dict[str, Any]) -> None:
    for key in (
        "raw_audio",
        "dialogue_audio",
        "bgm_audio",
        "diarization_rttm",
        "diarization_json",
        "diarization_refiner_report",
        "chunk_overrides_json",
        "speaker_chunks_json",
        "emotion_json",
        "asr_json",
        "translated_json",
        "master_timeline_json",
        "final_dub_audio",
        "output_video",
    ):
        candidate = deep_get(config, ("paths", key))
        if candidate:
            ensure_parent(candidate)
    for key in ("chunks_dir", "dub_dir"):
        candidate = deep_get(config, ("paths", key))
        if candidate:
            ensure_dir(candidate)
    for folder in ("input", "audio", "meta", "chunks", "dub", "output", "third_party", "docs"):
        ensure_dir(folder)


def select_audio_path(config: dict[str, Any], preferred_key: str) -> Path:
    preferred = resolve_project_path(require_value(config, ("paths", preferred_key)))
    if preferred.exists():
        return preferred
    return resolve_project_path(require_value(config, ("paths", "raw_audio")))


def existing_files(paths: Iterable[str | Path]) -> list[Path]:
    results: list[Path] = []
    for value in paths:
        candidate = resolve_project_path(value)
        if candidate.exists():
            results.append(candidate)
    return results
