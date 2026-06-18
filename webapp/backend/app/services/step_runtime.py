# run config 에서 단계별 docker service + 핵심 도구/모델을 추출 — Progress 페이지에 그대로 노출
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from common import deep_get

from webapp.backend.app.models import StepName, StepRuntime


def _basename(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PurePosixPath(text.replace("\\", "/")).name


def _read_env_value(env_file: str | None, key: str) -> str:
    """env_file 에서 KEY=VALUE 한 줄을 찾아 값을 반환. process env 에 이미 있으면 그걸 우선."""
    direct = os.environ.get(key, "").strip()
    if direct:
        return direct
    if not env_file:
        return ""
    path = Path(env_file)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return ""
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def _llm_label(mode: str, env_file: str | None) -> str:
    """mode 가 vectorengine_* 계열이면 백엔드+모델명을 합쳐 라벨 반환."""
    if not mode:
        return "translate"
    if mode.startswith("vectorengine"):
        model = _read_env_value(env_file, "VECTORENGINE_MODEL")
        return f"vectorengine / {model}" if model else "vectorengine"
    return mode


def build_step_runtime(config: dict[str, Any]) -> dict[StepName, StepRuntime]:
    """config 의 models / translation.mode / tts.* 에서 단계별 service+tool 채움."""
    use_separator = bool(deep_get(config, ("pipeline", "use_separator"), True))
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    translation_mode = str(deep_get(config, ("translation", "mode"), "") or "").strip()
    translation_env = str(deep_get(config, ("translation", "env_file"), ".env") or ".env").strip()
    instruction_mode = str(deep_get(config, ("tts", "instruction", "mode"), "") or "").strip()
    instruction_env = str(deep_get(config, ("tts", "instruction", "env_file"), ".env") or ".env").strip()
    tts_engine = str(deep_get(config, ("tts", "engine"), "cosyvoice") or "cosyvoice").strip()

    asr_tool = _basename(models.get("asr")) or "asr"
    emotion_tool = _basename(models.get("emotion")) or "emotion2vec"
    diarize_tool = _basename(models.get("diarization")) or "diarizen"
    tts_model = _basename(models.get("tts"))
    ensemble_models = deep_get(config, ("audio", "subtractive_ensemble", "models"), []) or []
    if isinstance(ensemble_models, list) and ensemble_models:
        separator_tool = " + ".join(_basename(m) for m in ensemble_models if m)
    else:
        separator_tool = "subtractive_ensemble"
    vad_model = _basename(deep_get(config, ("redirect_nonspeech", "model_path"), "")) or "silero_vad"

    run_tts_tool = f"{tts_engine} / {tts_model}".strip(" /") if tts_model else tts_engine

    runtime: dict[StepName, StepRuntime] = {
        "extract_audio":             StepRuntime(service="controller", tool="ffmpeg"),
        "separate_audio":            StepRuntime(
            service="separator" if use_separator else "controller",
            tool=separator_tool if use_separator else "passthrough",
        ),
        "redirect_nonspeech":        StepRuntime(service="separator", tool=vad_model),
        "diarize":                   StepRuntime(service="diarizer", tool=diarize_tool),
        "rttm_to_json":              StepRuntime(service="controller", tool=None),
        "face_clustering":           StepRuntime(service="face", tool="insightface+lightasd"),
        "build_repair_inputs":       StepRuntime(service="controller", tool=asr_tool),
        "apply_preserved_repair":    StepRuntime(
            service="diarizer",
            tool="4way-fusion+repair" if bool(deep_get(config, ("preserved_repair", "enabled"), False)) else "skipped",
        ),
        "apply_gapfilled":           StepRuntime(service="controller", tool=None),
        "merge_chunks":              StepRuntime(service="controller", tool=None),
        "cut_chunks":                StepRuntime(service="controller", tool="ffmpeg"),
        "extract_emotion":           StepRuntime(service="speaker", tool=emotion_tool),
        "run_asr":                   StepRuntime(service="speaker", tool=asr_tool),
        "translate":                 StepRuntime(service="controller", tool=_llm_label(translation_mode, translation_env)),
        "build_timeline":            StepRuntime(service="controller", tool=None),
        "generate_tts_instructions": StepRuntime(service="controller", tool=_llm_label(instruction_mode, instruction_env) if instruction_mode else "instruction"),
        "run_tts":                   StepRuntime(service="tts-cosyvoice", tool=run_tts_tool),
        "validate_tts":              StepRuntime(service="speaker", tool=asr_tool),
        "compose_audio":             StepRuntime(service="controller", tool="ffmpeg"),
        "mux":                       StepRuntime(service="controller", tool="ffmpeg"),
    }
    return runtime
