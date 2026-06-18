# UI overrides 6개 knob 을 base config 에 화이트리스트 머지 → configs/runs/{run_id}.json 으로 저장
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from webapp.backend.app.models import RunOverrides

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/project"))
RUN_CONFIG_DIR = PROJECT_ROOT / "configs" / "runs"


def _load_base(base_config: str) -> dict:
    path = PROJECT_ROOT / base_config
    if not path.exists():
        raise FileNotFoundError(f"Base config not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _set_nested(target: dict, dotted: str, value) -> None:
    """'tts.style_priority' → target['tts']['style_priority']=value (없으면 dict 생성)."""
    parts = dotted.split(".")
    cur = target
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _scope_paths_to_run(paths: dict, run_id: str) -> dict:
    """런별 격리: per-run 산출물(meta/chunks/dub/output) 경로에 run_id 를 끼워 넣어
    같은 영상의 여러 런이 서로 덮어쓰지 않게 한다. audio(분리본)·asd_tracks(얼굴+LightASD 점수)는
    소스 영상만의 함수라 결정적·재계산 비싸 영상 단위 공유 유지(격리 제외).
    {input_stem}/{tts_engine} 플레이스홀더는 그대로 두고 run_id 만 리터럴로 삽입(나중에 expand)."""
    shared_video_level = {"asd_tracks_json"}
    scoped: dict = {}
    for key, val in paths.items():
        if not isinstance(val, str) or key in shared_video_level:
            scoped[key] = val
            continue
        if val.startswith("meta/{input_stem}/"):
            val = "meta/{input_stem}/" + run_id + "/" + val[len("meta/{input_stem}/"):]
        elif val.startswith("output/{input_stem}/"):
            val = "output/{input_stem}/" + run_id + "/" + val[len("output/{input_stem}/"):]
        elif val.startswith("chunks/{input_stem}"):
            val = "chunks/{input_stem}/" + run_id + val[len("chunks/{input_stem}"):]
        elif val.startswith("dub/{input_stem}"):
            val = "dub/{input_stem}/" + run_id + val[len("dub/{input_stem}"):]
        # audio/... 및 그 외는 공유(미변경)
        scoped[key] = val
    return scoped


# UI knob 만 허용 — 그 외 base config 필드는 절대 덮어쓰지 않는다.
# 값이 list 면 같은 override 를 여러 config 경로에 동시 적용한다(예: source_language 는
# 번역과 ASR 양쪽이 봐야 함 — asr.language 가 명시된 config(kdrama)는 translation.source_language
# 만 바꾸면 ASR 이 안 따라오므로 둘 다 세팅).
_OVERRIDE_MAP: dict[str, str | list[str]] = {
    "target_language": "translation.target_language",
    "source_language": ["translation.source_language", "asr.language"],
    "fit_to_duration": "tts.fit_to_duration",
    "duration_fit_max_tempo": "tts.duration_fit_max_tempo",
    "use_separator": "pipeline.use_separator",
    "skip_existing": "tts.skip_existing",
}


def build_run_config(
    *,
    run_id: str,
    base_config: str,
    overrides: RunOverrides,
    input_video: str,
) -> Path:
    """base config 사본 → overrides 적용 → input_video 셋 → configs/runs/{run_id}.json 으로 저장."""
    config = copy.deepcopy(_load_base(base_config))
    config["input_video"] = input_video
    config["run_id"] = run_id
    # 런별 격리 — per-run 산출물 경로에 run_id 삽입(같은 영상 여러 런이 서로 덮어쓰던 문제 해결).
    config["paths"] = _scope_paths_to_run(config.get("paths", {}), run_id)

    for field, value in overrides.model_dump(exclude_none=True).items():
        if field not in _OVERRIDE_MAP:
            continue
        # 빈 문자열 override 는 무시 — source_language="" = 자동감지(base config 값 유지)
        if isinstance(value, str) and not value.strip():
            continue
        targets = _OVERRIDE_MAP[field]
        for dotted in (targets if isinstance(targets, list) else [targets]):
            _set_nested(config, dotted, value)

    # UI 런은 단계별 실행 + 청크별 instruct/감정 편집이라 '단일 invocation' 전용 최적화를 끈다:
    #  - tts.pipelined: generate_tts_instructions 를 run_tts 에 융합해 chunk_instruction_edit/
    #    chunk_emotion_edit 편집 단계를 없애므로 UI 편집 워크플로와 충돌 → 강제 OFF.
    #  - tts.prewarm: UI 는 단계마다 별도 subprocess 라 translate 에서 띄운 모델 프리워밍이
    #    run_tts(다른 프로세스)로 이어지지 않음(무효 + GPU 낭비) → 강제 OFF.
    # ※ translate/instruct LLM 호출 병렬화(TRANSLATE_LLM_CONCURRENCY)는 step 내부라 자동 적용 →
    #   UI 도 그대로 빨라진다(여긴 끄지 않음). pipelined/prewarm 은 headless 배치 전용.
    if isinstance(config.get("tts"), dict):
        config["tts"]["pipelined"] = False
        config["tts"]["prewarm"] = False

    RUN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_CONFIG_DIR / f"{run_id}.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)
    return out_path
