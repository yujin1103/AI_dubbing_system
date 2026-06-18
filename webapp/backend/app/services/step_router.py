# 단계 → GPU 서비스 라우팅 — scripts/docker/run_pipeline.ps1 의 Get-RequiredServices 로직 포팅
from __future__ import annotations

from typing import Iterable

from webapp.backend.app.models import PIPELINE_STEPS, StepName

# 각 단계가 실행되어야 하는 docker compose 서비스
STEP_SERVICE: dict[StepName, str] = {
    "extract_audio": "controller",
    "separate_audio": "separator",  # use_separator=False 면 controller 로 fallback
    "redirect_nonspeech": "separator",  # silero-vad 가 separator 컨테이너에 설치되어 있음
    "diarize": "diarizer",
    "rttm_to_json": "controller",
    "face_clustering": "face",
    "build_repair_inputs": "controller",  # fusion diar+vocals+faces+풀영상ASR 를 repair run_dir 형식으로 조립(ASR 데몬 호출)
    "apply_preserved_repair": "diarizer",  # 8 repair patch 를 /opt/venv_diarizen subprocess 로 실행 → diarizer 컨테이너
    "apply_gapfilled": "controller",  # gapfilled 화자분리를 청크 입력(diarization_json)으로 export(파일 변환)
    "merge_chunks": "controller",
    "cut_chunks": "controller",
    "extract_emotion": "speaker",
    "run_asr": "speaker",
    "translate": "controller",
    "build_timeline": "controller",
    "generate_tts_instructions": "controller",
    "run_tts": "tts-cosyvoice",
    "validate_tts": "speaker",
    "compose_audio": "controller",
    "mux": "controller",
}


def steps_in_range(from_step: StepName, to_step: StepName) -> list[StepName]:
    """from_step ~ to_step 사이의 단계 목록 반환 (포함)."""
    if from_step not in PIPELINE_STEPS or to_step not in PIPELINE_STEPS:
        raise ValueError(f"Unknown step: from={from_step} to={to_step}")
    start = PIPELINE_STEPS.index(from_step)
    end = PIPELINE_STEPS.index(to_step)
    if end < start:
        raise ValueError("to_step must be after or equal to from_step")
    return list(PIPELINE_STEPS[start : end + 1])


def resolve_service(step: StepName, *, use_separator: bool) -> str:
    """단계별 실제 실행 서비스 — separate_audio 만 use_separator 에 따라 분기.

    PIPELINE_ALL_IN_SERVICE env 가 설정되면 모든 단계를 그 단일 서비스로 라우팅한다.
    (예: 검증된 dubbing_pipeline:full 이미지를 controller 로 재사용 — GPU 서비스 6개를
    새로 빌드하지 않고, 데몬+venv_lipsync 인프로세스가 다 있는 단일 풀-환경 컨테이너에서 실행.)"""
    import os
    all_in = os.environ.get("PIPELINE_ALL_IN_SERVICE")
    if all_in:
        return all_in
    if step == "separate_audio" and not use_separator:
        return "controller"
    return STEP_SERVICE[step]


def group_consecutive_by_service(
    steps: Iterable[StepName],
    *,
    use_separator: bool,
) -> list[tuple[str, list[StepName]]]:
    """연속된 같은-서비스 단계를 묶어 [(service, [steps])] 형태로 반환.

    예: ["extract_audio", "separate_audio", "diarize"] + use_separator=True
        → [("controller", ["extract_audio"]), ("separator", ["separate_audio"]), ("diarizer", ["diarize"])]
        같은 서비스가 연속이면 한 chunk 로 묶여 docker compose exec 한 번으로 처리.
    """
    groups: list[tuple[str, list[StepName]]] = []
    current_service: str | None = None
    current_steps: list[StepName] = []

    for step in steps:
        service = resolve_service(step, use_separator=use_separator)
        if service == current_service:
            current_steps.append(step)
        else:
            if current_service is not None:
                groups.append((current_service, current_steps))
            current_service = service
            current_steps = [step]

    if current_service is not None and current_steps:
        groups.append((current_service, current_steps))

    return groups
