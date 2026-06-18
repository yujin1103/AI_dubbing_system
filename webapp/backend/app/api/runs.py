# /api/runs — 생성 / 조회 / 취소
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from webapp.backend.app.models import (
    BulkPatchChunksRequest,
    BulkRedubChunksRequest,
    ChunkRow,
    CreateRunRequest,
    LogResponse,
    MetricsResponse,
    PatchChunkPayload,
    PreviewInstructionRequest,
    PreviewInstructionResponse,
    ReferenceCandidate,
    RunRecord,
    StepDetailRecord,
    StepName,
)
from webapp.backend.app.services import activity, artifacts, pipeline_runner, run_store
from webapp.backend.app.services.config_builder import build_run_config
from webapp.backend.app.services.step_runtime import build_step_runtime
from webapp.backend.app.models import ActivityEvent

router = APIRouter(prefix="/api/runs", tags=["runs"])

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/project"))


def _get_run_or_404(run_id: str) -> RunRecord:
    record = run_store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    record = _ensure_step_runtime(record)
    return _sync_artifact_state(record)


def _ensure_step_runtime(record: RunRecord) -> RunRecord:
    """기존 run 마이그레이션 — step_runtime 비어있으면 config 에서 추출해 한 번 채워 저장."""
    if record.step_runtime:
        return record
    try:
        config = artifacts.load_run_config(record)
    except (FileNotFoundError, ValueError, OSError):
        return record
    record.step_runtime = build_step_runtime(config)
    run_store.save(record)
    return record


def _sync_artifact_state(record: RunRecord) -> RunRecord:
    if record.status not in ("queued", "running"):
        return record
    busy = pipeline_runner.is_busy()
    all_pending = all(step.state == "pending" for step in record.steps)
    if busy == record.run_id or not all_pending:
        return record

    completed_steps = set(artifacts.infer_completed_steps(record))
    if completed_steps:
        for step in record.steps:
            if step.name in completed_steps and step.state == "pending":
                run_store.update_step(record.run_id, step.name, "done")
    output_video = artifacts.output_video_path(record)
    if output_video:
        run_store.set_output_video(record.run_id, output_video)
        if "mux" in completed_steps and record.status in ("queued", "running"):
            run_store.update_status(record.run_id, "success", clear_error=True)
    return run_store.get(record.run_id) or record


def _ensure_editable(record: RunRecord) -> None:
    busy = pipeline_runner.is_busy()
    if busy is not None:
        raise HTTPException(status_code=409, detail={"error": "another run is in progress", "current_run_id": busy})
    if record.status in ("queued", "running"):
        raise HTTPException(status_code=409, detail="run is active")


def _force_tts_skip_existing(record: RunRecord) -> None:
    """Redub 의 의도(이 청크만 재합성) 보장 — run config 의 tts.skip_existing 을 True 로 in-place set.
    이 값이 False 면 run_tts 가 dub_stale 무관하게 모든 청크를 재합성한다."""
    config_path = (PROJECT_ROOT / record.config_path).resolve()
    if not config_path.exists():
        return
    with config_path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    tts = config.get("tts")
    if isinstance(tts, dict) and tts.get("skip_existing") is True:
        return
    config.setdefault("tts", {})["skip_existing"] = True
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)


def _queue_steps(run_id: str, from_step: StepName, *, to_step: StepName = "mux") -> RunRecord:
    steps = artifacts.steps_in_range(from_step, to_step)
    run_store.reset_steps(run_id, steps)
    run_store.clear_output_video(run_id)
    run_store.update_status(run_id, "queued", clear_error=True)
    pipeline_runner.schedule_run(run_id, from_step=from_step, to_step=to_step)
    return _get_run_or_404(run_id)


@router.post("", response_model=RunRecord, status_code=201)
async def create_run(payload: CreateRunRequest) -> RunRecord:
    busy = pipeline_runner.is_busy()
    if busy is not None:
        raise HTTPException(
            status_code=409,
            detail={"error": "another run is in progress", "current_run_id": busy},
        )

    input_video_abs = (PROJECT_ROOT / payload.input_path).resolve()
    try:
        input_video_abs.relative_to((PROJECT_ROOT / "input").resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="input_path must be under input/") from exc
    if not input_video_abs.exists():
        raise HTTPException(status_code=404, detail=f"input not found: {payload.input_path}")

    input_stem = input_video_abs.stem
    record = run_store.create(
        input_video=payload.input_path,
        input_stem=input_stem,
        config_path="",  # 아래에서 채움
    )
    config_path = build_run_config(
        run_id=record.run_id,
        base_config=payload.base_config,
        overrides=payload.overrides,
        input_video=payload.input_path,
    )
    record.config_path = str(config_path.relative_to(PROJECT_ROOT))
    config = artifacts.load_run_config(record)
    record.step_runtime = build_step_runtime(config)
    run_store.save(record)

    activity.record(
        record.run_id,
        "run_created",
        note=f"input={payload.input_path} base={payload.base_config}",
        after={"from_step": payload.from_step, "to_step": payload.to_step},
    )
    pipeline_runner.schedule_run(
        record.run_id,
        from_step=payload.from_step,
        to_step=payload.to_step,
    )
    return record


@router.get("", response_model=list[RunRecord])
def list_runs() -> list[RunRecord]:
    return [_sync_artifact_state(_ensure_step_runtime(record)) for record in run_store.list_runs()]


@router.get("/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    return _get_run_or_404(run_id)


@router.get("/{run_id}/chunks", response_model=list[ChunkRow])
def list_chunks(run_id: str) -> list[ChunkRow]:
    return artifacts.list_chunks(_get_run_or_404(run_id))


@router.get("/{run_id}/chunks/{chunk_id}", response_model=ChunkRow)
def get_chunk(run_id: str, chunk_id: str) -> ChunkRow:
    row = artifacts.get_chunk(_get_run_or_404(run_id), chunk_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return row


@router.get("/{run_id}/speaker-reference-bank", response_model=dict[str, list[ReferenceCandidate]])
def get_speaker_reference_bank(run_id: str) -> dict[str, list[ReferenceCandidate]]:
    return artifacts.list_speaker_reference_bank(_get_run_or_404(run_id))


@router.post("/{run_id}/speaker-reference-bank/score", response_model=dict[str, list[ReferenceCandidate]])
def score_speaker_reference_bank(run_id: str) -> dict[str, list[ReferenceCandidate]]:
    """controller 에서 MOS 모델(wav2vec2 + best.pt)로 각 화자 레퍼런스 후보의 음성품질을 채점하고
    화자별 최고 MOS 후보에 추천 표시를 붙인다. 기존 후보·선택 로직은 불변(추천은 보조 신호).
    blocking ~30-60s(모델 로드 1.1GB + 후보 채점)."""
    record = _get_run_or_404(run_id)
    _ensure_editable(record)
    cmd = [
        "docker", "compose",
        "-p", pipeline_runner.COMPOSE_PROJECT,
        "-f", pipeline_runner.COMPOSE_FILE,
        "exec", "-T", "controller",
        "python", "src/score_reference_mos.py", "--config", record.config_path,
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=900, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="MOS scoring timed out") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown error")[-500:]
        raise HTTPException(status_code=500, detail=f"MOS scoring failed: {detail}")
    activity.record(run_id, "mos_scored", note=(proc.stdout or "")[-200:])
    return artifacts.list_speaker_reference_bank(record)


@router.patch("/{run_id}/chunks/{chunk_id}", response_model=ChunkRow)
def patch_chunk(run_id: str, chunk_id: str, payload: PatchChunkPayload) -> ChunkRow:
    record = _get_run_or_404(run_id)
    _ensure_editable(record)
    before = artifacts.get_chunk(record, chunk_id)
    try:
        row = artifacts.patch_chunk(record, chunk_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="chunk not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_store.clear_output_video(run_id)
    # 어떤 필드가 변경됐는지에 따라 다른 kind 로 audit — 한 PATCH 가 두세 필드 동시에 바꾸기도 함
    if payload.translated_text is not None:
        activity.record(
            run_id, "chunk_text_edit", chunk_id=chunk_id,
            before=before.translated_text if before else None, after=row.translated_text,
        )
    if payload.tts_instruct_text is not None:
        activity.record(
            run_id, "chunk_instruction_edit", chunk_id=chunk_id,
            before=before.tts_instruct_text if before else None, after=row.tts_instruct_text,
        )
    if payload.emotion_label is not None or payload.emotion_scores is not None:
        activity.record(
            run_id, "chunk_emotion_edit", chunk_id=chunk_id,
            before={"label": before.emotion if before else None, "scores": before.emotion_scores if before else None},
            after={"label": row.emotion, "scores": row.emotion_scores},
        )
    if payload.speaker is not None:
        activity.record(
            run_id, "chunk_speaker_edit", chunk_id=chunk_id,
            before=before.speaker if before else None, after=row.speaker,
        )
    if payload.reference_mode is not None or payload.reference_chunk_id is not None:
        activity.record(
            run_id, "chunk_reference_edit", chunk_id=chunk_id,
            before={
                "mode": before.reference_override_mode if before else None,
                "chunk_id": before.reference_override_chunk_id if before else None,
            },
            after={
                "mode": row.reference_override_mode,
                "chunk_id": row.reference_override_chunk_id,
            },
        )
    return row


@router.patch("/{run_id}/chunks", response_model=list[ChunkRow])
def patch_chunks(run_id: str, payload: BulkPatchChunksRequest) -> list[ChunkRow]:
    record = _get_run_or_404(run_id)
    _ensure_editable(record)
    if not payload.updates:
        return []
    before_rows = {row.chunk_id: row for row in artifacts.list_chunks(record)}
    try:
        rows = artifacts.patch_chunks(record, payload.updates)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="chunk not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_store.clear_output_video(run_id)
    after_rows = {row.chunk_id: row for row in rows}
    for update in payload.updates:
        before = before_rows.get(update.chunk_id)
        after = after_rows.get(update.chunk_id)
        if after is None:
            continue
        if update.translated_text is not None:
            activity.record(
                run_id, "chunk_text_edit", chunk_id=update.chunk_id,
                before=before.translated_text if before else None, after=after.translated_text,
            )
        if update.tts_instruct_text is not None:
            activity.record(
                run_id, "chunk_instruction_edit", chunk_id=update.chunk_id,
                before=before.tts_instruct_text if before else None, after=after.tts_instruct_text,
            )
        if update.emotion_label is not None or update.emotion_scores is not None:
            activity.record(
                run_id, "chunk_emotion_edit", chunk_id=update.chunk_id,
                before={"label": before.emotion if before else None, "scores": before.emotion_scores if before else None},
                after={"label": after.emotion, "scores": after.emotion_scores},
            )
        if update.speaker is not None:
            activity.record(
                run_id, "chunk_speaker_edit", chunk_id=update.chunk_id,
                before=before.speaker if before else None, after=after.speaker,
            )
        if update.reference_mode is not None or update.reference_chunk_id is not None:
            activity.record(
                run_id, "chunk_reference_edit", chunk_id=update.chunk_id,
                before={
                    "mode": before.reference_override_mode if before else None,
                    "chunk_id": before.reference_override_chunk_id if before else None,
                },
                after={
                    "mode": after.reference_override_mode,
                    "chunk_id": after.reference_override_chunk_id,
                },
            )
    return rows


@router.post("/{run_id}/chunks/{chunk_id}/preview-instruction", response_model=PreviewInstructionResponse)
def preview_chunk_instruction(run_id: str, chunk_id: str, payload: PreviewInstructionRequest) -> PreviewInstructionResponse:
    """저장 없이 LLM 으로 instruction 미리보기. emotion 을 in-memory 로 overlay 한 row 로 1회 호출."""
    record = _get_run_or_404(run_id)
    row = artifacts.get_raw_chunk_row(record, chunk_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chunk not found")

    # 사용자가 EQ 편집 중인 미저장 값을 받았다면 source_emotion 을 그 값으로 overlay (in-memory only)
    if payload.emotion_label is not None or payload.emotion_scores is not None:
        existing = row.get("source_emotion") if isinstance(row.get("source_emotion"), dict) else {}
        new_emotion = dict(existing)
        if payload.emotion_scores is not None:
            clean = {str(k): max(0.0, min(1.0, float(v))) for k, v in payload.emotion_scores.items()}
            new_emotion["scores"] = clean
            if payload.emotion_label is None and clean:
                new_emotion["label"] = max(clean.items(), key=lambda kv: kv[1])[0]
        if payload.emotion_label is not None:
            new_emotion["label"] = str(payload.emotion_label)
        row["source_emotion"] = new_emotion

    config = artifacts.load_run_config(record)
    env_file = (
        ((config.get("tts") or {}).get("instruction") or {}).get("env_file")
        or (config.get("translation") or {}).get("env_file")
        or ".env"
    )

    # prev/next 대사 context 를 LLM 에 같이 넘기려면 master_timeline 전체 row 가 필요
    master_path = artifacts._config_value(config, "paths.master_timeline_json")
    all_rows = artifacts._read_json_list(master_path)

    from generate_tts_instructions import preview_instruction_for_row
    instruction, source = preview_instruction_for_row(
        row,
        all_rows=all_rows,
        env_file=env_file,
        timeout_sec=20,
    )
    return PreviewInstructionResponse(instruction=instruction, source=source)


@router.post("/{run_id}/chunks/{chunk_id}/redub", response_model=RunRecord)
async def redub_chunk(run_id: str, chunk_id: str) -> RunRecord:
    record = _get_run_or_404(run_id)
    _ensure_editable(record)
    _force_tts_skip_existing(record)
    try:
        artifacts.mark_chunk_stale(record, chunk_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="chunk not found") from exc
    activity.record(run_id, "chunk_redub", chunk_id=chunk_id)
    # gen_tts_instructions ~ mux 까지 자동 재실행 — 청크 dub 뿐 아니라 최종 합성/영상까지 갱신해
    # "청크는 반영됐는데 최종 영상은 stale" 한 부분반영 문제를 없앤다(단일·다중 리더빙 모두).
    _queue_steps(run_id, "generate_tts_instructions", to_step="mux")
    return _get_run_or_404(run_id)


@router.post("/{run_id}/chunks/redub", response_model=RunRecord)
async def redub_chunks(run_id: str, payload: BulkRedubChunksRequest) -> RunRecord:
    record = _get_run_or_404(run_id)
    _ensure_editable(record)
    if not payload.chunk_ids:
        raise HTTPException(status_code=400, detail="chunk_ids required")
    chunk_ids = list(dict.fromkeys(payload.chunk_ids))
    _force_tts_skip_existing(record)
    try:
        artifacts.mark_chunks_stale(record, chunk_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="chunk not found") from exc
    for chunk_id in chunk_ids:
        activity.record(run_id, "chunk_redub", chunk_id=chunk_id)
    # gen_tts_instructions ~ mux 까지 자동 재실행 — 청크 dub 뿐 아니라 최종 합성/영상까지 갱신해
    # "청크는 반영됐는데 최종 영상은 stale" 한 부분반영 문제를 없앤다(단일·다중 리더빙 모두).
    _queue_steps(run_id, "generate_tts_instructions", to_step="mux")
    return _get_run_or_404(run_id)


@router.get("/{run_id}/steps/{step}", response_model=StepDetailRecord)
def get_step_detail(run_id: str, step: StepName) -> StepDetailRecord:
    return artifacts.get_step_detail(_get_run_or_404(run_id), step)


@router.post("/{run_id}/steps/{step}/rerun", response_model=RunRecord)
async def rerun_step(run_id: str, step: StepName) -> RunRecord:
    record = _get_run_or_404(run_id)
    _ensure_editable(record)
    activity.record(run_id, "step_rerun", step=step)
    return _queue_steps(run_id, step)


@router.get("/{run_id}/log", response_model=LogResponse)
def get_log(run_id: str, limit: int = Query(default=500, ge=1, le=2000)) -> LogResponse:
    return LogResponse(lines=artifacts.read_log_lines(_get_run_or_404(run_id), limit=limit))


@router.get("/{run_id}/metrics", response_model=MetricsResponse)
def get_metrics(run_id: str) -> MetricsResponse:
    return artifacts.get_metrics(_get_run_or_404(run_id))


@router.get("/{run_id}/activity", response_model=list[ActivityEvent])
def get_activity(run_id: str, limit: int = Query(default=500, ge=1, le=2000)) -> list[ActivityEvent]:
    _get_run_or_404(run_id)
    return activity.list_events(run_id, limit=limit)


@router.post("/{run_id}/resume", response_model=RunRecord)
async def resume_run(run_id: str) -> RunRecord:
    """첫 done 이 아닌 단계부터 mux 까지 다시 큐잉. failed run 을 그 지점부터 이어 돌릴 때 사용."""
    record = _get_run_or_404(run_id)
    _ensure_editable(record)
    resume_from: StepName | None = None
    for step in record.steps:
        if step.state != "done":
            resume_from = step.name
            break
    if resume_from is None:
        raise HTTPException(status_code=400, detail="nothing to resume — all steps already done")
    activity.record(run_id, "run_resumed", step=resume_from)
    return _queue_steps(run_id, resume_from)


@router.post("/{run_id}/cancel", response_model=RunRecord)
async def cancel_run(run_id: str) -> RunRecord:
    record = _get_run_or_404(run_id)
    activity.record(run_id, "run_canceled")
    ok = await pipeline_runner.cancel(run_id)
    if not ok:
        if record.status not in ("queued", "running"):
            raise HTTPException(status_code=409, detail="run not active")
        run_store.skip_running_steps(run_id)
        run_store.clear_output_video(run_id)
        run_store.update_status(run_id, "canceled")
    else:
        run_store.clear_output_video(run_id)
    return run_store.get(run_id) or record
