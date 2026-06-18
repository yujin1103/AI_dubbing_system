# Run 산출물 JSON과 파일 메타데이터를 REST 응답 모델로 변환하는 서비스
from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Any

from common import deep_get, load_config
from quality_gate import assess_reference_candidate

from webapp.backend.app.models import (
    PIPELINE_STEPS,
    ChunkRow,
    MetricsResponse,
    PatchChunkPayload,
    PatchChunkUpdate,
    ReferenceCandidate,
    RunRecord,
    StepArtifact,
    StepDetailRecord,
    StepName,
)
from webapp.backend.app.services.step_router import STEP_SERVICE, steps_in_range

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[4])).resolve()

_PATH_KEYS: dict[StepName, tuple[list[tuple[str, str]], list[tuple[str, str]]]] = {
    "extract_audio": ([('input video', 'input_video')], [('raw audio', 'paths.raw_audio')]),
    "separate_audio": ([('raw audio', 'paths.raw_audio')], [('dialogue audio', 'paths.dialogue_audio'), ('background audio', 'paths.bgm_audio')]),
    "redirect_nonspeech": ([("dialogue audio", "paths.dialogue_audio"), ("background audio", "paths.bgm_audio")], [("background audio", "paths.bgm_audio")]),
    "diarize": ([('dialogue audio', 'paths.dialogue_audio')], [('diarization RTTM', 'paths.diarization_rttm')]),
    "rttm_to_json": ([('diarization RTTM', 'paths.diarization_rttm')], [('diarization JSON', 'paths.diarization_json')]),
    "merge_chunks": ([('diarization JSON', 'paths.diarization_json')], [('speaker chunks', 'paths.speaker_chunks_json')]),
    "cut_chunks": ([('speaker chunks', 'paths.speaker_chunks_json'), ('chunk source audio', 'audio.chunk_source')], [('chunk directory', 'paths.chunks_dir')]),
    "extract_emotion": ([('speaker chunks', 'paths.speaker_chunks_json'), ('chunk directory', 'paths.chunks_dir')], [('emotion JSON', 'paths.emotion_json')]),
    "run_asr": ([('speaker chunks', 'paths.speaker_chunks_json'), ('chunk directory', 'paths.chunks_dir')], [('ASR JSON', 'paths.asr_json')]),
    "translate": ([('ASR JSON', 'paths.asr_json')], [('translated JSON', 'paths.translated_json')]),
    "build_timeline": ([('speaker chunks', 'paths.speaker_chunks_json'), ('ASR JSON', 'paths.asr_json'), ('translated JSON', 'paths.translated_json'), ('emotion JSON', 'paths.emotion_json')], [('master timeline', 'paths.master_timeline_json')]),
    "generate_tts_instructions": ([('master timeline', 'paths.master_timeline_json')], [('master timeline', 'paths.master_timeline_json')]),
    "run_tts": ([('master timeline', 'paths.master_timeline_json')], [('dub directory', 'paths.dub_dir'), ('master timeline', 'paths.master_timeline_json')]),
    "validate_tts": ([('master timeline', 'paths.master_timeline_json'), ('dub directory', 'paths.dub_dir')], [('TTS validation JSON', 'paths.tts_validation_json'), ('master timeline', 'paths.master_timeline_json')]),
    "compose_audio": ([('master timeline', 'paths.master_timeline_json'), ('dub directory', 'paths.dub_dir'), ('background audio', 'paths.bgm_audio')], [('final dub audio', 'paths.final_dub_audio')]),
    "mux": ([('input video', 'input_video'), ('final dub audio', 'paths.final_dub_audio')], [('output video', 'paths.output_video')]),
}


def load_run_config(record: RunRecord) -> dict[str, Any]:
    return load_config(record.config_path)


def output_video_path(record: RunRecord) -> str | None:
    config = load_run_config(record)
    value = _config_value(config, "paths.output_video")
    if not value:
        return None
    path = _resolve_project_path(value)
    if not path.exists():
        return None
    return _project_relative(path)


def list_chunks(record: RunRecord) -> list[ChunkRow]:
    config = load_run_config(record)
    master_path = _config_value(config, "paths.master_timeline_json")
    master_rows = _read_json_list(master_path)
    if master_rows:
        return [_chunk_from_timeline(row) for row in master_rows if row.get("chunk_id")]

    chunk_rows = _read_json_list(_config_value(config, "paths.speaker_chunks_json"))
    if not chunk_rows:
        return []
    asr_rows = _by_chunk_id(_read_json_list(_config_value(config, "paths.asr_json")))
    translated_rows = _by_chunk_id(_read_json_list(_config_value(config, "paths.translated_json")))
    emotion_rows = _by_chunk_id(_read_json_list(_config_value(config, "paths.emotion_json")))

    rows: list[ChunkRow] = []
    for chunk in chunk_rows:
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id:
            continue
        asr = asr_rows.get(chunk_id, {})
        translated = translated_rows.get(chunk_id, {})
        emotion = emotion_rows.get(chunk_id, {})
        source_emotion = emotion.get("source_emotion")
        rows.append(ChunkRow(
            chunk_id=chunk_id,
            speaker=_string_or_none(chunk.get("speaker")),
            start=_float_or_none(chunk.get("start")),
            end=_float_or_none(chunk.get("end")),
            emotion=_emotion_label(source_emotion) or _string_or_none(emotion.get("emotion")),
            source_text=_string_or_none(asr.get("text_src") or translated.get("text_src")),
            translated_text=_string_or_none(translated.get("text_translated")),
            original_audio=_string_or_none(chunk.get("wav")),
            dubbed_audio=None,
            status="queued",
            duration_original=_duration_original(chunk),
            emotion_scores=_emotion_scores(source_emotion),
        ))
    return rows


def list_speaker_reference_bank(record: RunRecord) -> dict[str, list[ReferenceCandidate]]:
    config = load_run_config(record)
    master_path = _config_value(config, "paths.master_timeline_json")
    rows = _read_json_list(master_path)
    if not rows:
        return {}
    min_prompt_sec = float(deep_get(config, ("tts", "min_prompt_sec"), 1.2) or 1.2)
    # MOS 추천(additive): reference_mos.json 이 있으면 후보별 MOS 를 붙이고 화자별 최고 MOS 에 추천 표시.
    # 기존 후보·정렬·선택 로직은 그대로 둔다(MOS 는 보조 신호일 뿐).
    mos_raw = _read_json_object(_reference_mos_path_value(config))
    mos_map: dict[str, float] = {}
    for k, v in (mos_raw or {}).items():
        try:
            mos_map[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    grouped: dict[str, list[ReferenceCandidate]] = {}
    for row in rows:
        candidate = _reference_candidate_from_row(row, min_prompt_sec=min_prompt_sec)
        if candidate is None:
            continue
        candidate.mos = mos_map.get(candidate.chunk_id)
        speaker = candidate.speaker or ""
        if not speaker:
            continue
        grouped.setdefault(speaker, []).append(candidate)
    for speaker, candidates in grouped.items():
        grouped[speaker] = sorted(
            candidates,
            key=lambda item: (
                1 if item.accepted else 0,
                float(item.score or 0.0),
                float(item.duration or 0.0),
            ),
            reverse=True,
        )
        scored = [c for c in grouped[speaker] if c.mos is not None]
        if scored:
            max(scored, key=lambda c: float(c.mos or 0.0)).mos_recommended = True
    return grouped


def get_chunk(record: RunRecord, chunk_id: str) -> ChunkRow | None:
    return next((row for row in list_chunks(record) if row.chunk_id == chunk_id), None)


def get_raw_chunk_row(record: RunRecord, chunk_id: str) -> dict[str, Any] | None:
    """master_timeline 의 raw dict row — preview LLM 호출 등 src/ 함수로 그대로 넘기기 위함."""
    config = load_run_config(record)
    master_path = _config_value(config, "paths.master_timeline_json")
    rows = _read_json_list(master_path)
    for row in rows:
        if str(row.get("chunk_id", "")) == chunk_id:
            return row
    return None


def infer_completed_steps(record: RunRecord) -> list[StepName]:
    config = load_run_config(record)
    output_video = _config_value(config, "paths.output_video")
    if output_video and _path_exists(output_video):
        return list(PIPELINE_STEPS)

    completed: list[StepName] = []
    for step in PIPELINE_STEPS:
        output_specs = _PATH_KEYS[step][1]
        if output_specs and all(_artifact(config, label, ref, "output").exists for label, ref in output_specs):
            completed.append(step)
    return completed


def get_step_detail(record: RunRecord, step: StepName) -> StepDetailRecord:
    config = load_run_config(record)
    step_record = next((entry for entry in record.steps if entry.name == step), None)
    input_specs, output_specs = _PATH_KEYS[step]
    inputs = [_artifact(config, label, ref, "input") for label, ref in input_specs]
    outputs = [_artifact(config, label, ref, "output") for label, ref in output_specs]
    return StepDetailRecord(
        step=step,
        status=step_record.state if step_record else None,
        service=STEP_SERVICE[step],
        inputs=[item.path for item in inputs],
        outputs=[item.path for item in outputs],
        artifacts=inputs + outputs,
        log_excerpt=read_log_lines(record, limit=80),
        error=step_record.error if step_record else None,
    )


def read_log_lines(record: RunRecord, *, limit: int = 500) -> list[str]:
    path = Path(record.log_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max(1, min(limit, 2000)):]


def get_metrics(record: RunRecord) -> MetricsResponse:
    rows = list_chunks(record)
    ratios: list[float] = []
    emotions: dict[str, int] = {}
    ready = stale = errors = validation_failures = 0
    for row in rows:
        if row.status == "done":
            ready += 1
        if row.dub_stale or row.status == "stale":
            stale += 1
        if row.status == "error" or row.error:
            errors += 1
        if row.error and "validation" in row.error.lower():
            validation_failures += 1
        if row.duration_original and row.duration_dub and row.duration_original > 0:
            ratios.append(row.duration_dub / row.duration_original)
        if row.emotion:
            emotions[row.emotion] = emotions.get(row.emotion, 0) + 1
    return MetricsResponse(
        total_chunks=len(rows),
        ready_chunks=ready,
        stale_chunks=stale,
        error_chunks=errors,
        average_duration_ratio=round(sum(ratios) / len(ratios), 3) if ratios else None,
        validation_failures=validation_failures,
        emotion_distribution=emotions,
    )


def patch_chunk(record: RunRecord, chunk_id: str, payload: PatchChunkPayload) -> ChunkRow:
    update = PatchChunkUpdate(chunk_id=chunk_id, **payload.dict())
    return patch_chunks(record, [update])[0]


def patch_chunks(record: RunRecord, updates: list[PatchChunkUpdate]) -> list[ChunkRow]:
    if not updates:
        return []
    if any(update.emotion is not None for update in updates):
        raise ValueError("emotion(legacy) editing is not supported")

    chunk_ids = [update.chunk_id for update in updates]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("duplicate chunk_id")

    existing_ids = {row.chunk_id for row in list_chunks(record)}
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in existing_ids]
    if missing:
        raise KeyError(missing[0])

    config = load_run_config(record)
    master_path = _config_value(config, "paths.master_timeline_json")
    master_rows = _read_json_list(master_path)
    overrides_path = _chunk_overrides_path_value(config)
    chunk_overrides = _read_json_object(overrides_path)
    translated_path = _config_value(config, "paths.translated_json")
    translated_rows = _read_json_list(translated_path)
    master_changed = False
    overrides_changed = False
    translated_changed = False

    for update in updates:
        nothing_to_do = (
            update.speaker is None
            and update.reference_mode is None
            and update.reference_chunk_id is None
            and update.translated_text is None
            and update.tts_instruct_text is None
            and update.emotion_label is None
            and update.emotion_scores is None
        )
        if nothing_to_do:
            continue

        changed = False
        if update.translated_text is not None:
            if translated_rows and _patch_translation_rows(translated_rows, update.chunk_id, update.translated_text, stale=False):
                translated_changed = True
                changed = True
            if master_rows and _patch_translation_rows(master_rows, update.chunk_id, update.translated_text, stale=True):
                master_changed = True
                changed = True

        if update.speaker is not None or update.reference_mode is not None or update.reference_chunk_id is not None:
            if not master_rows:
                raise KeyError(update.chunk_id)
            if _patch_chunk_override(
                chunk_overrides,
                update.chunk_id,
                speaker=update.speaker,
                reference_mode=update.reference_mode,
                reference_chunk_id=update.reference_chunk_id,
            ):
                overrides_changed = True
            if _patch_master_for_speaker_reference(
                master_rows,
                update.chunk_id,
                speaker=update.speaker,
                reference_mode=update.reference_mode,
                reference_chunk_id=update.reference_chunk_id,
            ):
                master_changed = True
                changed = True

        if update.tts_instruct_text is not None or update.emotion_label is not None or update.emotion_scores is not None:
            if not master_rows:
                raise KeyError(update.chunk_id)
            if _patch_master_for_advanced_fields(
                master_rows,
                update.chunk_id,
                tts_instruct_text=update.tts_instruct_text,
                emotion_label=update.emotion_label,
                emotion_scores=update.emotion_scores,
            ):
                master_changed = True
                changed = True

        if not changed:
            raise KeyError(update.chunk_id)

    if translated_changed:
        _write_json(translated_path, translated_rows)
    if overrides_changed:
        _write_json(overrides_path, chunk_overrides)
    if master_changed:
        _write_json(master_path, master_rows)

    updated = {row.chunk_id: row for row in list_chunks(record)}
    return [updated[chunk_id] for chunk_id in chunk_ids]


def _chunk_overrides_path_value(config: dict[str, Any]) -> str:
    configured = _config_value(config, "paths.chunk_overrides_json")
    if configured:
        return configured
    master_path = _config_value(config, "paths.master_timeline_json")
    if master_path:
        return _project_relative(_resolve_project_path(master_path).with_name("chunk_overrides.json"))
    return "meta/input/chunk_overrides.json"


def _reference_mos_path_value(config: dict[str, Any]) -> str | None:
    """스피커 뱅크 MOS 채점 결과(reference_mos.json) — master_timeline 과 같은 디렉토리."""
    master_path = _config_value(config, "paths.master_timeline_json")
    if master_path:
        return _project_relative(_resolve_project_path(master_path).with_name("reference_mos.json"))
    return None


def _read_json_object(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = _resolve_project_path(path_value)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    return data if isinstance(data, dict) else {}


def _normalize_reference_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto", "default"}:
        return "auto"
    if normalized in {"self", "self_only", "force_self"}:
        return "self"
    if normalized in {"speaker_bank", "speaker_best", "bank", "best"}:
        return "speaker_bank"
    if normalized == "manual":
        return "manual"
    raise ValueError(f"unsupported reference_mode: {value}")


def _patch_chunk_override(
    overrides: dict[str, Any],
    chunk_id: str,
    *,
    speaker: str | None,
    reference_mode: str | None,
    reference_chunk_id: str | None,
) -> bool:
    current = dict(overrides.get(chunk_id) or {})
    before = json.dumps(current, ensure_ascii=False, sort_keys=True)

    if speaker is not None:
        clean_speaker = str(speaker).strip()
        if not clean_speaker:
            raise ValueError("speaker must not be blank")
        current["speaker"] = clean_speaker

    if reference_mode is not None:
        clean_mode = _normalize_reference_mode(reference_mode)
        if clean_mode == "auto":
            current.pop("reference_mode", None)
            current.pop("reference_chunk_id", None)
        else:
            current["reference_mode"] = clean_mode
            if clean_mode == "self":
                current.pop("reference_chunk_id", None)
    if reference_chunk_id is not None:
        clean_reference = str(reference_chunk_id).strip()
        if clean_reference:
            current["reference_chunk_id"] = clean_reference
        else:
            current.pop("reference_chunk_id", None)

    if current:
        overrides[chunk_id] = current
    else:
        overrides.pop(chunk_id, None)
    after = json.dumps(overrides.get(chunk_id) or {}, ensure_ascii=False, sort_keys=True)
    return before != after


def _patch_master_for_speaker_reference(
    rows: list[dict[str, Any]],
    chunk_id: str,
    *,
    speaker: str | None,
    reference_mode: str | None,
    reference_chunk_id: str | None,
) -> bool:
    for row in rows:
        if str(row.get("chunk_id", "")) != chunk_id:
            continue
        before = json.dumps(
            {
                "speaker": row.get("speaker"),
                "reference_mode_override": row.get("reference_mode_override"),
                "reference_chunk_id_override": row.get("reference_chunk_id_override"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if speaker is not None:
            clean_speaker = str(speaker).strip()
            if not clean_speaker:
                raise ValueError("speaker must not be blank")
            if clean_speaker != str(row.get("speaker", "") or ""):
                row.setdefault("speaker_original", row.get("speaker"))
                row["speaker"] = clean_speaker
                row["speaker_override"] = True

        if reference_mode is not None:
            clean_mode = _normalize_reference_mode(reference_mode)
            if clean_mode == "auto":
                row.pop("reference_mode_override", None)
                row.pop("reference_chunk_id_override", None)
            else:
                row["reference_mode_override"] = clean_mode
                if clean_mode == "self":
                    row.pop("reference_chunk_id_override", None)
        if reference_chunk_id is not None:
            clean_reference = str(reference_chunk_id).strip()
            if clean_reference:
                row["reference_chunk_id_override"] = clean_reference
            else:
                row.pop("reference_chunk_id_override", None)

        after = json.dumps(
            {
                "speaker": row.get("speaker"),
                "reference_mode_override": row.get("reference_mode_override"),
                "reference_chunk_id_override": row.get("reference_chunk_id_override"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if before != after:
            row["dub_stale"] = True
            row.pop("dub_error", None)
            row.pop("dub_input_signature", None)
            row.pop("dub_reference_chunk_id", None)
            row.pop("dub_reference_wav", None)
            row.pop("dub_reference_mode", None)
        return True
    return False


def _patch_master_for_advanced_fields(
    rows: list[dict[str, Any]],
    chunk_id: str,
    *,
    tts_instruct_text: str | None,
    emotion_label: str | None,
    emotion_scores: dict[str, float] | None,
) -> bool:
    for row in rows:
        if str(row.get("chunk_id", "")) != chunk_id:
            continue
        # emotion 편집 — source_emotion dict 갱신, label은 명시값 우선, 없으면 max-score로 자동
        if emotion_label is not None or emotion_scores is not None:
            existing = row.get("source_emotion") if isinstance(row.get("source_emotion"), dict) else {}
            new_emotion: dict[str, Any] = dict(existing)
            if emotion_scores is not None:
                clean_scores: dict[str, float] = {}
                for key, value in emotion_scores.items():
                    try:
                        clean_scores[str(key)] = max(0.0, min(1.0, float(value)))
                    except (TypeError, ValueError):
                        continue
                if clean_scores:
                    new_emotion["scores"] = clean_scores
                    new_emotion["confidence"] = max(clean_scores.values())
                    if emotion_label is None:
                        new_emotion["label"] = max(clean_scores.items(), key=lambda kv: kv[1])[0]
            if emotion_label is not None:
                new_emotion["label"] = str(emotion_label)
            new_emotion["source"] = "manual"
            row["source_emotion"] = new_emotion
            # 감정이 바뀌면 LLM/fallback 프롬프트는 재생성. 사용자가 직접 만든 manual instruction 은 보존.
            instruct_source = row.get("tts_instruct_source")
            if tts_instruct_text is None and instruct_source != "manual":
                row.pop("tts_instruct_text", None)
                row.pop("tts_instruct_source", None)
                row.pop("tts_instruct_emotion_label", None)
                row.pop("tts_instruct_applied", None)
                row.pop("tts_instruct_error", None)
            row["dub_stale"] = True
            row.pop("dub_input_signature", None)

        # tts_instruct_text 편집 — sanitize 로 영어/endofprompt 강제. CJK 만 있던 입력은 빈 결과 → manual 마킹 안 하고 다음 redub 가 LLM 으로 채우게
        if tts_instruct_text is not None:
            from generate_tts_instructions import sanitize_instruction
            cleaned = sanitize_instruction(tts_instruct_text)
            if cleaned:
                row["tts_instruct_text"] = cleaned
                row["tts_instruct_source"] = "manual"
            else:
                row.pop("tts_instruct_text", None)
                row.pop("tts_instruct_source", None)
            row.pop("tts_instruct_error", None)
            row.pop("tts_instruct_applied", None)
            row["dub_stale"] = True
            row.pop("dub_input_signature", None)
        return True
    return False


def mark_chunk_stale(record: RunRecord, chunk_id: str) -> None:
    mark_chunks_stale(record, [chunk_id])


def mark_chunks_stale(record: RunRecord, chunk_ids: list[str]) -> None:
    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return
    config = load_run_config(record)
    master_path = _config_value(config, "paths.master_timeline_json")
    rows = _read_json_list(master_path)
    found: set[str] = set()
    for row in rows:
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id in ids:
            row["dub_stale"] = True
            found.add(chunk_id)
    missing = [chunk_id for chunk_id in ids if chunk_id not in found]
    if missing:
        raise KeyError(missing[0])
    _write_json(master_path, rows)


def downstream_steps(from_step: StepName) -> list[StepName]:
    return steps_in_range(from_step, "mux")


def _patch_translation_rows(rows: list[dict[str, Any]], chunk_id: str, text: str, *, stale: bool) -> bool:
    for row in rows:
        if str(row.get("chunk_id", "")) != chunk_id:
            continue
        row["text_translated"] = text
        row["text_tts"] = text
        # 사용자가 빈 텍스트 → 번역 본문을 채워 넣은 경우 blocked 마킹 해제 — translate 단계 재실행 없이 다음 단계 진행 가능.
        if text.strip():
            row.pop("translation_blocked", None)
            row.pop("translation_blocked_reason", None)
        if stale:
            row["dub_stale"] = True
            for key in (
                "tts_instruct_text",
                "tts_instruct_source",
                "tts_instruct_emotion_label",
                "tts_instruct_error",
                "tts_instruct_applied",
                "dub_input_signature",
            ):
                row.pop(key, None)
        return True
    return False


def _reference_candidate_from_row(row: dict[str, Any], *, min_prompt_sec: float) -> ReferenceCandidate | None:
    chunk_id = str(row.get("chunk_id", "") or "").strip()
    wav = _string_or_none(row.get("wav"))
    if not chunk_id or not wav:
        return None
    duration = _duration_original(row)
    feature = row.get("source_audio_summary") if isinstance(row.get("source_audio_summary"), dict) else None
    assessment = assess_reference_candidate(
        {
            "wav": wav,
            "text_src": str(row.get("text_src", "") or ""),
            "duration": duration or 0.0,
        },
        chunk_feature=feature,
        min_prompt_sec=min_prompt_sec,
    )
    return ReferenceCandidate(
        chunk_id=chunk_id,
        speaker=_string_or_none(row.get("speaker")),
        start=_float_or_none(row.get("start")),
        end=_float_or_none(row.get("end")),
        duration=duration,
        text_src=_string_or_none(row.get("text_src")),
        wav=wav,
        accepted=bool(assessment.get("accepted")),
        score=_float_or_none(assessment.get("score")),
        warnings=[str(item) for item in assessment.get("warnings", [])],
        critical_flags=[str(item) for item in assessment.get("critical_flags", [])],
    )


def _chunk_from_timeline(row: dict[str, Any]) -> ChunkRow:
    source_emotion = row.get("source_emotion")
    error = _string_or_none(row.get("dub_error") or row.get("tts_validation_error") or row.get("emotion_error"))
    dubbed_audio = _string_or_none(row.get("dub_wav"))
    stale = bool(row.get("dub_stale"))
    blocked = bool(row.get("translation_blocked"))
    status = _chunk_status(error=error, stale=stale, dubbed_audio=dubbed_audio, blocked=blocked)
    return ChunkRow(
        chunk_id=str(row["chunk_id"]),
        speaker=_string_or_none(row.get("speaker")),
        start=_float_or_none(row.get("start")),
        end=_float_or_none(row.get("end")),
        emotion=_emotion_label(source_emotion) or _string_or_none(row.get("emotion")),
        source_text=_string_or_none(row.get("text_src")),
        translated_text=_string_or_none(row.get("text_translated")),
        original_audio=_string_or_none(row.get("wav")),
        dubbed_audio=dubbed_audio,
        status=status,
        dub_stale=stale,
        error=error,
        duration_original=_duration_original(row),
        duration_dub=_duration_dub(row, dubbed_audio),
        emotion_scores=_emotion_scores(source_emotion),
        tts_instruct_text=_string_or_none(row.get("tts_instruct_text")),
        tts_instruct_source=_string_or_none(row.get("tts_instruct_source")),
        reference_mode=_string_or_none(row.get("reference_mode_override") or row.get("dub_reference_mode")),
        reference_chunk_id=_string_or_none(row.get("reference_chunk_id_override") or row.get("dub_reference_chunk_id")),
        reference_audio=_string_or_none(row.get("dub_reference_wav")),
        reference_override_mode=_string_or_none(row.get("reference_mode_override")),
        reference_override_chunk_id=_string_or_none(row.get("reference_chunk_id_override")),
        translation_blocked=blocked,
        translation_blocked_reason=_string_or_none(row.get("translation_blocked_reason")),
    )


def _chunk_status(*, error: str | None, stale: bool, dubbed_audio: str | None, blocked: bool = False) -> str:
    if blocked:
        return "blocked"
    if error:
        return "error"
    if stale:
        return "stale"
    if dubbed_audio and _path_exists(dubbed_audio):
        return "done"
    return "queued"


def _artifact(config: dict[str, Any], label: str, ref: str, kind: str) -> StepArtifact:
    value = _config_value(config, ref)
    if not value:
        return StepArtifact(label=label, path="", kind=kind)
    path = _resolve_project_path(value)
    exists = path.exists()
    stat = path.stat() if exists else None
    return StepArtifact(
        label=label,
        path=_project_relative(path),
        kind=kind,
        exists=exists,
        size_bytes=stat.st_size if stat and path.is_file() else None,
        mtime=stat.st_mtime if stat else None,
    )


def _config_value(config: dict[str, Any], ref: str) -> str | None:
    if "." not in ref:
        return _string_or_none(config.get(ref))
    value = deep_get(config, tuple(ref.split(".")))
    return _string_or_none(value)


def _read_json_list(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = _resolve_project_path(path_value)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _write_json(path_value: str | None, data: Any) -> None:
    if not path_value:
        raise ValueError("missing artifact path")
    path = _resolve_project_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    tmp.replace(path)


def _by_chunk_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("chunk_id", "")): row for row in rows if row.get("chunk_id")}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes project root: {value}") from exc
    return resolved


def _project_relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def _path_exists(path_value: str) -> bool:
    try:
        return _resolve_project_path(path_value).exists()
    except ValueError:
        return False


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_original(row: dict[str, Any]) -> float | None:
    direct = _float_or_none(row.get("duration") or row.get("duration_original"))
    if direct is not None:
        return direct
    start = _float_or_none(row.get("start"))
    end = _float_or_none(row.get("end"))
    if start is None or end is None:
        return None
    return round(end - start, 3)


def _duration_dub(row: dict[str, Any], dubbed_audio: str | None) -> float | None:
    for value in (
        row.get("duration_dub"),
        row.get("dub_duration"),
        deep_get(row, ("quality_gates", "tts", "metrics", "actual_duration")),
        deep_get(row, ("quality_gates", "tts_asr", "metrics", "actual_duration")),
    ):
        duration = _float_or_none(value)
        if duration is not None:
            return duration
    if not dubbed_audio:
        return None
    try:
        wav_path = _resolve_project_path(dubbed_audio)
        if not wav_path.exists():
            return None
        with wave.open(str(wav_path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 3)
    except (OSError, wave.Error, ValueError, ZeroDivisionError):
        return None


def _emotion_label(value: Any) -> str | None:
    if isinstance(value, dict):
        return _string_or_none(value.get("label") or value.get("emotion"))
    return _string_or_none(value)


def _emotion_scores(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    scores = value.get("scores")
    if not isinstance(scores, dict):
        return None
    out: dict[str, float] = {}
    for key, score in scores.items():
        parsed = _float_or_none(score)
        if parsed is not None:
            out[str(key)] = parsed
    return out or None
