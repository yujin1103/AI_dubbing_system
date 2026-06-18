# Pydantic 데이터 모델 — REST/WS 페이로드 + 디스크 영속화 스키마
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# 20단계 파이프라인 step 이름 — src/pipeline.py:STEP_FUNCTIONS 와 동기 유지
# (face_clustering 은 face service. build_repair_inputs/apply_preserved_repair/apply_gapfilled 은
#  검증된 4-way fusion + 8 repair patch 를 모듈 pipeline 안에서 재현하는 화자분리 보정 단계 —
#  preserved_repair.enabled=false 인 config 면 각 단계가 no-op 스킵된다.)
PIPELINE_STEPS: tuple[str, ...] = (
    "extract_audio",
    "separate_audio",
    "redirect_nonspeech",
    "diarize",
    "rttm_to_json",
    "face_clustering",
    "build_repair_inputs",
    "apply_preserved_repair",
    "apply_gapfilled",
    "merge_chunks",
    "cut_chunks",
    "extract_emotion",
    "run_asr",
    "translate",
    "build_timeline",
    "generate_tts_instructions",
    "run_tts",
    "validate_tts",
    "compose_audio",
    "mux",
)

StepName = Literal[
    "extract_audio", "separate_audio", "redirect_nonspeech", "diarize", "rttm_to_json",
    "face_clustering", "build_repair_inputs", "apply_preserved_repair", "apply_gapfilled", "merge_chunks",
    "cut_chunks", "extract_emotion", "run_asr", "translate", "build_timeline",
    "generate_tts_instructions", "run_tts", "validate_tts", "compose_audio", "mux",
]
StepState = Literal["pending", "running", "done", "failed", "skipped"]
RunStatus = Literal["queued", "running", "success", "failed", "canceled"]


class RunOverrides(BaseModel):
    """UI 가 노출하는 knob — 이 외 필드는 base config 그대로 유지."""

    target_language: Optional[str] = None
    source_language: Optional[str] = None  # 원본 영상 언어 수동 지정(빈 문자열=자동감지/미override)
    fit_to_duration: Optional[bool] = None
    duration_fit_max_tempo: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    use_separator: Optional[bool] = None
    skip_existing: Optional[bool] = None


class CreateRunRequest(BaseModel):
    input_path: str  # input/ 아래 상대 경로
    base_config: str = "configs/cosyvoice3-docker-draft.json"
    overrides: RunOverrides = Field(default_factory=RunOverrides)
    from_step: StepName = "extract_audio"
    # mux 직전(validate_tts)에서 멈춤 — 사용자가 청크를 검수·수정 후 직접 Run Final Output 트리거
    to_step: StepName = "validate_tts"


class StepRecord(BaseModel):
    name: StepName
    state: StepState = "pending"
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None


class StepRuntime(BaseModel):
    """단계별 docker compose service + 그 단계의 핵심 도구/모델 — UI Progress 카드에서 노출."""

    service: str
    tool: Optional[str] = None  # ffmpeg / 모델 파일명 / LLM mode 등. 별도 도구 없는 단계는 None


class RunRecord(BaseModel):
    run_id: str
    created_at: float
    input_video: str
    input_stem: str
    tts_engine: str = "cosyvoice"
    config_path: str
    status: RunStatus = "queued"
    steps: list[StepRecord]
    log_path: str
    output_video: Optional[str] = None
    error: Optional[str] = None
    # 단계별 service/tool — run 생성 시점에 config 에서 추출, 이후 lazy 갱신
    step_runtime: dict[StepName, StepRuntime] = Field(default_factory=dict)


class ChunkRow(BaseModel):
    chunk_id: str
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    emotion: Optional[str] = None
    source_text: Optional[str] = None
    translated_text: Optional[str] = None
    original_audio: Optional[str] = None
    dubbed_audio: Optional[str] = None
    status: str = "queued"
    dub_stale: bool = False
    error: Optional[str] = None
    duration_original: Optional[float] = None
    duration_dub: Optional[float] = None
    emotion_scores: Optional[dict[str, float]] = None
    tts_instruct_text: Optional[str] = None
    tts_instruct_source: Optional[str] = None  # llm | fallback | manual
    reference_mode: Optional[str] = None
    reference_chunk_id: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_override_mode: Optional[str] = None
    reference_override_chunk_id: Optional[str] = None
    # 번역기가 content_filter 등으로 차단해 본문이 비어 있는 청크. UI 가 수동 입력을 유도하기 위해 노출.
    translation_blocked: bool = False
    translation_blocked_reason: Optional[str] = None


class ReferenceCandidate(BaseModel):
    chunk_id: str
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    duration: Optional[float] = None
    text_src: Optional[str] = None
    wav: Optional[str] = None
    accepted: bool = False
    score: Optional[float] = None
    mos: Optional[float] = None  # MOS 품질점수(1~5) — reference_mos.json 채점 결과(있을 때만)
    mos_recommended: bool = False  # 해당 화자 후보 중 MOS 최고(추천 배지) — 기존 선택 로직 불변
    warnings: list[str] = Field(default_factory=list)
    critical_flags: list[str] = Field(default_factory=list)


class PreviewInstructionRequest(BaseModel):
    """Preview LLM instruction — emotion 을 in-memory 로 overlay 한 후 LLM 호출.
    값이 둘 다 None 이면 master_timeline 의 현재 emotion 사용."""

    emotion_label: Optional[str] = None
    emotion_scores: Optional[dict[str, float]] = None


class PreviewInstructionResponse(BaseModel):
    instruction: str  # <|endofprompt|> 포함된 형식 그대로 반환 — frontend 가 표시 시 strip
    source: Literal["llm", "fallback"] = "llm"
    error: Optional[str] = None


class PatchChunkPayload(BaseModel):
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    translated_text: Optional[str] = None
    reference_mode: Optional[str] = None
    reference_chunk_id: Optional[str] = None
    # TTS 프롬프트 직접 편집 — 저장 시 source=manual 마킹, dub_stale=true
    tts_instruct_text: Optional[str] = None
    # 감정 라벨 직접 편집 (max-score 기반 자동 갱신과 무관하게 사용자 지정 가능)
    emotion_label: Optional[str] = None
    # 9개 감정 score (0~1) — 이퀄라이저 편집 결과
    emotion_scores: Optional[dict[str, float]] = None


class PatchChunkUpdate(PatchChunkPayload):
    chunk_id: str


class BulkPatchChunksRequest(BaseModel):
    updates: list[PatchChunkUpdate] = Field(default_factory=list)


class BulkRedubChunksRequest(BaseModel):
    chunk_ids: list[str] = Field(default_factory=list)


class StepArtifact(BaseModel):
    label: str
    path: str
    kind: Optional[str] = None
    exists: bool = False
    size_bytes: Optional[int] = None
    mtime: Optional[float] = None


class StepDetailRecord(BaseModel):
    step: StepName
    status: Optional[StepState] = None
    service: Optional[str] = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    artifacts: list[StepArtifact] = Field(default_factory=list)
    log_excerpt: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class LogResponse(BaseModel):
    lines: list[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    total_chunks: int = 0
    ready_chunks: int = 0
    stale_chunks: int = 0
    error_chunks: int = 0
    average_duration_ratio: Optional[float] = None
    validation_failures: int = 0
    emotion_distribution: dict[str, int] = Field(default_factory=dict)


class StepEvent(BaseModel):
    type: Literal["step_start", "step_done", "step_error"]
    step: StepName
    index: int
    ts: float
    duration_ms: Optional[float] = None
    message: Optional[str] = None


class LogEvent(BaseModel):
    type: Literal["log"] = "log"
    line: str
    ts: float


class HeartbeatEvent(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"
    ts: float


class RunDoneEvent(BaseModel):
    type: Literal["run_done"] = "run_done"
    status: RunStatus
    ts: float


ActivityKind = Literal[
    "run_created",
    "status_change",
    "run_canceled",
    "run_resumed",
    "chunk_text_edit",
    "chunk_instruction_edit",
    "chunk_emotion_edit",
    "chunk_speaker_edit",
    "chunk_reference_edit",
    "chunk_redub",
    "step_rerun",
    "mos_scored",
]


class ActivityEvent(BaseModel):
    ts: float
    kind: ActivityKind
    chunk_id: Optional[str] = None
    step: Optional[StepName] = None
    status: Optional[RunStatus] = None
    before: Optional[Any] = None
    after: Optional[Any] = None
    note: Optional[str] = None
    # cross-run feed 에서만 채워지는 필드 — 단일 run 응답에서는 None
    run_id: Optional[str] = None


class ProjectSummary(BaseModel):
    """input_stem 단위로 묶은 가상 프로젝트 — 백엔드는 별도 영속화 안 함, 매 요청 집계."""

    input_stem: str
    input_video: str  # 그룹 대표 — 가장 최근 run 의 input_video
    run_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    canceled_count: int = 0
    last_run_id: Optional[str] = None
    last_status: Optional[RunStatus] = None
    last_progress_pct: int = 0
    created_at: float = 0.0  # 가장 오래된 run 의 created_at
    updated_at: float = 0.0  # 가장 최근 run 의 created_at
