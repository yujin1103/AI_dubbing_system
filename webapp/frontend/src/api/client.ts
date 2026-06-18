// Docker Vite proxy를 통해 백엔드 REST와 optional artifact API를 호출하는 클라이언트
export const PIPELINE_STEPS = [
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
] as const;

export type StepName = (typeof PIPELINE_STEPS)[number];
export type StepState = "pending" | "running" | "done" | "failed" | "skipped";
export type RunStatus = "queued" | "running" | "success" | "failed" | "canceled";

export interface StepRecord {
  name: StepName;
  state: StepState;
  started_at?: number | null;
  ended_at?: number | null;
  error?: string | null;
}

export interface StepRuntime {
  service: string;
  tool?: string | null;
}

export interface RunRecord {
  run_id: string;
  created_at: number;
  input_video: string;
  input_stem: string;
  tts_engine: string;
  config_path: string;
  status: RunStatus;
  steps: StepRecord[];
  log_path: string;
  output_video?: string | null;
  error?: string | null;
  step_runtime?: Partial<Record<StepName, StepRuntime>>;
}

export interface RunOverrides {
  target_language?: string;
  source_language?: string;
  fit_to_duration?: boolean;
  duration_fit_max_tempo?: number;
  use_separator?: boolean;
  skip_existing?: boolean;
}

export interface CreateRunRequest {
  input_path: string;
  base_config?: string;
  overrides?: RunOverrides;
  from_step?: StepName;
  to_step?: StepName;
}

export interface InputFile {
  input_path: string;
  name: string;
  size_bytes: number;
}

export interface ConfigEntry {
  name: string;
  path: string;
}

export interface ChunkRow {
  chunk_id: string;
  speaker?: string | null;
  start?: number | null;
  end?: number | null;
  emotion?: string | null;
  source_text?: string | null;
  translated_text?: string | null;
  original_audio?: string | null;
  dubbed_audio?: string | null;
  status?: "queued" | "running" | "done" | "error" | "stale" | "blocked" | string;
  dub_stale?: boolean;
  error?: string | null;
  duration_original?: number | null;
  duration_dub?: number | null;
  emotion_scores?: Record<string, number> | null;
  tts_instruct_text?: string | null;
  tts_instruct_source?: string | null;
  reference_mode?: string | null;
  reference_chunk_id?: string | null;
  reference_audio?: string | null;
  reference_override_mode?: string | null;
  reference_override_chunk_id?: string | null;
  translation_blocked?: boolean;
  translation_blocked_reason?: string | null;
}

export interface ReferenceCandidate {
  chunk_id: string;
  speaker?: string | null;
  start?: number | null;
  end?: number | null;
  duration?: number | null;
  text_src?: string | null;
  wav?: string | null;
  accepted: boolean;
  score?: number | null;
  mos?: number | null;
  mos_recommended?: boolean;
  warnings: string[];
  critical_flags: string[];
}

export interface PreviewInstructionRequest {
  emotion_label?: string | null;
  emotion_scores?: Record<string, number> | null;
}

export interface PreviewInstructionResponse {
  instruction: string;
  source: "llm" | "fallback";
  error?: string | null;
}

export interface PatchChunkPayload {
  speaker?: string;
  emotion?: string;
  translated_text?: string;
  reference_mode?: string;
  reference_chunk_id?: string | null;
  tts_instruct_text?: string;
  emotion_label?: string;
  emotion_scores?: Record<string, number>;
}

export interface PatchChunkUpdate extends PatchChunkPayload {
  chunk_id: string;
}

export interface BulkPatchChunksRequest {
  updates: PatchChunkUpdate[];
}

export interface BulkRedubChunksRequest {
  chunk_ids: string[];
}

export interface StepArtifact {
  label: string;
  path: string;
  kind?: string | null;
  exists: boolean;
  size_bytes?: number | null;
  mtime?: number | null;
}

export interface StepDetailRecord {
  step: StepName;
  status?: StepState | null;
  service?: string | null;
  inputs?: string[];
  outputs?: string[];
  artifacts?: StepArtifact[];
  log_excerpt?: string[];
  error?: string | null;
}

export type ActivityKind =
  | "run_created"
  | "status_change"
  | "run_canceled"
  | "run_resumed"
  | "chunk_text_edit"
  | "chunk_instruction_edit"
  | "chunk_emotion_edit"
  | "chunk_speaker_edit"
  | "chunk_reference_edit"
  | "chunk_redub"
  | "step_rerun"
  | "mos_scored";

export interface ActivityEvent {
  ts: number;
  kind: ActivityKind;
  chunk_id?: string | null;
  step?: StepName | null;
  status?: RunStatus | null;
  before?: unknown;
  after?: unknown;
  note?: string | null;
  run_id?: string | null;
}

export interface ProjectSummary {
  input_stem: string;
  input_video: string;
  run_count: number;
  success_count: number;
  failed_count: number;
  canceled_count: number;
  last_run_id?: string | null;
  last_status?: RunStatus | null;
  last_progress_pct: number;
  created_at: number;
  updated_at: number;
}

export interface MetricsResponse {
  total_chunks: number;
  ready_chunks: number;
  stale_chunks: number;
  error_chunks: number;
  average_duration_ratio?: number | null;
  validation_failures: number;
  emotion_distribution: Record<string, number>;
}

export class ApiError extends Error {
  constructor(
    public method: string,
    public path: string,
    public status: number,
    public responseText: string,
  ) {
    super(`${method} ${path} → ${status}: ${responseText}`);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(method, path, res.status, text);
  }
  return res.json() as Promise<T>;
}

export function isOptionalEndpointMissing(error: unknown): boolean {
  return error instanceof ApiError && [404, 405, 501].includes(error.status);
}

export const api = {
  health: () => request<{ status: string; docker_cli: boolean; services: Record<string, string> }>("GET", "/api/health"),
  listConfigs: () => request<ConfigEntry[]>("GET", "/api/configs"),
  listInputs: () => request<InputFile[]>("GET", "/api/inputs"),
  uploadVideo: async (file: File): Promise<{ input_path: string; size_bytes: number }> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/uploads", { method: "POST", body: form });
    if (!res.ok) throw new ApiError("POST", "/api/uploads", res.status, await res.text());
    return res.json();
  },
  createRun: (payload: CreateRunRequest) => request<RunRecord>("POST", "/api/runs", payload),
  listRuns: () => request<RunRecord[]>("GET", "/api/runs"),
  getRun: (id: string) => request<RunRecord>("GET", `/api/runs/${id}`),
  cancelRun: (id: string) => request<RunRecord>("POST", `/api/runs/${id}/cancel`),
  resumeRun: (id: string) => request<RunRecord>("POST", `/api/runs/${id}/resume`),
  listChunks: (id: string) => request<ChunkRow[]>("GET", `/api/runs/${id}/chunks`),
  getChunk: (id: string, chunkId: string) => request<ChunkRow>("GET", `/api/runs/${id}/chunks/${chunkId}`),
  getSpeakerReferenceBank: (id: string) => request<Record<string, ReferenceCandidate[]>>("GET", `/api/runs/${id}/speaker-reference-bank`),
  scoreSpeakerReferenceBank: (id: string) => request<Record<string, ReferenceCandidate[]>>("POST", `/api/runs/${id}/speaker-reference-bank/score`),
  patchChunk: (id: string, chunkId: string, payload: PatchChunkPayload) => request<ChunkRow>("PATCH", `/api/runs/${id}/chunks/${chunkId}`, payload),
  patchChunks: (id: string, payload: BulkPatchChunksRequest) => request<ChunkRow[]>("PATCH", `/api/runs/${id}/chunks`, payload),
  redubChunk: (id: string, chunkId: string) => request<RunRecord>("POST", `/api/runs/${id}/chunks/${chunkId}/redub`),
  redubChunks: (id: string, payload: BulkRedubChunksRequest) => request<RunRecord>("POST", `/api/runs/${id}/chunks/redub`, payload),
  previewInstruction: (id: string, chunkId: string, payload: PreviewInstructionRequest) =>
    request<PreviewInstructionResponse>("POST", `/api/runs/${id}/chunks/${chunkId}/preview-instruction`, payload),
  getStepDetail: (id: string, step: StepName) => request<StepDetailRecord>("GET", `/api/runs/${id}/steps/${step}`),
  rerunStep: (id: string, step: StepName) => request<RunRecord>("POST", `/api/runs/${id}/steps/${step}/rerun`),
  getLog: (id: string, limit = 500) => request<{ lines: string[] }>("GET", `/api/runs/${id}/log?limit=${limit}`),
  getMetrics: (id: string) => request<MetricsResponse>("GET", `/api/runs/${id}/metrics`),
  getActivity: (id: string, limit = 500) => request<ActivityEvent[]>("GET", `/api/runs/${id}/activity?limit=${limit}`),
  listProjects: () => request<ProjectSummary[]>("GET", "/api/projects"),
  getProjectRuns: (inputStem: string) => request<RunRecord[]>("GET", `/api/projects/${encodeURIComponent(inputStem)}/runs`),
  getCrossActivity: (limit = 200) => request<ActivityEvent[]>("GET", `/api/activity?limit=${limit}`),
};
