// 20단계 더빙 파이프라인의 화면 표시 메타데이터 (src/pipeline.py:STEP_FUNCTIONS 와 동기)
import { PIPELINE_STEPS, type StepName } from "@/api/client";

export interface StepMeta {
  label: string;
  shortLabel: string;
  description: string;
  service: string;
  inputs: string[];
  outputs: string[];
}

export interface PipelinePhase {
  id: string;
  label: string;
  steps: StepName[];
}

export const PIPELINE_STEP_META: Record<StepName, StepMeta> = {
  extract_audio: {
    label: "Extract Audio",
    shortLabel: "extract_audio",
    description: "입력 영상에서 원본 오디오 트랙을 추출합니다.",
    service: "controller",
    inputs: ["input.mp4"],
    outputs: ["raw.wav"],
  },
  separate_audio: {
    label: "Separate Audio",
    shortLabel: "separate_audio",
    description: "BS-RoFormer + MDX23C-InstVoc_HQ 앙상블로 보컬과 배경음을 분리합니다.",
    service: "separator",
    inputs: ["raw.wav"],
    outputs: ["dialogue.wav", "bgm.wav"],
  },
  redirect_nonspeech: {
    label: "Redirect Nonspeech",
    shortLabel: "redirect_nonspeech",
    description: "분리된 무음·비발화 구간을 배경 트랙으로 되돌려 후속 화자 처리 입력을 정리합니다.",
    service: "controller",
    inputs: ["dialogue.wav", "bgm.wav"],
    outputs: ["dialogue.wav", "bgm.wav"],
  },
  diarize: {
    label: "Diarize",
    shortLabel: "diarize",
    description: "화자 구간을 추정하고 RTTM 결과를 만듭니다.",
    service: "diarizer",
    inputs: ["dialogue.wav"],
    outputs: ["diarization.rttm"],
  },
  rttm_to_json: {
    label: "RTTM to JSON",
    shortLabel: "rttm_to_json",
    description: "RTTM을 파이프라인에서 쓰는 JSON chunk 구조로 변환합니다.",
    service: "controller",
    inputs: ["diarization.rttm"],
    outputs: ["chunks.json"],
  },
  face_clustering: {
    label: "Face Clustering",
    shortLabel: "face_clustering",
    description: "InsightFace 임베딩 + LightASD 활성화자 검출로 얼굴 클러스터를 만들어 화자분리 보정 신호로 씁니다.",
    service: "face",
    inputs: ["input.mp4", "diarization.json"],
    outputs: ["faces.json", "face_clusters.json"],
  },
  build_repair_inputs: {
    label: "Build Repair Inputs",
    shortLabel: "repair_inputs",
    description: "fusion 화자분리·보컬·얼굴·풀영상 ASR을 검증된 repair 입력(run_dir) 형식으로 조립합니다.",
    service: "controller",
    inputs: ["diarization.json", "dialogue.wav", "faces.json"],
    outputs: ["repair_run_dir/"],
  },
  apply_preserved_repair: {
    label: "Apply Preserved Repair",
    shortLabel: "preserved_repair",
    description: "검증된 4-way fusion + 8 repair patch(gap-fill·face-identity·word-split 등)로 화자분리를 보정합니다. config에서 비활성이면 건너뜁니다.",
    service: "diarizer",
    inputs: ["repair_run_dir/"],
    outputs: ["*_segments_gapfilled.json"],
  },
  apply_gapfilled: {
    label: "Apply Gapfilled",
    shortLabel: "apply_gapfilled",
    description: "보정된 gapfilled 화자분리를 청크 단계 입력으로 export합니다.",
    service: "controller",
    inputs: ["*_segments_gapfilled.json"],
    outputs: ["diarization.json"],
  },
  merge_chunks: {
    label: "Merge Chunks",
    shortLabel: "merge_chunks",
    description: "짧거나 인접한 발화 구간을 더빙 단위로 병합합니다.",
    service: "controller",
    inputs: ["chunks.json"],
    outputs: ["merged.json"],
  },
  cut_chunks: {
    label: "Cut Chunks",
    shortLabel: "cut_chunks",
    description: "각 발화 구간의 원본 오디오 조각을 저장합니다.",
    service: "controller",
    inputs: ["merged.json", "dialogue.wav"],
    outputs: ["chunk_*.wav"],
  },
  extract_emotion: {
    label: "Extract Emotion",
    shortLabel: "extract_emotion",
    description: "발화별 감정 벡터와 스타일 힌트를 추출합니다.",
    service: "speaker",
    inputs: ["chunk_*.wav"],
    outputs: ["emotion.json"],
  },
  run_asr: {
    label: "Run ASR",
    shortLabel: "run_asr",
    description: "원본 발화의 텍스트를 인식합니다.",
    service: "speaker",
    inputs: ["chunk_*.wav"],
    outputs: ["asr.json"],
  },
  translate: {
    label: "Translate",
    shortLabel: "translate",
    description: "ASR 텍스트를 목표 언어로 번역합니다.",
    service: "controller",
    inputs: ["asr.json"],
    outputs: ["translated.json"],
  },
  build_timeline: {
    label: "Build Timeline",
    shortLabel: "build_timeline",
    description: "번역문, 화자, 감정, 타이밍을 하나의 master timeline으로 합칩니다.",
    service: "controller",
    inputs: ["translated.json", "emotion.json"],
    outputs: ["timeline.json"],
  },
  generate_tts_instructions: {
    label: "Generate TTS Instructions",
    shortLabel: "tts_instructions",
    description: "CosyVoice 입력 프롬프트와 스타일 지시문을 생성합니다.",
    service: "controller",
    inputs: ["timeline.json"],
    outputs: ["tts_jobs.json"],
  },
  run_tts: {
    label: "Run TTS",
    shortLabel: "run_tts",
    description: "각 chunk의 한국어 더빙 음성을 합성합니다.",
    service: "tts-cosyvoice",
    inputs: ["tts_jobs.json"],
    outputs: ["chunk_*_dub.wav"],
  },
  validate_tts: {
    label: "Validate TTS",
    shortLabel: "validate_tts",
    description: "합성 음성 길이와 실패 chunk를 검증합니다.",
    service: "speaker",
    inputs: ["chunk_*_dub.wav"],
    outputs: ["tts_report.json"],
  },
  compose_audio: {
    label: "Compose Audio",
    shortLabel: "compose_audio",
    description: "합성 음성을 배경음과 맞춰 최종 오디오로 합성합니다.",
    service: "controller",
    inputs: ["chunk_*_dub.wav", "bgm.wav"],
    outputs: ["dubbed.wav"],
  },
  mux: {
    label: "Mux",
    shortLabel: "mux",
    description: "최종 오디오를 원본 영상과 mux하여 결과 영상을 만듭니다.",
    service: "controller",
    inputs: ["dubbed.wav", "input.mp4"],
    outputs: ["output.mp4"],
  },
};

export const PIPELINE_PHASES: PipelinePhase[] = [
  { id: "audio", label: "Audio Prep", steps: ["extract_audio", "separate_audio", "redirect_nonspeech"] },
  { id: "diarization", label: "Diarization & ASR", steps: ["diarize", "rttm_to_json", "face_clustering", "build_repair_inputs", "apply_preserved_repair", "apply_gapfilled", "merge_chunks", "cut_chunks", "extract_emotion"] },
  { id: "text", label: "Text Processing", steps: ["run_asr", "translate", "build_timeline"] },
  { id: "synthesis", label: "Synthesis", steps: ["generate_tts_instructions", "run_tts", "validate_tts", "compose_audio", "mux"] },
];

export function getStepMeta(step: StepName): StepMeta {
  // backend 가 새 단계를 추가해도 UI 가 죽지 않도록 안전한 기본값 fallback
  return (
    PIPELINE_STEP_META[step] ?? {
      label: step,
      shortLabel: step,
      description: "",
      service: "controller",
      inputs: [],
      outputs: [],
    }
  );
}

export function isStepName(value: string | undefined): value is StepName {
  return Boolean(value && (PIPELINE_STEPS as readonly string[]).includes(value));
}
