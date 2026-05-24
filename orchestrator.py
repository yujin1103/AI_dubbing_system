"""
AI 더빙 파이프라인 — 단일 Python 오케스트레이터
=====================================================

현재 구조 (단일 Python):
  orchestrator.py — 모든 모델을 직접 로드하여 함수 호출로 파이프라인 실행

실행 방법:
    # 기본 (LatentSync 1.6, 512x512 마스킹 부드러움)
    python orchestrator.py --input /workspace/media/input/test.mp4 --lang ko \
        --speakers 1 --content-type lecture --enable-lipsync

    # 한국어 LoRA 파인튜닝 가중치 사용 (자동 인식: /workspace/media/lora/latentsync_ko.pt)
    python orchestrator.py --input /workspace/media/input/test.mp4 --lang ko \
        --enable-lipsync   # → ko LoRA 자동 사용

    # 명시적 가중치 지정
    python orchestrator.py --input video.mp4 --lang ko --enable-lipsync \
        --lipsync-ckpt /workspace/media/training_outputs/.../checkpoint-25000.pt

의존성 설치:
  pip install qwen-asr pyannote.audio funasr deep-translator
  pip install soundfile librosa numpy torch demucs
  pip install git+https://github.com/FunAudioLLM/CosyVoice.git
"""

import os
import sys
import re
import uuid
import json
import shutil
import tempfile
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime

import numpy as np
import soundfile as sf
import librosa
import torch

# ─── High-quality time-stretch (WSOLA) ──────────────────────
# librosa phase-vocoder는 한국어 TTS에서 "같은 구 반복 / 기계음" 아티팩트를 만듦.
# rubberband-cli (3.3.0 이상) + pyrubberband 조합이 있으면 WSOLA로 고품질 처리.
# 없으면 librosa로 폴백.
try:
    import pyrubberband as _pyrb
    import shutil as _shutil
    if _shutil.which("rubberband"):
        _HAS_RUBBERBAND = True
        print("[Init] rubberband-cli 감지됨 → 고품질 time-stretch 활성화 ✅")
    else:
        _HAS_RUBBERBAND = False
        print("[Init] ⚠️  rubberband-cli 없음 → librosa phase-vocoder 폴백")
except ImportError:
    _HAS_RUBBERBAND = False


def high_quality_time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rubberband 우선, librosa 폴백. pitch 보존, WSOLA 기반."""
    if abs(rate - 1.0) < 0.01:
        return audio
    if _HAS_RUBBERBAND:
        try:
            return _pyrb.time_stretch(audio, sr, rate).astype(audio.dtype)
        except Exception as e:
            print(f"  [time_stretch] rubberband 실패 → librosa 폴백: {e}")
    return librosa.effects.time_stretch(audio, rate=rate)


def normalize_peak(audio: np.ndarray, target: float = 0.9) -> np.ndarray:
    """time-stretch 후 peak 재정규화 (볼륨 균일화)."""
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        return audio * (target / peak)
    return audio


def count_korean_syllables(text: str) -> int:
    """한국어 음절(한글 글자) 수. 구두점/공백 제외.
    다국어 확장 시 language별 분기 필요 (상단 주석 참고)."""
    return sum(1 for c in text if '\uac00' <= c <= '\ud7a3')


def emotion_desc_rate_factor(emotion_desc: str) -> float:
    """
    🔥 LLM 자연어 감정 묘사에서 발화 속도 힌트 추출.
    "slow, reflective" 같은 묘사 → 0.85 (느림)
    "quick, urgent"     같은 묘사 → 1.05 (빠름)
    그 외                          → 1.00
    카테고리 배율(EMOTION_RATE_FACTOR)과 곱해서 사용.
    """
    if not emotion_desc:
        return 1.0
    desc = emotion_desc.lower()
    slow_words = ['slow', 'reflective', 'gentle', 'soft', 'quiet', 'tender',
                  'melancholy', 'thoughtful', 'measured', 'careful', 'somber',
                  'subdued', 'calm', 'tranquil', 'still', 'languid']
    fast_words = ['quick', 'fast', 'rushed', 'excited', 'urgent', 'energetic',
                  'frantic', 'animated', 'lively', 'snappy', 'rapid']
    if any(w in desc for w in slow_words):
        return 0.85
    if any(w in desc for w in fast_words):
        return 1.05
    return 1.0


# ─── 환경 설정 ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR  = os.path.join(BASE_DIR, "media")

# ── 공유 디렉터리 (실행 간 보존) ──
INPUT_DIR  = os.path.join(MEDIA_DIR, "input")       # 원본 영상
OUTPUT_DIR = os.path.join(MEDIA_DIR, "output")      # 최종 더빙 영상 (run별 보존)
REPORT_DIR = os.path.join(MEDIA_DIR, "reports")     # 파이프라인 리포트 (run별 보존)
RUNS_DIR   = os.path.join(MEDIA_DIR, "runs")        # 🆕 실행별 격리 루트

# ── 실행별 디렉터리 (run_pipeline() 진입 시 RunContext.activate()로 덮어씀) ──
# 주의: 초기값 None. 파이프라인 밖에서 이 상수를 직접 쓰지 말 것.
CHUNKS_DIR: Optional[str] = None
VOCALS_DIR: Optional[str] = None
BGM_DIR:    Optional[str] = None
DUBBED_DIR: Optional[str] = None
REF_DIR:    Optional[str] = None

# 공유 디렉터리만 미리 생성 (run-scoped는 RunContext.create()에서 만듦)
for d in [INPUT_DIR, OUTPUT_DIR, REPORT_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)

# 현재 활성 run_id (concat_chunks, save_pipeline_report에서 파일명에 사용)
CURRENT_RUN_ID: Optional[str] = None

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# MOS 모델 경로 (공유, 영구 보존)
MOS_CHECKPOINT = os.path.join(MEDIA_DIR, "model_cache", "mos_model", "best.pt")

# 세그먼트 분할 기준
MAX_SEG_DURATION = 12.0   # 초
MAX_SEG_WORDS    = 30    # 단어 수

# 감정 레이블 정규화
EMOTION_LABEL_MAP = {
    "angry": "Angry", "sad": "Sad", "happy": "Happy",
    "surprise": "Surprised", "fear": "Scared",
    "disgust": "Angry", "neutral": "Neutral", "others": "Neutral",
}

# 언어 코드 매핑
LANG_CODE_TO_NAME = {
    "en": "English", "ko": "Korean", "ja": "Japanese",
    "zh": "Chinese", "fr": "French", "de": "German",
    "es": "Spanish", "ru": "Russian", "ar": "Arabic",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
}

DEEP_LANG_MAP = {
    "ko": "ko", "ja": "ja", "zh": "zh-CN",
    "es": "es", "fr": "fr", "de": "de",
    "en": "en", "ru": "ru", "pt": "pt",
    "it": "it", "ar": "ar", "nl": "nl",
}

TTS_SAMPLE_RATE = 24000  # CosyVoice3 기본 샘플레이트


# ─── 데이터 클래스 ────────────────────────────────────────────

@dataclass
class WordTiming:
    word:  str
    start: float
    end:   float


@dataclass
class Segment:
    """
    파이프라인을 흐르는 기본 단위.
    ASR → pyannote → emotion2vec+ → 번역 → TTS 순으로 필드가 채워짐.
    """
    id:            int
    speaker:       str
    start:         float
    end:           float
    text:          str               # 원본 텍스트 (영어)
    translated:    str = ""          # 번역된 텍스트
    emotion:       str = "Neutral"   # 실제 TTS에 적용된 감정 (정책 적용 후)
    emotion_score: float = 0.0
    # 🔥 콘텐츠 타입 정책: raw 감정은 항상 기록, 정책에 의해 emotion이 덮어써질 수 있음
    raw_emotion:       str = "Neutral"   # emotion2vec+ 원본 감지값
    raw_emotion_score: float = 0.0
    # v29 (5/14): 팀원 master_timeline_cosyvoice.json 형식 — 9개 감정 점수 (UI 조정용)
    raw_emotion_scores: Dict[str, float] = field(default_factory=dict)
    # 🔥 LLM 자연어 감정 묘사 (passthrough 정책에서만 채워짐, neutral_only면 빈 문자열)
    #   CosyVoice3는 6-key 카테고리가 아니라 자유로운 자연어 instruction을 받음.
    #   LLM이 텍스트+화자+emotion2vec+ 결과를 종합해 풍부한 묘사를 만듦.
    tts_context:   str = ""          # 예: "The speaker recalls a major life decision."
    tts_emotion:   str = ""          # 예: "reflective, lightly nostalgic, casual"
    speed:         float = 1.0
    words:         List[WordTiming] = field(default_factory=list)


@dataclass
class SpeakerProfile:
    """
    화자별 감정별 레퍼런스 음성 파일 경로.
    CosyVoice 전환 시 감정마다 다른 레퍼런스를 사용.
    """
    speaker_id: str
    references: Dict[str, str] = field(default_factory=dict)
    # 예: {"Neutral": "/data/reference/SPEAKER_00_Neutral.wav",
    #       "Sad": "/data/reference/SPEAKER_00_Sad.wav"}

    def get_ref(self, emotion: str) -> str:
        """감정에 맞는 레퍼런스 반환. 없으면 Neutral 반환.
        v29 (5/14): LATENTSYNC_REF_SINGLE_PER_SPEAKER=1 면 emotion 무관 best 1개만 사용
        (음색 일관성 ↑, 기계음 ↓).
        """
        import os as _os_ref
        if _os_ref.environ.get("LATENTSYNC_REF_SINGLE_PER_SPEAKER", "0") == "1":
            # emotion 무관 — Neutral 우선, 없으면 첫 번째
            if "Neutral" in self.references:
                return self.references["Neutral"]
            if self.references:
                return next(iter(self.references.values()))
            return ""
        return self.references.get(emotion) or self.references.get("Neutral", "")


# ─── 실행 컨텍스트 (Run Isolation) ──────────────────────────
# 각 run_pipeline() 호출은 고유한 run_id와 격리된 디렉터리를 받는다.
# 중간 산출물은 media/runs/{run_id}/ 하위에만 기록되고,
# 최종 output/report는 공유 디렉터리에 {run_id} suffix로 저장.

@dataclass
class RunContext:
    run_id: str
    root:   str      # media/runs/{run_id}
    chunks: str
    vocals: str
    bgm:    str
    dubbed: str
    ref:    str
    temp:   str      # demucs 등 실행별 임시 공간

    @classmethod
    def create(cls, file_name: str, explicit_id: Optional[str] = None) -> "RunContext":
        """
        새 run_id를 발급하고 격리된 작업 디렉터리를 생성한다.
        형식: YYYYMMDD_HHMMSS_{파일명축약}_{hex6}
          - 해시 6자로 동시 실행 시에도 충돌 확률 0에 수렴
          - 파일명 축약은 한/영/숫자만, 최대 10자
        """
        if explicit_id:
            run_id = explicit_id
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            short = re.sub(r'[^a-zA-Z0-9가-힣]', '', file_name)[:10] or "run"
            rand = uuid.uuid4().hex[:6]
            run_id = f"{ts}_{short}_{rand}"

        root = os.path.join(RUNS_DIR, run_id)
        ctx = cls(
            run_id=run_id,
            root=root,
            chunks=os.path.join(root, "chunks"),
            vocals=os.path.join(root, "vocals"),
            bgm=os.path.join(root, "bgm"),
            dubbed=os.path.join(root, "dubbed"),
            ref=os.path.join(root, "reference"),
            temp=os.path.join(root, "temp"),
        )
        for d in [ctx.root, ctx.chunks, ctx.vocals, ctx.bgm,
                  ctx.dubbed, ctx.ref, ctx.temp]:
            os.makedirs(d, exist_ok=True)
        return ctx

    def activate(self):
        """전역 디렉터리 상수를 이 컨텍스트로 덮어쓴다.
        같은 프로세스 내 순차 실행만 지원 (동시 실행 비지원)."""
        global CHUNKS_DIR, VOCALS_DIR, BGM_DIR, DUBBED_DIR, REF_DIR, CURRENT_RUN_ID
        CHUNKS_DIR = self.chunks
        VOCALS_DIR = self.vocals
        BGM_DIR    = self.bgm
        DUBBED_DIR = self.dubbed
        REF_DIR    = self.ref
        CURRENT_RUN_ID = self.run_id
        print(f"[Run] ID: {self.run_id}")
        print(f"[Run] 작업 공간: {self.root}")


# ─── 모델 관리 (순차 로드/언로드) ─────────────────────────────
# GPU 16GB 환경에서 모든 모델을 동시에 올릴 수 없으므로
# 파이프라인 단계별로 필요한 모델만 로드하고 끝나면 해제.

_diarization_model  = None
_emotion_model      = None
_cosy_model         = None   # CosyVoice3
_google_translator  = None
_mos_evaluator      = None   # MOS 품질 평가
_vad_model          = None   # VAD 모델 전역 변수

# ASR은 venv_asr에서 subprocess로 실행 (transformers 4.57.6 필요)
ASR_VENV_PYTHON = "/opt/venv_asr/bin/python"
ASR_WORKER_PATH = os.path.join(BASE_DIR, "asr_worker.py")

# LatentSync는 venv_lipsync에서 subprocess로 실행 (PEFT + 격리된 transformers 4.48.0)
LATENT_SYNC_DIR = "/opt/LatentSync"
LATENT_SYNC_PYTHON = "/opt/venv_lipsync/bin/python"
LATENT_SYNC_CKPT = "/opt/LatentSync/checkpoints/latentsync_unet.pt"  # 베이스 (resolve_lipsync_ckpt에서 lang별 자동 override)


def _unload(name: str):
    """GPU 모델을 메모리에서 해제."""
    global _diarization_model, _emotion_model, _cosy_model, _mos_evaluator

    model_map = {
        "diarization": "_diarization_model",
        "emotion": "_emotion_model",
        "cosy": "_cosy_model",
        "mos": "_mos_evaluator",
        "vad": "_vad_model",
    
    }
    var_name = model_map.get(name)
    if not var_name:
        return

    obj = globals().get(var_name)
    if obj is not None:
        if hasattr(obj, 'unload'):
            obj.unload()
        else:
            del obj
        globals()[var_name] = None
        if name == "diarization":
            _diarization_model = None
        elif name == "emotion":
            _emotion_model = None
        elif name == "cosy":
            _cosy_model = None
        elif name == "mos":
            _mos_evaluator = None
        elif name == "vad":          # <--- [추가]
            _vad_model = None        # <--- [추가]

        import gc
        gc.collect()                 # 파이썬 가비지 컬렉터 강제 실행
        torch.cuda.empty_cache()
        torch.cuda.synchronize()     # 🔥 추가: GPU가 메모리를 완전히 뱉을 때까지 대기
        print(f"[Memory] {name} 모델 해제 완료 🗑️")


def load_mos():
    """MOS 품질 평가 모델 로드."""
    global _mos_evaluator
    if _mos_evaluator is not None:
        return

    if not os.path.exists(MOS_CHECKPOINT):
        print(f"[Load] MOS 모델 없음: {MOS_CHECKPOINT} → MOS 평가 스킵")
        return

    print("[Load] MOS 평가 모델 로딩...")
    try:
        from mos_evaluator import MOSEvaluator
        _mos_evaluator = MOSEvaluator(MOS_CHECKPOINT, threshold=3.5)
        print("[Load] MOS 평가 모델 로드 완료 ✅")
    except Exception as e:
        print(f"[Load] MOS 로드 실패: {e} ⚠️")


def load_diarization():
    """pyannote 화자 분리 로드."""
    global _diarization_model
    if _diarization_model is not None:
        return

    if not HF_TOKEN:
        print("[Load] HF_TOKEN 없음 → pyannote 스킵 ⚠️")
        return

    print("[Load] pyannote speaker-diarization 로딩...")
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        _diarization_model = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=HF_TOKEN,
        )
        _diarization_model.to(torch.device(DEVICE))
        print("[Load] pyannote 로드 완료 ✅")
    except Exception as e:
        print(f"[Load] pyannote 로드 실패: {e} ⚠️")


def load_emotion():
    """emotion2vec+ large 로드."""
    global _emotion_model
    if _emotion_model is not None:
        return

    print("[Load] emotion2vec+ large 로딩...")
    try:
        from funasr import AutoModel
        _emotion_model = AutoModel(
            model="iic/emotion2vec_plus_large",
            device=DEVICE,
            disable_update=True,
        )
        print("[Load] emotion2vec+ 로드 완료 ✅")
    except Exception as e:
        print(f"[Load] emotion2vec+ 로드 실패: {e} ⚠️")

# ─────────────────────────────────────────────────────────────
# Silero VAD로 교체 — 기존 load_vad / apply_vad_filter 대체
# ─────────────────────────────────────────────────────────────
# pip install silero-vad

def load_vad():
    """Silero VAD v5 로드 (다국어, 경량)."""
    global _vad_model
    if _vad_model is not None:
        return

    print("[Load] Silero VAD v5 로딩...")
    try:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad(onnx=False)  # PyTorch 버전
        # GPU로 올리려면: _vad_model = _vad_model.to(DEVICE)
        # 근데 너무 가벼워서 CPU가 오히려 오버헤드 적음
        print("[Load] Silero VAD 로드 완료 ✅")
    except Exception as e:
        print(f"[Load] Silero VAD 로드 실패: {e} ⚠️")
        _vad_model = None


def apply_vad_filter(vocals_path: str) -> str:
    """
    Silero VAD로 발화 구간만 살리고 나머지는 묵음 처리.
    16kHz로 리샘플링 후 VAD 돌리고, 원본 샘플레이트로 다시 매핑.
    """
    if _vad_model is None:
        return vocals_path

    try:
        from silero_vad import get_speech_timestamps, read_audio

        # Silero는 16kHz 필수
        wav_16k = read_audio(vocals_path, sampling_rate=16000)

        speech_ts = get_speech_timestamps(
            wav_16k,
            _vad_model,
            sampling_rate=16000,
            threshold=0.5,              # 0.3~0.6 튜닝 가능 (낮출수록 관대)
            min_speech_duration_ms=250, # 250ms 미만 발화는 무시 (숨/클릭)
            min_silence_duration_ms=100,# 100ms 이상 침묵으로 구간 분리
            speech_pad_ms=30,           # 구간 앞뒤 30ms 패딩 (단어 끝부분 보호)
            return_seconds=True,
        )

        if not speech_ts:
            print("[VAD] 발화 구간 없음 → 원본 유지")
            return vocals_path

        # 원본 파일 로드 (샘플레이트 유지)
        audio, sr = sf.read(vocals_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        clean_audio = np.zeros_like(audio)
        for seg in speech_ts:
            s = int(seg['start'] * sr)
            e = int(seg['end']   * sr)
            clean_audio[s:e] = audio[s:e]

        # v135+: peak normalize clean vocals (조용한 segment 음량 균일화 → diarize 정확도 향상)
        # 사용자 보고: "How hard can be" 68s 소리 작아서 잘 안 잡힘.
        # 환경변수: LATENTSYNC_VOCALS_NORMALIZE (default 1, 끄려면 0)
        _normalize = os.environ.get("LATENTSYNC_VOCALS_NORMALIZE", "1") == "1"
        if _normalize:
            peak = float(np.abs(clean_audio).max())
            if peak > 1e-6 and peak < 0.7:
                gain = min(0.9 / peak, 5.0)  # 최대 5x boost (extreme noise 회피)
                clean_audio = np.clip(clean_audio * gain, -1.0, 1.0)
                print(f"[VAD] peak normalize ×{gain:.2f} ({peak:.3f}→{min(0.9, gain*peak):.2f})")
            # 추가: per-segment RMS-based gain (조용한 발화 구간 더 균일화)
            _per_seg_norm = os.environ.get("LATENTSYNC_VOCALS_SEG_NORM", "1") == "1"
            if _per_seg_norm:
                from numpy import sqrt as _sqrt
                target_rms = 0.10  # 평균 RMS 목표
                for seg in speech_ts:
                    s = int(seg['start'] * sr); e = int(seg['end'] * sr)
                    chunk = clean_audio[s:e]
                    if len(chunk) == 0:
                        continue
                    rms = float(_sqrt(np.mean(chunk * chunk)))
                    if rms < 1e-6:
                        continue
                    seg_gain = min(target_rms / rms, 3.0)  # 최대 3x boost per segment
                    if seg_gain > 1.2:  # 20% 이상 boost 필요할 때만
                        clean_audio[s:e] = np.clip(chunk * seg_gain, -1.0, 1.0)

        clean_path = vocals_path.replace("_vocals.wav", "_clean_vocals.wav")
        sf.write(clean_path, clean_audio, sr)

        total_speech = sum(s['end'] - s['start'] for s in speech_ts)
        print(f"[VAD] 발화 {len(speech_ts)}개 구간, 총 {total_speech:.1f}s 보존")
        return clean_path

    except Exception as e:
        print(f"[VAD] Silero 처리 실패: {e} → 원본 유지")
        return vocals_path

def _fix_modelscope_symlinks():
    """
    Windows NTFS에서 modelscope 심링크가 깨지는 문제 자동 수정.
    '0.5B' → '0___5B' 같은 경로 변환이 발생하면
    원본 폴더를 깨진 경로로 복사해준다.
    """
    import glob
    cache_dir = os.environ.get("MODELSCOPE_CACHE", "/root/.cache/modelscope")
    hub_dir = os.path.join(cache_dir, "hub", "FunAudioLLM")
    if not os.path.isdir(hub_dir):
        return

    for entry in os.listdir(hub_dir):
        if "___" not in entry:
            continue
        broken_path = os.path.join(hub_dir, entry)
        # '0___5B' → '0.5B' 로 원본 이름 복원
        original_name = entry.replace("___", ".")
        original_path = os.path.join(hub_dir, original_name)

        if not os.path.isdir(original_path):
            continue

        # 깨진 폴더가 비어있거나 yaml이 없으면 원본에서 복사
        yaml_check = os.path.join(broken_path, "cosyvoice3.yaml")
        if not os.path.exists(yaml_check):
            print(f"[Fix] 심링크 수정: {entry} ← {original_name}")
            shutil.rmtree(broken_path, ignore_errors=True)
            shutil.copytree(original_path, broken_path)
            print(f"[Fix] 복사 완료 ✅")


def load_cosy():
    global _cosy_model
    if _cosy_model is not None:
        return

    _fix_modelscope_symlinks()

    print("[Load] CosyVoice3 (Fun-CosyVoice3-0.5B-2512) 로딩...")
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
        import inspect
        init_params = inspect.signature(CosyVoice3.__init__).parameters
        kwargs = {}
        if "load_jit" in init_params:
            kwargs["load_jit"] = False
        if "load_onnx" in init_params:
            kwargs["load_onnx"] = False
        if "load_trt" in init_params:
            kwargs["load_trt"] = False

        # modelscope 경로 해석 우회 → 실제 다운로드 폴더를 직접 지정
        cache_dir = os.environ.get("MODELSCOPE_CACHE", "/root/.cache/modelscope")
        local_model_dir = os.path.join(
            cache_dir, "hub", "FunAudioLLM", "Fun-CosyVoice3-0.5B-2512"
        )

        if os.path.isdir(local_model_dir):
            print(f"[Load] 로컬 캐시 사용: {local_model_dir}")
            _cosy_model = CosyVoice3(local_model_dir, **kwargs)
        else:
            print("[Load] 로컬 캐시 없음 → modelscope에서 다운로드")
            _cosy_model = CosyVoice3(
                "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", **kwargs
            )

        print("[Load] CosyVoice3 로드 완료 ✅")
    except Exception as e:
        print(f"[Load] CosyVoice3 로드 실패: {e} ⚠️")
        raise RuntimeError(f"CosyVoice3 로드 필수인데 실패: {e}")


def load_translator():
    """번역기 초기화 — VectorEngine LLM 또는 Deep Translator 폴백."""
    global _google_translator
    if _google_translator is not None:
        return

    api_key = os.environ.get("VECTORENGINE_API_KEY", "")
    if api_key:
        print("[Load] VectorEngine LLM 번역 초기화 완료 ✅")
        _google_translator = "vectorengine"
        return

    try:
        from deep_translator import GoogleTranslator
        _google_translator = GoogleTranslator
        print("[Load] Deep Translator 폴백 초기화 완료 ✅")
    except Exception as e:
        print(f"[Load] 번역기 초기화 실패: {e} ⚠️")


# ─── Step 1: 영상 분할 ────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    """ffprobe로 비디오 길이(초) 추출. 실패 시 0.0 반환.
    🔥 수정 N에 필요: 마지막 세그먼트 TTS가 비디오 끝 넘지 않게 하기 위한 절대 경계."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[Duration] ffprobe 실패 ({video_path}): {e}")
        return 0.0


def split_video(video_path: str, file_name: str, segment_time: int = 300) -> List[str]:
    """
    FFmpeg로 영상을 segment_time 초 단위 청크로 분할.

    INPUT:
      video_path   : str  — /data/input/movie.mp4
      file_name    : str  — "movie" (확장자 없음)
      segment_time : int  — 300 (초, 기본 5분)

    OUTPUT:
      chunk_paths  : List[str] — ["/data/chunks/movie_chunk_000.mp4", ...]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"입력 파일 없음: {video_path}")

    output_pattern = os.path.join(CHUNKS_DIR, f"{file_name}_chunk_%03d.mp4")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-c", "copy", "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern, "-y"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg split 실패:\n{result.stderr}")

    chunks = sorted([
        os.path.join(CHUNKS_DIR, f)
        for f in os.listdir(CHUNKS_DIR)
        if f.startswith(file_name) and f.endswith(".mp4") and "final" not in f
    ])
    print(f"[Split] {len(chunks)}개 청크 생성 완료")
    return chunks


# ─── Step 2: 음원 분리 ────────────────────────────────────────

def separate_audio(chunk_path: str) -> Tuple[str, str]:
    """
    BS-Roformer로 청크에서 목소리(vocals)와 배경음(BGM) 분리.

    Why BS-Roformer over Demucs htdemucs_ft:
      - htdemucs_ft: SDR ~9.5 (음악 vocals 학습, 영화 OOD)
      - BS-Roformer (model_bs_roformer_ep_317_sdr_12.9755): SDR 12.97
      - test3 검증: htdemucs 1명 vs BS-Roformer 3명 화자 detect (quiet 화자 보존)

    INPUT:
      chunk_path  : str — /data/chunks/movie_chunk_000.mp4
    OUTPUT:
      vocals_path : str — /data/vocals/movie_chunk_000_vocals.wav
      bgm_path    : str — /data/bgm/movie_chunk_000_bgm.wav
    """
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"청크 파일 없음: {chunk_path}")

    chunk_name = os.path.basename(chunk_path).replace(".mp4", "")
    if CURRENT_RUN_ID:
        temp_base = os.path.join(RUNS_DIR, CURRENT_RUN_ID, "temp")
    else:
        temp_base = tempfile.gettempdir()
    out_dir = os.path.join(temp_base, "bsroformer", chunk_name)
    os.makedirs(out_dir, exist_ok=True)

    # === SEP_FAST_PATCH (v28): content_type별 분기 ===
    # 환경변수 SEP_FAST=1 또는 lecture/news/auto/단순 영상이면 htdemucs (-60초)
    # drama/movie 등 BGM 강한 영상은 BS-Roformer 유지 (default)
    # v24 (5/14): LATENTSYNC_SEP_ENSEMBLE=1 → BS-RoFormer + MDX23C Subtractive Ensemble
    #             팀원 PPT 검증된 형식, 음원 품질 향상 (vocal leak 감소)
    use_fast = os.environ.get("SEP_FAST", "0") == "1"
    use_ensemble = os.environ.get("LATENTSYNC_SEP_ENSEMBLE", "0") == "1"

    def _run_separator(model_filename, out_subdir):
        """단일 모델로 separation 실행, vocals/bgm 경로 반환."""
        sub_dir = os.path.join(out_dir, out_subdir)
        os.makedirs(sub_dir, exist_ok=True)
        sep_script = (
            "import os; "
            "from audio_separator.separator import Separator; "
            "sep = Separator(output_dir=os.environ['OUT_DIR'], "
            "model_file_dir='/workspace/media/model_cache/audio_separator', "
            "log_level=30, use_autocast=True); "
            f"sep.load_model(model_filename='{model_filename}'); "
            "sep.separate(os.environ['INPUT_PATH'])"
        )
        r = subprocess.run(
            ["/opt/venv_lipsync/bin/python", "-c", sep_script],
            capture_output=True, text=True,
            env={**os.environ, "OUT_DIR": sub_dir, "INPUT_PATH": chunk_path,
                 "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}
        )
        if r.returncode != 0:
            raise RuntimeError(f"{model_filename} 실패:\n{r.stderr}")
        v_src = b_src = None
        for f in os.listdir(sub_dir):
            if "(Vocals)" in f:
                v_src = os.path.join(sub_dir, f)
            elif "(Instrumental)" in f or "(Other)" in f:
                b_src = os.path.join(sub_dir, f)
        if not v_src:
            raise FileNotFoundError(f"{model_filename} vocals 누락: {os.listdir(sub_dir)}")
        return v_src, b_src

    vocals_dst = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
    bgm_dst    = os.path.join(BGM_DIR,    f"{chunk_name}_bgm.wav")

    if use_ensemble:
        # v26 (5/14): BS-RoFormer vocals + subtractive bgm (사용자 선택 Option A)
        # 이전 v25 max-magnitude ensemble 도 SPEAKER_05 약화 → 5명 detect 문제.
        # 가장 안전한 조합:
        #   - vocals: BS-RoFormer 단독 (v4 6명 detect 동일, 약한 화자 보존)
        #   - bgm: 원본 - vocals_BSR (subtractive, vocal leak 제거)
        # MDX23C 호출 안 함 → 시간 절감 (+3분 → ~5초)
        print(f"[Separate-Ensemble] v26 시작: BSR vocals + subtractive bgm")
        bsr_v, bsr_b = _run_separator(
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "bsr"
        )
        print(f"[Separate-Ensemble] BS-RoFormer 완료")
        try:
            import soundfile as sf
            import numpy as np
            # vocals = BSR (그대로)
            shutil.move(bsr_v, vocals_dst)
            print(f"[Separate-Ensemble] vocals = BSR (약한 화자 보존)")
            # subtractive bgm: 원본 - BSR vocals
            v_bsr, sr1 = sf.read(vocals_dst)
            orig_wav = os.path.join(out_dir, f"{chunk_name}_orig.wav")
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", chunk_path,
                "-ar", str(sr1), "-vn", orig_wav,
            ], check=True, capture_output=True)
            orig, sr_o = sf.read(orig_wav)
            n = min(len(orig), len(v_bsr))
            orig = orig[:n]
            v_bsr = v_bsr[:n]
            if orig.ndim != v_bsr.ndim:
                if orig.ndim == 1:
                    orig = np.stack([orig, orig], axis=-1)
                elif v_bsr.ndim == 1:
                    v_bsr = np.stack([v_bsr, v_bsr], axis=-1)
            bgm_subtract = (orig - v_bsr).astype(np.float32)
            sf.write(bgm_dst, bgm_subtract, sr1)
            print(f"[Separate-Ensemble] subtractive bgm 생성 완료 (vocal leak 제거)")
        except Exception as _e:
            print(f"[Separate-Ensemble] subtractive 실패 ({_e}) → BSR bgm 사용")
            shutil.move(bsr_b, bgm_dst)
    elif use_fast:
        # htdemucs_ft: SDR 9.5, 30초 (BS-Roformer 90초 대비 -60초)
        print(f"[Separate] SEP_FAST=1 → htdemucs_ft (-60초)")
        v_src, b_src = _run_separator("htdemucs_ft.yaml", "htdemucs")
        shutil.move(v_src, vocals_dst)
        shutil.move(b_src, bgm_dst)
    else:
        # BS-Roformer 단독 (default)
        v_src, b_src = _run_separator(
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "bsr"
        )
        shutil.move(v_src, vocals_dst)
        shutil.move(b_src, bgm_dst)

    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"[Separate-BSR] vocals: {vocals_dst}")
    print(f"[Separate-BSR] bgm:    {bgm_dst}")
    return vocals_dst, bgm_dst


# ─── Step 3: 음성 인식 + 타임스탬프 ─────────────────────────

# === ASR_DAEMON_CLIENT (v28) ===
ASR_DAEMON_URL = os.environ.get("ASR_DAEMON_URL", "http://127.0.0.1:8902")
_asr_daemon_alive: Optional[bool] = None

def _check_asr_daemon() -> bool:
    global _asr_daemon_alive
    if _asr_daemon_alive is not None:
        return _asr_daemon_alive
    try:
        import requests as _rq
        r = _rq.get(f"{ASR_DAEMON_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("model_loaded"):
            _asr_daemon_alive = True
            print(f"[ASR] daemon alive at {ASR_DAEMON_URL} ⚡")
            return True
    except Exception:
        pass
    _asr_daemon_alive = False
    return False


def transcribe(vocals_path: str, language: Optional[str] = None) -> Tuple[str, List[WordTiming]]:
    """
    Qwen3-ASR + ForcedAligner로 음성을 텍스트로 변환.
    데몬 우선 사용 (모델 로딩 30-45초 절감), 없으면 subprocess fallback.
    """
    lang_name = LANG_CODE_TO_NAME.get(language, language) if language else None

    # 데몬 우선
    if _check_asr_daemon():
        try:
            import requests as _rq
            print(f"[ASR] daemon 호출 ({os.path.basename(vocals_path)})...")
            r = _rq.post(f"{ASR_DAEMON_URL}/transcribe", json={
                "audio_path": os.path.abspath(vocals_path),
                "language": lang_name or "auto",
            }, timeout=600)
            r.raise_for_status()
            data = r.json()
            if data.get("success"):
                detected_lang = data.get("detected_language", "en") or "en"
                lang_code = next(
                    (k for k, v in LANG_CODE_TO_NAME.items() if v.lower() == detected_lang.lower()),
                    detected_lang.lower()
                )
                words = []
                for w in data.get("words", []):
                    words.append(WordTiming(
                        word=w["word"], start=float(w["start"]), end=float(w["end"])
                    ))
                for w in words[:20]:
                    print(f"  [{w.start:.2f}~{w.end:.2f}] duration={w.end-w.start:.2f}s → {w.word}")
                print(f"[ASR] 감지 언어: {lang_code}, 단어 수: {len(words)} (daemon)")
                return lang_code, words
            else:
                print(f"[ASR] daemon 실패: {data.get('error')} → subprocess fallback")
        except Exception as e:
            print(f"[ASR] daemon 호출 실패: {e} → subprocess fallback")

    print(f"[ASR] {os.path.basename(vocals_path)} 전사 중... (venv_asr subprocess)")

    cmd = [ASR_VENV_PYTHON, ASR_WORKER_PATH, vocals_path]
    if lang_name:
        cmd += ["--language", lang_name]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": ""}
    )

    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"  {line}")

    if result.returncode != 0:
        raise RuntimeError(f"ASR Worker 실패 (exit {result.returncode}):\n{result.stderr}")

    try:
        asr_output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ASR Worker JSON 파싱 실패: {e}\nstdout: {result.stdout[:500]}")

    if not asr_output:
        print("[ASR] 결과 없음")
        return "en", []

    first = asr_output[0]
    detected_lang = first.get("language", "en") or "en"
    lang_code = next(
        (k for k, v in LANG_CODE_TO_NAME.items() if v.lower() == detected_lang.lower()),
        detected_lang.lower()
    )

    words = []
    for w in first.get("words", []):
        words.append(WordTiming(
            word=w["word"],
            start=float(w["start"]),
            end=float(w["end"])
        ))
    for w in words[:20]:
        print(f"  [{w.start:.2f}~{w.end:.2f}] duration={w.end-w.start:.2f}s → {w.word}")

    print(f"[ASR] 감지 언어: {lang_code}, 단어 수: {len(words)}")
    
    # 디버깅: 비정상 긴 단어 확인
    long_words = [w for w in words if (w.end - w.start) > 3.0]
    if long_words:
        print(f"[ASR] ⚠️  3초 이상 단어 {len(long_words)}개 감지 (ForcedAligner 오염 의심)")
        for w in long_words[:5]:
            print(f"  [{w.start:.2f}~{w.end:.2f}] {w.end-w.start:.1f}s → '{w.word}'")
    
    return lang_code, words

# ─── Step 4: 화자 분리 ────────────────────────────────────────

def diarize(vocals_path: str, num_speakers: int = None) -> list:
    """
    pyannote로 누가 언제 말했는지 구분.

    INPUT:
      vocals_path : str — /data/vocals/chunk_000_vocals.wav

    OUTPUT:
      diarization : pyannote Annotation 객체
                    (itertracks()로 (turn, _, speaker) 순회 가능)
                    pyannote 없으면 None 반환
    """
    if _diarization_model is None:
        print("[Diarize] pyannote 없음 → SPEAKER_00으로 통일")
        return None

    print(f"[Diarize] {os.path.basename(vocals_path)} 화자 분리 중...")

    # === DIARIZE_DAEMON_CLIENT (v28): DiariZen daemon 우선 (모델 로딩 20-30초 절감) ===
    diarize_daemon_url = os.environ.get("DIARIZE_DAEMON_URL", "http://127.0.0.1:8903")
    try:
        import requests as _rq
        _h = _rq.get(f"{diarize_daemon_url}/health", timeout=2)
        if _h.status_code == 200 and _h.json().get("model_loaded"):
            print(f"[Diarize] daemon alive at {diarize_daemon_url} ⚡")
            r = _rq.post(f"{diarize_daemon_url}/diarize", json={
                "vocals_wav": os.path.abspath(vocals_path),
                "num_speakers": num_speakers,
            }, timeout=600)
            r.raise_for_status()
            data = r.json()
            if data.get("success") and data.get("segments"):
                from pyannote.core import Annotation, Segment
                diar = Annotation()
                for seg in data["segments"]:
                    diar[Segment(seg["start"], seg["end"])] = seg["speaker"]
                print(f"[Diarize] DiariZen daemon: {data['n_speakers']}명 detect "
                      f"({len(data['segments'])} turns)")
                return diar
            else:
                print(f"[Diarize] daemon 결과 부족 → subprocess fallback")
    except Exception as _de:
        # daemon 없거나 health 실패 → 조용히 fallback (정상 흐름)
        pass

    # === DiariZen 단독 사용 (v23 5/14: pyannote fallback 제거) ===
    # 사용자 결정: DiariZen 이 더 정확. pyannote fallback 안 함.
    # subprocess로 venv_diarizen 호출 (sm_120 호환 환경변수 포함)
    diarizen_worker = "/workspace/scripts/diarize_worker_diarizen.py"
    diarizen_python = "/opt/venv_diarizen/bin/python"
    if os.path.exists(diarizen_worker) and os.path.exists(diarizen_python):
        try:
            cmd = [diarizen_python, diarizen_worker, vocals_path]
            if num_speakers:
                cmd.extend(["--num-speakers", str(num_speakers)])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and r.stdout.strip():
                json_line = r.stdout.strip().split("\n")[-1]
                data = json.loads(json_line)
                if "segments" in data and data["segments"]:
                    from pyannote.core import Annotation, Segment
                    diar = Annotation()
                    for seg in data["segments"]:
                        diar[Segment(seg["start"], seg["end"])] = seg["speaker"]
                    print(f"[Diarize] DiariZen 사용: {data['n_speakers']}명 detect "
                          f"({len(data['segments'])} turns)")
                    return diar
                else:
                    print(f"[Diarize] ⚠️ DiariZen 결과 비어있음: {data}")
            else:
                err = (r.stderr or r.stdout)[:500]
                print(f"[Diarize] ❌ DiariZen 실패: {err}")
        except Exception as e:
            print(f"[Diarize] ❌ DiariZen 호출 실패: {e}")
    else:
        print(f"[Diarize] ❌ DiariZen worker/venv 없음: {diarizen_worker}")

    # DiariZen 실패 시 → SPEAKER_00 단일 화자로 fallback (pyannote 사용 X)
    print(f"[Diarize] DiariZen 실패 → SPEAKER_00 단일 화자 처리")
    return None


def _get_speaker_at(diarization, time: float) -> str:
    """특정 시간에 말하는 화자 반환. pyannote 없으면 SPEAKER_00.

    SPEAKER_UNK 방지: 정확 매칭 실패 시 가장 가까운 turn의 화자 할당.
    """
    if diarization is None:
        return "SPEAKER_00"
    # 1차: 정확 시간 매칭
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= time <= turn.end:
            return speaker
    # 2차: 가장 가까운 turn 찾기 (UNK 방지)
    best_speaker = None
    best_dist = float("inf")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # 거리 = time과 turn 중간점의 차이
        if time < turn.start:
            d = turn.start - time
        elif time > turn.end:
            d = time - turn.end
        else:
            d = 0
        if d < best_dist:
            best_dist = d
            best_speaker = speaker
    return best_speaker if best_speaker else "SPEAKER_00"


# ───────────────────────────────────────────────────────────────────
# 🔥 화자 분리 후처리 (ECAPA-TDNN centroid 기반)
# ───────────────────────────────────────────────────────────────────
# 문제:
#   pyannote가 같은 화자의 톤 변화(예: 강조/속삭임)를 다른 화자로 과분할.
#   짧은 segment(<1s)의 boundary 오류로 화자 라벨 잘못 붙음.
#
# 해결:
#   1) 모든 turn에서 ECAPA-TDNN(speechbrain) 임베딩 추출
#   2) 화자별 centroid 계산 (긴 turn만 사용 → centroid 안정성 ↑)
#   3) centroid 거리가 가까운 화자 쌍 병합 (over-segmentation 해결)
#   4) 짧은 turn을 가장 가까운 centroid로 재할당 (boundary 정확도 ↑)
#   5) 결과를 새로운 Annotation으로 반환
# ───────────────────────────────────────────────────────────────────

_ecapa_model = None


def _load_ecapa():
    """SpeechBrain ECAPA-TDNN 화자 임베딩 모델 로드 (1회)."""
    global _ecapa_model
    if _ecapa_model is not None:
        return _ecapa_model
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        _ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/workspace/media/model_cache/speechbrain/ecapa",
            run_opts={"device": DEVICE},
        )
        print("[Diarize] ECAPA-TDNN 임베딩 모델 로드 ✅")
        return _ecapa_model
    except Exception as e:
        print(f"[Diarize] ECAPA 로드 실패 ({type(e).__name__}: {e}) → 후처리 스킵")
        return None


def _ecapa_embedding(audio: np.ndarray, sr: int, model) -> Optional[np.ndarray]:
    """단일 오디오 청크의 ECAPA 임베딩 (192-dim, L2-normalized)."""
    try:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            # ECAPA는 16kHz 고정. resample.
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(audio) < sr * 0.4:  # 최소 0.4초
            return None
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(wav).squeeze().cpu().numpy()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return None
        return emb / norm
    except Exception as e:
        print(f"[Diarize] embedding 실패: {e}")
        return None


def post_process_diarization(
    diarization,
    vocals_path: str,
    min_dur_for_centroid: float = 0.5,  # 2.0 → 0.5 (짧은 화자도 centroid 포함, 여자 1.2s 발화 보존)
    merge_threshold: float = 0.50,  # 0.70 → 0.50 (drama over-detect 적극 병합. v8에서 0.55는 SPEAKER_03↔05 cos=0.54 못 잡음)
    short_turn_threshold: float = 0.8,
    min_segment_duration: float = 0.5,
):
    """화자 분리 결과를 ECAPA centroid 기반으로 정제.

    INPUT:
      diarization            : pyannote Annotation (또는 None)
      vocals_path            : 화자 분리에 쓴 vocals.wav
      min_dur_for_centroid   : centroid 계산에 쓸 turn 최소 길이 (안정성 ↑)
      merge_threshold        : centroid cosine similarity가 이 이상이면 동일 화자로 병합
      short_turn_threshold   : 이보다 짧은 turn은 centroid 기반 재할당
      min_segment_duration   : 인접 같은 화자 turn 병합 후 이 이하면 흡수

    OUTPUT:
      동일한 Annotation 인터페이스 (.itertracks(yield_label=True))
      또는 None (실패시 원본 그대로 사용 권장)
    """
    if diarization is None:
        print("[Diarize] 화자 분리 결과 없음 → 후처리 스킵")
        return None, {}

    n_speakers = len(set(s for _, _, s in diarization.itertracks(yield_label=True)))
    apply_merge = n_speakers >= 3
    if not apply_merge:
        print(f"[Diarize] DiariZen {n_speakers}명 detect → 병합 skip, outlier 감지만 적용")

    model = _load_ecapa()
    if model is None:
        return diarization, {}  # ECAPA 로드 실패 시 원본 반환

    # 1) 오디오 로드
    try:
        audio, sr = sf.read(vocals_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    except Exception as e:
        print(f"[Diarize] vocals 로드 실패: {e} → 후처리 스킵")
        return diarization, {}

    # 2) 모든 turn 수집 + 임베딩 추출
    turns = []  # list of (start, end, orig_speaker, embedding)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        s = max(0.0, float(turn.start))
        e = min(len(audio) / sr, float(turn.end))
        if e - s < 0.3:
            turns.append((s, e, speaker, None))
            continue
        s_idx, e_idx = int(s * sr), int(e * sr)
        emb = _ecapa_embedding(audio[s_idx:e_idx], sr, model)
        turns.append((s, e, speaker, emb))

    n_turns = len(turns)
    n_with_emb = sum(1 for t in turns if t[3] is not None)
    print(f"[Diarize] post: {n_turns} turns, embedding 추출 {n_with_emb}개")

    if n_with_emb < 2:
        print("[Diarize] 임베딩 부족 → 후처리 스킵")
        return diarization, {}

    # 3) 화자별 centroid (긴 turn만 사용)
    speaker_embs = {}
    for s, e, spk, emb in turns:
        if emb is None:
            continue
        if (e - s) < min_dur_for_centroid:
            continue
        speaker_embs.setdefault(spk, []).append(emb)

    # 긴 turn이 부족한 화자는 모든 임베딩 사용
    for s, e, spk, emb in turns:
        if emb is None or spk in speaker_embs:
            continue
        speaker_embs.setdefault(spk, []).append(emb)

    if not speaker_embs:
        return diarization, {}

    centroids = {spk: np.mean(embs, axis=0) for spk, embs in speaker_embs.items()}
    # L2 normalize centroids
    for spk in centroids:
        norm = np.linalg.norm(centroids[spk])
        if norm > 1e-8:
            centroids[spk] = centroids[spk] / norm

    speaker_list = sorted(centroids.keys())
    print(f"[Diarize] pyannote 검출 화자: {len(speaker_list)}명 {speaker_list}")

    # 4) 화자 병합 (centroid 거리 가까운 쌍)
    #    Union-Find 방식으로 병합 그룹 형성
    parent = {spk: spk for spk in speaker_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # 사전식 작은 쪽으로 통일 (재현성)
            parent[max(ra, rb)] = min(ra, rb)

    merged_pairs = []
    # ≤2명도 cosine 매우 높으면 (>=0.65) 병합 — DiariZen이 같은 화자를 다른 ID로 over-detect 처리
    # 여자 like ID 8 (max_sim 0.33)은 안 합쳐짐, 같은 남자 톤 변화 (cosine 0.65+) 합쳐짐
    effective_threshold = merge_threshold if apply_merge else 0.65
    for i in range(len(speaker_list)):
        for j in range(i + 1, len(speaker_list)):
            a, b = speaker_list[i], speaker_list[j]
            sim = float(np.dot(centroids[a], centroids[b]))
            if sim >= effective_threshold:
                union(a, b)
                merged_pairs.append((a, b, sim))

    # 매핑: orig_speaker → canonical_speaker
    speaker_map = {spk: find(spk) for spk in speaker_list}
    merged_speakers = sorted(set(speaker_map.values()))
    n_merged = len(speaker_list) - len(merged_speakers)
    if merged_pairs:
        print(f"[Diarize] 병합된 쌍 ({n_merged}개 화자 사라짐):")
        for a, b, sim in merged_pairs:
            print(f"  {a} ↔ {b}  (cosine={sim:.3f})")
    print(f"[Diarize] 최종 화자 수: {len(merged_speakers)}명 {merged_speakers}")

    # 병합 후 centroid 재계산
    final_centroids = {}
    for canonical in merged_speakers:
        embs = []
        for orig, c in speaker_map.items():
            if c == canonical and orig in speaker_embs:
                embs.extend(speaker_embs[orig])
        if embs:
            mean = np.mean(embs, axis=0)
            n = np.linalg.norm(mean)
            final_centroids[canonical] = mean / n if n > 1e-8 else mean

    # 5) 짧은 turn 재할당 + 모든 turn에 canonical 라벨 부여
    new_turns = []  # list of (start, end, canonical_speaker)
    for s, e, orig_spk, emb in turns:
        canonical = speaker_map.get(orig_spk, orig_spk)
        # 짧은 turn이고 임베딩 있으면 centroid 거리로 재할당
        if (e - s) < short_turn_threshold and emb is not None and final_centroids:
            sims = {c: float(np.dot(emb, cv)) for c, cv in final_centroids.items()}
            best = max(sims, key=sims.get)
            if sims[best] > sims.get(canonical, -1):
                canonical = best
        new_turns.append((s, e, canonical))

    # 6) 인접 같은 화자 turn 병합
    # v27 (5/15): 사용자 분석 — "분할 메커니즘이 같은 발화를 쪼갬"
    # 0.3s 하드코딩 → 환경변수화. default 1.0s로 키워서 같은 화자 연속 발화 보호.
    # 안전: 같은 화자만 병합하므로 다른 의미 발화 섞임 위험 없음.
    same_spk_gap = float(os.environ.get("LATENTSYNC_POSTPROC_SAME_SPK_GAP", "1.0"))
    new_turns.sort(key=lambda t: t[0])
    merged_turns = []
    for s, e, spk in new_turns:
        if merged_turns and merged_turns[-1][2] == spk and (s - merged_turns[-1][1]) < same_spk_gap:
            ps, pe, pspk = merged_turns.pop()
            merged_turns.append((ps, max(pe, e), pspk))
        else:
            merged_turns.append((s, e, spk))

    # 7) 너무 짧은 turn(<min_segment_duration)을 인접 turn에 흡수
    if len(merged_turns) > 1:
        cleaned = []
        for s, e, spk in merged_turns:
            if (e - s) < min_segment_duration and cleaned:
                # 이전 turn에 흡수
                ps, pe, pspk = cleaned.pop()
                cleaned.append((ps, e, pspk))  # 라벨은 이전 화자로
            else:
                cleaned.append((s, e, spk))
        merged_turns = cleaned

    print(f"[Diarize] turn 수: {n_turns} → {len(merged_turns)} (병합/정제)")

    # === OUTLIER DETECTION (짧은 다른 화자 발화 회수) ===
    # v23 (5/14): LATENTSYNC_OUTLIER_FAR_THRESH env var 로 조정 가능
    # 낮을수록 outlier 적게 (over-segmentation 회피)
    # v21 (5/15): LATENTSYNC_OUTLIER_OFF=1 → 검출 자체 비활성 (가짜 SPEAKER_99/98/97 회귀 fix)
    if os.environ.get("LATENTSYNC_OUTLIER_OFF", "0") == "1":
        print("[Diarize] Outlier 검출 비활성화 (LATENTSYNC_OUTLIER_OFF=1)")
    else:
        OUTLIER_FAR_THRESH = float(os.environ.get("LATENTSYNC_OUTLIER_FAR_THRESH", "0.40"))
        OUTLIER_MIN_DUR = 0.5      # 이 이상 segment만 검사
        new_speaker_idx = 99
        outlier_count = 0
        for i in range(len(merged_turns)):
            s, e, spk = merged_turns[i]
            if (e - s) < OUTLIER_MIN_DUR:
                continue
            s_idx, e_idx = int(s * sr), int(e * sr)
            emb = _ecapa_embedding(audio[s_idx:e_idx], sr, model)
            if emb is None or not final_centroids:
                continue
            # 모든 centroid에서의 거리 (cosine sim, 클수록 가까움)
            max_sim = max(float(np.dot(emb, cv)) for cv in final_centroids.values())
            if max_sim < OUTLIER_FAR_THRESH:
                new_spk = f"SPEAKER_{new_speaker_idx:02d}"
                print(f"  [Outlier] turn{i} [{s:.2f}~{e:.2f}] {spk}→{new_spk} (max_sim={max_sim:.2f} < {OUTLIER_FAR_THRESH})")
                merged_turns[i] = (s, e, new_spk)
                # outlier 화자도 centroid bank에 등록 (segment_refiner에서 재할당 가능)
                final_centroids[new_spk] = emb
                outlier_count += 1
                new_speaker_idx -= 1
        if outlier_count:
            print(f"[Diarize] Outlier 재분류: {outlier_count}개 segment → 새 화자")

    # 8) 새로운 Annotation 객체 구성 (pyannote 호환)
    try:
        from pyannote.core import Annotation, Segment
        new_anno = Annotation(uri=getattr(diarization, "uri", None))
        for s, e, spk in merged_turns:
            new_anno[Segment(s, e)] = spk
        return new_anno, final_centroids
    except Exception as e:
        print(f"[Diarize] Annotation 재구성 실패: {e} → 원본 반환")
        return diarization, final_centroids

# ─── 오염 탐지 보조 함수 (build_segments 위에 추가) ───────────
def _detect_contaminated_words(
    words: List[WordTiming],
    vocals_path: str,
) -> set:
    """
    ForcedAligner/ASR 환각 의심 단어 인덱스 반환.
      1) duration > 3.0s  또는  duration < 0.02s
      2) 3개 이상 연속 단어의 duration/gap이 거의 동일(균등분할 fallback)
      3) 해당 구간의 RMS 에너지 ≈ 0 (VAD 이후 묵음 구간의 ASR 환각)
    """
    contaminated = set()
    n = len(words)
    if n == 0:
        return contaminated
 
    # 1) duration 양 극단
    for i, w in enumerate(words):
        d = w.end - w.start
        if d > 3.0 or d < 0.02:
            contaminated.add(i)
 
    # 2) 균등분할 패턴: 3개 연속 duration/gap 동일 + duration > 1.0s
    for i in range(n - 2):
        d0 = words[i].end     - words[i].start
        d1 = words[i + 1].end - words[i + 1].start
        d2 = words[i + 2].end - words[i + 2].start
        g0 = words[i + 1].start - words[i].end
        g1 = words[i + 2].start - words[i + 1].end
        if (d0 > 1.0
            and abs(d0 - d1) < 0.05 and abs(d1 - d2) < 0.05
            and abs(g0 - g1) < 0.05 and g0 > 0.3):
            contaminated.update([i, i + 1, i + 2])
 
    # 3) RMS 에너지 검증 (VAD 이후 묵음 구간 = 확실한 환각)
    try:
        audio, sr = sf.read(vocals_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
 
        for i, w in enumerate(words):
            if i in contaminated:
                continue
            s = int(w.start * sr)
            e = int(w.end   * sr)
            if e <= s or s >= len(audio):
                contaminated.add(i)
                continue
            chunk = audio[s:min(e, len(audio))]
            if len(chunk) == 0:
                contaminated.add(i)
                continue
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            if rms < 0.005:          # VAD로 0 채워진 구간 허들
                contaminated.add(i)
    except Exception as e:
        print(f"[Segments] RMS 검증 실패 (계속 진행): {e}")
 
    return contaminated
    
# ─── Step 5: 세그먼트 조합 (문장 단위) ──────────────────────
# 🔥 수정 P: 시간/단어수 기반 기계적 분할 → LLM 구두점 복원으로 문장 단위 분할.
#     이점: 완결된 문장으로 TTS/번역해서 "문장 중간 끊김/반복" 사라짐.
#     폴백: LLM 실패 시 gap 기반(0.4s)으로 분할.
# ────────────────────────────────────────────────────────────

# 문장 기준 파라미터 (다국어 확장 시 language별 값 필요 — 주석 참고)
SENT_MIN_DURATION = 1.5    # 초. 이보다 짧은 문장은 앞 문장과 병합
SENT_MAX_DURATION = 15.0   # 초. 이보다 긴 문장은 쉼표/접속사/gap으로 보조 분할
SENT_MERGE_CAP    = 12.0   # 초. 짧은 문장 병합 상한
# 문장 종결 문자 (다국어 확장: LANG_SENTENCE_END 테이블 참고)
SENTENCE_END_CHARS = ".?!。？！"
SENTENCE_PAUSE_CHARS = ",;:，；：、"  # 보조 분할용
# 영어 접속사(긴 문장 추가 분할 힌트) — 다국어 확장 시 언어별 리스트
ENG_CONNECTIVES = {"and", "but", "so", "because", "however", "although",
                   "while", "when", "if", "then", "also"}

# ─── 콘텐츠 타입별 감정 정책 ────────────────────────────────
# CosyVoice3의 Happy instruction이 긴 문장에서 토큰 반복 환각을 유발하는 문제,
# 그리고 emotion2vec+가 강연 톤을 Angry/Sad로 과잉 감지하는 문제를 동시에 해결.
#
#   passthrough:  emotion2vec+ 원본 감지 그대로 사용 (영화/드라마용)
#   neutral_only: 모든 세그먼트를 Neutral로 고정 (강연/인터뷰/뉴스용)
#
# 다국어 확장 시에도 그대로 사용 — 언어와 무관한 정책.
EMOTION_POLICIES = {
    "auto":      "passthrough",     # 기본값: 감지 그대로
    "lecture":   "neutral_only",    # 강연: Neutral 고정
    "interview": "neutral_only",    # 인터뷰: Neutral 고정
    "news":      "neutral_only",    # 뉴스: Neutral 고정
    "movie":     "passthrough",     # 영화: 감지 그대로 (감정 표현 유지)
    "drama":     "passthrough",     # 드라마: 감지 그대로
}


def _restore_punctuation_llm(text: str, src_lang_name: str = "English") -> Optional[str]:
    """
    LLM에 구두점 복원 요청. 실패 시 None 반환 → 폴백 유도.
    다국어 확장: src_lang_name만 바꾸면 됨 (예: "Japanese", "Chinese").
    """
    api_key = os.environ.get("VECTORENGINE_API_KEY", "")
    if not api_key:
        return None  # 번역이 Google Translator 폴백이면 이것도 폴백

    import requests
    base_url = os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai")
    model = os.environ.get("VECTORENGINE_MODEL", "gpt-5.4")

    system = (
        f"You are a professional editor. Restore natural punctuation to the following "
        f"raw {src_lang_name} transcription. Rules:\n"
        f"- Add periods, question marks, exclamation marks at sentence ends.\n"
        f"- Add commas/semicolons at natural pauses.\n"
        f"- PRESERVE EVERY WORD EXACTLY — do not add, remove, or change words.\n"
        f"- Preserve original casing and numbers.\n"
        f"- Output ONLY the punctuated text, nothing else."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        # reasoning 모델 여유: 입력 토큰 × 3 + 베이스 4096
        "max_tokens": 4096,  # 충분 (구두점만 추가, 본문 길이 변화 거의 없음)
        "reasoning_effort": "low",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 최대 2회 시도 (timeout/일시적 네트워크 오류 대응)
    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers, json=payload,
                timeout=300,  # 600 → 300 (빨리 실패하고 폴백)
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            out = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            return out or None
        except Exception as e:
            last_err = e
            if attempt == 0:
                print(f"[Punctuate] 1차 시도 실패 ({type(e).__name__}) → 재시도")
            else:
                print(f"[Punctuate] LLM 실패 ({type(e).__name__}): {e}")
                return None
    return None


def _match_punctuated_to_words(
    punctuated: str,
    words: List["WordTiming"]
) -> List[List[int]]:
    """
    구두점 복원된 문장들을 원본 단어 인덱스 리스트로 매핑.
    반환: [[첫 문장 단어 idx들], [둘째 문장 단어 idx들], ...]
    전략:
      1) 문장 단위로 분할 (SENTENCE_END_CHARS)
      2) 각 문장에서 단어만 추출해 원본과 순서 매칭
      3) 매칭 실패 허용: 단어가 조금 달라도 (lowercase + alnum) 맞추기
    다국어 확장: 공백 기반 토큰화는 CJK에선 다른 방법 필요 (주석 참고).
    """
    if not punctuated or not words:
        return []

    # 문장 단위 분할: 종결 문자 뒤에서 자름
    import re as _re
    sent_pattern = _re.compile(f"[^{_re.escape(SENTENCE_END_CHARS)}]+"
                               f"[{_re.escape(SENTENCE_END_CHARS)}]*")
    sents = [s.strip() for s in sent_pattern.findall(punctuated) if s.strip()]
    if not sents:
        sents = [punctuated]

    def norm(s: str) -> str:
        """비교용 정규화: lowercase + alnum만."""
        return "".join(ch.lower() for ch in s if ch.isalnum())

    word_norms = [norm(w.word) for w in words]
    cursor = 0
    result = []
    for sent in sents:
        toks = [norm(t) for t in sent.split() if norm(t)]
        if not toks:
            continue
        # 첫 토큰이 word_norms[cursor:]에서 시작하는지 확인, 순차 매칭
        indices = []
        ci = cursor
        for tok in toks:
            # 현재 위치부터 ±3 범위에서 찾기 (LLM이 단어 합칠 수 있어서)
            found = -1
            for delta in range(0, 4):
                if ci + delta < len(word_norms) and word_norms[ci + delta] == tok:
                    found = ci + delta
                    break
                # 합성어 대응: "twenty-seven" → "twentyseven"
                if ci + delta + 1 < len(word_norms):
                    combined = word_norms[ci + delta] + word_norms[ci + delta + 1]
                    if combined == tok:
                        indices.append(ci + delta)
                        indices.append(ci + delta + 1)
                        ci = ci + delta + 2
                        found = -2  # 이미 처리됨
                        break
            if found >= 0:
                indices.append(found)
                ci = found + 1
            elif found == -2:
                pass  # 위에서 처리됨
            # 매칭 실패 → 건너뛰기 (LLM이 토큰 바꿨을 수 있음)
        if indices:
            # indices를 연속 구간으로 보장 (min~max 전체 포함)
            lo, hi = min(indices), max(indices)
            result.append(list(range(lo, hi + 1)))
            cursor = hi + 1

    # 마지막에 남은 단어들이 있으면 마지막 문장에 추가
    if cursor < len(words) and result:
        result[-1].extend(range(cursor, len(words)))

    return result


def _split_long_sentence(
    word_indices: List[int],
    words: List["WordTiming"],
    max_dur: float = SENT_MAX_DURATION,
) -> List[List[int]]:
    """
    긴 문장을 쉼표 → 접속사 → gap 순서로 보조 분할.
    """
    if not word_indices:
        return []

    def duration_of(idxs):
        return words[idxs[-1]].end - words[idxs[0]].start

    if duration_of(word_indices) <= max_dur:
        return [word_indices]

    # 1) 쉼표/세미콜론 등에서 자르기 (단어 말미에 해당 문자가 있는지)
    chunks = []
    current = []
    for i in word_indices:
        current.append(i)
        if words[i].word.rstrip().endswith(tuple(SENTENCE_PAUSE_CHARS)):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    # 여전히 긴 청크는 접속사 앞에서 자르기
    def split_by_connective(idxs):
        if duration_of(idxs) <= max_dur:
            return [idxs]
        result = []
        current = []
        for i in idxs:
            if current and words[i].word.lower().strip(".,?!") in ENG_CONNECTIVES:
                if duration_of(current) >= 2.0:  # 너무 짧게 자르지 않게
                    result.append(current)
                    current = []
            current.append(i)
        if current:
            result.append(current)
        return result

    refined = []
    for ch in chunks:
        refined.extend(split_by_connective(ch))

    # 여전히 긴 청크는 gap 기반 (0.3s+)
    def split_by_gap(idxs):
        if duration_of(idxs) <= max_dur or len(idxs) < 2:
            return [idxs]
        result = []
        current = [idxs[0]]
        for i in idxs[1:]:
            prev = current[-1]
            gap = words[i].start - words[prev].end
            if gap > 0.3 and duration_of(current) >= 2.0:
                result.append(current)
                current = []
            current.append(i)
        if current:
            result.append(current)
        return result

    final = []
    for ch in refined:
        final.extend(split_by_gap(ch))
    return final


def _merge_short_sentences(
    sentence_word_indices: List[List[int]],
    words: List["WordTiming"],
    min_dur: float = SENT_MIN_DURATION,
    cap: float = SENT_MERGE_CAP,
    diarization=None,
) -> List[List[int]]:
    """짧은 문장(<min_dur)은 인접 문장과 병합. 총 길이 cap 초과 금지.

    v29 (5/14): env var 로 조정 가능 + 단어 수 기반 강제 병합 추가
      LATENTSYNC_SENT_MIN_DURATION: 기본 1.5초 (이하 병합)
      LATENTSYNC_SENT_MIN_WORDS: 기본 3 단어 (미만이면 강제 병합, 짧은 발화 보존)
      LATENTSYNC_SENT_MERGE_CAP: 기본 SENT_MERGE_CAP 초 (총 길이 상한)
    사용자 피드백: 'stop being a stupid rabbit' 같은 짧은 문장이 두 개로 잘려서
    번역이 어색해지는 문제. 짧은 segment 적극 병합으로 해결.
    """
    import os as _os_ms
    _env_min_dur = float(_os_ms.environ.get("LATENTSYNC_SENT_MIN_DURATION", str(min_dur)))
    _env_min_words = int(_os_ms.environ.get("LATENTSYNC_SENT_MIN_WORDS", "3"))
    _env_cap = float(_os_ms.environ.get("LATENTSYNC_SENT_MERGE_CAP", str(cap)))
    min_dur = _env_min_dur
    cap = _env_cap

    if not sentence_word_indices:
        return []

    def dur(idxs):
        return words[idxs[-1]].end - words[idxs[0]].start

    merged = []
    for s in sentence_word_indices:
        if not s:
            continue
        # v29: 단어 수 < min_words 또는 duration < min_dur 면 병합 시도
        should_merge = (len(s) < _env_min_words) or (dur(s) < min_dur)
        if merged:
            prev_should_merge = (len(merged[-1]) < _env_min_words) or (dur(merged[-1]) < min_dur)
        else:
            prev_should_merge = False
        if merged and (should_merge or prev_should_merge):
            combined_end = words[s[-1]].end
            combined_start = words[merged[-1][0]].start
            if (combined_end - combined_start) <= cap:
                merged[-1] = merged[-1] + s
                continue
        merged.append(list(s))

    # v22 (safe): 마지막 sentence 안전 병합
    # 위험: 다른 화자/다른 의미를 무작정 병합하면 음색/맥락 회귀 발생 (사용자 피드백)
    # 조건: 같은 화자 + gap < 1.0s + 이전 sentence와 자연 연결될 때만 병합
    # diarization 없으면 보호 비활성 (안전 우선)
    import os as _os_last
    _protect_last_merge = _os_last.environ.get("LATENTSYNC_PROTECT_LAST_MERGE", "0") != "0"
    if _protect_last_merge and len(merged) >= 2 and diarization is not None:
        last = merged[-1]
        prev = merged[-2]
        last_dur = dur(last)
        last_words = len(last)
        if last_words <= 4 or last_dur < 2.0:
            # 같은 화자 확인 — last와 prev의 중간 word 시점 기준
            def _spk_at(grp_indices):
                if not grp_indices:
                    return None
                mid = words[grp_indices[len(grp_indices) // 2]]
                return _get_speaker_at(diarization, (mid.start + mid.end) / 2)
            spk_last = _spk_at(last)
            spk_prev = _spk_at(prev)
            # gap = 이전 sentence 끝 ~ 마지막 sentence 시작
            gap = words[last[0]].start - words[prev[-1]].end
            same_speaker = (spk_last == spk_prev and spk_last is not None)
            close_gap = (gap < 1.0)
            combined_end = words[last[-1]].end
            combined_start = words[prev[0]].start
            if same_speaker and close_gap and (combined_end - combined_start) <= cap * 1.5:
                merged[-2] = prev + last
                merged.pop()
                print(f"[Merge] 마지막 sentence 안전 병합 (same spk={spk_last}, gap={gap:.2f}s): {last_words} 단어")
            else:
                if not same_speaker:
                    print(f"[Merge] 마지막 sentence 병합 skip (다른 화자: {spk_prev} vs {spk_last})")
                elif not close_gap:
                    print(f"[Merge] 마지막 sentence 병합 skip (gap {gap:.2f}s 큼)")
    return merged


def _merge_same_speaker_adjacent(
    sentence_groups: List[List[int]],
    words: List["WordTiming"],
    diarization,
    max_gap: float = 1.0,
    max_merged_duration: float = 14.0,
    verbose: bool = True,
) -> List[List[int]]:
    """같은 화자의 인접 sentence를 안전하게 병합.

    v27 (5/15): 사용자 통찰 — "문장 길이 제한 때문에 SPEAKER_05가 5번으로 분할됨"
    문제 진단:
      - 우리 코드의 여러 분할 메커니즘 (post_process gap 0.3s, LLM punct, fallback
        gap 0.4s)이 같은 화자의 *연속 발화*를 *gap 0.4~1.5s에서 분할*시킴.
      - 예: SPEAKER_05 [59.20~59.84]+[60.48~69.36]+[69.92~72.00]가 한 발화인데
        gap 0.64s, 0.56s에서 3 sentences로 분리됨. SENT_MERGE_CAP=10s 라서
        합쳐도 12.8s가 cap 초과 → 다시 합쳐지지 않음.
    해결:
      - 같은 화자 + gap < max_gap + 합쳐도 max_merged_duration 이내면 무조건 병합
      - 다른 화자 boundary는 절대 침범 안 함 (안전)
      - "단어 길이 제한 때문에 다른 의미 발화 병합" 위험 회피 (화자 일치만 검사)

    Args:
        sentence_groups: 이전 단계 결과
        words: WordTiming 리스트
        diarization: pyannote Annotation (None이면 병합 안 함, 안전)
        max_gap: 이 이하 gap만 병합 후보
        max_merged_duration: 합쳐도 이 이내만

    Returns:
        병합된 sentence_groups 리스트
    """
    if diarization is None or not sentence_groups:
        return sentence_groups

    import os as _os_merge
    max_gap = float(_os_merge.environ.get("LATENTSYNC_SAME_SPK_GAP", str(max_gap)))
    max_merged_duration = float(_os_merge.environ.get("LATENTSYNC_SAME_SPK_MERGE_CAP", str(max_merged_duration)))

    def _spk(grp):
        if not grp:
            return None
        # 중간 word 시점 기준 화자
        mid_idx = grp[len(grp) // 2]
        mid_word = words[mid_idx]
        return _get_speaker_at(diarization, (mid_word.start + mid_word.end) / 2)

    merged: List[List[int]] = [list(sentence_groups[0])] if sentence_groups[0] else []
    n_merges = 0
    for grp in sentence_groups[1:]:
        if not grp:
            continue
        if not merged:
            merged.append(list(grp))
            continue
        prev = merged[-1]
        prev_spk = _spk(prev)
        cur_spk = _spk(grp)
        gap = words[grp[0]].start - words[prev[-1]].end
        combined_dur = words[grp[-1]].end - words[prev[0]].start

        # 같은 화자 + 짧은 gap + 합쳐도 cap 이내 → 안전 병합
        if (prev_spk == cur_spk and prev_spk is not None
                and gap < max_gap
                and combined_dur <= max_merged_duration):
            merged[-1] = prev + grp
            n_merges += 1
        else:
            merged.append(list(grp))

    if verbose and n_merges > 0:
        print(f"[Merge] 같은 화자 인접 sentence {n_merges}건 안전 병합 "
              f"(gap<{max_gap}s, merged_dur<{max_merged_duration}s)")
    return merged


def _split_groups_by_speaker(
    groups: List[List[int]],
    words: List["WordTiming"],
    diarization,
) -> List[List[int]]:
    """sentence group 내부에서 화자가 바뀌는 word 경계에서 강제 split.

    LLM 구두점 복원이 두 화자 발화를 한 문장으로 묶어버린 경우를 보정.
    1-word 깜빡임(A B A 패턴)은 smoothing으로 무시 → false split 방지.

    v30 (5/14): env var 옵션
      LATENTSYNC_SPLIT_BY_SPEAKER_OFF=1 — split 완전 비활성화
        (사용자 피드백: 짧은 단편 대량 발생, 모든 문제 근본)
      LATENTSYNC_SPLIT_MIN_SUB_WORDS — split 결과 sub-group 최소 단어 수 (기본 3)
        이 미만 sub-group 은 인접 sub-group 과 강제 병합
    """
    import os as _os_split
    if _os_split.environ.get("LATENTSYNC_SPLIT_BY_SPEAKER_OFF", "0") == "1":
        print("[Segments] _split_groups_by_speaker 비활성화 (LATENTSYNC_SPLIT_BY_SPEAKER_OFF=1)")
        return groups
    if diarization is None:
        return groups
    # v20: 짧은 group은 split 비활성 (마지막 토끼 split 회귀 방지)
    # LATENTSYNC_SPLIT_NO_SHORT_WORDS=N: N 단어 이하 group은 split 안 함
    # majority 통일 임계 (LATENTSYNC_SPLIT_MAJORITY_TH): 한 화자가 이 비율 이상이면 전체 통일
    _no_split_short = int(_os_split.environ.get("LATENTSYNC_SPLIT_NO_SHORT_WORDS", "6"))
    _majority_th = float(_os_split.environ.get("LATENTSYNC_SPLIT_MAJORITY_TH", "0.80"))
    result = []
    n_split = 0
    n_skipped_short = 0
    # v22 (safe): "마지막 group 절대 보호" 제거 — 같은 group 안에 두 화자 word가
    # 섞이면 절대 보호가 잘못된 화자 통합 회귀를 일으킴 (사용자 피드백).
    # 단일 화자면 어차피 split 안 일어나므로 절대 보호는 효과 없고 위험만 있음.
    # 대신 짧은 group 보호 (SPLIT_NO_SHORT_WORDS) + majority 통일 (SPLIT_MAJORITY_TH)
    # + 1/2-word smoothing 으로 word-diarization noise는 처리.
    for grp_idx, grp in enumerate(groups):
        if len(grp) < 2:
            result.append(grp)
            continue
        # v20: 짧은 group 보호 — "stop being a stupid rabbit" 같은 5-6 단어 짧은
        # 문장이 word-level diarization noise로 split되는 회귀 방지.
        if len(grp) <= _no_split_short:
            result.append(grp)
            n_skipped_short += 1
            continue
        # word별 speaker 매핑 (word 중간점 기준)
        word_spk = []
        for idx in grp:
            w = words[idx]
            t = (w.start + w.end) / 2
            word_spk.append(_get_speaker_at(diarization, t))
        # 1-word 깜빡임 smoothing: A B A → A A A
        smoothed = list(word_spk)
        for i in range(1, len(smoothed) - 1):
            if smoothed[i - 1] == smoothed[i + 1] and smoothed[i] != smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]
        # v20: 2-word 깜빡임 smoothing: A B B A → A A A A
        for i in range(1, len(smoothed) - 2):
            if (smoothed[i - 1] == smoothed[i + 2]
                and smoothed[i] != smoothed[i - 1]
                and smoothed[i + 1] != smoothed[i - 1]):
                smoothed[i] = smoothed[i - 1]
                smoothed[i + 1] = smoothed[i - 1]
        # v20: majority 통일 — 한 화자가 80%+ 면 group 전체를 그 화자로 (split 안 함)
        from collections import Counter as _Counter
        _cnt = _Counter(smoothed)
        if _cnt:
            _top_spk, _top_n = _cnt.most_common(1)[0]
            if _top_n / max(1, len(smoothed)) >= _majority_th:
                smoothed = [_top_spk] * len(smoothed)
        # speaker 변화 지점에서 split
        sub_grps = []
        cur = [grp[0]]
        prev_spk = smoothed[0]
        for i in range(1, len(grp)):
            if smoothed[i] != prev_spk:
                if cur:
                    sub_grps.append(cur)
                cur = []
                prev_spk = smoothed[i]
            cur.append(grp[i])
        if cur:
            sub_grps.append(cur)

        # v32 (5/14): split 후 같은 화자 인접 짧은 sub-group 만 병합 (다른 화자 절대 병합 X)
        # v15 사용자 피드백: 다른 화자 강제 병합 → 음색 잘못 + 기계음.
        # 같은 화자 짧은 sub 만 안전 병합 → 화자 매칭 유지.
        _min_sub_words = int(_os_split.environ.get("LATENTSYNC_SPLIT_MIN_SUB_WORDS", "0"))
        if _min_sub_words > 0 and len(sub_grps) > 1:
            def _sub_speaker(sg):
                # sub-group 의 대표 화자 = 중간 word 시점 화자
                if not sg:
                    return None
                mid_word = words[sg[len(sg) // 2]]
                t = (mid_word.start + mid_word.end) / 2
                return _get_speaker_at(diarization, t)

            merged_sub = [sub_grps[0]]
            for sg in sub_grps[1:]:
                if not sg:
                    continue
                # 같은 화자 + 어느 한쪽 짧으면 병합 (다른 화자는 절대 X)
                prev_spk = _sub_speaker(merged_sub[-1])
                cur_spk = _sub_speaker(sg)
                same_speaker = (prev_spk == cur_spk)
                if same_speaker and (len(sg) < _min_sub_words or len(merged_sub[-1]) < _min_sub_words):
                    merged_sub[-1] = merged_sub[-1] + sg
                else:
                    merged_sub.append(sg)
            sub_grps = merged_sub

        if len(sub_grps) > 1:
            n_split += 1
        result.extend(sub_grps)
    if n_split:
        print(f"[Segments] sentence-내부 화자 split: {n_split}개 그룹 → 총 {len(result)}개")
    return result


def _fallback_gap_segments(
    words: List["WordTiming"],
    max_dur: float = SENT_MAX_DURATION,
) -> List[List[int]]:
    """
    LLM 실패 시 폴백: gap 기반 분할 (수정 L).
    기준: gap > 0.4s AND 현재 세그먼트가 3s 이상 진행됨 → 분할.
    """
    if not words:
        return []
    result = []
    current = [0]
    for i in range(1, len(words)):
        prev_idx = current[-1]
        gap = words[i].start - words[prev_idx].end
        elapsed = words[prev_idx].end - words[current[0]].start
        # 큰 gap(1.5s+)은 무조건 분할, 작은 gap(0.4s+)은 3초+ 경과 시에만
        if gap > 1.5 or (gap > 0.4 and elapsed >= 3.0) or elapsed >= max_dur:
            result.append(current)
            current = []
        current.append(i)
    if current:
        result.append(current)
    return result


def build_segments(
    words: List[WordTiming],
    diarization,
    vocals_path: str,
    max_duration: float = SENT_MAX_DURATION,
    max_words: int = MAX_SEG_WORDS,
    src_lang: Optional[str] = None,
) -> List[Segment]:
    """
    문장 단위 세그먼트 생성.
    1) 오염 단어 제거
    2) 구두점 복원 (LLM) — 실패 시 gap 기반 폴백
    3) 문장 단위로 word index 그룹화
    4) 긴 문장 보조 분할, 짧은 문장 병합
    5) 세그먼트 객체 생성 (start/end = 단어 타임스탬프 기반)
    """
    if not words:
        duration = sf.info(vocals_path).duration
        speaker  = _get_speaker_at(diarization, duration / 2)
        return [Segment(id=0, speaker=speaker, start=0.0, end=duration, text="")]

    # ── 오염 단어 필터링 (기존 로직 유지) ──
    bad_idx = _detect_contaminated_words(words, vocals_path)
    if bad_idx:
        print(f"[Segments] 오염 의심 단어 {len(bad_idx)}개 제거:")
        for i in sorted(bad_idx)[:10]:
            w = words[i]
            print(f"  [{w.start:.2f}~{w.end:.2f}] ({w.end - w.start:.2f}s) → '{w.word}'")
        if len(bad_idx) > 10:
            print(f"  ... 외 {len(bad_idx) - 10}개")

    filtered_words = [w for i, w in enumerate(words) if i not in bad_idx]
    if not filtered_words:
        print("[Segments] 모든 단어가 오염됨 → 빈 세그먼트")
        return []
    words = filtered_words

    # ── 구두점 복원 LLM 호출 ──
    src_lang_name = LANG_CODE_TO_NAME.get((src_lang or "en").lower(), "English")
    raw_text = " ".join(w.word for w in words)
    print(f"[Segments] 구두점 복원 요청 ({src_lang_name}, {len(words)}단어)...")
    punctuated = _restore_punctuation_llm(raw_text, src_lang_name=src_lang_name)

    # ── 문장 단위 단어 인덱스 그룹 ──
    sentence_groups: List[List[int]] = []
    if punctuated:
        print(f"[Segments] 복원됨: '{punctuated[:80]}...'")
        sentence_groups = _match_punctuated_to_words(punctuated, words)
        if not sentence_groups:
            print("[Segments] ⚠️ 단어 매칭 실패 → gap 폴백")
            sentence_groups = _fallback_gap_segments(words)
    else:
        print("[Segments] ⚠️ LLM 구두점 실패 → gap 기반 폴백")
        sentence_groups = _fallback_gap_segments(words)

    # ── 긴 문장 보조 분할 + 짧은 문장 병합 ──
    expanded = []
    for grp in sentence_groups:
        expanded.extend(_split_long_sentence(grp, words, max_duration))
    final_groups = _merge_short_sentences(expanded, words, diarization=diarization)

    # ── v27: 같은 화자 인접 sentence 안전 병합 (분할 회복) ──
    # 우리 코드의 다양한 분할 메커니즘 (post_process 0.3s gap, LLM punct, fallback 0.4s)
    # 이 같은 화자의 연속 발화를 분할시킨 케이스를 복원.
    # 다른 화자 boundary는 절대 침범 안 함.
    final_groups = _merge_same_speaker_adjacent(final_groups, words, diarization)

    # ── sentence-내부 화자 변화 강제 split ──
    # LLM 구두점 복원이 두 화자 발화를 한 문장으로 묶은 경우 보정
    final_groups = _split_groups_by_speaker(final_groups, words, diarization)

    # ── 세그먼트 객체 생성 ──
    segments = []
    for seg_id, grp in enumerate(final_groups):
        if not grp:
            continue
        seg_words = [words[i] for i in grp]
        seg_start = seg_words[0].start
        seg_end   = seg_words[-1].end
        if seg_end <= seg_start:
            seg_end = seg_start + 0.1
        seg_text = " ".join(w.word for w in seg_words).strip()
        speaker  = _get_speaker_at(diarization, (seg_start + seg_end) / 2)
        segments.append(Segment(
            id=seg_id,
            speaker=speaker,
            start=round(seg_start, 3),
            end=round(seg_end, 3),
            text=seg_text,
            words=list(seg_words),
        ))

    print(f"[Segments] {len(segments)}개 세그먼트 생성 (문장 단위)")
    return segments


# ─── Step 6: 감정 추출 ────────────────────────────────────────

def extract_emotion(vocals_path: str, start: float, end: float) -> Tuple[str, float, Dict[str, float]]:
    """
    emotion2vec+로 오디오 구간의 감정 추출.
    v29 (5/14): 팀원 master_timeline_cosyvoice.json 형식 — 9개 감정 scores 모두 반환.

    OUTPUT:
      emotion       : str   — "Neutral" / "Angry" / "Sad" / "Happy" / ...
      emotion_score : float — 최고 점수 (0.0 ~ 1.0)
      all_scores    : dict  — {label: score} 9개 감정 (angry/disgusted/fearful/happy/
                              neutral/other/sad/surprised/unknown). UI 조정용
    """
    empty_scores = {"angry": 0.0, "disgusted": 0.0, "fearful": 0.0,
                    "happy": 0.0, "neutral": 1.0, "other": 0.0,
                    "sad": 0.0, "surprised": 0.0, "unknown": 0.0}
    if _emotion_model is None:
        return "Neutral", 0.0, dict(empty_scores)

    try:
        audio, sr = sf.read(vocals_path)
        chunk = audio[int(start * sr):int(end * sr)]

        if len(chunk) < sr * 0.3:
            return "Neutral", 0.0, dict(empty_scores)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, sr)
            tmp_path = tmp.name

        result = _emotion_model.generate(
            tmp_path,
            output_dir=None,
            granularity="utterance",
            extract_embedding=False
        )
        os.unlink(tmp_path)

        if result and len(result) > 0:
            labels = result[0].get("labels", [])
            scores = result[0].get("scores", [])
            if labels and scores:
                # 모든 9개 score 저장 (UI 조정용)
                all_scores = {}
                for lab, sc in zip(labels, scores):
                    key = lab.lower()
                    if "/" in key:
                        key = key.split("/")[-1]
                    all_scores[key] = round(float(sc), 6)
                # 누락된 라벨은 0.0 으로 채움
                for k in empty_scores:
                    all_scores.setdefault(k, 0.0)

                best_idx = scores.index(max(scores))
                raw_label = labels[best_idx].lower()
                if "/" in raw_label:
                    raw_label = raw_label.split("/")[-1]
                emotion = EMOTION_LABEL_MAP.get(raw_label, "Neutral")
                return emotion, round(float(scores[best_idx]), 3), all_scores

    except Exception as e:
        print(f"[Emotion] 추출 실패: {e}")

    return "Neutral", 0.0, dict(empty_scores)


def fill_emotions(
    segments: List[Segment],
    vocals_path: str,
    content_type: str = "auto",
) -> List[Segment]:
    """
    세그먼트 리스트 전체에 감정 정보 채움.

    INPUT:
      segments     : List[Segment] — emotion이 "Neutral"인 상태
      vocals_path  : str
      content_type : str — "auto/lecture/interview/news/movie/drama"
                           EMOTION_POLICIES 참고.

    OUTPUT:
      segments : List[Segment] — emotion, emotion_score (적용값) +
                                 raw_emotion, raw_emotion_score (원본 감지값) 채워진 상태

    정책:
      - raw_emotion/raw_emotion_score는 항상 emotion2vec+ 감지값으로 저장 (리포트/분석용)
      - emotion/emotion_score는 content_type에 따라:
        · neutral_only → "Neutral", 1.0 으로 덮어씀
        · passthrough  → raw와 동일
    """
    policy = EMOTION_POLICIES.get(content_type, "passthrough")
    print(f"[Emotion] 콘텐츠 타입: {content_type} → 정책: {policy}")

    # v22 (5/14): 감정 수치 조절 env vars
    #   LATENTSYNC_EMOTION_THRESHOLD: 0.0~1.0. raw_score < 임계값 이면 Neutral 로 강등
    #   LATENTSYNC_EMOTION_OVERRIDE: 강제 emotion (Neutral/Happy/Sad/Angry/Surprised/Scared)
    #   LATENTSYNC_EMOTION_INTENSITY: mild/moderate/strong (instruct_text 톤 강도)
    _emotion_threshold = float(os.environ.get("LATENTSYNC_EMOTION_THRESHOLD", "0.0"))
    _emotion_override = os.environ.get("LATENTSYNC_EMOTION_OVERRIDE", "").strip()
    if _emotion_threshold > 0 or _emotion_override:
        print(f"[Emotion] 수치 조절: threshold={_emotion_threshold}, override='{_emotion_override}'")

    override_count = 0
    for seg in segments:
        raw_emo, raw_score, raw_scores = extract_emotion(vocals_path, seg.start, seg.end)
        seg.raw_emotion = raw_emo
        seg.raw_emotion_score = raw_score
        seg.raw_emotion_scores = raw_scores

        # 강제 override 가 최우선
        if _emotion_override:
            seg.emotion = _emotion_override
            seg.emotion_score = 1.0
            print(f"[Emotion] seg {seg.id}: {raw_emo} ({raw_score:.2f}) → {_emotion_override} [override]")
        elif policy == "neutral_only":
            seg.emotion = "Neutral"
            seg.emotion_score = 1.0
            if raw_emo != "Neutral":
                override_count += 1
            print(f"[Emotion] seg {seg.id}: {raw_emo} ({raw_score:.2f}) → Neutral [policy]")
        else:  # passthrough
            # threshold 미만이면 Neutral 로 강등 (강한 감정만 인정)
            if _emotion_threshold > 0 and raw_emo != "Neutral" and raw_score < _emotion_threshold:
                seg.emotion = "Neutral"
                seg.emotion_score = 1.0
                print(f"[Emotion] seg {seg.id}: {raw_emo} ({raw_score:.2f}) → Neutral [threshold<{_emotion_threshold}]")
            else:
                seg.emotion = raw_emo
                seg.emotion_score = raw_score
                print(f"[Emotion] seg {seg.id}: {raw_emo} ({raw_score:.2f})")

    if policy == "neutral_only" and override_count > 0:
        print(f"[Emotion] 정책 적용으로 {override_count}/{len(segments)}개 세그먼트가 Neutral로 변경됨")
    return segments


# ─── Step 7: Speaker Profile Bank 구성 ───────────────────────

def _quality_gate(clip: np.ndarray, sr: int) -> Dict[str, float]:
    """v28 (5/14): Quality Gate 휴리스틱 (사용자 피드백 반영 — 완화).
    완화 이유: silence 40% 너무 strict → reference 후보 과도 제외 →
    짧은 ref → CosyVoice3 한국어 prosody 부족 → 어눌/짤림.
    env var:
      LATENTSYNC_QG_OFF=1: Quality Gate 완전 비활성화 (모든 candidate 채택)
      LATENTSYNC_QG_SILENCE_MAX: 무음 비율 임계 (기본 0.70, v9 0.40)
      LATENTSYNC_QG_SNR_MIN: SNR 임계 dB (기본 5.0, v9 10.0)
    """
    import os as _os_qg
    # mono로 변환 (stereo면 channel 평균)
    if clip.ndim > 1:
        mono = clip.mean(axis=-1)
    else:
        mono = clip

    silence_thresh = 0.01
    silence_ratio = float(np.mean(np.abs(mono) < silence_thresh))

    clipping_thresh = 0.99
    clipping_ratio = float(np.mean(np.abs(mono) > clipping_thresh))

    abs_mono = np.abs(mono)
    sorted_abs = np.sort(abs_mono)
    n = len(sorted_abs)
    if n < 20:
        snr_db = 0.0
    else:
        noise_rms = float(np.sqrt(np.mean(sorted_abs[:n // 10] ** 2))) + 1e-10
        voice_rms = float(np.sqrt(np.mean(sorted_abs[n // 2:] ** 2)))
        snr_db = float(20.0 * np.log10(voice_rms / noise_rms))

    # v28 완화: 명백한 결함만 필터 (극단 무음/clipping/저SNR)
    qg_off = _os_qg.environ.get("LATENTSYNC_QG_OFF", "0") == "1"
    silence_max = float(_os_qg.environ.get("LATENTSYNC_QG_SILENCE_MAX", "0.70"))
    snr_min = float(_os_qg.environ.get("LATENTSYNC_QG_SNR_MIN", "5.0"))
    if qg_off:
        pass_gate = True
    else:
        pass_gate = (
            silence_ratio < silence_max and   # 기본 70% 무음 미만 (40 → 70 완화)
            clipping_ratio < 0.005 and         # clipping 0.5% 미만 (0.1 → 0.5 완화)
            snr_db > snr_min                   # SNR 5dB 이상 (10 → 5 완화)
        )

    # quality score (0~1, 높을수록 좋음). MOS와 결합용
    silence_score = max(0.0, 1.0 - silence_ratio / 0.4)   # 무음 40% = score 0
    clipping_score = max(0.0, 1.0 - clipping_ratio / 0.001)  # clipping 0.1% = score 0
    snr_score = min(1.0, max(0.0, (snr_db - 5.0) / 20.0))  # 5dB=0, 25dB=1
    quality_score = (silence_score + clipping_score + snr_score) / 3.0

    return {
        'silence_ratio': silence_ratio,
        'clipping_ratio': clipping_ratio,
        'snr_db': snr_db,
        'pass': pass_gate,
        'score': quality_score,
    }


def build_speaker_profiles(
    segments: List[Segment],
    vocals_path: str,
    exclude_intervals: Optional[List[Tuple[float, float, str]]] = None,
) -> Dict[str, SpeakerProfile]:
    """
    화자별 + 감정별 레퍼런스 음성 파일 자동 추출.
    v27 (5/14): MOS + Quality Gate 휴리스틱 (팀원 PPT 형식).
    3~15초 사이 구간만 후보로 사용 (CosyVoice3 제한: 30초).

    v19 (5/15): exclude_intervals 옵션 추가.
        AV-Reassign으로 라벨이 변경된 segment는 원래 라벨이 정확하지 않을 가능성이
        있으므로 ref bank 후보에서 제외 (ref audio 오염 방지).
        v18 회귀: SPEAKER_02 ref bank에 SPEAKER_01 segment 섞여 남성화된 사례 fix.

    Args:
        exclude_intervals: [(start, end, old_spk), ...] 재할당된 영역.
                           이와 50%+ 겹치는 segment는 ref 후보에서 skip.
    """
    # v20: ref audio 길이 하한 환경변수화
    # CosyVoice 공식 권장: 6~10초 (4초대는 권장 하한, BGM leak시 metallic ring 위험)
    # 검색 결과 (Issue #1704, #862): 짧은 ref + noisy → first-word vocoder artifact
    _ref_min_dur = float(os.environ.get("LATENTSYNC_REF_MIN_DUR", "3.0"))
    _ref_max_dur = float(os.environ.get("LATENTSYNC_REF_MAX_DUR", "15.0"))
    _ref_fallback_min = float(os.environ.get("LATENTSYNC_REF_FALLBACK_MIN", "2.0"))
    _ref_fallback_max = float(os.environ.get("LATENTSYNC_REF_FALLBACK_MAX", "25.0"))

    # v19: 재할당된 영역 제외 헬퍼 (av_fusion이 import 안 됐을 때 fallback)
    def _is_excluded(seg) -> bool:
        if not exclude_intervals:
            return False
        try:
            import sys as _sys
            if "/workspace/scripts" not in _sys.path:
                _sys.path.insert(0, "/workspace/scripts")
            from av_fusion import overlaps_excluded_intervals
            return overlaps_excluded_intervals(
                seg.start, seg.end, exclude_intervals, min_overlap_ratio=0.5
            )
        except Exception:
            seg_dur = max(1e-3, seg.end - seg.start)
            for ex_s, ex_e, _old in exclude_intervals:
                overlap = max(0.0, min(seg.end, ex_e) - max(seg.start, ex_s))
                if overlap / seg_dur >= 0.5:
                    return True
            return False

    # 화자별 감정별로 후보 구간 수집
    # 5/12 audio quality fix: 짧은 reference (1-2초) 는 voice cloning 부정확
    # → 기계음/단조로운 prosody 원인. 최소 3초 strict, fallback 도 2초까지만.
    candidates: Dict[str, Dict[str, List[Tuple[float, Segment]]]] = {}
    n_excluded = 0

    for seg in segments:
        duration = seg.end - seg.start
        if duration < _ref_min_dur or duration > _ref_max_dur:
            continue
        if _is_excluded(seg):
            n_excluded += 1
            continue
        if seg.speaker not in candidates:
            candidates[seg.speaker] = {}
        if seg.emotion not in candidates[seg.speaker]:
            candidates[seg.speaker][seg.emotion] = []
        candidates[seg.speaker][seg.emotion].append((duration, seg))

    if n_excluded > 0:
        print(f"[Profiles] AV-Reassign 영역 {n_excluded}개 segment 제외 (ref 오염 방지)")

    # v20: per-speaker 보강 — speaker가 후보 없으면 그 화자만 fallback 적용
    # (기존: 전체 candidates 비면 fallback — 일부 화자만 6초+ 없는 케이스 보호 못 함)
    all_speakers = set(seg.speaker for seg in segments)
    missing_speakers = all_speakers - set(candidates.keys())
    if missing_speakers:
        print(f"[Profiles] 6초+ ref 없는 화자 {len(missing_speakers)}명: {sorted(missing_speakers)} → fallback")
        for seg in segments:
            if seg.speaker not in missing_speakers:
                continue
            duration = seg.end - seg.start
            if duration < _ref_fallback_min or duration > _ref_fallback_max:
                continue
            if _is_excluded(seg):
                continue
            if seg.speaker not in candidates:
                candidates[seg.speaker] = {}
            if seg.emotion not in candidates[seg.speaker]:
                candidates[seg.speaker][seg.emotion] = []
            candidates[seg.speaker][seg.emotion].append((duration, seg))

    # 후보가 전혀 없으면 전체 fallback
    if not candidates:
        for seg in segments:
            duration = seg.end - seg.start
            if duration < _ref_fallback_min or duration > _ref_fallback_max:
                continue
            if _is_excluded(seg):
                continue
            if seg.speaker not in candidates:
                candidates[seg.speaker] = {}
            if seg.emotion not in candidates[seg.speaker]:
                candidates[seg.speaker][seg.emotion] = []
            candidates[seg.speaker][seg.emotion].append((duration, seg))

    profiles = {}
    audio, sr = sf.read(vocals_path)

    for speaker, emotion_map in candidates.items():
        profile = SpeakerProfile(speaker_id=speaker)

        for emotion, seg_list in emotion_map.items():
            best_seg = None
            best_mos = -1
            best_duration = 0

            for duration, seg in seg_list:
                # 레퍼런스 후보를 임시 파일로 저장
                start_sample = int(seg.start * sr)
                end_sample = int(seg.end * sr)
                max_ref_samples = int(15.0 * sr)
                if (end_sample - start_sample) > max_ref_samples:
                    end_sample = start_sample + max_ref_samples
                clip = audio[start_sample:end_sample]

                # v27 (5/14): Quality Gate 휴리스틱 우선 필터 (팀원 PPT)
                # 무음 비율 + clipping + SNR 검증 → 통과한 candidate 만 MOS 평가
                qg = _quality_gate(clip, sr)
                if not qg['pass']:
                    print(f"  [QualityGate] {speaker}/{emotion} seg {seg.id} skip: "
                          f"silence={qg['silence_ratio']:.2f}, clip={qg['clipping_ratio']:.4f}, "
                          f"snr={qg['snr_db']:.1f}dB")
                    continue

                # MOS 평가 + Quality Gate score + duration bonus 결합
                if _mos_evaluator is not None:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, clip, sr)
                        mos_score = _mos_evaluator.evaluate(tmp.name)
                        os.unlink(tmp.name)

                    duration_bonus = 0.05 * min(duration, 10.0)
                    quality_bonus = 0.5 * qg['score']   # Quality Gate 가중치
                    combined = mos_score + duration_bonus + quality_bonus
                    # best_combined 도 같은 metric (이전 best segment 의 qg/duration 재계산 안 함 → 단순화)
                    if combined > best_mos + 0.05 * min(best_duration, 10.0) if best_seg else -1:
                        best_mos = mos_score
                        best_seg = seg
                        best_duration = duration
                else:
                    # MOS 없으면 Quality Gate score + duration 으로 선택
                    candidate_score = qg['score'] + 0.05 * min(duration, 10.0)
                    best_candidate_score = best_mos + 0.05 * min(best_duration, 10.0) if best_seg else -1
                    if candidate_score > best_candidate_score:
                        best_mos = qg['score']
                        best_seg = seg
                        best_duration = duration
                        best_duration = duration
                        best_mos = 0.0

            if best_seg is None:
                continue

            ref_filename = f"{speaker}_{emotion}.wav"
            ref_path = os.path.join(REF_DIR, ref_filename)

            start_sample = int(best_seg.start * sr)
            end_sample = int(best_seg.end * sr)
            max_ref_samples = int(15.0 * sr)
            if (end_sample - start_sample) > max_ref_samples:
                end_sample = start_sample + max_ref_samples
            clip = audio[start_sample:end_sample]

            # 앞뒤 침묵 제거
            clip_trimmed, _ = librosa.effects.trim(clip, top_db=20)
            if len(clip_trimmed) > sr:  # 1초 이상이면 사용
                clip = clip_trimmed

            # 5/7 VOLUME FIX: peak normalize for clean reference
            # 작은 vocal (BS-Roformer 분리 후 종종 quiet)을 boost
            # CosyVoice voice cloning quality 향상 (기계음 방지)
            peak = float(np.abs(clip).max())
            if peak > 0 and peak < 0.5:
                gain = min(0.7 / peak, 4.0)  # 최대 4배 boost (extreme noise 방지)
                clip = clip * gain
                clip = np.clip(clip, -1.0, 1.0)
                print(f"[Profile] {speaker}/{emotion} reference boost ×{gain:.2f} (peak {peak:.2f}→{0.7 if gain*peak<0.7 else gain*peak:.2f})")

            sf.write(ref_path, clip, sr)
            profile.references[emotion] = ref_path

            mos_str = f", MOS={best_mos:.2f}" if best_mos > 0 else ""
            print(f"[Profile] {speaker} / {emotion}: {ref_filename} ({best_duration:.1f}s{mos_str})")

        profiles[speaker] = profile

    return profiles


# ─── Step 8: 번역 ─────────────────────────────────────────────

def translate_segments(
    segments: List[Segment],
    tgt_lang: str,
    content_type: str = "auto",
) -> List[Segment]:
    """세그먼트 전체 번역 — LLM은 묶어서 한 번에. content_type에 따라 멀티필드 출력."""
    # LLM 번역 강제 (Google fallback 제거됨 - 사용자 요구)
    if _google_translator != "vectorengine":
        print("[Translate] ⚠️ vectorengine LLM 미설정 — translation skip "
              "(seg.translated 빈 상태 유지, TTS 단계에서 자동 skip)")
        for seg in segments:
            seg.translated = ""
        return segments
    return _translate_segments_llm(segments, tgt_lang, content_type=content_type)


def _translate_segments_llm(
    segments: List[Segment],
    tgt_lang: str,
    content_type: str = "auto",
) -> List[Segment]:
    """
    VectorEngine GPT로 청크 전체 세그먼트를 배치 분할하여 번역.

    🔥 호출 안정성 보강:
      - 배치 분할 (5개씩): 호출당 reasoning 부담 감소 → timeout 위험 감소
      - 다단계 폴백:
          1차: 멀티필드(passthrough) 또는 단일필드(neutral_only)
          2차: 같은 배치를 단일필드로 재시도 (passthrough였다면)
          3차: 해당 배치만 Google Translator 폴백
      - max_tokens 4096 (이전 16384 → reasoning 폭주 방지)
      - timeout 300초 (이전 600초 → 빨리 실패하고 폴백)
    """
    api_key = os.environ.get("VECTORENGINE_API_KEY", "")
    base_url = os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai")
    model = os.environ.get("VECTORENGINE_MODEL", "gpt-5.4")

    lang_names = {
        "ko": "한국어", "ja": "일본어", "zh": "중국어", "fr": "프랑스어",
        "de": "독일어", "es": "스페인어", "en": "영어", "ru": "러시아어",
        "pt": "포르투갈어", "it": "이탈리아어", "ar": "아랍어", "nl": "네덜란드어",
    }
    lang_name = lang_names.get(tgt_lang, tgt_lang)

    # DIALOGUE_FILTER: 한숨/탄식/단순 의성어는 dubbing 안 함 (원본 유지)
    NOISE_PATTERNS = {
        # English noise words (ASR이 한숨/탄식을 잘못 텍스트화하는 경우)
        "uh", "uhh", "uhhh", "um", "umm", "oh", "ohh", "ah", "ahh", "ahhh",
        "huh", "mm", "mmm", "hm", "hmm", "hmmm", "eh", "ehh",
        "wow", "oof", "ouch", "ow", "oww", "ugh", "ugh",
        # Korean noise (한국어로 번역되어 들어올 경우)
        "어", "음", "아", "오", "허", "후", "흠", "헉",
        # Common laughs/cries
        "haha", "hehe", "hihi", "lol",
    }

    def _is_dialogue(seg) -> bool:
        text = seg.text.strip()
        if not text:
            return False
        # 의성어 정확 매칭 (전체 텍스트가 패턴)
        text_lower = text.lower().rstrip(".,!?~")
        if text_lower in NOISE_PATTERNS:
            return False
        # 너무 짧음 (3글자 이하 + 단어 수 1개 이하)
        if len(text) < 3 and len(text.split()) <= 1:
            return False
        return True

    to_translate = [(i, seg) for i, seg in enumerate(segments) if _is_dialogue(seg)]
    skipped = len(segments) - len(to_translate)
    if skipped > 0:
        print(f"[Translate] {skipped}개 non-dialogue segment skip (한숨/탄식 → 원본 유지)")
    if not to_translate:
        return segments

    policy = EMOTION_POLICIES.get(content_type, "passthrough")
    # v20 (5/14): LATENTSYNC_FORCE_LLM_DESC=1 면 interview/lecture/news 에서도
    # LLM 자연어 묘사 활성화 → 발화 prosody 강화.
    _force_llm_desc = os.environ.get("LATENTSYNC_FORCE_LLM_DESC", "0") == "1"
    use_emotion_desc = (policy == "passthrough") or _force_llm_desc
    if _force_llm_desc and policy != "passthrough":
        print(f"[Translate] LLM 자연어 묘사 강제 활성화 (LATENTSYNC_FORCE_LLM_DESC=1, policy={policy})")

    BATCH_SIZE = 7  # 5/7: 5→7 sweet spot (-30% LLM 시간, 길이 정확도 95%+ 유지)
    total_batches = (len(to_translate) + BATCH_SIZE - 1) // BATCH_SIZE

    if total_batches > 1:
        print(f"[Translate] {len(to_translate)}개 세그먼트를 {total_batches}개 배치로 분할 (배치당 ≤{BATCH_SIZE}개)")

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(to_translate))
        batch = to_translate[batch_start:batch_end]
        batch_label = f"batch {batch_idx + 1}/{total_batches}"
        if total_batches > 1:
            print(f"[Translate] === {batch_label} ({len(batch)}개) ===")

        # 1차: 정책 그대로 시도 (멀티필드 or 단일필드)
        parsed = _llm_translate_batch(
            batch=batch, lang_name=lang_name,
            api_key=api_key, base_url=base_url, model=model,
            use_emotion_desc=use_emotion_desc,
        )

        # 2차 폴백: 멀티필드 실패면 단일필드로 재시도
        if parsed is None and use_emotion_desc:
            print(f"[Translate] {batch_label} 멀티필드 실패 → 단일필드 재시도")
            parsed = _llm_translate_batch(
                batch=batch, lang_name=lang_name,
                api_key=api_key, base_url=base_url, model=model,
                use_emotion_desc=False,
            )

        # 3차 폴백 제거 (Google translate 사용 안 함 - 사용자 요구)
        # LLM 완전 실패 → translated 빈 상태 유지 (TTS 자동 skip)
        if parsed is None:
            print(f"[Translate] ⚠️ {batch_label} LLM 완전 실패 — segments translation skip "
                  f"(원본 audio 유지)")
            for batch_local_idx, (i, seg) in enumerate(batch):
                seg.translated = ""
            continue

        # 결과 적용 (배치 내 인덱스 → 글로벌 인덱스 매핑)
        for batch_local_idx, (i, seg) in enumerate(batch):
            entry = parsed.get(batch_local_idx, {})
            kor = entry.get('korean', '')
            if kor:
                seg.translated = kor
                if use_emotion_desc:
                    seg.tts_context = entry.get('context', '') or ''
                    # 'tone' 우선 (v15 새 방식), 없으면 'emotion' fallback
                    seg.tts_emotion = entry.get('tone', '') or entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:60]}]")
                    else:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"⚠️ emotion 누락")
                else:
                    print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
            else:
                # LLM이 한국어 미생성 → translation skip (Google fallback 제거)
                seg.translated = ""
                print(f"[Translate] ⚠️ LLM korean 누락 → skip: {seg.text[:30]}...")

    return segments


def _llm_translate_batch(
    batch: list,
    lang_name: str,
    api_key: str,
    base_url: str,
    model: str,
    use_emotion_desc: bool,
) -> Optional[Dict[int, Dict[str, str]]]:
    """
    한 배치를 LLM에 보내고 파싱된 결과를 반환.
    실패 시 None — 호출자가 폴백 처리.
    배치 내 인덱스(0부터)로 결과 반환.
    """
    import requests
    import re

    if not api_key:
        return None

    # ── 입력 라인 구성 (배치 로컬 인덱스 사용) ──
    # v16: v15 rollback (prompt 강화 → 압축/잘림 발생). v14 상태로 복원.
    # 한국어 짧음 fix는 LLM이 아닌 CosyVoice speed retry로 (synthesize_chunk).
    LLM_EMOTION_FACTOR = {
        "Sad": 0.85, "Scared": 0.85, "Angry": 0.95,
        "Happy": 1.0, "Surprised": 1.0, "Neutral": 1.0,
    }
    BASE_SYL_RATE = 6.0  # v14 상태

    # v17 (5/14): community-verified soft range + CAPEL countdown for short utterances.
    # v15 strict 0.3 + HARD CONSTRAINT 피드백 → 의미 손상, 첫부분 잘림 회귀.
    # 학계+업계 합의 (VideoDubber AAAI 2023, CAPEL arxiv 2508.13805):
    #   - hard syllable constraint는 증명된 실패 패턴
    #   - 길이는 LLM이 아닌 TTS speed + time-stretch가 흡수 (already implemented)
    #   - 의미 보존 > 음절 정확도 (TTS retry stage가 ±15% 처리)
    #   - 짧은 utterance(<1.5s)만 CAPEL countdown 마커로 LLM 길이 준수율 향상
    import os as _os_strict
    _strict_range = float(_os_strict.environ.get("LATENTSYNC_SYL_RANGE", "0.5"))
    _short_utt_capel = _os_strict.environ.get("LATENTSYNC_CAPEL_SHORT", "1") != "0"

    lines = []
    for batch_local_idx, (i, seg) in enumerate(batch):
        duration = max(0.3, seg.end - seg.start)
        seg_emotion = getattr(seg, 'emotion', 'Neutral') or 'Neutral'
        emo_factor = LLM_EMOTION_FACTOR.get(seg_emotion, 1.0)
        rate = BASE_SYL_RATE * emo_factor
        target_min = max(1, int(duration * (rate - _strict_range)))
        target_max = max(target_min + 1, int(duration * (rate + _strict_range)))
        target_mid = (target_min + target_max) // 2

        # CAPEL countdown for short utterances (target ≤ 4 syl, duration < 1.5s).
        # Increases LLM length compliance 30%→95% on short outputs (arxiv 2508.13805).
        # Example: 'Good' (target 2) → countdown '<2>_<1>_<0>' guides '좋_아' = '좋아'.
        countdown_hint = ""
        if _short_utt_capel and target_max <= 4 and duration < 1.5:
            countdown_hint = f", countdown: " + "_".join(f"<{n}>" for n in range(max(1, target_mid), 0, -1)) + "<0>"

        if use_emotion_desc:
            raw_e = getattr(seg, 'raw_emotion', seg.emotion) or seg.emotion
            raw_s = getattr(seg, 'raw_emotion_score', seg.emotion_score) or 0.0
            lines.append(
                f"[{batch_local_idx}] (duration: {duration:.1f}s, target ~{target_mid} syl (range {target_min}-{target_max}){countdown_hint}, "
                f"emotion2vec_hint: {raw_e} {raw_s:.2f}, speaker: {seg.speaker}) {seg.text}"
            )
        else:
            lines.append(
                f"[{batch_local_idx}] (duration: {duration:.1f}s, target ~{target_mid} syl (range {target_min}-{target_max}){countdown_hint}) {seg.text}"
            )
    batch_text = "\n".join(lines)

    # ── 시스템 프롬프트 ──
    if use_emotion_desc:
        system_prompt = (
            f"You are a professional dubbing translator and emotion designer.\n"
            f"For each numbered line, output TWO fields:\n"
            f"  1. korean: natural spoken {lang_name} translation for dubbing\n"
            f"  2. tone: SHORT natural English phrase describing delivery.\n"
            f"\n"
            f"OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\n"
            f"[N]\n"
            f"korean: <translation>\n"
            f"tone: <phrase>\n"
            f"\n"
            f"TRANSLATION RULES (length-aware, meaning-first):\n"
            f"- Each line shows a soft syllable target: 'target ~N syl (range MIN-MAX)'.\n"
            f"- AIM for ~N. Range MIN-MAX is acceptable. **Meaning preservation > exact syllable count.**\n"
            f"- The TTS engine adjusts speed downstream (±15% absorbed automatically),\n"
            f"   so do NOT force unnatural Korean to hit a number.\n"
            f"- For VERY SHORT source (1 word: 'Good', 'No', 'Bull') → SHORT translation:\n"
            f"   'Good'→'좋아' (2 syl) · 'No'→'아니' (2 syl) · 'Bull'→'말이 돼' (3 syl).\n"
            f"   Do NOT pad: '좋아요 정말 그래요' (8 syl) for 'Good' is WRONG.\n"
            f"- If line shows 'countdown: <N>_<N-1>_..._<0>', fill EACH slot with exactly 1 syllable.\n"
            f"   Example for target 2: countdown <2>_<1>_<0> → '좋_아' → output '좋아'.\n"
            f"   This is a length scaffold for very short utterances only.\n"
            f"- If natural Korean cannot fit in range, output the SHORTEST natural Korean.\n"
            f"   The downstream TTS speed adjustment will absorb the gap. Never pad with empty particles.\n"
            f"\n"
            f"SPEAKER CONSISTENCY (화자 일관성):\n"
            f"- SAME speaker → SAME 종결어미 (반말 OR 존댓말) throughout entire output.\n"
            f"- Drama context: child/peer speakers use 반말, formal speakers use 존댓말.\n"
            f"- DO NOT switch styles within same speaker (e.g., '~한다 → ~해요' transition is FAILURE).\n"
            f"- Output in {lang_name} ONLY. No original text leak.\n"
            f"- Make consecutive lines from same speaker flow naturally.\n"
            f"\n"
            f"TONE RULES (팀원 검증된 형식, ratio 0.93-1.46 정상):\n"
            f"- SHORT phrase (5-10 words), starts with 'with' or 'in a'.\n"
            f"- Describes delivery STYLE + subtle EMOTION cue.\n"
            f"- AVOID slow-down keywords: 'slow pacing', 'restrained voice', 'measured cadence',\n"
            f"   'drawn out', 'lingering' — these slow LLM delivery and break timing.\n"
            f"- USE pace-neutral or pace-positive: 'natural delivery', 'conversational', 'lively',\n"
            f"   'crisp', 'energetic', 'brisk'.\n"
            f"\n"
            f"GOOD EXAMPLES (팀원 스타일):\n"
            f"  'with a faint hint of delighted curiosity'\n"
            f"  'with a subtle note of pleased discovery'\n"
            f"  'with a gentle note of caution'\n"
            f"  'with a subtle wry edge'\n"
            f"  'close to the speaker natural delivery'\n"
            f"  'with a calm professional tone'\n"
            f"  'in a brisk conversational manner'\n"
            f"\n"
            f"BAD EXAMPLES (피해야 함):\n"
            f"  'angry' (too short — no nuance)\n"
            f"  'in a casual, pleased, lightly confident tone with warm but restrained excitement' (too long, breaks timing)\n"
            f"  'with deep, anguished sadness, slow pacing, restrained voice' (slow keywords)\n"
            f"  '낮은 톤' (Korean — must be English)\n"
            f"\n"
            f"TONE CONTEXT:\n"
            f"- emotion2vec_hint = audio classifier reference (use as hint).\n"
            f"- text content may override the audio hint if mismatched.\n"
            f"- TTS prefix template: 'You are a helpful assistant. Please say it close to the speaker natural delivery, {{tone}}.<|endofprompt|>'\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."
        )
    else:
        system_prompt = (
            f"You are a professional dubbing translator. Translate each numbered line into "
            f"natural spoken {lang_name} for dubbing with lip-sync.\n"
            f"RULES (length-aware, meaning-first):\n"
            f"- Keep the same numbering format [0], [1], [2]...\n"
            f"- Each line shows a soft syllable target: 'target ~N syl (range MIN-MAX)'.\n"
            f"- AIM for ~N. Range MIN-MAX is acceptable. Meaning preservation > exact count.\n"
            f"- The TTS engine adjusts speed downstream (±15%), so don't force unnatural Korean.\n"
            f"- For VERY SHORT source (1 word) → SHORT translation: 'Good'→'좋아', 'No'→'아니'.\n"
            f"   Never pad short source with empty particles to hit a number.\n"
            f"- If countdown markers appear ('<N>_..._<0>'), fill each slot with 1 syllable.\n"
            f"- If natural Korean cannot fit range, output shortest natural Korean; TTS absorbs the rest.\n"
            f"- You MUST output in {lang_name}. DO NOT output the original text.\n"
            f"- Do NOT merge content between lines.\n"
            f"- Make consecutive lines from the same speaker sound natural in sequence.\n"
            f"- Output ONLY translations. No thinking, no explanation."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": batch_text},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,             # 16384 → 4096 (reasoning 폭주 방지)
        "reasoning_effort": "low",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 최대 2회 시도 (timeout/일시적 오류)
    for attempt in range(2):
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers, json=payload,
                timeout=300,  # 600 → 300 (빨리 실패하고 폴백)
            )
            resp.raise_for_status()
            resp_json = resp.json()
            msg = resp_json["choices"][0]["message"]
            result = (msg.get("content") or "").strip()
            if not result:
                result = (msg.get("reasoning_content") or "").strip()
                if result:
                    print("[Translate] ⚠️ content 비었음 → reasoning_content 사용")

            if not result:
                choice = resp_json["choices"][0]
                usage = resp_json.get("usage", {})
                print(f"[Translate] ⚠️ LLM 빈 응답:")
                print(f"  finish_reason: {choice.get('finish_reason')!r}")
                print(f"  usage: {usage}")
                if attempt == 0:
                    print(f"[Translate] 빈 응답 → 재시도")
                    continue
                return None

            if "<think>" in result:
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

            # v26: CAPEL countdown 마커 제거 (LLM이 가이드를 출력에 포함시킨 경우)
            # "<3>거<2>기<1>로<0>가" → "거기로가"
            # 우리가 LLM에 "<N>_<N-1>...<0>" 형식 가이드를 줬는데 LLM이 결과에 넣음
            if '<' in result and '>' in result:
                # <숫자> 또는 <숫자>_ 또는 _ 마커 제거
                result = re.sub(r'<\d+>_?', '', result)

            if use_emotion_desc:
                parsed = _parse_translation_with_emotion(result)
            else:
                parsed = _parse_translation_only(result)

            expected = set(range(len(batch)))
            got = set(k for k, v in parsed.items() if v.get('korean'))
            missing = sorted(expected - got)
            if missing:
                print(f"[Translate] ⚠️ LLM 응답 누락 번호 {len(missing)}개: {missing[:20]}")
                if len(missing) > len(batch) // 2 and attempt == 0:
                    # 절반 이상 누락이면 재시도
                    print(f"[Translate] 누락 과다 → 재시도")
                    continue

            return parsed

        except Exception as e:
            err_type = type(e).__name__
            if attempt == 0:
                print(f"[Translate] 1차 시도 실패 ({err_type}: {str(e)[:80]}) → 재시도")
            else:
                print(f"[Translate] 2차 시도도 실패 ({err_type}): {e}")
                return None

    return None


def _parse_translation_only(result: str) -> Dict[int, Dict[str, str]]:
    """[N] korean_text 형식 파싱."""
    import re as _re
    out: Dict[int, Dict[str, str]] = {}
    current_idx = None
    current_buf = []
    num_pat = _re.compile(r'^\s*[\[\(]?(\d+)[\]\)]?\s*[\.\:\)]?\s+(.*)$')

    for line in result.split("\n"):
        line = line.rstrip()
        m = num_pat.match(line)
        if m:
            if current_idx is not None:
                out[current_idx] = {'korean': " ".join(current_buf).strip()}
            current_idx = int(m.group(1))
            current_buf = [m.group(2)] if m.group(2) else []
        elif current_idx is not None and line.strip():
            current_buf.append(line.strip())
    if current_idx is not None:
        out[current_idx] = {'korean': " ".join(current_buf).strip()}
    return out


def _parse_translation_with_emotion(result: str) -> Dict[int, Dict[str, str]]:
    """
    [N] / korean: ... / context: ... / emotion: ... 형식 파싱.
    LLM 응답 형식이 약간 흔들려도 받아주도록 유연하게 처리.
    """
    import re as _re
    out: Dict[int, Dict[str, str]] = {}

    # 인덱스 패턴: 줄 시작에 [N] 또는 (N) 또는 N. 등
    idx_pat = _re.compile(r'^\s*[\[\(]?(\d+)[\]\)]?\s*[\.\:\)]?\s*$')
    # 인덱스 + 인라인: [N] korean: ... 같은 한 줄 형식도 허용
    idx_inline = _re.compile(r'^\s*[\[\(]?(\d+)[\]\)]?\s*[\.\:\)]?\s+(.*)$')
    # 필드 패턴: korean:, context:, emotion: (대소문자 무시)
    field_pat = _re.compile(r'^\s*(korean|context|emotion|tone)\s*[:：]\s*(.*)$', _re.IGNORECASE)

    current_idx = None
    current_field = None  # 마지막 인식한 필드 (멀티라인 값 처리용)
    fields: Dict[str, List[str]] = {}

    def flush():
        nonlocal current_idx, fields, current_field
        if current_idx is not None:
            entry = {k: " ".join(v).strip() for k, v in fields.items()}
            # 빈 필드는 없는 걸로
            entry = {k: v for k, v in entry.items() if v}
            if entry:
                out[current_idx] = entry
        fields = {}
        current_field = None

    for raw_line in result.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            current_field = None  # 빈 줄이면 멀티라인 종료
            continue

        # 인덱스 헤더 줄?
        m_idx = idx_pat.match(line)
        if m_idx:
            flush()
            current_idx = int(m_idx.group(1))
            continue

        # 인라인 인덱스: [N] 그 뒤에 콘텐츠
        m_inline = idx_inline.match(line)
        if m_inline and current_idx is None:
            # 이 경우 잘 안 쓰이지만 보호적으로 처리
            flush()
            current_idx = int(m_inline.group(1))
            rest = m_inline.group(2)
            # rest가 'korean: ...' 형식인지 확인
            mf = field_pat.match(rest)
            if mf:
                current_field = mf.group(1).lower()
                fields.setdefault(current_field, []).append(mf.group(2).strip())
            else:
                # 인덱스 + 한국어가 그냥 같은 줄 → korean으로 간주
                current_field = 'korean'
                fields.setdefault('korean', []).append(rest.strip())
            continue

        # 필드 줄?
        m_field = field_pat.match(line)
        if m_field and current_idx is not None:
            current_field = m_field.group(1).lower()
            fields.setdefault(current_field, []).append(m_field.group(2).strip())
            continue

        # 멀티라인 연속? (필드 값이 다음 줄로 이어지는 경우)
        if current_idx is not None and current_field is not None:
            fields.setdefault(current_field, []).append(line.strip())

    flush()
    return out


def _translate_google(text: str, tgt_lang: str, max_retries: int = 3) -> str:
    """Deep Translator (Google 무료) 폴백."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return text

    target_code = DEEP_LANG_MAP.get(tgt_lang, "ko")

    for attempt in range(max_retries):
        try:
            result = GoogleTranslator(source="auto", target=target_code).translate(text)
            if result and result.strip():
                return result.strip()
        except Exception as e:
            print(f"[Translate] Google 오류 {attempt+1}/{max_retries}: {e}")

    return text


def retranslate_shorter(
    original_text: str,
    tgt_lang: str,
    max_syllables: int,
    current_translation: str = "",
) -> Optional[str]:
    """
    🔥 수정 M: overflow 예상 시 번역을 짧게 재요청.
    성공 시 축약된 번역 반환, 실패 시 None.
    다국어 확장: 음절 카운팅/속도 테이블을 언어별로 분기 필요.
    """
    api_key = os.environ.get("VECTORENGINE_API_KEY", "")
    if not api_key:
        return None

    import requests
    base_url = os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai")
    model = os.environ.get("VECTORENGINE_MODEL", "gpt-5.4")

    lang_names = {
        "ko": "Korean", "ja": "Japanese", "zh": "Chinese", "fr": "French",
        "de": "German", "es": "Spanish", "en": "English", "ru": "Russian",
        "pt": "Portuguese", "it": "Italian", "ar": "Arabic", "nl": "Dutch",
    }
    lang_name = lang_names.get(tgt_lang, tgt_lang)

    current_hint = ""
    if current_translation:
        current_hint = (
            f"\nCurrent (too long) translation: {current_translation}\n"
            f"Shorten this while keeping core meaning."
        )

    system = (
        f"You are a dubbing translator under strict length constraint.\n"
        f"Translate the source into natural spoken {lang_name}, MAX {max_syllables} syllables.\n"
        f"The result will be spoken aloud — it MUST fit within the syllable limit.\n"
        f"Drop non-essential words, use contractions, prefer shorter synonyms.\n"
        f"Preserve the core meaning. Output ONLY the translation, nothing else.{current_hint}"
    )

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": original_text},
                ],
                "temperature": 0.2,
                "max_tokens": 4096,  # reasoning 모델 여유 (2048 → 4096)
                "reasoning_effort": "low",
            },
            timeout=300,  # 600 → 300 (빨리 실패하고 폴백)
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        out = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        # 프롬프트 잔재 제거 (혹시 LLM이 말풍선 같이 출력하면)
        import re as _re
        out = _re.sub(r'^["\'\[\(]*|["\'\]\)]*$', '', out).strip()
        return out or None
    except Exception as e:
        print(f"[Retranslate] 실패 ({type(e).__name__}): {e}")
        return None


# ─── Step 9: 음성 합성 ────────────────────────────────────────



# === Whisper validation helper (외국어 환각 detect) ===
_whisper_validator = None

def _load_whisper_validator():
    global _whisper_validator
    if _whisper_validator is not None:
        return _whisper_validator
    try:
        import torch
        from transformers import pipeline
        _whisper_validator = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=0 if torch.cuda.is_available() else -1,
        )
        print("[Validator] whisper-tiny 로드 ✅")
        return _whisper_validator
    except Exception as e:
        print(f"[Validator] whisper 로드 실패: {e}")
        return None


def _is_korean_audio(audio_arr, sample_rate, korean_ratio_threshold: float = 0.5) -> tuple:
    """audio가 한국어 발화인지 검증.

    Returns:
        (is_korean: bool, korean_ratio: float, transcribed: str)
    """
    val = _load_whisper_validator()
    if val is None:
        return True, 1.0, ""  # validator 없으면 통과
    if len(audio_arr) < 16000 * 0.5:  # 0.5초 미만은 스킵
        return True, 1.0, ""
    try:
        # whisper input: 16kHz mono
        if sample_rate != 16000:
            import librosa
            audio_16k = librosa.resample(audio_arr.astype("float32"),
                                         orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio_arr.astype("float32")
        result = val({"array": audio_16k, "sampling_rate": 16000})
        text = result.get("text", "").strip()
        if not text:
            return True, 1.0, ""

        # 한국어 글자 비율
        total_letters = sum(1 for c in text if c.isalpha() or 0xAC00 <= ord(c) <= 0xD7A3)
        if total_letters == 0:
            return True, 1.0, text  # whisper noise
        korean_letters = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7A3)
        ratio = korean_letters / total_letters
        return ratio >= korean_ratio_threshold, ratio, text
    except Exception as e:
        return True, 1.0, ""  # 에러 시 통과 (안전)


# === COSY_DAEMON_CLIENT (v28): TTS 데몬 사용으로 모델 로딩 60-90초 절감 ===
COSY_DAEMON_URL = os.environ.get("COSY_DAEMON_URL", "http://127.0.0.1:8901")
_cosy_daemon_alive: Optional[bool] = None  # 캐시: None=미확인, True=alive, False=dead

def _check_cosy_daemon() -> bool:
    """데몬 alive 검사 (한 번만, 결과 캐시)."""
    global _cosy_daemon_alive
    if _cosy_daemon_alive is not None:
        return _cosy_daemon_alive
    try:
        import requests as _rq
        r = _rq.get(f"{COSY_DAEMON_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("model_loaded"):
            _cosy_daemon_alive = True
            print(f"[Cosy] daemon alive at {COSY_DAEMON_URL} ⚡ (60-90s 모델 로딩 절감)")
            return True
    except Exception:
        pass
    _cosy_daemon_alive = False
    return False


def _synthesize_via_daemon(text, ref_audio, speed, tone, emotion):
    """데몬 HTTP 호출로 TTS 합성. 실패 시 None."""
    try:
        import requests as _rq
        import base64 as _b64
        import io as _io
        r = _rq.post(f"{COSY_DAEMON_URL}/synthesize", json={
            "text": text,
            "ref_audio_path": ref_audio,
            "speed": speed,
            "tone": tone,
            "emotion": emotion,
        }, timeout=300)
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            print(f"[Cosy] daemon error: {data.get('error')}")
            return None
        wav_bytes = _b64.b64decode(data["audio_b64"])
        wav, _sr = sf.read(_io.BytesIO(wav_bytes))
        return wav.astype(np.float32)
    except Exception as e:
        print(f"[Cosy] daemon call 실패: {e}")
        return None


def synthesize_segment_cosy(
    text: str,
    ref_audio: str,
    lang: str,
    speed: float = 1.0,
    emotion: str = "Neutral",
    tts_context: str = "",
    tts_emotion: str = "",
) -> np.ndarray:
    """
    CosyVoice3로 세그먼트 1개 합성.

    Instruction 우선순위:
      1. tts_context + tts_emotion 둘 중 하나라도 있으면 → LLM 자연어 묘사 사용
      2. 없으면 → emotion 카테고리 6-key 매핑 (폴백/하위 호환)
      3. emotion=Neutral 이면 → 빈 instruction (강제 자연 합성)

    데몬 사용 우선:
      COSY_DAEMON_URL alive면 HTTP 호출로 60-90초 모델 로딩 절감.
      데몬 없으면 inline 합성 (이전 동작).
    """
    if not ref_audio or not os.path.exists(ref_audio):
        print(f"[TTS] 레퍼런스 파일 없음: {ref_audio}")
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    # 데몬 alive면 데몬으로 합성
    if _check_cosy_daemon():
        result = _synthesize_via_daemon(
            text=text, ref_audio=ref_audio, speed=speed,
            tone=tts_emotion or "", emotion=emotion,
        )
        if result is not None:
            return result
        print(f"[Cosy] daemon 합성 실패 → inline fallback")

    if _cosy_model is None:
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    # ── instruction 결정 ──
    # 폴백 카테고리 매핑 (LLM 묘사 없을 때만 사용)
    # 다국어 확장 포인트: tgt_lang별로 instruction을 target 언어로 작성 가능
    emotion_instruction = {
        "Angry": "Speak with firm conviction.",
        "Sad": "Speak reflectively and thoughtfully.",
        "Happy": "Speak with a warm, engaging tone.",
        "Surprised": "Speak with genuine curiosity.",
        "Scared": "Speak with quiet tension.",
        "Neutral": "",
    }

    # === v15: 팀원 검증 imperative format prefix ===
    # 학습 분포 매칭: "You are a helpful assistant. Please say a sentence as loudly as possible."
    # 팀원 적용:     "You are a helpful assistant. Please say this sentence in a casual, ..."
    # 우리 적용:     "You are a helpful assistant. Please say this sentence {tone}."
    #
    # tts_emotion (= tone, LLM이 v15 prompt로 출력)이 풀 imperative phrase

    # === v22 (5/14): 팀원 master_timeline_cosyvoice.json 형식 적용 ===
    # LATENTSYNC_EMOTION_INTENSITY: mild/moderate/strong (instruct_text 톤 강도)
    _intensity = os.environ.get("LATENTSYNC_EMOTION_INTENSITY", "moderate").lower()
    intensity_tone_maps = {
        "mild": {
            "Sad":       "with a faint hint of sadness",
            "Angry":     "with a slight edge",
            "Happy":     "with a faint hint of warmth",
            "Surprised": "with a faint raised intonation",
            "Scared":    "with a faint breathy hesitation",
            "Neutral":   "",
        },
        "moderate": {
            "Sad":       "with a gentle note of sadness",
            "Angry":     "with sharp confrontational intensity",
            "Happy":     "with a bright energetic edge",
            "Surprised": "with a subtle raised intonation",
            "Scared":    "with a tense breathy quality",
            "Neutral":   "",
        },
        "strong": {
            "Sad":       "with deep, anguished sadness",
            "Angry":     "with sharp, raised, confrontational tone, intense urgency",
            "Happy":     "in a bright, energetic tone with lively prosody",
            "Surprised": "with sudden, sharp surprise, raised intonation",
            "Scared":    "with raw, tense fear, shaky breathy voice",
            "Neutral":   "",
        },
    }
    cat_tone_map = intensity_tone_maps.get(_intensity, intensity_tone_maps["moderate"])
    if tts_emotion:
        # LLM 풍부 묘사 (이미 짧고 자연스러운 형식)
        tone_phrase = tts_emotion.strip().rstrip(".")
    else:
        tone_phrase = cat_tone_map.get(emotion, cat_tone_map["Neutral"])

    # 팀원 형식: "Please say it close to the speaker's natural delivery, with [톤]"
    # v129+: F5-TTS Issue #315 + 연구 결과: instruction text가 target 언어와 다르면 cross-lingual accent leakage
    # → tgt_lang 별로 instruction을 target 언어로 작성 (기계음 방지)
    _instruct_by_lang = {
        "ko": ("자연스러운 화자 음성에 가깝게 말해주세요", "톤"),
        "ja": ("話者の自然な声に近いトーンで話してください", "トーン"),
        "zh": ("请用接近说话人自然语调的方式说", "语调"),
        "en": ("Please say it close to the speaker's natural delivery", "tone"),
    }
    _base_phrase, _tone_label = _instruct_by_lang.get(lang, _instruct_by_lang["en"])
    if tone_phrase:
        instruct_phrase = f"{_base_phrase}, {tone_phrase}"
    else:
        instruct_phrase = _base_phrase
    instruct_text = f"You are a helpful assistant. {instruct_phrase}.<|endofprompt|>"

    ref_16k = os.path.join(tempfile.gettempdir(), "ref_16k_temp.wav")
    try:
        # 레퍼런스를 16kHz 모노로 변환 (CosyVoice3 요구사항)
        subprocess.run([
            "ffmpeg", "-y", "-i", ref_audio,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", ref_16k
        ], capture_output=True)

        output = []
        # v21: instruct2 + pace 명시
        for result in _cosy_model.inference_instruct2(
            tts_text=text,
            instruct_text=instruct_text,
            prompt_wav=ref_16k,
            stream=False,
            speed=speed
        ):
            output.append(result["tts_speech"].squeeze().numpy())

        if not output:
            return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

        wav = np.concatenate(output, axis=0).astype(np.float32)

        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav * (0.9 / peak)

        # 5/12 first-word fix: CosyVoice cold-start 시 첫 50ms 가 high-entropy
        # → 기계음/click 들림. 짧은 fade-in (50ms ramp) 으로 부드럽게.
        # v20 (5/15): 끝에도 짧은 fade-out 추가 — trim 발생 시 abrupt cut 완화.
        # 사용자 피드백 "발화가 끝에서 자꾸 짤려" → 마지막 30ms ramp-down으로
        # 청각적 잘림 인상 감소.
        _native_sr = _cosy_model.sample_rate
        _fade_n = int(0.05 * _native_sr)
        _fade_out_n = int(0.03 * _native_sr)
        if len(wav) > _fade_n * 2 and _fade_n > 0:
            ramp = np.linspace(0.0, 1.0, _fade_n, dtype=np.float32)
            wav[:_fade_n] *= ramp
        if len(wav) > _fade_out_n * 2 and _fade_out_n > 0:
            ramp_out = np.linspace(1.0, 0.0, _fade_out_n, dtype=np.float32)
            wav[-_fade_out_n:] *= ramp_out

        # CosyVoice3 출력(24000Hz)과 TTS_SAMPLE_RATE가 다르면 리샘플링
        if _native_sr != TTS_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=_native_sr, target_sr=TTS_SAMPLE_RATE)

        return wav

    except Exception as e:
        print(f"[TTS] CosyVoice3 합성 실패: {e}")
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)
    finally:
        if os.path.exists(ref_16k):
            os.unlink(ref_16k)


def synthesize_chunk(segments, profiles, chunk_name, tgt_lang,
                     video_duration: Optional[float] = None):
    """
    TTS 합성 + MOS 평가 + 길이 조정 + 배치.
    수정 통합:
      - H: rubberband WSOLA (아티팩트 방지)
      - I: MAX_STRETCH=1.15 (보수적 압축 한도)
      - D: time_stretch 후 peak normalize
      - M: overflow 예상 시 번역 축약 재요청
      - N: 마지막 세그먼트가 video_duration 넘지 않게 보호
    """
    if not segments:
        raise ValueError("segments가 비어 있습니다")

    # 🔥 수정 N: 비디오 길이 제약
    #    청크 비디오 길이가 주어지면 마지막 세그먼트의 합성이
    #    비디오 밖으로 나가지 않게 보장.
    if video_duration is None:
        # 안 받으면 세그먼트 끝 + 여유 5초로 추정 (하위 호환)
        video_duration = max(seg.end for seg in segments) + 5.0

    # 버퍼: 비디오 길이 + 5초 (overflow 허용 여유)
    total_samples = int((video_duration + 5.0) * TTS_SAMPLE_RATE)
    output_audio = np.zeros(total_samples, dtype=np.float32)

    # 같은 화자 연속 세그먼트 묶기 (문장 단위 분할 후에도 병합 로직 유지)
    # 병합 기준: 같은 화자 + 같은 감정 + gap < 0.3s (문장 내부 호흡만 묶기)
    merged = []
    i = 0
    while i < len(segments):
        group = [segments[i]]
        while (i + 1 < len(segments) and
               segments[i + 1].speaker == segments[i].speaker and
               segments[i + 1].emotion == segments[i].emotion and
               segments[i + 1].start - segments[i].end < 0.3):
            i += 1
            group.append(segments[i])
        merged.append(group)
        i += 1

    # 길이 조정 파라미터 (5/12 audio quality fix: 기계음 방지)
    # atempo > 1.07 또는 < 0.95 부터 phase vocoder artifact (metallic ring)
    # 가 들리기 시작 → 입 sync 일부 손실하더라도 음질 우선.
    # 너무 짧거나 긴 segment 는 PREDICT_THRESHOLD 가 잡아서 재번역 유도.
    # v17: 환경변수화. rubberband WSOLA는 1.15까지 자연스러움 (VideoDubber AAAI 2023).
    #   LATENTSYNC_MAX_STRETCH (default 1.07 안전, 1.15 권장 with rubberband)
    #   LATENTSYNC_MIN_STRETCH (default 0.95 안전, 0.90 권장)
    #   LATENTSYNC_TOL_LATE   (default 0.15s, 0.20s = LLM soft range와 매칭)
    import os as _os_stretch
    MAX_STRETCH = float(_os_stretch.environ.get("LATENTSYNC_MAX_STRETCH", "1.07"))
    MIN_STRETCH = float(_os_stretch.environ.get("LATENTSYNC_MIN_STRETCH", "0.95"))
    TOL_LATE = float(_os_stretch.environ.get("LATENTSYNC_TOL_LATE", "0.15"))
    # 다국어 확장: 타겟 언어별 발화 속도. 현재는 한국어만.
    LANG_SPEECH_RATE = {
        "ko": 5.5, "ja": 7.5, "zh": 5.0, "en": 3.5,
        "es": 6.5, "fr": 6.0, "de": 4.5, "ru": 5.0,
        "pt": 5.5, "it": 6.0, "ar": 5.0, "nl": 4.5,
    }
    # 🔥 수정 M-2: 감정 지시에 따른 발화 속도 보정 배율
    #   Sad/Scared 감정 instruction은 CosyVoice3가 천천히 합성함 → 예측 속도를 낮춰야
    #   overflow 예측이 정확해진다. 이 값 없이는 ratio=2.17 같은 극단 overflow 못 잡음.
    EMOTION_RATE_FACTOR = {
        "Sad": 0.75,        # "Speak reflectively" → 느림
        "Scared": 0.80,     # "quiet tension" → 느림
        "Neutral": 1.00,
        "Happy": 1.00,
        "Angry": 0.95,      # "firm conviction" → 약간 느림
        "Surprised": 1.00,
    }
    speech_rate = LANG_SPEECH_RATE.get(tgt_lang, 5.5)

    for gi, group in enumerate(merged):
        combined_text = " ".join(seg.translated for seg in group if seg.translated.strip())
        if not combined_text.strip():
            continue

        first_seg = group[0]
        last_seg = group[-1]

        profile = profiles.get(first_seg.speaker)
        ref_path = profile.get_ref(first_seg.emotion) if profile else ""
        if not ref_path or not os.path.exists(ref_path):
            ref_path = profile.get_ref("Neutral") if profile else ""

        # SELF_REF_FALLBACK: profile 없으면 (짧은 outlier 화자 등)
        # v30 (5/15): 같은 화자의 모든 segment 를 concat 해서 reference 생성.
        # 기존 동작 (단일 segment + padding) 은 매우 짧은 화자에게 기계음 유발
        # (예: SPK_05 0.64s segment 자체 audio → CosyVoice3 voice cloning 빈약).
        # concat 으로 3s+ 누적되면 zero-shot cloning 안정성 ↑.
        if not ref_path or not os.path.exists(ref_path):
            try:
                vocals_path = os.path.join(VOCALS_DIR, f"{chunk_name}_clean_vocals.wav")
                if not os.path.exists(vocals_path):
                    vocals_path = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
                if os.path.exists(vocals_path):
                    self_ref = os.path.join(
                        tempfile.gettempdir(),
                        f"selfref_{chunk_name}_{first_seg.speaker}_{gi}.wav"
                    )
                    # 같은 화자의 모든 segment 수집 (전체 segments 인자에서)
                    same_spk = [s for s in segments if s.speaker == first_seg.speaker]
                    total_dur_same = sum(s.end - s.start for s in same_spk)

                    # v30: concat 시도 — 총 길이 ≥1.5s 면 concat (CosyVoice3 안정 하한)
                    used_concat = False
                    if total_dur_same >= 1.5 and len(same_spk) >= 2:
                        import soundfile as _sf_ref
                        import numpy as _np_ref
                        try:
                            audio_ref, sr_ref = _sf_ref.read(vocals_path)
                            if audio_ref.ndim > 1:
                                audio_ref = _np_ref.mean(audio_ref, axis=1)
                            chunks_ref = []
                            silence_n = int(0.15 * sr_ref)  # 150ms silence between
                            silence = _np_ref.zeros(silence_n, dtype=audio_ref.dtype)
                            for s in sorted(same_spk, key=lambda x: x.start):
                                s_idx = int(s.start * sr_ref)
                                e_idx = int(s.end * sr_ref)
                                if e_idx > s_idx:
                                    chunks_ref.append(audio_ref[s_idx:e_idx])
                                    chunks_ref.append(silence)
                            if chunks_ref:
                                concat_audio = _np_ref.concatenate(chunks_ref[:-1])  # drop trailing silence
                                # 16kHz mono cast for CosyVoice3 (sf 직접 write 후 ffmpeg resample)
                                tmp_concat = self_ref + ".raw.wav"
                                _sf_ref.write(tmp_concat, concat_audio, sr_ref)
                                r = subprocess.run([
                                    "ffmpeg", "-y", "-loglevel", "error", "-i", tmp_concat,
                                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", self_ref,
                                ], capture_output=True, text=True)
                                if os.path.exists(tmp_concat):
                                    os.unlink(tmp_concat)
                                if r.returncode == 0 and os.path.exists(self_ref):
                                    ref_path = self_ref
                                    used_concat = True
                                    print(f"  ↳ self-ref concat: {first_seg.speaker} "
                                          f"({len(same_spk)} segs, total {total_dur_same:.2f}s)")
                        except Exception as _ce:
                            print(f"  ↳ self-ref concat 실패: {_ce} → 단일 segment fallback")
                            used_concat = False

                    # concat 실패 또는 segment 1개 뿐이면 기존 단일 segment 동작
                    if not used_concat:
                        seg_dur = last_seg.end - first_seg.start
                        pad = 0.5 if seg_dur < 2.0 else 0.0
                        ext_start = max(0, first_seg.start - pad)
                        ext_end = last_seg.end + pad
                        r = subprocess.run([
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-ss", str(ext_start), "-to", str(ext_end),
                            "-i", vocals_path,
                            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                            self_ref,
                        ], capture_output=True, text=True)
                        if r.returncode == 0 and os.path.exists(self_ref):
                            ref_path = self_ref
                            print(f"  ↳ self-ref single: {first_seg.speaker} ({seg_dur:.2f}s, "
                                  f"+{pad*2:.1f}s padding)")

                    # v126+: ref < 3s 면 audio loop 적용 (CosyVoice3 기계음 방지)
                    # 본인 voice 그대로 + 0.2s silence 끼고 반복 → ≥ min_ref_dur
                    if ref_path and os.path.exists(ref_path):
                        try:
                            import soundfile as _sf_loop
                            import numpy as _np_loop
                            _min_ref = float(os.environ.get("LATENTSYNC_MIN_REF_DUR", "3.0"))
                            _loop_audio, _loop_sr = _sf_loop.read(ref_path)
                            if _loop_audio.ndim > 1:
                                _loop_audio = _np_loop.mean(_loop_audio, axis=1)
                            _cur_dur = len(_loop_audio) / _loop_sr
                            if _cur_dur > 0 and _cur_dur < _min_ref:
                                _loop_count = int(_min_ref / _cur_dur) + 1
                                _silence = _np_loop.zeros(int(0.2 * _loop_sr), dtype=_loop_audio.dtype)
                                _looped = _loop_audio.copy()
                                for _ in range(_loop_count - 1):
                                    _looped = _np_loop.concatenate([_looped, _silence, _loop_audio])
                                _sf_loop.write(ref_path, _looped, _loop_sr)
                                _new_dur = len(_looped) / _loop_sr
                                print(f"  ↳ self-ref loop: {first_seg.speaker} "
                                      f"{_cur_dur:.2f}s × {_loop_count} → {_new_dur:.2f}s "
                                      f"(min_ref={_min_ref}s)", flush=True)
                        except Exception as _le:
                            print(f"  ↳ self-ref loop 실패: {_le} (원본 ref 유지)")
            except Exception as _e:
                print(f"  ↳ self-ref 실패: {_e}")

        if not ref_path or not os.path.exists(ref_path):
            print(f"  ⚠️ {first_seg.speaker} reference 없음 — segment skip")
            continue

        # v126+: profile ref도 < 3s면 loop (silence trim으로 짧아진 경우)
        # 자기 본인 voice 그대로 반복 → CosyVoice3 클로닝 안정성 보장
        try:
            import soundfile as _sf_p
            import numpy as _np_p
            _min_ref_p = float(os.environ.get("LATENTSYNC_MIN_REF_DUR", "3.0"))
            _p_audio, _p_sr = _sf_p.read(ref_path)
            if _p_audio.ndim > 1:
                _p_audio = _np_p.mean(_p_audio, axis=1)
            _p_dur = len(_p_audio) / _p_sr
            if _p_dur > 0 and _p_dur < _min_ref_p:
                _p_count = int(_min_ref_p / _p_dur) + 1
                _p_silence = _np_p.zeros(int(0.2 * _p_sr), dtype=_p_audio.dtype)
                _p_looped = _p_audio.copy()
                for _ in range(_p_count - 1):
                    _p_looped = _np_p.concatenate([_p_looped, _p_silence, _p_audio])
                # 새 파일 (원본 보존)
                _loop_path = ref_path.replace(".wav", "_loop.wav")
                _sf_p.write(_loop_path, _p_looped, _p_sr)
                ref_path = _loop_path
                _p_new = len(_p_looped) / _p_sr
                print(f"  ↳ profile-ref loop: {first_seg.speaker} "
                      f"{_p_dur:.2f}s × {_p_count} → {_p_new:.2f}s (min={_min_ref_p}s)", flush=True)
        except Exception as _ple:
            print(f"  ↳ profile-ref duration check 실패: {_ple} (원본 유지)")

        group_start = first_seg.start
        group_end   = last_seg.end
        group_duration = group_end - group_start

        # 🔥 수정 N: 다음 그룹 start vs 비디오 끝, 더 이른 쪽을 상한으로
        if gi + 1 < len(merged):
            next_boundary = merged[gi + 1][0].start
        else:
            # 마지막 그룹: 비디오 끝을 절대 경계로 사용
            next_boundary = video_duration
        max_allowed_duration = max(0.1, next_boundary - group_start - 0.05)
        max_allowed_samples = int(max_allowed_duration * TTS_SAMPLE_RATE)
        is_last_group = (gi == len(merged) - 1)

        print(f"[TTS] [{first_seg.speaker}][{first_seg.emotion}] "
              f"{group_start:.2f}~{group_end:.2f}s (max {max_allowed_duration:.2f}s"
              f"{', LAST' if is_last_group else ''}) "
              f"→ '{combined_text[:50]}'")

        # 🔥 수정 M: overflow 예상 시 사전 축약
        #    자연 발화 속도 예측: 한국어는 ~5.5 음절/초
        #    🔥 수정 M-2: 감정별 속도 보정 (Sad/Scared는 느림)
        #    🔥 수정 M-3: PREDICT_THRESHOLD 1.10 → 1.00 (MAX_STRETCH 수준으로 타이트)
        #    🔥 LLM 묘사 배율 추가 — 묘사 텍스트에 'slow/reflective' 등이 있으면 0.85, 'quick/urgent' 1.05
        combined_syllables = count_korean_syllables(combined_text) if tgt_lang == "ko" \
                             else len(combined_text)  # 다국어 임시 fallback
        emotion_factor = EMOTION_RATE_FACTOR.get(first_seg.emotion, 1.0)
        # LLM 묘사가 있으면 카테고리와 곱해서 더 정확한 예측
        desc_factor = emotion_desc_rate_factor(getattr(first_seg, 'tts_emotion', '') or '')
        combined_factor = emotion_factor * desc_factor
        effective_rate = speech_rate * combined_factor
        predicted_dur = combined_syllables / effective_rate if effective_rate > 0 else 0
        # v17: PREDICT_THRESHOLD를 MAX_STRETCH 에서 분리.
        # 사전 재번역은 좀 더 관대해도 됨 — TTS speed retry + atempo가 후처리에서 흡수.
        # MAX_STRETCH 와 동일 묶음(1.07)은 너무 적극적 재번역 → 의미 손상.
        PREDICT_THRESHOLD = float(os.environ.get("LATENTSYNC_PREDICT_THRESHOLD", "1.20"))

        if predicted_dur > max_allowed_duration * PREDICT_THRESHOLD and len(group) == 1:
            # 단일 세그먼트 + overflow 예상 → 재번역 시도
            target_syl = int(max_allowed_duration * effective_rate)
            desc_part = f", 묘사='{first_seg.tts_emotion[:30]}'" if getattr(first_seg, 'tts_emotion', '') else ""
            print(f"  🔄 overflow 예상 ({predicted_dur:.1f}s > {max_allowed_duration:.1f}s, "
                  f"감정={first_seg.emotion} cat={emotion_factor:.2f} desc={desc_factor:.2f}{desc_part}) — "
                  f"축약 재번역 요청 (target ≤{target_syl}음절)")
            shorter = retranslate_shorter(
                original_text=group[0].text,
                tgt_lang=tgt_lang,
                max_syllables=target_syl,
                current_translation=group[0].translated,
            )
            if shorter:
                new_syl = count_korean_syllables(shorter) if tgt_lang == "ko" else len(shorter)
                # v17: 축약 결과 검증 — 너무 짧으면 의미 손상 의심 → 거부
                # target_syl × 0.5 미만이면 LLM이 의미 버리고 음절만 맞춘 것
                min_accept = max(2, int(target_syl * 0.5))
                if new_syl < combined_syllables and new_syl >= min_accept:
                    print(f"  ✅ 축약됨: {combined_syllables}음절 → {new_syl}음절 (target {target_syl})")
                    combined_text = shorter
                    group[0].translated = shorter
                elif new_syl < min_accept:
                    print(f"  ⚠️ 축약 거부 ({new_syl}음절 < {min_accept} 의미 손상 의심) — 원본 사용")
                else:
                    print(f"  ⚠️ 축약 실패 (여전히 {new_syl}음절) — 원본 사용")

        # POST_TTS_LENGTH_CHECK: speed=1.0 합성 후 길이 측정
        # overflow 시 speed=1.10, 1.15로 재합성하여 자연스럽게 줄이기 시도
        # (time_stretch보다 음질 좋음 — TTS는 phoneme duration 자연 조절, time_stretch는 사후 압축)
        MOS_RESYNTH_ENABLED = False
        mos_score = 0.0
        retry_count = 0

        # 1차: emotion-based speed (v13)
        audio_chunk = synthesize_segment_cosy(
            text=combined_text,
            ref_audio=ref_path,
            lang=tgt_lang,
            speed=getattr(first_seg, "speed", 1.0),
            emotion=first_seg.emotion,
            tts_context=getattr(first_seg, 'tts_context', '') or '',
            tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
        )

        # 길이 측정 + 효율적 speed retry (OPTIMIZED)
        cur_duration = len(audio_chunk) / TTS_SAMPLE_RATE
        ratio = cur_duration / max(0.1, max_allowed_duration)

        # ratio 따라 retry 전략 결정 (불필요한 합성 회피)
        if ratio < 0.80:
            # v16 NEW: 한국어 너무 짧음 → speed=0.90으로 자연 늘림 (atempo 회피)
            # 사용자 페인 "한글 짧으니 일부러 길게 말하는 인상" 직접 fix
            retry_audio = synthesize_segment_cosy(
                text=combined_text,
                ref_audio=ref_path,
                lang=tgt_lang,
                speed=0.90,
                emotion=first_seg.emotion,
                tts_context=getattr(first_seg, 'tts_context', '') or '',
                tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
            )
            retry_dur = len(retry_audio) / TTS_SAMPLE_RATE
            retry_ratio = retry_dur / max(0.1, max_allowed_duration)
            # retry가 더 길고 max 안 침범하면 채택
            if retry_dur > cur_duration and retry_ratio <= 1.10:
                print(f"  ↳ ratio<0.80 → speed=0.90 retry ({cur_duration:.2f}s → {retry_dur:.2f}s, ratio {ratio:.2f}→{retry_ratio:.2f})")
                audio_chunk = retry_audio
        elif ratio < 1.05:
            pass  # 이미 OK, retry 안 함
        elif ratio < 1.40:
            # 1.15x 한 번만 시도 (speed retry로 해결 가능한 범위)
            retry_audio = synthesize_segment_cosy(
                text=combined_text,
                ref_audio=ref_path,
                lang=tgt_lang,
                speed=1.15,
                emotion=first_seg.emotion,
                tts_context=getattr(first_seg, 'tts_context', '') or '',
                tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
            )
            retry_dur = len(retry_audio) / TTS_SAMPLE_RATE
            if retry_dur < cur_duration:
                print(f"  ↳ speed=1.15 재합성 ({cur_duration:.2f}s → {retry_dur:.2f}s)")
                audio_chunk = retry_audio
        else:
            # ratio >= 1.40: speed retry로 해결 불가능한 격차 → 즉시 time_stretch fallback
            print(f"  ↳ ratio={ratio:.2f} 너무 큼 → speed retry skip, "
                  f"time_stretch + trim으로 fallback")

        if MOS_RESYNTH_ENABLED and _mos_evaluator is not None:
            MAX_RETRIES = 2
            MOS_THRESHOLD = 3.5
            for attempt in range(MAX_RETRIES + 1):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_chunk, TTS_SAMPLE_RATE)
                    try:
                        mos_score = _mos_evaluator.evaluate(tmp.name)
                    except Exception as e:
                        print(f"[MOS] 평가 실패: {e}")
                        mos_score = 0.0
                    finally:
                        os.unlink(tmp.name)

                if mos_score >= MOS_THRESHOLD:
                    print(f"[MOS] {first_seg.speaker}/{first_seg.emotion} "
                          f"MOS={mos_score:.2f} ✅ (시도 {attempt + 1})")
                    break

                retry_count = attempt + 1
                if attempt < MAX_RETRIES:
                    print(f"[MOS] MOS={mos_score:.2f} < {MOS_THRESHOLD} "
                          f"→ 재합성 ({retry_count}/{MAX_RETRIES})")
                    audio_chunk = synthesize_segment_cosy(
                        text=combined_text,
                        ref_audio=ref_path,
                        lang=tgt_lang,
                        speed=speed,
                        emotion=first_seg.emotion,
                        tts_context=getattr(first_seg, 'tts_context', '') or '',
                        tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
                    )
                else:
                    print(f"[MOS] MOS={mos_score:.2f} → 최대 재시도 도달")

        for seg in group:
            seg._tts_mos = mos_score
            seg._tts_retries = retry_count

        # ─── 길이 조정 (LIP_SYNC v27: 입 움직임에 정확히 맞추기) ─────────
        # target = group_duration (영어 원문 발화 시간 = 입 움직이는 시간)
        # 짧으면 atempo로 늘리고, 길면 압축 → 정확한 lip-sync
        # 단, max_allowed_duration (다음 segment 시작 전)은 절대 침범 금지
        natural_samples = len(audio_chunk)
        natural_duration = natural_samples / TTS_SAMPLE_RATE
        target_duration = group_duration  # 입 움직임 끝
        target_samples = int(target_duration * TTS_SAMPLE_RATE)

        # ratio: 1.0 = 입 움직임과 정확히 일치
        ratio = natural_duration / max(0.1, target_duration)

        # tolerance: target * (1 ± TOL_LATE/target) 안이면 "거의 맞음"
        tol_ratio_late  = 1.0 + (TOL_LATE / max(0.1, target_duration))  # 살짝 길음 허용
        tol_ratio_early = 1.0 - (TOL_LATE / max(0.1, target_duration))  # 살짝 짧음 허용

        if tol_ratio_early <= ratio <= tol_ratio_late:
            # 거의 맞음 → 그대로 (자연스러움)
            print(f"  ↳ lip-sync OK (target={target_duration:.2f}s, "
                  f"actual={natural_duration:.2f}s, ratio={ratio:.2f})")

        elif ratio < tol_ratio_early:
            # 한국어가 짧음 → atempo로 늘리기 (입 움직이는 동안 말 채우기)
            stretch_ratio = max(MIN_STRETCH, ratio)  # 0.85 한계
            if stretch_ratio < ratio:
                # 한계 도달 (ratio < 0.85) — 더 못 늘림, silence 일부 발생
                print(f"  ⚠️ 한국어 너무 짧음 (ratio={ratio:.2f} < {MIN_STRETCH}) — "
                      f"atempo {MIN_STRETCH}x 한계, silence 일부 발생")
                audio_chunk = high_quality_time_stretch(
                    audio_chunk, TTS_SAMPLE_RATE, MIN_STRETCH
                )
            else:
                audio_chunk = high_quality_time_stretch(
                    audio_chunk, TTS_SAMPLE_RATE, stretch_ratio
                )
                print(f"  ↳ 짧음 → atempo {stretch_ratio:.2f}x 늘림 "
                      f"(target {target_duration:.2f}s 맞춤)")
            audio_chunk = normalize_peak(audio_chunk, target=0.9)

        elif ratio <= MAX_STRETCH:
            # 한국어가 김 → atempo로 압축 (1.25x까지 자연스러움 한계)
            audio_chunk = high_quality_time_stretch(
                audio_chunk, TTS_SAMPLE_RATE, ratio
            )
            audio_chunk = normalize_peak(audio_chunk, target=0.9)
            print(f"  ↳ 김 → atempo {ratio:.2f}x 압축 (lip-sync 맞춤)")

        else:
            # 1.25x로도 부족한 큰 overflow
            # 1.25x 압축 후 max_allowed (다음 segment 침범 방지)까지 trim
            audio_chunk = high_quality_time_stretch(
                audio_chunk, TTS_SAMPLE_RATE, MAX_STRETCH
            )
            audio_chunk = normalize_peak(audio_chunk, target=0.9)

            if len(audio_chunk) > max_allowed_samples:
                trimmed_sec = (len(audio_chunk) - max_allowed_samples) / TTS_SAMPLE_RATE
                audio_chunk = audio_chunk[:max_allowed_samples]
                # v20: trim 후 마지막 30ms fade-out — abrupt cut 청각 인상 완화
                _fadeout_n = int(0.03 * TTS_SAMPLE_RATE)
                if len(audio_chunk) > _fadeout_n * 2 and _fadeout_n > 0:
                    _ramp_out = np.linspace(1.0, 0.0, _fadeout_n, dtype=np.float32)
                    audio_chunk[-_fadeout_n:] = audio_chunk[-_fadeout_n:] * _ramp_out
                if is_last_group:
                    print(f"  ⚠️ 마지막 segment ratio={ratio:.2f} — {MAX_STRETCH}x 압축 "
                          f"+ 비디오 끝 맞춰 {trimmed_sec:.2f}s trim (fade-out 30ms)")
                else:
                    print(f"  ⚠️ ratio={ratio:.2f} 너무 큼 — {MAX_STRETCH}x 압축 "
                          f"+ 다음 segment 침범 방지 {trimmed_sec:.2f}s trim (fade-out 30ms)")
            else:
                # 1.25x 압축으로 max_allowed 안에는 들어감 (단, group_duration 넘음)
                overflow_after = (len(audio_chunk) - target_samples) / TTS_SAMPLE_RATE
                print(f"  ⚠️ ratio={ratio:.2f} → {MAX_STRETCH}x 압축, "
                      f"입 멈춤 후 {overflow_after:.2f}s 더 발화")

        # ─── 배치 ──
        start_sample = int(group_start * TTS_SAMPLE_RATE)
        end_sample = min(start_sample + len(audio_chunk), total_samples)
        copy_len = end_sample - start_sample
        if copy_len > 0:
            output_audio[start_sample:end_sample] = audio_chunk[:copy_len]

        # v28 (5/14): Redub 지원 — segment 별 wav + 메타 저장
        try:
            from pathlib import Path as _Path
            _seg_dir = _Path(DUBBED_DIR) / f"{chunk_name}_segments"
            _seg_dir.mkdir(exist_ok=True)
            _seg_wav = _seg_dir / f"group_{gi:03d}_{first_seg.speaker}.wav"
            sf.write(str(_seg_wav), audio_chunk, TTS_SAMPLE_RATE)
            if not hasattr(synthesize_chunk, '_seg_meta_list'):
                synthesize_chunk._seg_meta_list = []
            synthesize_chunk._seg_meta_list.append({
                'group_idx': gi,
                'speaker': first_seg.speaker,
                'emotion': first_seg.emotion,
                'text': combined_text,
                'tts_emotion': getattr(first_seg, 'tts_emotion', '') or '',
                'tts_context': getattr(first_seg, 'tts_context', '') or '',
                'speed': float(getattr(first_seg, 'speed', 1.0)),
                'group_start': float(group_start),
                'group_end': float(group_end),
                'max_allowed_duration': float(max_allowed_duration),
                'segment_ids': [s.id for s in group],
                'wav_path': str(_seg_wav),
                'ref_path': ref_path,
            })
        except Exception as _e_redub:
            print(f"  [Redub] segment 메타 저장 실패: {_e_redub}")

    # v28 (5/14): segments.json 저장 (Redub 위해)
    try:
        from pathlib import Path as _Path
        _meta_dir = _Path(RUNS_DIR) / CURRENT_RUN_ID / "meta"
        _meta_dir.mkdir(exist_ok=True)
        _segments_json = _meta_dir / f"{chunk_name}_segments.json"
        seg_meta_list = getattr(synthesize_chunk, '_seg_meta_list', [])
        with open(_segments_json, 'w', encoding='utf-8') as _f:
            json.dump({
                'chunk_name': chunk_name,
                'video_duration': float(video_duration),
                'tts_sample_rate': TTS_SAMPLE_RATE,
                'groups': seg_meta_list,
            }, _f, ensure_ascii=False, indent=2)
        print(f"[Redub] segments meta 저장: {_segments_json} ({len(seg_meta_list)} groups)")
        # 다음 chunk 처리 위해 reset
        synthesize_chunk._seg_meta_list = []
    except Exception as _e:
        print(f"[Redub] segments.json 저장 실패: {_e}")

    # 🔥 SYNC FIX: dubbed.wav를 정확히 video_duration으로 트림.
    #    이전: total_samples = (video_duration + 5.0) * SR → 5초 trailing buffer가 그대로 저장
    #          → mix_audio의 amix `duration=first`가 이걸 따라가서 영상(64s)+오디오(69s) 불일치
    #          → MuseTalk이 영상 frame을 loop으로 5초 채우며 입은 새 audio 따라감 (시각적 disconnect)
    #    수정: 정확히 video_duration까지만 저장. 마지막 segment의 trim 보호와 중복 안전장치.
    final_samples = int(video_duration * TTS_SAMPLE_RATE)
    if len(output_audio) > final_samples:
        output_audio = output_audio[:final_samples]
        print(f"[TTS] dubbed.wav 트림: video_duration={video_duration:.3f}s "
              f"({final_samples} samples)")

    dubbed_path = os.path.join(DUBBED_DIR, f"{chunk_name}_dubbed.wav")
    sf.write(dubbed_path, output_audio, TTS_SAMPLE_RATE)
    print(f"[TTS] 더빙 저장: {dubbed_path}")
    return dubbed_path


# ─── Step 10: 믹싱 + 합치기 ──────────────────────────────────

def mix_audio(
    chunk_path: str,
    dubbed_path: str,
    bgm_path: str,
    output_path: str,
    dubbed_volume: float = 0.7,    # legacy fallback (loudnorm OFF 시)
    bgm_volume: float = 0.9,       # 5/7: BGM 원본 dynamics 보존 (0.6 → 0.9)
    use_loudnorm: bool = True,     # 5/7 NEW: perceptual loudness 매칭
    target_lufs: float = -23.0,    # 표준 대화 dialogue level (EBU R128)
) -> str:
    """
    FFmpeg로 더빙 오디오 + BGM 믹싱 후 원본 영상 트랙에 붙이기.

    INPUT:
      chunk_path    : str   — 원본 영상 청크 (비디오 트랙 사용)
      dubbed_path   : str   — 더빙 오디오
      bgm_path      : str   — 배경음
      output_path   : str   — 출력 파일
      dubbed_volume : float — 더빙 볼륨 (loudnorm OFF 시만 사용)
      bgm_volume    : float — BGM 볼륨 (기본 0.9, 원본 dynamics 보존)
      use_loudnorm  : bool  — True면 EBU R128 loudnorm으로 dubbed 자동 매칭 (권장)
      target_lufs   : float — 목표 perceptual loudness (-23 dialogue, -16 broadcast)

    OUTPUT:
      output_path : str — /data/chunks/movie_chunk_000_final.mp4
    """
    # 🔥 v24 (5/14): 팀원 compose_audio 파이프라인 형식 적용
    # 핵심 변경:
    #   1) 원본 sr 자동 감지 (16k/44.1k/48k 등) 후 그것으로 통일
    #   2) stereo 2ch 출력 (mono 압축 X, BGM stereo 보존)
    #   3) LC-AAC 강제 (HE-AAC SBR 비활성 → 96k 비정상 표시 X)
    #   4) dubbed mono → stereo (양쪽 동일 채널, 가운데 정위)
    try:
        _sr_probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate", "-of", "csv=p=0", chunk_path,
        ], text=True).strip()
        orig_sr = int(_sr_probe) if _sr_probe else 44100
    except Exception:
        orig_sr = 44100
    print(f"[Mix] 팀원 형식 적용: sr={orig_sr}Hz, stereo 2ch, LC-AAC 192k")

    # dubbed (24k mono) → stereo, BGM (원본 stereo) 유지
    if use_loudnorm:
        dubbed_filter = f"loudnorm=I={target_lufs}:TP=-2:LRA=11"
        filter_complex = (
            f"[1:a]aformat=channel_layouts=stereo:sample_rates={orig_sr},{dubbed_filter}[dub];"
            f"[2:a]aformat=channel_layouts=stereo:sample_rates={orig_sr},volume={bgm_volume}[bgm];"
            "[dub][bgm]amix=inputs=2:duration=shortest:normalize=0[a]"
        )
    else:
        filter_complex = (
            f"[1:a]aformat=channel_layouts=stereo:sample_rates={orig_sr},volume={dubbed_volume}[dub];"
            f"[2:a]aformat=channel_layouts=stereo:sample_rates={orig_sr},volume={bgm_volume}[bgm];"
            "[dub][bgm]amix=inputs=2:duration=shortest:normalize=0[a]"
        )
    cmd = [
        "ffmpeg",
        "-i", chunk_path,
        "-i", dubbed_path,
        "-i", bgm_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-profile:a", "aac_low",  # LC-AAC 강제 (HE-AAC SBR → 96k 비정상 방지)
        "-b:a", "192k",
        "-ar", str(orig_sr),       # 원본 sr 유지
        "-ac", "2",                # stereo 출력 (팀원 형식)
        "-shortest",
        output_path, "-y"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg mix 실패:\n{result.stderr}")

    print(f"[Mix] 완료: {output_path}")
    return output_path


# ─── 🎬 LatentSync 1.6 (diffusion) 립싱크 적용 ────────────────
# MuseTalk 대비 장점: 마스킹 부드러움 ↑, 시간 일관성 ↑, 파인튜닝 가능
# 단점: 느림 (~10x), 메모리 더 많이 사용
def apply_latent_sync(
    dubbed_video_path: str,
    output_path: str,
    inference_steps: int = 10,   # 5/11 update: DPM++ 10-step (TRT + TeaCache 와 함께)
    guidance_scale: float = 1.5,
    seed: int = 1247,
    config_name: str = "stage2_512_nf16.yaml",  # v27 default: 16 frames @ 512
    ckpt_path: Optional[str] = None,        # None이면 베이스 (LoRA 안 씀)
    enable_deepcache: bool = False,         # 5/7 sm_120 CUDA stream issue — OFF default
    # 5/11 update: 새 가속·품질 옵션 (env var 로 inference.py 에 전달)
    use_trt: bool = True,                   # TRT FP16 엔진 (2.55GB) 사용 — 3.18× 가속
    scheduler: str = "dpm",                 # DPMSolver++ (10 step = DDIM 20 동등)
    teacache_threshold: float = 0.1,        # TeaCache (timestep skip) — 추가 33% 가속
    profile_threshold: float = 0.35,        # 5/12: 정면만 lipsync (측면 yaw>0.35 skip, 입 떠다님 방지)
    face_diag_min_ratio: float = 0.0,       # 5/11: ASD 통합 후 비활성화 (drama medium-shot 살리기)
    chunk_seconds: int = 0,                 # >0 시 chunked inference (장편 영상 메모리 절약)
    face_strict: bool = False,              # 5/11: face_detector strict mode (드라마 안전)
) -> Optional[str]:
    """
    LatentSync 1.6으로 립싱크 적용 (diffusion 기반, 512x512 추론).

    INPUT:
      dubbed_video_path : 더빙된 한국어 오디오 포함 영상
      output_path       : 출력 경로
      config_name       : stage2_512.yaml (512 출력) 또는 stage2_efficient.yaml (256)
      ckpt_path         : 가중치 .pt. None이면 LATENT_SYNC_CKPT (기본)

    OUTPUT:
      성공 시 output_path, 실패 시 None
    """
    if not os.path.isdir(LATENT_SYNC_DIR):
        print(f"[Lipsync] ⚠️  LatentSync 레포 없음: {LATENT_SYNC_DIR}")
        return None
    if not os.path.isfile(LATENT_SYNC_PYTHON):
        print(f"[Lipsync] ⚠️  venv 없음: {LATENT_SYNC_PYTHON}")
        return None

    ckpt = ckpt_path or LATENT_SYNC_CKPT
    if not os.path.isfile(ckpt):
        print(f"[Lipsync] ⚠️  가중치 없음: {ckpt}")
        return None

    if not os.path.isfile(dubbed_video_path):
        print(f"[Lipsync] ❌ 입력 비디오 없음: {dubbed_video_path}")
        return None

    print(f"\n--- [Lipsync] LatentSync 1.6 적용 시작 ---")
    print(f"[Lipsync] 입력: {dubbed_video_path}")
    print(f"[Lipsync] 출력: {output_path}")
    print(f"[Lipsync] 가중치: {ckpt}")
    print(f"[Lipsync] config: {config_name}, steps={inference_steps}, guidance={guidance_scale}")

    # === 5/11 update: 가속·품질 옵션 env var 로 inference.py 에 전달 ===
    if use_trt:
        os.environ["LATENTSYNC_USE_TRT"] = "1"
        trt_engine = "/workspace/trt_work/engines/unet_fp16.trt"
        if os.path.isfile(trt_engine):
            os.environ["LATENTSYNC_TRT_ENGINE"] = trt_engine
            print(f"[Lipsync] TRT engine: {trt_engine}")
        else:
            print(f"[Lipsync] ⚠️  TRT engine not found, falling back to PyTorch UNet")
            os.environ["LATENTSYNC_USE_TRT"] = "0"
    else:
        os.environ.pop("LATENTSYNC_USE_TRT", None)

    os.environ["LATENTSYNC_SCHEDULER"] = scheduler  # "dpm" or "ddim"
    if teacache_threshold > 0:
        os.environ["LATENTSYNC_TEACACHE"] = str(teacache_threshold)
        print(f"[Lipsync] TeaCache rel_l1={teacache_threshold}")
    else:
        os.environ.pop("LATENTSYNC_TEACACHE", None)

    if profile_threshold > 0:
        os.environ["LATENTSYNC_PROFILE_THRESHOLD"] = str(profile_threshold)
        print(f"[Lipsync] Profile skip threshold={profile_threshold} (yaw)")
    else:
        os.environ.pop("LATENTSYNC_PROFILE_THRESHOLD", None)

    if face_diag_min_ratio > 0:
        os.environ["LATENTSYNC_FACE_DIAG_MIN_RATIO"] = str(face_diag_min_ratio)
        print(f"[Lipsync] Face distance min ratio={face_diag_min_ratio}")
    else:
        os.environ.pop("LATENTSYNC_FACE_DIAG_MIN_RATIO", None)

    if chunk_seconds > 0:
        os.environ["LATENTSYNC_CHUNK_SECONDS"] = str(chunk_seconds)
        print(f"[Lipsync] Chunked inference: {chunk_seconds}s per chunk")
    else:
        os.environ.pop("LATENTSYNC_CHUNK_SECONDS", None)

    # 5/11 update: face detector strict mode (드라마 artifact 방지)
    if face_strict:
        os.environ["LATENTSYNC_FACE_STRICT"] = "1"
        print(f"[Lipsync] Face detector strict (det_score≥0.85, w/h≥0.55, roll±30°)")
    else:
        os.environ.pop("LATENTSYNC_FACE_STRICT", None)

    print(f"[Lipsync] Scheduler: {scheduler}")

    # === DAEMON CLEANUP (defense-in-depth): direct call에서도 안전 ===
    # run_pipeline에서 이미 호출되지만, apply_lipsync()를 단독 사용 시 보호
    # pkill no-op (daemon 없으면 무해), CUDA context fragmentation 방지
    try:
        _stop_daemons()
    except Exception as _e:
        print(f"[Lipsync] daemon cleanup 실패: {_e} (진행)")

    # 5/12 audio quality fix: 24kHz mono 추출 (기존 16kHz는 기계음 원인).
    # Whisper audio2feat 이 내부에서 16k 로 자동 resample 하므로 24k 입력 OK.
    # LatentSync 가 최종 mp4 mux 할 때 audio_temp 를 그대로 사용하므로
    # 출력 영상 음성도 24kHz 유지 (CosyVoice 네이티브 = 24kHz).
    audio_temp = os.path.join("/tmp", f"latentsync_audio_{os.getpid()}.wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", dubbed_video_path,
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
            audio_temp,
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:200] if e.stderr else str(e)
        print(f"[Lipsync] ❌ 오디오 추출 실패: {err}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        LATENT_SYNC_PYTHON, "-m", "scripts.inference",
        "--unet_config_path", f"configs/unet/{config_name}",
        "--inference_ckpt_path", ckpt,
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--video_path",  os.path.abspath(dubbed_video_path),
        "--audio_path",  os.path.abspath(audio_temp),
        "--video_out_path", os.path.abspath(output_path),
        "--seed", str(seed),
    ]
    # v27: DeepCache는 메모리 여유 시에만 (사용자 명시 옵션)
    if enable_deepcache:
        cmd.append("--enable_deepcache")

    import time as _time
    start = _time.time()
    try:
        subprocess.run(cmd, cwd=LATENT_SYNC_DIR, check=True)
        elapsed = _time.time() - start
        print(f"[Lipsync] ✅ LatentSync 완료 ({elapsed:.1f}초): {output_path}")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1024:
            print(f"[Lipsync] ⚠️  출력 비정상 (1KB 미만)")
            return None
        return output_path
    except subprocess.CalledProcessError as e:
        elapsed = _time.time() - start
        print(f"[Lipsync] ❌ LatentSync 실패 ({elapsed:.1f}s 후, exit {e.returncode})")
        return None
    except Exception as e:
        print(f"[Lipsync] ❌ 예외 ({type(e).__name__}): {e}")
        return None
    finally:
        try:
            os.unlink(audio_temp)
        except OSError:
            pass


# ─── 다국어 LoRA 자동 인식 ─────────────────────────────────
def resolve_lipsync_ckpt(tgt_lang: str, explicit: Optional[str] = None) -> str:
    """tgt_lang에 따른 LatentSync 가중치 경로 자동 선택.

    우선순위:
      1. 사용자 명시 경로 (--lipsync-ckpt)
      2. media/lora/latentsync_<lang>.pt 자동 인식
      3. 베이스 모델 (LATENT_SYNC_CKPT) 폴백
    """
    if explicit:
        return explicit

    if tgt_lang:
        lang_ckpt = os.path.join(MEDIA_DIR, "lora", f"latentsync_{tgt_lang}.pt")
        if os.path.isfile(lang_ckpt):
            print(f"[Lipsync] 🌐 자동 인식: {lang_ckpt} (lang={tgt_lang})", flush=True)
            return lang_ckpt

    print(f"[Lipsync] 베이스 사용: {LATENT_SYNC_CKPT} (lang={tgt_lang}, LoRA 없음)", flush=True)
    return LATENT_SYNC_CKPT


# ─── 립싱크 wrapper (LatentSync 단일 엔진) ──────────────────
def apply_lipsync(
    dubbed_video_path: str,
    output_path: str,
    tgt_lang: Optional[str] = None,   # lang별 가중치 자동 선택용
    **kwargs,
) -> Optional[str]:
    """LatentSync 1.6 적용. tgt_lang 기반 가중치 자동 선택.

    공식 옵션 (kwargs로 전달):
      inference_steps, guidance_scale, seed, config_name, ckpt_path, enable_deepcache
    """
    ls_keys = {"inference_steps", "guidance_scale", "seed",
               "config_name", "ckpt_path", "enable_deepcache",
               # 5/11: 새 가속·품질 옵션
               "use_trt", "scheduler", "teacache_threshold",
               "profile_threshold", "face_diag_min_ratio", "chunk_seconds",
               "face_strict"}
    ls_kwargs = {k: v for k, v in kwargs.items() if k in ls_keys}

    # 🌐 lang별 가중치 자동 선택 (latentsync_<lang>.pt 자동 인식)
    if tgt_lang:
        explicit = ls_kwargs.get("ckpt_path")
        ls_kwargs["ckpt_path"] = resolve_lipsync_ckpt(tgt_lang, explicit)

    return apply_latent_sync(dubbed_video_path, output_path, **ls_kwargs)


# === GFPGAN 후처리 (face quality 향상) ===
# 5/7 update: async I/O 버전 default (sequential 대비 -22% 시간, 동일 품질)
GFPGAN_PYTHON = "/opt/venv_gfpgan/bin/python"
# 5/11 priority: v3 (TRT + GPU paste + optional downscale-detect)
#                → v2 (TRT + GPU paste)
#                → original async (PyTorch)
GFPGAN_SCRIPT_V3 = "/workspace/patches/gfpgan_async_postprocess_trt_v3.py"
GFPGAN_SCRIPT_V2 = "/workspace/patches/gfpgan_async_postprocess_trt_v2.py"
GFPGAN_SCRIPT_LEGACY = "/opt/LatentSync/scripts/gfpgan_async_postprocess.py"
GFPGAN_SCRIPT_FALLBACK = "/opt/LatentSync/scripts/gfpgan_postprocess.py"
GFPGAN_MODEL  = "/opt/gfpgan_models/GFPGANv1.4.pth"

def apply_gfpgan_postprocess(
    lipsync_video_path: str,
    output_path: str,
    upscale: int = 1,
    downscale_detect: int = 0,  # 5/11: v3 only, detection input 다운스케일 (2=540p detect)
) -> Optional[str]:
    """GFPGAN 후처리 — lipsync 결과의 face quality 향상.

    INPUT:
      lipsync_video_path : LatentSync 결과 mp4 (입술 작은 회색 artifact 등)
      output_path        : 후처리 결과 mp4
      upscale            : 1 (해상도 유지) / 2 (2x SR)

    OUTPUT:
      성공 시 output_path, 실패 시 None
    """
    if not os.path.isfile(GFPGAN_PYTHON):
        print(f"[GFPGAN] venv 없음: {GFPGAN_PYTHON}")
        return None
    # 5/11: v3 (TRT + GPU paste + optional downscale-detect) 우선
    #       → v2 (TRT + GPU paste)
    #       → legacy async (PyTorch)
    if os.path.isfile(GFPGAN_SCRIPT_V3):
        gfpgan_script = GFPGAN_SCRIPT_V3
        gfpgan_variant = "v3"
    elif os.path.isfile(GFPGAN_SCRIPT_V2):
        gfpgan_script = GFPGAN_SCRIPT_V2
        gfpgan_variant = "v2"
    elif os.path.isfile(GFPGAN_SCRIPT_LEGACY):
        gfpgan_script = GFPGAN_SCRIPT_LEGACY
        gfpgan_variant = "legacy"
    elif os.path.isfile(GFPGAN_SCRIPT_FALLBACK):
        gfpgan_script = GFPGAN_SCRIPT_FALLBACK
        gfpgan_variant = "fallback"
    else:
        print(f"[GFPGAN] script 없음")
        return None
    print(f"[GFPGAN] using: {os.path.basename(gfpgan_script)} ({gfpgan_variant})")
    if not os.path.isfile(GFPGAN_MODEL):
        print(f"[GFPGAN] model 없음: {GFPGAN_MODEL}")
        return None
    if not os.path.isfile(lipsync_video_path):
        print(f"[GFPGAN] 입력 비디오 없음: {lipsync_video_path}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\n--- [GFPGAN] 후처리 적용 시작 ---")
    print(f"[GFPGAN] 입력: {lipsync_video_path}")
    print(f"[GFPGAN] 출력: {output_path}")

    cmd = [
        GFPGAN_PYTHON, gfpgan_script,
        "--input", os.path.abspath(lipsync_video_path),
        "--output", os.path.abspath(output_path),
        "--upscale", str(upscale),
    ]
    # 5/11: v3 만 --downscale-detect 지원
    if gfpgan_variant == "v3" and downscale_detect > 1:
        cmd += ["--downscale-detect", str(downscale_detect)]
        print(f"[GFPGAN] downscale-detect: {downscale_detect} (detection 만 1/{downscale_detect} 해상도)")
    import time as _time
    start = _time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = _time.time() - start
        print(f"[GFPGAN] ✅ 완료 ({elapsed:.1f}초): {output_path}")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1024:
            print(f"[GFPGAN] ⚠️ 출력 비정상 (1KB 미만)")
            return None
        return output_path
    except subprocess.CalledProcessError as e:
        elapsed = _time.time() - start
        print(f"[GFPGAN] ❌ 실패 ({elapsed:.1f}s, exit {e.returncode})")
        return None
    except Exception as e:
        print(f"[GFPGAN] ❌ 예외: {e}")
        return None


def concat_chunks(file_name: str, tgt_lang: str) -> str:
    """
    모든 _final.mp4 청크를 이어붙여 최종 영상 생성.
    CHUNKS_DIR은 RunContext.activate()에 의해 runs/{run_id}/chunks/로 설정됨.

    INPUT:
      file_name : str — "movie"
      tgt_lang  : str — "ko"

    OUTPUT:
      output_path : str — media/output/{file_name}_{tgt_lang}_{run_id}.mp4
    """
    # file_name 필터 + _final.mp4 필터 (run 디렉터리라 이미 격리되어 있지만 안전장치)
    finals = sorted([
        f for f in os.listdir(CHUNKS_DIR)
        if f.startswith(file_name) and "_final.mp4" in f
    ])
    if not finals:
        raise FileNotFoundError(
            f"final 청크가 없습니다 (CHUNKS_DIR={CHUNKS_DIR}, file_name={file_name})"
        )

    # concat_list.txt는 run 내부 temp 공간에 (OUTPUT_DIR 오염 방지)
    concat_list_path = os.path.join(CHUNKS_DIR, f"{file_name}_concat_list.txt")
    with open(concat_list_path, "w") as f:
        for chunk in finals:
            f.write(f"file '{os.path.join(CHUNKS_DIR, chunk)}'\n")

    # run_id를 파일명에 포함 → 같은 입력 재실행해도 결과 덮어쓰기 없음
    run_suffix = f"_{CURRENT_RUN_ID}" if CURRENT_RUN_ID else ""
    output_path = os.path.join(
        OUTPUT_DIR, f"{file_name}_{tgt_lang}{run_suffix}.mp4"
    )
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy", output_path, "-y"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat 실패:\n{result.stderr}")

    print(f"\n[Concat] 최종 영상 완성: {output_path}")
    return output_path

    


# ─── 메인 파이프라인 ──────────────────────────────────────────

def save_pipeline_report(
    file_name: str,
    tgt_lang: str,
    chunk_data: dict,
    output_path: str
):
    """파이프라인 중간 결과를 JSON으로 저장.

    v30 (5/15): cross-modal speaker matching 결과 + trim 적용값을 JSON에 노출.
        chunk 레벨: face_clusters, face_speaker_remap, av_reassigned_intervals,
                     av_fusion_stats, speaker_face_count, audio_gender_stats
        segment 레벨: reassigned_from, face_match_track, audio_gender,
                       trim_applied_s, pad_applied_s, lipsync_coverage_pct
    """
    report = {
        "run_id": CURRENT_RUN_ID,
        "run_root": os.path.join(RUNS_DIR, CURRENT_RUN_ID) if CURRENT_RUN_ID else None,
        "input": file_name,
        "target_lang": tgt_lang,
        "timestamp": datetime.now().isoformat(),
        "output": output_path,
        "chunks": []
    }

    # v30: audio_f0_gender.json 한 번만 읽기 (전체 영상 단위 1개)
    audio_gender_data = None
    audio_gender_fps = 25.0
    try:
        _run_root = os.path.join(RUNS_DIR, CURRENT_RUN_ID) if CURRENT_RUN_ID else None
        if _run_root:
            _f0_json = os.path.join(_run_root, "meta", "audio_f0_gender.json")
            if os.path.isfile(_f0_json):
                with open(_f0_json, "r", encoding="utf-8") as _f:
                    _f0_doc = json.load(_f)
                audio_gender_data = _f0_doc.get("gender_per_frame", [])
                audio_gender_fps = float(_f0_doc.get("fps", 25.0))
    except Exception as _e_f0:
        print(f"[Report] audio_f0_gender.json 로드 실패 (계속): {_e_f0}")

    def _seg_audio_gender(start_s: float, end_s: float):
        """segment 구간의 audio gender majority vote (audio_gender_data 있을 때만)."""
        if not audio_gender_data:
            return None
        f1 = max(0, int(start_s * audio_gender_fps))
        f2 = min(len(audio_gender_data), int(end_s * audio_gender_fps + 1))
        if f2 <= f1:
            return None
        win = audio_gender_data[f1:f2]
        if not win:
            return None
        male_n = sum(1 for g in win if g == "male")
        female_n = sum(1 for g in win if g == "female")
        if male_n == 0 and female_n == 0:
            return "unknown"
        return "male" if male_n >= female_n else "female"

    for chunk_name, data in chunk_data.items():
        # v30: AV-Fusion 결과 추출 (없으면 빈 dict)
        fusion = data.get("av_fusion") or {}
        asd_result = data.get("asd_result") or {}
        face_clusters = data.get("face_clusters") or {}
        face_speaker_remap = data.get("face_speaker_remap") or {}
        av_reassigned = data.get("av_reassigned_intervals") or []
        speaker_face_count = fusion.get("speaker_face_count") or {}

        # segment별 reassign source lookup: (round(start,3), round(end,3)) → old_spk
        reassign_lookup = {
            (round(s, 3), round(e, 3)): old
            for s, e, old in av_reassigned
        }

        # segment별 dominant face track lookup
        def _seg_dom_face(seg_start: float, seg_end: float):
            tracks = asd_result.get("tracks") or []
            fps = asd_result.get("fps", 25.0)
            if not tracks:
                return None, 0.0
            f1 = int(seg_start * fps)
            f2 = int(seg_end * fps + 1)
            face_freq = {}
            seg_frames = max(1, f2 - f1)
            for tidx, t in enumerate(tracks):
                for fi in t.get("frames", []):
                    if f1 <= fi < f2:
                        face_freq[tidx] = face_freq.get(tidx, 0) + 1
            if not face_freq:
                return None, 0.0
            dom_t, dom_n = max(face_freq.items(), key=lambda x: x[1])
            return int(dom_t), round(dom_n / seg_frames, 3)

        chunk_report = {
            "name": chunk_name,
            "vocals_path": data.get("vocals_path", ""),
            "bgm_path": data.get("bgm_path", ""),
            "detected_lang": data.get("detected_lang", ""),
            "video_duration_s": round(float(data.get("video_duration", 0.0)), 3),
            "segments": [],
            "speaker_profiles": {},
            # v30: cross-modal speaker matching 결과
            "av_fusion_stats": {
                "n_face_tracks": len(asd_result.get("tracks", [])),
                "n_frames": int(fusion.get("n_frames", asd_result.get("n_frames", 0))),
                "fps": float(fusion.get("fps", asd_result.get("fps", 25.0))),
                "lipsync_target_frames": (
                    sum(1 for t in fusion.get("per_frame_target", []) if t is not None)
                    if fusion.get("per_frame_target") else 0
                ),
                "audio_active_frames": (
                    sum(1 for s in fusion.get("frame_active_speaker", []) if s is not None)
                    if fusion.get("frame_active_speaker") else 0
                ),
                "spurious_speakers": fusion.get("spurious_speakers", []),
            },
            "face_clusters": {str(k): int(v) for k, v in face_clusters.items()},
            "face_speaker_remap": dict(face_speaker_remap),
            "av_reassigned_intervals": [
                {"start": round(s, 3), "end": round(e, 3), "old_speaker": old}
                for s, e, old in av_reassigned
            ],
            "speaker_face_count": {
                spk: {str(t): int(n) for t, n in faces.items()}
                for spk, faces in speaker_face_count.items()
            },
        }

        # v30: lipsync coverage % (검증 1순위 지표)
        _nf = chunk_report["av_fusion_stats"]["n_frames"]
        if _nf > 0:
            chunk_report["av_fusion_stats"]["lipsync_coverage_pct"] = round(
                100.0 * chunk_report["av_fusion_stats"]["lipsync_target_frames"] / _nf, 2
            )

        for seg in data.get("segments", []):
            seg_key = (round(seg.start, 3), round(seg.end, 3))
            dom_track, dom_share = _seg_dom_face(seg.start, seg.end)
            aud_gender = _seg_audio_gender(seg.start, seg.end)

            seg_info = {
                "id": seg.id,
                "speaker": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "duration": round(seg.end - seg.start, 3),
                "original_text": seg.text,
                "translated_text": seg.translated,
                "emotion": seg.emotion,
                "emotion_score": seg.emotion_score,
                "raw_emotion": getattr(seg, 'raw_emotion', seg.emotion),
                "raw_emotion_score": getattr(seg, 'raw_emotion_score', seg.emotion_score),
                # v29: 팀원 master_timeline 형식 (UI 조정용)
                "source_emotion": {
                    "label": getattr(seg, 'raw_emotion', 'Neutral').lower(),
                    "confidence": getattr(seg, 'raw_emotion_score', 0.0),
                    "scores": getattr(seg, 'raw_emotion_scores', {}),
                },
                "tts_context": getattr(seg, 'tts_context', ''),
                "tts_emotion": getattr(seg, 'tts_emotion', ''),
                "tts_instruct_text": f"You are a helpful assistant. Please say it close to the speaker's natural delivery"
                                     + (f", {seg.tts_emotion}" if getattr(seg, 'tts_emotion', '') else "")
                                     + ".<|endofprompt|>",
                "speed": seg.speed,
                "tts_mos": getattr(seg, '_tts_mos', 0.0),
                "tts_retries": getattr(seg, '_tts_retries', 0),
                # v30: cross-modal speaker matching trace
                "reassigned_from": reassign_lookup.get(seg_key),
                "face_match_track": dom_track,
                "face_match_share": dom_share,
                "audio_gender": aud_gender,
                # v30: trim/padding 실제 적용값 (synthesize_chunk가 채움)
                "tts_actual_duration_s": getattr(seg, '_tts_actual_duration', None),
                "tts_target_duration_s": getattr(seg, '_tts_target_duration', None),
                "tts_stretch_ratio": getattr(seg, '_tts_stretch_ratio', None),
                "tts_trim_applied_s": getattr(seg, '_tts_trim_applied', 0.0),
                "tts_pad_applied_s": getattr(seg, '_tts_pad_applied', 0.0),
            }
            chunk_report["segments"].append(seg_info)

        for speaker, profile in data.get("profiles", {}).items():
            chunk_report["speaker_profiles"][speaker] = {
                "references": {
                    emotion: os.path.basename(path)
                    for emotion, path in profile.references.items()
                }
            }

        report["chunks"].append(chunk_report)

    run_suffix = f"_{CURRENT_RUN_ID}" if CURRENT_RUN_ID else ""
    report_path = os.path.join(
        REPORT_DIR, f"{file_name}_{tgt_lang}{run_suffix}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Report] 저장: {report_path}")
    return report_path


# === DAEMON AUTO-LAUNCH ===
DAEMON_CONFIGS = [
    {
        "name": "Cosy",
        "url":  "http://127.0.0.1:8901",
        "venv": "/usr/bin/python",  # main container python (CosyVoice 그대로 사용)
        "script": "/workspace/patches/cosyvoice_daemon.py",
        "port": 8901,
        "log":  "/workspace/media/logs/cosy_daemon.log",
        "load_wait": 90,
    },
    {
        "name": "ASR",
        "url":  "http://127.0.0.1:8902",
        "venv": "/opt/venv_asr/bin/python",
        "script": "/workspace/patches/asr_daemon.py",
        "port": 8902,
        "log":  "/workspace/media/logs/asr_daemon.log",
        "load_wait": 60,
    },
    {
        "name": "Diarize",
        "url":  "http://127.0.0.1:8903",
        "venv": "/opt/venv_diarizen/bin/python",
        "script": "/workspace/patches/diarize_daemon.py",
        "port": 8903,
        "log":  "/workspace/media/logs/diarize_daemon.log",
        "load_wait": 60,
    },
]


def _ensure_daemons_running():
    """데몬 alive 체크, 없으면 background로 시작 후 모델 로딩 대기."""
    import requests as _rq
    os.makedirs("/workspace/media/logs", exist_ok=True)
    started_any = False
    for cfg in DAEMON_CONFIGS:
        # 1. 이미 alive인지 체크
        try:
            r = _rq.get(f"{cfg['url']}/health", timeout=2)
            if r.status_code == 200 and r.json().get("model_loaded"):
                print(f"[Daemon] {cfg['name']} already alive ⚡")
                continue
        except Exception:
            pass

        # 2. venv/script 존재 확인
        if not os.path.isfile(cfg["venv"]):
            print(f"[Daemon] {cfg['name']} venv 없음 ({cfg['venv']}) → skip")
            continue
        if not os.path.isfile(cfg["script"]):
            print(f"[Daemon] {cfg['name']} script 없음 ({cfg['script']}) → skip")
            continue

        # 3. background launch
        print(f"[Daemon] {cfg['name']} 시작 (port {cfg['port']})...")
        log_f = open(cfg["log"], "w")
        subprocess.Popen(
            [cfg["venv"], cfg["script"], "--port", str(cfg["port"])],
            stdout=log_f, stderr=log_f, start_new_session=True,
        )
        started_any = True

    if started_any:
        # 4. 모델 로딩 대기 (최대 90초)
        print(f"[Daemon] 모델 로딩 대기 (최대 90초)...")
        import time as _time
        deadline = _time.time() + 95
        ready = set()
        while _time.time() < deadline and len(ready) < len(DAEMON_CONFIGS):
            for cfg in DAEMON_CONFIGS:
                if cfg["name"] in ready:
                    continue
                try:
                    r = _rq.get(f"{cfg['url']}/health", timeout=2)
                    if r.status_code == 200 and r.json().get("model_loaded"):
                        ready.add(cfg["name"])
                        print(f"[Daemon] {cfg['name']} ready ✅")
                except Exception:
                    pass
            _time.sleep(2)
        not_ready = [c["name"] for c in DAEMON_CONFIGS if c["name"] not in ready]
        if not_ready:
            print(f"[Daemon] timeout: {not_ready} → subprocess fallback")


def _stop_daemons(timeout: int = 10):
    """모든 데몬을 graceful 종료 + GPU 메모리 회수.

    Lipsync 시작 전 호출 → GPU 16GB 모두 lipsync에 사용 가능.
    SIGTERM (graceful) → SIGKILL (강제) 순서.
    """
    import signal as _signal
    import time as _time

    daemon_names = [cfg["name"] for cfg in DAEMON_CONFIGS]
    # v124+: 7-daemon 구성 전체 종료 (whisperx, nemo, pyannote, fusion, vbx 포함)
    # 누락 시 lipsync GPU 메모리 부족 → SIGKILL (exit -9)
    daemon_scripts = [
        "cosyvoice_daemon.py",
        "asr_daemon.py",
        "whisperx_daemon.py",
        "diarize_daemon.py",
        "nemo_diarize_daemon.py",
        "pyannote_diarize_daemon.py",
        "fusion_diarize_daemon.py",
        "vbx_diarize_daemon.py",
        "campplus_diarize_daemon.py",
    ]
    pgrep_pattern = "(cosyvoice_daemon|asr_daemon|whisperx_daemon|diarize_daemon|nemo_diarize_daemon|pyannote_diarize_daemon|fusion_diarize_daemon|vbx_diarize_daemon|campplus_diarize_daemon)"

    # 1. SIGTERM (graceful)
    print(f"[Daemon] stopping all daemons (graceful)...")
    for script_name in daemon_scripts:
        subprocess.run(
            ["pkill", "-SIGTERM", "-f", script_name],
            capture_output=True
        )

    # 2. 종료 대기 (최대 timeout 초)
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        result = subprocess.run(
            ["pgrep", "-f", pgrep_pattern],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            print(f"[Daemon] all stopped ✅")
            break
        _time.sleep(1)
    else:
        # 3. SIGKILL (강제)
        print(f"[Daemon] graceful timeout → SIGKILL")
        for script_name in daemon_scripts:
            subprocess.run(
                ["pkill", "-SIGKILL", "-f", script_name],
                capture_output=True
            )

    # 4. GPU memory 명시 정리 (orchestrator 자체 메모리)
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
            _t.cuda.synchronize()
            free = _t.cuda.mem_get_info()[0] / 1024**3
            print(f"[Daemon] GPU memory 회수: {free:.1f}GB free")
    except Exception:
        pass


def run_pipeline(
    video_path: str,
    file_name: str,
    tgt_lang: str = "ko",
    src_lang: Optional[str] = None,
    segment_time: int = 300,
    num_speakers: int = None,
    run_id: Optional[str] = None,
    content_type: str = "auto",
    enable_lipsync: bool = False,
    # LatentSync 옵션
    lipsync_steps: int = 10,             # 5/11: DPM++ 10-step default (TRT+TeaCache 와 조합)
    lipsync_guidance: float = 1.5,       # guidance scale
    lipsync_seed: int = 1247,
    lipsync_config: str = "stage2_512_nf16.yaml",   # v27 default: 16 frames @ 512
    lipsync_ckpt: Optional[str] = None,        # None=베이스 (v27 default, use_lora 시 lang별 자동)
    use_lora: bool = False,                    # True면 LoRA 자동 인식
    lipsync_deepcache: bool = False,           # 5/7 sm_120 호환성 issue — OFF default
    lipsync_vae_chunk: int = 2,                # 5/7 final: chunk=2 (v42 정확 매칭, chunk=4는 quality 미세 저하 가능)
    # 5/11 new: 가속·품질 옵션 (apply_lipsync 로 통과)
    lipsync_use_trt: bool = True,              # TRT FP16 engine (3.18× 가속)
    lipsync_scheduler: str = "dpm",            # DPMSolver++ (default)
    lipsync_teacache: float = 0.1,             # TeaCache rel_l1 threshold (0=off)
    lipsync_profile_threshold: float = 0.35,   # 5/12: 정면만 lipsync (측면 입 떠다님 방지)
    lipsync_face_diag_min_ratio: float = 0.0,  # 5/11: ASD 통합 후 default off (face miss → mask artifact 방지)
    lipsync_chunk_seconds: int = 0,            # >0 시 chunked inference (long video memory)
    lipsync_face_strict: bool = False,         # 5/11: 드라마 artifact 방지 (det_score 0.85, roll 체크)
    enable_postprocess: bool = False,          # GFPGAN 후처리 (face quality, v42 setup)
    postprocess_upscale: int = 1,              # 5/11: 1x default (v3 + GPU paste 안전)
    postprocess_downscale_detect: int = 0,     # 5/11: detection 만 다운스케일 (0=off, 2=540p detect)
    smart_daemon: bool = False,                # 더빙 단계만 daemon 사용, lipsync 전 stop
):
    """
    전체 더빙 파이프라인 실행.
    GPU 16GB 환경을 위해 모델을 단계별로 로드/언로드.

    각 실행은 고유한 run_id를 받고 media/runs/{run_id}/ 아래에 격리된
    작업 공간을 갖는다. 최종 output/report는 공유 디렉터리에 {run_id}
    suffix로 저장되어 이전 실행과 충돌하지 않는다.

    content_type: "auto/lecture/interview/news/movie/drama"
      EMOTION_POLICIES에 따라 감정 처리 방식이 달라짐.
      lecture/interview/news는 모든 세그먼트를 Neutral로 고정 (오탐지 방지).

    enable_lipsync: True면 concat_chunks 후 LatentSync 1.6 으로 입 모양 동기화.
      lipsync_steps: diffusion inference steps (10~50, 기본 20)
      lipsync_guidance: guidance scale (1.0~3.0, 기본 1.5)
      lipsync_config: stage2_512.yaml (512 출력, 기본) / stage2_efficient.yaml (256, 빠름)
      lipsync_ckpt: 가중치 .pt. None이면 tgt_lang 기반 자동 (latentsync_<lang>.pt) 또는 base
    """
    # ── 🔥 Run Context 생성 + 전역 디렉터리 상수 덮어쓰기 ──
    ctx = RunContext.create(file_name, explicit_id=run_id)
    ctx.activate()

    # === SMART_DAEMON (v28): 더빙 단계만 daemon, lipsync 전 stop ===
    # smart_daemon=True 시:
    #   1. 시작 시 데몬 launch (90초 모델 로딩)
    #   2. 더빙 단계: TTS/ASR/Diarize daemon 사용 (-2-3분 절감)
    #   3. Lipsync 시작 직전: 데몬 stop (GPU 메모리 회수)
    #   4. Lipsync: 16GB 모두 활용 (OOM 안전)
    if smart_daemon:
        try:
            print(f"[SmartDaemon] 더빙 단계 daemon 시작 (lipsync 전 자동 stop)")
            _ensure_daemons_running()
        except Exception as _e:
            print(f"[SmartDaemon] launch 실패: {_e} (subprocess fallback)")
    elif os.environ.get("AUTOLAUNCH_DAEMONS", "0") == "1":
        try:
            _ensure_daemons_running()
        except Exception as _e:
            print(f"[Daemon] auto-launch 실패: {_e} (subprocess fallback 사용)")

    print(f"\n{'='*50}")
    print(f"[Pipeline] 시작: {video_path} → {tgt_lang}")
    print(f"[Pipeline] device: {DEVICE}")
    print(f"[Pipeline] run_id: {ctx.run_id}")
    print(f"[Pipeline] content_type: {content_type} "
          f"(policy: {EMOTION_POLICIES.get(content_type, 'passthrough')})")
    if enable_lipsync:
        # LatentSync 환경 간단 체크 (apply_latent_sync 내부에 자세한 체크 있음)
        if not os.path.isdir(LATENT_SYNC_DIR):
            print(f"[Pipeline] ⚠️  LatentSync 레포 없음 ({LATENT_SYNC_DIR}) → 립싱크 비활성화")
            enable_lipsync = False
        elif not os.path.isfile(LATENT_SYNC_CKPT):
            print(f"[Pipeline] ⚠️  베이스 가중치 없음 ({LATENT_SYNC_CKPT}) → 립싱크 비활성화")
            enable_lipsync = False
        else:
            print(f"[Pipeline] 립싱크: ON (LatentSync 1.6, config={lipsync_config}, steps={lipsync_steps})")
    else:
        print(f"[Pipeline] 립싱크: OFF")
    print(f"{'='*50}\n")

    # Translator는 CPU만 사용하므로 미리 로드 (해제 불필요)
    load_translator()

    # Step 1: 영상 분할
    chunks = split_video(video_path, file_name, segment_time)

    # ── 1단계: 음원 분리 (모든 청크 먼저) ────────────────
    # v28 (5/14): Resume 지원 — vocals/bgm 파일 존재 시 skip
    _resume_mode = os.environ.get("LATENTSYNC_RESUME", "0") == "1"
    if _resume_mode:
        print("[Resume] LATENTSYNC_RESUME=1 — 결과 파일 존재 시 단계 skip")
    chunk_data = {}
    for chunk_path in chunks:
        chunk_name = os.path.basename(chunk_path).replace(".mp4", "")
        # Resume check: vocals/bgm 존재
        expected_vocals = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
        expected_bgm = os.path.join(BGM_DIR, f"{chunk_name}_bgm.wav")
        if _resume_mode and os.path.exists(expected_vocals) and os.path.exists(expected_bgm):
            print(f"\n--- [Separate] 청크: {chunk_name} [Resume skip] ⚡")
            vocals_path, bgm_path = expected_vocals, expected_bgm
        else:
            print(f"\n--- [Separate] 청크: {chunk_name} ---")
            vocals_path, bgm_path = separate_audio(chunk_path)
        video_dur = get_video_duration(chunk_path)
        chunk_data[chunk_name] = {
            "chunk_path": chunk_path,
            "vocals_path": vocals_path,
            "bgm_path": bgm_path,
            "video_duration": video_dur,
        }
        print(f"[Duration] {chunk_name}: {video_dur:.2f}s")

    # GPU 정리
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # 🔥 추가: Demucs가 쓴 VRAM이 완벽히 반환되도록 강제 동기화
    import time; time.sleep(2)

    load_vad()
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [VAD] 청크: {chunk_name} ---")
        clean_vocals_path = apply_vad_filter(data["vocals_path"])
        
        # ASR과 pyannote가 폭발음 섞인 원본 대신 '깨끗한 파일'을 쓰도록 경로 덮어쓰기!
        data["vocals_path"] = clean_vocals_path 
    _unload("vad")

    # ── 2단계: ASR (모든 청크) ───────────────────────────
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [ASR] 청크: {chunk_name} ---")
        detected_lang, words = transcribe(data["vocals_path"], src_lang)
        data["words"] = words
        data["detected_lang"] = src_lang or detected_lang
        
    # ── 2단계: 화자 분리 (모든 청크) ─────────────────────
    load_diarization()
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [Diarize] 청크: {chunk_name} ---")

        # Step 4: 화자 분리
        diarization = diarize(data["vocals_path"], num_speakers=num_speakers)

        # 🔥 Step 4-2: ECAPA centroid 후처리 (over-segmentation 해결)
        # v22 (5/14): env var 로 threshold 조정 가능
        #   LATENTSYNC_DIARIZE_MERGE_THRESHOLD: cosine sim 임계값 (기본 0.50)
        #     낮을수록 더 적극 병합 (under-segment 위험)
        #     높을수록 보수적 (over-segment 잔존)
        #   LATENTSYNC_DIARIZE_SHORT_TURN: 짧은 turn 재할당 임계값 (기본 0.8s)
        speaker_centroids: Dict[str, np.ndarray] = {}
        if num_speakers is None or num_speakers > 1:
            _merge_thr = float(os.environ.get("LATENTSYNC_DIARIZE_MERGE_THRESHOLD", "0.50"))
            _short_turn = float(os.environ.get("LATENTSYNC_DIARIZE_SHORT_TURN", "0.8"))
            _min_dur_centroid = float(os.environ.get("LATENTSYNC_DIARIZE_MIN_DUR_CENTROID", "0.5"))
            print(f"[Diarize] 후처리 파라미터: merge_threshold={_merge_thr}, "
                  f"short_turn={_short_turn}, min_dur_centroid={_min_dur_centroid}")
            diarization, speaker_centroids = post_process_diarization(
                diarization, data["vocals_path"],
                merge_threshold=_merge_thr,
                short_turn_threshold=_short_turn,
                min_dur_for_centroid=_min_dur_centroid,
            )
        data["speaker_centroids"] = speaker_centroids

        # 🔥 Step 4-3: AV Fusion (LightASD) — 화자 분리 정확도 향상
        #   Visual ASD로 화면 내 발화자를 식별하여 audio diarization 검증.
        #   spurious speaker (한숨/짧은 noise를 별개 화자로 잘못 인식) 자동 제거.
        try:
            import sys as _sys
            if "/workspace/scripts" not in _sys.path:
                _sys.path.insert(0, "/workspace/scripts")
            from asd_runner import run_asd
            from av_fusion import fuse_av_diarization, detect_face_based_merges
            # === LIGHTASD_CACHE_PATCH (v28): 영상 hash 기반 cache ===
            # 같은 영상 재실행 시 LightASD 1-2분 절감
            import hashlib as _hashlib
            import pickle as _pickle
            asd_cache_dir = os.path.join(MEDIA_DIR, "cache", "lightasd")
            os.makedirs(asd_cache_dir, exist_ok=True)
            with open(data["chunk_path"], "rb") as _f:
                # 파일 첫 1MB + 마지막 1MB hash (전체 hash는 너무 느림)
                _f.seek(0)
                _head = _f.read(1024 * 1024)
                _f.seek(-min(1024 * 1024, os.path.getsize(data["chunk_path"])), 2)
                _tail = _f.read()
            _video_hash = _hashlib.md5(_head + _tail + str(os.path.getsize(data["chunk_path"])).encode()).hexdigest()[:16]
            asd_cache_path = os.path.join(asd_cache_dir, f"{_video_hash}.pkl")
            asd_result = None
            if os.path.isfile(asd_cache_path):
                try:
                    with open(asd_cache_path, "rb") as _f:
                        asd_result = _pickle.load(_f)
                    print(f"[AV-Fusion] cache hit ({_video_hash}) → LightASD skip ⚡")
                except Exception as _ce:
                    print(f"[AV-Fusion] cache load 실패 ({_ce}) → 재실행")
                    asd_result = None
            if asd_result is None:
                print(f"[AV-Fusion] LightASD on {chunk_name}...")
                asd_result = run_asd(data["chunk_path"])
                # cache 저장
                if asd_result is not None:
                    try:
                        with open(asd_cache_path, "wb") as _f:
                            _pickle.dump(asd_result, _f)
                        print(f"[AV-Fusion] cached → {asd_cache_path}")
                    except Exception as _ce:
                        print(f"[AV-Fusion] cache 저장 실패: {_ce}")
            if asd_result and diarization is not None:
                audio_segments_list = [
                    (turn.start, turn.end, spk)
                    for turn, _, spk in diarization.itertracks(yield_label=True)
                ]
                fusion = fuse_av_diarization(audio_segments_list, asd_result, verbose=False)
                # spurious speaker 제거
                if fusion["spurious_speakers"]:
                    from pyannote.core import Annotation
                    new_diar = Annotation()
                    for turn, track, spk in diarization.itertracks(yield_label=True):
                        if spk not in fusion["spurious_speakers"]:
                            new_diar[turn, track] = spk
                    print(f"[AV-Fusion] spurious 화자 제거: {fusion['spurious_speakers']}")
                    diarization = new_diar

                # === v14 NEW: AV-Fusion 자동 화자 병합 ===
                # 같은 face track에 매핑된 두 speaker = 같은 사람 가능성 높음
                # ECAPA centroid가 못 잡은 over-detect를 시각 정보로 보정
                face_count = fusion.get("speaker_face_count", {})
                if face_count:
                    # v23 (5/14): AV-Fusion face-based auto merge env var 조정
                    # LATENTSYNC_AV_MERGE_FRAMES (기본 10): 공유 face frame 임계값 ↑ → 병합 보수적
                    # LATENTSYNC_AV_MERGE_RATIO (기본 0.30): 공유 비율 임계값 ↑ → 병합 보수적
                    # LATENTSYNC_AV_MERGE_OFF=1: face merge 완전 비활성화 (DiariZen 결과 그대로)
                    _av_merge_frames = int(os.environ.get("LATENTSYNC_AV_MERGE_FRAMES", "10"))
                    _av_merge_ratio = float(os.environ.get("LATENTSYNC_AV_MERGE_RATIO", "0.30"))
                    _av_merge_off = os.environ.get("LATENTSYNC_AV_MERGE_OFF", "0") == "1"
                    if _av_merge_off:
                        print("[AV-Fusion] auto merge 비활성화 (LATENTSYNC_AV_MERGE_OFF=1)")
                        merge_pairs = []
                    else:
                        merge_pairs = detect_face_based_merges(face_count, min_shared_frames=_av_merge_frames, min_share_ratio=_av_merge_ratio)
                    if merge_pairs:
                        # union-find로 병합
                        parent = {spk: spk for spk in face_count.keys()}
                        def _find(x):
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x
                        for s1, s2, shared in merge_pairs:
                            r1, r2 = _find(s1), _find(s2)
                            if r1 != r2:
                                parent[max(r1, r2)] = min(r1, r2)
                                print(f"[AV-Fusion] 자동 병합: {s2} → {s1} (공유 face frames: {shared})")
                        # 병합 매핑 적용
                        speaker_remap = {spk: _find(spk) for spk in face_count.keys()}
                        # diarization 라벨 변경
                        if any(v != k for k, v in speaker_remap.items()):
                            from pyannote.core import Annotation, Segment as PyaSeg
                            new_diar2 = Annotation()
                            for turn, track, spk in diarization.itertracks(yield_label=True):
                                canon = speaker_remap.get(spk, spk)
                                new_diar2[turn, track] = canon
                            diarization = new_diar2
                            # speaker_centroids도 병합
                            if speaker_centroids:
                                merged_centroids = {}
                                for spk, c in speaker_centroids.items():
                                    canon = speaker_remap.get(spk, spk)
                                    if canon not in merged_centroids:
                                        merged_centroids[canon] = c
                                speaker_centroids = merged_centroids
                                data["speaker_centroids"] = speaker_centroids
                            print(f"[AV-Fusion] 화자 병합 후: {sorted(set(speaker_remap.values()))}")

                # === v23 NEW: Face Recognition cluster (audio over-split fix) ===
                # 사용자 지적: "화자 분리가 제일 중요한 관건"
                # audio diarization은 같은 사람의 다른 톤 변화를 다른 speaker로 분리할 수 있음.
                # face_id (ArcFace) embedding으로 face track의 *진짜 정체성* 결정 →
                # cluster_id 같은데 audio speaker 다르면 통합 (audio over-split fix).
                # 환경변수:
                #   LATENTSYNC_FACE_ID_OFF=1 → 비활성
                #   LATENTSYNC_FACE_ID_SIM (default 0.50) ArcFace cosine sim 임계 (동일인)
                #   LATENTSYNC_FACE_ID_CLUSTER_SHARE (default 0.50) cluster 대표 speaker 최소 점유
                # 부수 산출물:
                #   data["face_clusters"]: Dict[track_idx, cluster_id]
                #   data["face_speaker_remap"]: Dict[old_speaker, canonical]
                data["face_clusters"] = {}
                data["face_speaker_remap"] = {}
                _faceid_off = os.environ.get("LATENTSYNC_FACE_ID_OFF", "0") == "1"
                if _faceid_off:
                    print("[FaceID] 비활성화 (LATENTSYNC_FACE_ID_OFF=1)")
                else:
                    try:
                        from face_id_embedder import (
                            compute_track_face_embeddings,
                            cluster_tracks_by_face,
                            derive_speaker_remap_from_face_clusters,
                            apply_speaker_remap_to_diarization,
                            reassign_segments_by_face_voting,
                            absorb_spurious_speakers,
                        )
                        _faceid_sim = float(os.environ.get("LATENTSYNC_FACE_ID_SIM", "0.50"))
                        _faceid_share = float(os.environ.get("LATENTSYNC_FACE_ID_CLUSTER_SHARE", "0.50"))
                        track_embs = compute_track_face_embeddings(
                            asd_result, data["chunk_path"], verbose=True
                        )
                        if track_embs:
                            track_to_cluster = cluster_tracks_by_face(
                                track_embs, similarity_threshold=_faceid_sim, verbose=True,
                            )
                            data["face_clusters"] = track_to_cluster
                            faceid_remap = derive_speaker_remap_from_face_clusters(
                                fusion, track_to_cluster,
                                min_cluster_share=_faceid_share, verbose=True,
                            )
                            # v68+: face_id remap만 OFF (FaceVoting/absorb는 유지)
                            if os.environ.get("LATENTSYNC_FACE_ID_REMAP_OFF", "0") == "1":
                                print("[FaceID] speaker remap 비활성화 (LATENTSYNC_FACE_ID_REMAP_OFF=1) — FaceVoting/absorb_spurious는 유지")
                                faceid_remap = {}
                            if faceid_remap:
                                diarization = apply_speaker_remap_to_diarization(
                                    diarization, faceid_remap
                                )
                                data["face_speaker_remap"] = faceid_remap
                                # speaker_centroids remap
                                if speaker_centroids:
                                    new_cents = {}
                                    for spk, c in speaker_centroids.items():
                                        canon = faceid_remap.get(spk, spk)
                                        if canon not in new_cents:
                                            new_cents[canon] = c
                                    speaker_centroids = new_cents
                                    data["speaker_centroids"] = speaker_centroids
                                # fusion의 speaker_face_count도 remap (downstream AV-Reassign용)
                                if "speaker_face_count" in fusion:
                                    old_sfc = fusion["speaker_face_count"]
                                    new_sfc: Dict[str, Dict[int, int]] = {}
                                    for spk, face_map in old_sfc.items():
                                        canon = faceid_remap.get(spk, spk)
                                        new_sfc.setdefault(canon, {})
                                        for face_idx, n in face_map.items():
                                            new_sfc[canon][face_idx] = new_sfc[canon].get(face_idx, 0) + n
                                    fusion["speaker_face_count"] = new_sfc
                                print(f"[FaceID] {len(faceid_remap)}개 speaker remap 적용 "
                                      f"→ 화자 수 {len(set(faceid_remap.values()) | (set(spk for _, _, spk in diarization.itertracks(yield_label=True)) - set(faceid_remap.keys())))}")

                            # === v49 NEW: spurious 1-turn speaker auto-absorb ===
                            # LATENTSYNC_SPURIOUS_OFF=1 → 비활성. default 활성.
                            _spurious_off = os.environ.get("LATENTSYNC_SPURIOUS_OFF", "0") == "1"
                            if not _spurious_off:
                                try:
                                    from av_fusion import apply_reassignment_to_diarization as _apply_re
                                    current_audio = [
                                        (turn.start, turn.end, spk)
                                        for turn, _, spk in diarization.itertracks(yield_label=True)
                                    ]
                                    sp_reassign = absorb_spurious_speakers(
                                        current_audio, asd_result, track_embs,
                                        fusion.get("speaker_face_count", {}),
                                        verbose=True,
                                    )
                                    if sp_reassign:
                                        diarization, _sp_intervals = _apply_re(
                                            diarization, sp_reassign, return_intervals=True
                                        )
                                        prev_iv = data.get("av_reassigned_intervals", []) or []
                                        data["av_reassigned_intervals"] = prev_iv + list(_sp_intervals)
                                        print(f"[Spurious] {len(sp_reassign)} segments absorbed")
                                except Exception as _se:
                                    import traceback as _stb
                                    print(f"[Spurious] 실패 (계속 진행): {_se}")
                                    _stb.print_exc()

                            # === v36 NEW: per-segment ArcFace voting (가설 C) ===
                            # cluster_canonical safety 우회. segment 마다 face embedding
                            # 직접 비교 → 가장 가까운 speaker로 reassign.
                            # 환경변수: LATENTSYNC_FACE_VOTING_OFF=1 → 비활성
                            #   LATENTSYNC_FACE_VOTING_THRESHOLD (default 0.55)
                            #   LATENTSYNC_FACE_VOTING_MARGIN (default 0.05)
                            #   LATENTSYNC_FACE_VOTING_MIN_SHARE (default 0.50)
                            _voting_off = os.environ.get("LATENTSYNC_FACE_VOTING_OFF", "0") == "1"
                            if not _voting_off:
                                try:
                                    from av_fusion import apply_reassignment_to_diarization as _apply_reassign
                                    current_audio_segs = [
                                        (turn.start, turn.end, spk)
                                        for turn, _, spk in diarization.itertracks(yield_label=True)
                                    ]
                                    voting_reassignments = reassign_segments_by_face_voting(
                                        current_audio_segs, asd_result, track_embs,
                                        fusion.get("speaker_face_count", {}),
                                        verbose=True,
                                    )
                                    if voting_reassignments:
                                        diarization, _voting_intervals = _apply_reassign(
                                            diarization, voting_reassignments, return_intervals=True
                                        )
                                        # av_reassigned_intervals 에 추가 (ref bank 오염 방지)
                                        prev_intervals = data.get("av_reassigned_intervals", []) or []
                                        data["av_reassigned_intervals"] = prev_intervals + list(_voting_intervals)
                                        # speaker_face_count 도 voting 결과 반영
                                        if "speaker_face_count" in fusion:
                                            voting_map = {(round(s, 3), round(e, 3)): (new, old)
                                                          for s, e, old, new in voting_reassignments}
                                            # downstream AV-Reassign 가 fresh state 보도록 fusion 재계산은 생략
                                            # (voting 결과 자체는 diarization에 이미 적용됨)
                                            pass
                                        print(f"[FaceVoting] {len(voting_reassignments)} segments reassigned")
                                except Exception as _ve:
                                    import traceback as _vtb
                                    print(f"[FaceVoting] 실패 (계속 진행): {_ve}")
                                    _vtb.print_exc()
                    except Exception as _fie:
                        import traceback as _ftb
                        print(f"[FaceID] 실패 (계속 진행): {_fie}")
                        _ftb.print_exc()

                # === v17/v19 per-segment AV reassignment (재구조) ===
                # face track 기반 라벨 교정. SPEAKER_3 → 5/0/1 오인 패턴 fix.
                # 환경변수:
                #   LATENTSYNC_AV_REASSIGN_OFF=1 → 비활성
                #   LATENTSYNC_AV_REASSIGN_DOMINANT (default 0.6) face owner threshold
                #   LATENTSYNC_AV_REASSIGN_SHARE   (default 0.5) seg dominant share
                #   LATENTSYNC_AV_REASSIGN_MAX_DUR (default 미설정=무제한)
                #                                  값 있으면 그보다 긴 segment skip
                #                                  (v18 회귀: 긴 발화의 audio 라벨 신뢰)
                # 부수 산출물:
                #   data["av_reassigned_intervals"] = [(start, end, old_spk), ...]
                #   → build_speaker_profiles에서 ref bank 오염 방지용으로 활용
                data["av_reassigned_intervals"] = []
                _reassign_off = os.environ.get("LATENTSYNC_AV_REASSIGN_OFF", "0") == "1"
                if _reassign_off:
                    print("[AV-Reassign] 비활성화 (LATENTSYNC_AV_REASSIGN_OFF=1)")
                else:
                    try:
                        from av_fusion import (
                            reassign_segments_by_face_owner,
                            apply_reassignment_to_diarization,
                        )
                        _reassign_dom = float(os.environ.get("LATENTSYNC_AV_REASSIGN_DOMINANT", "0.6"))
                        _reassign_share = float(os.environ.get("LATENTSYNC_AV_REASSIGN_SHARE", "0.5"))
                        _reassign_max_dur_str = os.environ.get("LATENTSYNC_AV_REASSIGN_MAX_DUR", "").strip()
                        _reassign_max_dur = float(_reassign_max_dur_str) if _reassign_max_dur_str else None
                        current_audio_segs = [
                            (turn.start, turn.end, spk)
                            for turn, _, spk in diarization.itertracks(yield_label=True)
                        ]
                        reassignments = reassign_segments_by_face_owner(
                            current_audio_segs, fusion, asd_result,
                            min_dominant_ratio=_reassign_dom,
                            min_seg_face_share=_reassign_share,
                            max_seg_duration=_reassign_max_dur,
                            verbose=True,
                        )
                        if reassignments:
                            diarization, reassigned_intervals = apply_reassignment_to_diarization(
                                diarization, reassignments, return_intervals=True
                            )
                            data["av_reassigned_intervals"] = reassigned_intervals
                            print(f"[AV-Reassign] {len(reassigned_intervals)}/{len(current_audio_segs)} segments 재할당 "
                                  f"(dom>={_reassign_dom:.0%}, share>={_reassign_share:.0%}, "
                                  f"max_dur={_reassign_max_dur if _reassign_max_dur else 'inf'})")
                    except Exception as _re:
                        import traceback as _rtb
                        print(f"[AV-Reassign] 실패 (계속 진행): {_re}")
                        _rtb.print_exc()

                # 결과 저장 (lipsync 단계에서 활용 가능)
                data["av_fusion"] = fusion
                data["asd_result"] = asd_result
                # === ASD_INDEX_PATCH:capture ===
                # Record this chunk's ASD pickle location + frame count for the
                # lipsync-side filter (asd_filter_runtime.LipsyncASDFilter).
                data["_asd_index_entry"] = {
                    "stem": chunk_name,
                    "asd_path": asd_cache_path if os.path.isfile(asd_cache_path) else None,
                    "n_frames": int(asd_result.get("n_frames", 0)),
                }
                # === ASD_INDEX_PATCH:capture end ===
                print(f"[AV-Fusion] face tracks={len(asd_result['tracks'])}, "
                      f"lipsync target frames={sum(1 for t in fusion['per_frame_target'] if t is not None)}/"
                      f"{fusion['n_frames']}")
            else:
                print(f"[AV-Fusion] ASD 결과 없음 → skip")
        except Exception as _e:
            import traceback as _tb
            print(f"[AV-Fusion] 실패 (계속 진행): {_e}")
            _tb.print_exc()

        # Step 5: 세그먼트 조합 (문장 단위, 수정 P)
        segments = build_segments(
            data["words"], diarization, data["vocals_path"],
            src_lang=data.get("detected_lang", src_lang or "en"),
        )
        # 5/7: ASD-guided segment refinement
        # 빠른 화자 교차 (drama)에서 face_id 변화 시점에서 split
        # 짧은 발화 over-extension 방지
        try:
            import sys as _sys
            if "/workspace/scripts" not in _sys.path:
                _sys.path.insert(0, "/workspace/scripts")
            from segment_refiner import refine_segments
            asd_for_refine = data.get("asd_result")
            # 5/7 fix: WordTiming dataclass 또는 dict 모두 지원
            def _w_get(w, key, default=None):
                if hasattr(w, key):
                    return getattr(w, key)
                if isinstance(w, dict):
                    return w.get(key, default)
                return default
            # word timestamps per segment 추출
            words_by_seg = []
            for seg in segments:
                seg_words = [
                    {"word": _w_get(w, "word", _w_get(w, "text", "")),
                     "start": _w_get(w, "start", 0.0),
                     "end": _w_get(w, "end", 0.0)}
                    for w in data["words"]
                    if _w_get(w, "start", 0) >= seg.start and _w_get(w, "end", 0) <= seg.end
                ]
                words_by_seg.append(seg_words)
            ecapa_centroids = data.get("speaker_centroids", {})

            # v87+: SPK별 face centroid 계산 (SPK merge 시 voice + face 결합용)
            # v112+: dominant face cluster의 face만 사용 (mixed centroid 회피)
            # v114+: SPK별 dominant gender 추출 (다른 gender SPK는 merge 금지)
            spk_face_centroid = {}
            spk_gender = {}
            spk_shape_feats = {}
            try:
                from scripts.face_id_embedder import get_last_track_genders as _get_tg
                _track_genders = _get_tg()
                _sfc2 = fusion.get("speaker_face_count", {}) if 'fusion' in locals() else {}
                if _track_genders and _sfc2:
                    from collections import Counter as _C2
                    for _spk, _face_map in _sfc2.items():
                        _gender_votes = []
                        for _f_idx, _n in _face_map.items():
                            g = _track_genders.get(_f_idx)
                            if g:
                                _gender_votes.extend([g] * int(_n))
                        if _gender_votes:
                            spk_gender[_spk] = _C2(_gender_votes).most_common(1)[0][0]
                    if spk_gender:
                        print(f"[Refine] SPK dominant gender: {spk_gender}")
            except Exception as _ge:
                print(f"[Refine] gender 계산 실패: {_ge}")
            # v120+: SPK별 shape feature centroid 계산 (mouth/face geometry)
            try:
                from scripts.face_id_embedder import get_last_track_shape_feats as _get_tsf
                _track_shapes = _get_tsf()
                _sfc3 = fusion.get("speaker_face_count", {}) if 'fusion' in locals() else {}
                if _track_shapes and _sfc3:
                    import numpy as _np_s
                    for _spk, _face_map in _sfc3.items():
                        _accs, _ws = [], []
                        for _f_idx, _n in _face_map.items():
                            sf = _track_shapes.get(_f_idx)
                            if sf is not None:
                                _accs.append(sf * float(_n))
                                _ws.append(float(_n))
                        if _accs and sum(_ws) > 0:
                            spk_shape_feats[_spk] = (_np_s.sum(_accs, axis=0) / sum(_ws)).astype(_np_s.float32)
                    if spk_shape_feats:
                        print(f"[Refine] SPK shape feats: {list(spk_shape_feats.keys())}")
            except Exception as _shge:
                print(f"[Refine] shape feats 계산 실패: {_shge}")
            try:
                _sfc = fusion.get("speaker_face_count", {}) if 'fusion' in locals() else {}
                _te = locals().get("track_embs", {})
                _ttc = locals().get("track_to_cluster", {})
                _use_dom_cluster = os.environ.get("LATENTSYNC_SPK_FACE_DOMINANT_CLUSTER", "0") == "1"
                if _sfc and _te:
                    import numpy as _np
                    from collections import defaultdict as _dd
                    # SPK별 dominant cluster 결정
                    spk_dom_cluster = {}
                    if _use_dom_cluster and _ttc:
                        for _spk, _face_map in _sfc.items():
                            _cluster_count = _dd(int)
                            for _f_idx, _n in _face_map.items():
                                _cid = _ttc.get(_f_idx)
                                if _cid is not None:
                                    _cluster_count[_cid] += _n
                            if _cluster_count:
                                spk_dom_cluster[_spk] = max(_cluster_count, key=_cluster_count.get)
                        print(f"[Refine] SPK dominant cluster: {spk_dom_cluster}")
                    for _spk, _face_map in _sfc.items():
                        _embs, _w = [], []
                        _dom_cid = spk_dom_cluster.get(_spk) if _use_dom_cluster else None
                        for _f_idx, _n in _face_map.items():
                            if _use_dom_cluster and _dom_cid is not None and _ttc:
                                if _ttc.get(_f_idx) != _dom_cid:
                                    continue
                            if _f_idx in _te:
                                _embs.append(_te[_f_idx])
                                _w.append(float(_n))
                        if _embs:
                            _avg = _np.average(_np.stack(_embs), axis=0, weights=_w)
                            _norm = max(_np.linalg.norm(_avg), 1e-9)
                            spk_face_centroid[_spk] = _avg / _norm
                    print(f"[Refine] spk_face_centroid 계산 완료: {list(spk_face_centroid.keys())} (dominant_cluster={_use_dom_cluster})")
            except Exception as _se:
                print(f"[Refine] spk_face_centroid 계산 실패: {_se}")

            # v98+: wespeaker model load (옵션, ENV로 활성화)
            _wespeaker_model = None
            if os.environ.get("LATENTSYNC_WESPEAKER", "0") == "1":
                try:
                    import wespeakerruntime as _wr
                    _ws_onnx = os.environ.get("LATENTSYNC_WESPEAKER_ONNX", "")
                    if _ws_onnx and os.path.exists(_ws_onnx):
                        _wespeaker_model = _wr.Speaker(onnx_path=_ws_onnx, lang="en")
                        print(f"[Refine] wespeaker model loaded ({os.path.basename(_ws_onnx)})")
                    else:
                        _wespeaker_model = _wr.Speaker(lang="en")
                        print("[Refine] wespeaker model loaded (voxceleb_resnet34_LM default)")
                except Exception as _wse:
                    print(f"[Refine] wespeaker load 실패: {_wse}")

            refined = refine_segments(
                segments,
                asd_for_refine,
                words_by_seg,
                speaker_centroids=ecapa_centroids,
                vocals_path=data["vocals_path"],
                ecapa_model=_ecapa_model,
                spk_face_centroid=spk_face_centroid,
                wespeaker_model=_wespeaker_model,
                spk_gender=spk_gender,
                spk_shape_feats=spk_shape_feats,
            )
            if len(refined) != len(segments):
                print(f"[Refine] {len(segments)} → {len(refined)} segments after ASD refinement")
            segments = refined
        except Exception as _re:
            print(f"[Refine] failed ({_re}) — 기존 segments 유지")
            import traceback as _tb
            _tb.print_exc()

        data["diarization"] = diarization
        data["segments"] = segments
    _unload("diarization")

    # v75+: DIARIZE_ONLY 모드 — 화자 분리 sweep용 fast iteration (TTS skip)
    if os.environ.get("LATENTSYNC_DIARIZE_ONLY", "0") == "1":
        from pathlib import Path as _Path
        _meta_dir = _Path(RUNS_DIR) / CURRENT_RUN_ID / "meta"
        _meta_dir.mkdir(exist_ok=True)
        for _cn, _d in chunk_data.items():
            _segs = _d.get("segments", [])
            _out = _meta_dir / f"{_cn}_diarize_only.json"
            with open(_out, 'w', encoding='utf-8') as _f:
                json.dump({
                    "chunk_name": _cn,
                    "n_segments": len(_segs),
                    "groups": [{
                        "speaker": getattr(s, "speaker", ""),
                        "start": float(getattr(s, "start", 0)),
                        "end": float(getattr(s, "end", 0)),
                        "text": getattr(s, "text", ""),
                    } for s in _segs],
                }, _f, ensure_ascii=False, indent=2)
            print(f"[DIARIZE_ONLY] {_cn}: {len(_segs)} segments → {_out}")
        print("[DIARIZE_ONLY] Exit before TTS (LATENTSYNC_DIARIZE_ONLY=1).")
        import sys as _sys
        _sys.exit(0)

    # ── 3단계: 감정 추출 + MOS 레퍼런스 선택 ──────────────
    load_emotion()
    load_mos()  # MOS 모델 로드 (레퍼런스 평가용)
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [Emotion] 청크: {chunk_name} ---")

        # Step 6: 감정 추출 (콘텐츠 타입 정책 적용)
        data["segments"] = fill_emotions(
            data["segments"], data["vocals_path"], content_type=content_type
        )

        # Step 7: Speaker Profile Bank 구성 (MOS 필터 적용)
        # v19: AV-Reassign 영역은 ref 후보에서 제외 (ref 오염 방지).
        # LATENTSYNC_REF_EXCLUDE_REASSIGNED=0 → 제외 비활성
        _ref_exclude_off = os.environ.get("LATENTSYNC_REF_EXCLUDE_REASSIGNED", "1") == "0"
        _exclude = None if _ref_exclude_off else data.get("av_reassigned_intervals") or None
        data["profiles"] = build_speaker_profiles(
            data["segments"], data["vocals_path"],
            exclude_intervals=_exclude,
        )
    _unload("emotion")
    _unload("mos")  # 레퍼런스 선택 끝 → MOS 해제 (TTS 평가 시 다시 로드)

    # ── 4단계: 번역 + 속도 조절 (CPU, 모든 청크) ─────────
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [Translate] 청크: {chunk_name} ---")

        # Step 8: 번역
        data["segments"] = translate_segments(data["segments"], tgt_lang, content_type=content_type)

        # === v13: emotion-based speed (감정 표현 강화) ===
        # 기존(v12): 모든 seg.speed = 1.0 강제
        # 변경: emotion에 따라 미세 조정 (range 0.92~1.05 — 교수님 지적 회피)
        #   Sad: 0.92 (천천히, atempo 늘림 줄임 효과도 있음)
        #   Angry/Surprised: 1.05 (격렬한 감정 + atempo 압축 회피)
        # 길이 보정은 여전히 synthesize_chunk()의 time_stretch가 담당
        EMOTION_SPEED = {
            "Sad":       0.92,
            "Angry":     1.05,
            "Surprised": 1.05,
            "Happy":     1.0,
            "Scared":    0.95,
            "Neutral":   1.0,
        }
        for seg in data["segments"]:
            seg.speed = EMOTION_SPEED.get(seg.emotion, 1.0)

    # ── 5단계: TTS + 믹싱 (MOS reload 제거 - MOS_RESYNTH 비활성 상태) ───
    # v28 (5/14): Resume — dubbed wav + chunk_final.mp4 존재 시 skip
    _cosy_loaded = False
    for chunk_name, data in chunk_data.items():
        # Resume check: dubbed wav 존재
        expected_dubbed = os.path.join(DUBBED_DIR, f"{chunk_name}_dubbed.wav")
        expected_final = os.path.join(CHUNKS_DIR, f"{chunk_name}_final.mp4")
        if _resume_mode and os.path.exists(expected_dubbed):
            print(f"\n--- [TTS] 청크: {chunk_name} [Resume skip] ⚡ (dubbed 존재)")
            dubbed_path = expected_dubbed
        else:
            if not _cosy_loaded:
                load_cosy()
                _cosy_loaded = True
            print(f"\n--- [TTS] 청크: {chunk_name} ---")
            dubbed_path = synthesize_chunk(
                data["segments"], data["profiles"], chunk_name, tgt_lang,
                video_duration=data.get("video_duration"),
            )

        # Mix: chunk_final.mp4 존재 시 skip
        if _resume_mode and os.path.exists(expected_final):
            print(f"[Mix] 청크: {chunk_name} [Resume skip] ⚡ ({expected_final})")
            final_path = expected_final
        else:
            final_path = os.path.join(CHUNKS_DIR, f"{chunk_name}_final.mp4")
            mix_audio(data["chunk_path"], dubbed_path, data["bgm_path"], final_path)
        data["chunk_final_path"] = final_path
    if _cosy_loaded:
        _unload("cosy")

    # 전체 청크 합치기
    output_path = concat_chunks(file_name, tgt_lang)

    # ─── 🎬 LatentSync 1.6 립싱크 적용 ──────────────────────────
    # enable_lipsync=True 일 때만 실행. 실패 시 원본 output_path 유지.
    # 결과는 _lipsync.mp4 별도 파일로 저장 → 비교 가능 + 롤백 용이.
    if enable_lipsync:
        # === DAEMON CLEANUP (방어적, 5/7 추가): lipsync 진입 시 항상 daemon 정리
        # smart_daemon flag 무관 — 이전 run의 stale daemon이 GPU context 점유 시
        # CUDA "device not ready" 발생 가능 (sm_120 driver 민감)
        # 데몬이 없어도 pkill은 무해 (no-op)
        print(f"[Lipsync-Pre] daemon 정리 (방어적, GPU context fragmentation 방지)")
        try:
            _stop_daemons()
        except Exception as _e:
            print(f"[Lipsync-Pre] daemon stop 실패: {_e} (lipsync는 진행)")

        lipsync_out = output_path.replace(".mp4", "_lipsync.mp4")
        # v27: use_lora=False (기본) 시 베이스 모델 사용 = 깨끗한 1.6 결과
        # use_lora=True 시 lang별 자동 선택 (latentsync_<lang>.pt)
        effective_tgt_lang = tgt_lang if use_lora else None
        # VAE chunk + VAE variant는 환경변수로 inference.py에 전달
        os.environ["LATENTSYNC_VAE_CHUNK"] = str(lipsync_vae_chunk)
        # 5/7: EMA VAE default (v42 한국어 검증 — mse 대비 부드러움, GFPGAN과 궁합 ↑)
        os.environ.setdefault("LATENTSYNC_VAE_VARIANT", "ema")
        # === ASD_INDEX_PATCH:dump ===
        # Materialize per-chunk ASD info as
        #   runs/<run_id>/meta/asd_filter_index.json
        # so the LatentSync subprocess can map global frame indices back to
        # the right ASD pickle. Order matches concat_chunks() (sorted name).
        try:
            import json as _json_asd
            run_root = os.path.join(RUNS_DIR, CURRENT_RUN_ID) if CURRENT_RUN_ID else None
            if run_root and os.path.isdir(run_root):
                _entries = []
                for _ck, _cd in sorted(chunk_data.items()):
                    _e = _cd.get("_asd_index_entry")
                    if _e and _e.get("n_frames"):
                        _entries.append(_e)
                if _entries:
                    _meta_dir = os.path.join(run_root, "meta")
                    os.makedirs(_meta_dir, exist_ok=True)
                    _idx_path = os.path.join(_meta_dir, "asd_filter_index.json")
                    with open(_idx_path, "w", encoding="utf-8") as _f:
                        _json_asd.dump({
                            "version": 1,
                            "fps": 25.0,
                            "score_threshold": float(
                                os.environ.get("LATENTSYNC_ASD_THRESHOLD", "0.0")
                            ),
                            "chunks": _entries,
                        }, _f, indent=2)
                    print(f"[ASD-Filter] index dumped: {_idx_path} "
                          f"(chunks={len(_entries)}, total_frames="
                          f"{sum(e['n_frames'] for e in _entries)})")
                    if os.environ.get("LATENTSYNC_ASD_FILTER_DISABLE", "0") != "1":
                        os.environ["LATENTSYNC_ASD_FILTER_RUN_DIR"] = run_root + "/"
                        print(f"[ASD-Filter] ENABLED for lipsync subprocess "
                              f"(LATENTSYNC_ASD_FILTER_RUN_DIR={run_root}/)")
                    else:
                        print("[ASD-Filter] disabled by LATENTSYNC_ASD_FILTER_DISABLE=1")
                else:
                    print("[ASD-Filter] no chunks have ASD output, skipping index dump")
            else:
                print(f"[ASD-Filter] run_root unavailable, skipping index dump")
        except Exception as _e_asd:
            import traceback as _tb_asd
            print(f"[ASD-Filter] index dump failed (continuing without filter): {_e_asd}")
            _tb_asd.print_exc()
        # === ASD_INDEX_PATCH:dump end ===

        # === SPEAKER_PROFILE_PATCH:dump (5/12) ===
        # Build face_profiles + audio F0 gender JSONs and set env vars so the
        # LatentSync subprocess applies per-face speaker matching.
        # Uses the standalone scripts under /workspace/patches/ which read
        # the ASD pickles + diarized vocals.
        try:
            _spk_run_root = os.path.join(RUNS_DIR, CURRENT_RUN_ID) if CURRENT_RUN_ID else None
            if _spk_run_root and os.path.isdir(_spk_run_root) and \
               os.environ.get("LATENTSYNC_SPEAKER_PROFILES_DISABLE", "0") != "1":
                import subprocess as _sub_sp
                from concurrent.futures import ThreadPoolExecutor as _TPE
                _build_script = "/workspace/patches/build_face_profiles.py"
                _f0_script = "/workspace/patches/audio_f0_gender.py"
                _meta_dir = os.path.join(_spk_run_root, "meta")
                os.makedirs(_meta_dir, exist_ok=True)

                # 5/12 PARALLEL: face profile (GPU/insightface) + audio F0 (CPU/librosa)
                # 둘이 자원 경쟁 안 하므로 동시 실행 → ~2분 절약
                def _build_face_profiles_task():
                    if not os.path.isfile(_build_script):
                        return None
                    _n_spks = locals().get("num_speakers", None)
                    _cmd = [LATENT_SYNC_PYTHON, _build_script,
                            "--run-dir", _spk_run_root,
                            "--cluster-threshold", "0.4",
                            "--n-samples", "10"]
                    if _n_spks:
                        _cmd += ["--num-speakers", str(int(_n_spks))]
                    _env = dict(os.environ)
                    _env["LATENTSYNC_ENABLE_FACE_RECOGNITION"] = "1"
                    print("[SpeakerProfile] building speaker_face_profiles.json (parallel) ...", flush=True)
                    return _sub_sp.run(_cmd, env=_env,
                                       capture_output=True, text=True, timeout=600)

                def _build_audio_gender_task():
                    if not os.path.isfile(_f0_script):
                        return None
                    _dubbed_dir = os.path.join(_spk_run_root, "dubbed")
                    if not os.path.isdir(_dubbed_dir):
                        return None
                    _wavs = sorted(_g for _g in os.listdir(_dubbed_dir)
                                   if _g.endswith("_dubbed.wav"))
                    if not _wavs:
                        return None
                    _wav_path = os.path.join(_dubbed_dir, _wavs[0])
                    _f0_json_local = os.path.join(_meta_dir, "audio_f0_gender.json")
                    print(f"[SpeakerProfile] building audio_f0_gender.json on {_wavs[0]} (parallel) ...", flush=True)
                    return _sub_sp.run([LATENT_SYNC_PYTHON, _f0_script,
                                        "--wav", _wav_path,
                                        "--fps", "25",
                                        "--out", _f0_json_local],
                                       capture_output=True, text=True, timeout=300)

                with _TPE(max_workers=2) as _ex:
                    _fut_face = _ex.submit(_build_face_profiles_task)
                    _fut_audio = _ex.submit(_build_audio_gender_task)
                    _rc_face = _fut_face.result()
                    _rc_audio = _fut_audio.result()

                # Process face profile result
                _spk_json = os.path.join(_meta_dir, "speaker_face_profiles.json")
                if _rc_face is not None and _rc_face.returncode == 0 and os.path.isfile(_spk_json):
                    os.environ["LATENTSYNC_SPEAKER_PROFILES_PATH"] = _spk_json
                    print(f"[SpeakerProfile] ENABLED ({_spk_json})")
                else:
                    rc = _rc_face.returncode if _rc_face else "(skipped)"
                    err = (_rc_face.stderr[-200:] if _rc_face and _rc_face.stderr else "")
                    print(f"[SpeakerProfile] face build failed: rc={rc}, stderr: {err!r}")
                # Process audio gender result
                _f0_json = os.path.join(_meta_dir, "audio_f0_gender.json")
                if _rc_audio is not None and _rc_audio.returncode == 0 and os.path.isfile(_f0_json):
                    os.environ["LATENTSYNC_AUDIO_F0_GENDER_PATH"] = _f0_json
                    print(f"[SpeakerProfile] audio gender ENABLED ({_f0_json})")
                else:
                    rc = _rc_audio.returncode if _rc_audio else "(skipped)"
                    print(f"[SpeakerProfile] F0 build failed: rc={rc}")
                # Recognition auto-enables when SPEAKER_PROFILES_PATH set (face_detector.py)
        except Exception as _e_sp:
            import traceback as _tb_sp
            print(f"[SpeakerProfile] integration failed (continuing): {_e_sp}")
            _tb_sp.print_exc()
        # === SPEAKER_PROFILE_PATCH:dump end ===
        result = apply_lipsync(
            dubbed_video_path=output_path,
            output_path=lipsync_out,
            tgt_lang=effective_tgt_lang,
            inference_steps=lipsync_steps,
            guidance_scale=lipsync_guidance,
            seed=lipsync_seed,
            config_name=lipsync_config,
            ckpt_path=lipsync_ckpt,
            enable_deepcache=lipsync_deepcache,
            # 5/11 new: 가속·품질 옵션
            use_trt=lipsync_use_trt,
            scheduler=lipsync_scheduler,
            teacache_threshold=lipsync_teacache,
            profile_threshold=lipsync_profile_threshold,
            face_diag_min_ratio=lipsync_face_diag_min_ratio,
            chunk_seconds=lipsync_chunk_seconds,
            face_strict=lipsync_face_strict,
        )
        if result:
            print(f"[Pipeline] 립싱크 적용된 영상: {result}")
            print(f"[Pipeline] 원본 (오디오만): {output_path}")
            output_path = result  # 최종 출력은 립싱크 버전
        else:
            print(f"[Pipeline] ⚠️  립싱크 실패 — 오디오만 더빙된 원본 유지")
    # ──────────────────────────────────────────────────────────

    # ─── 🎨 Mouth-only enhance (5/12: 사용자 요청 — 립싱크 영역만 화질 향상) ──
    # 전체 GFPGAN 대비 빠르고 (1080p 6분 ~10분), mask 영역만 GFPGAN +
    # Gaussian feather (σ=6) 로 마스크 자국 완화.
    # enable_postprocess=True 이면 자동 적용 (기존 GFPGAN 대체).
    # Lipsync succeeded iff output_path was updated to a *_lipsync.mp4
    _lipsync_ok = enable_lipsync and output_path.endswith("_lipsync.mp4")
    if enable_lipsync and enable_postprocess and _lipsync_ok:
        try:
            # Original frame source for non-mask area (BGM-mixed pre-lipsync)
            _chunk_final = None
            for _ck, _cd in sorted(chunk_data.items()):
                _cf = _cd.get("chunk_final_path") if isinstance(_cd, dict) else None
                if _cf and os.path.isfile(_cf):
                    _chunk_final = _cf
                    break
            _mouth_script = "/workspace/patches/mouth_only_enhance_v3.py"
            if _chunk_final and os.path.isfile(_mouth_script):
                _gfpgan_py = "/opt/venv_gfpgan/bin/python"
                _mouth_out = output_path.replace(".mp4", "_mouth_enhanced.mp4")
                # 5/12 update: poisson_mixed (MIXED_CLONE) — 양쪽 gradient 유지.
                # NORMAL_CLONE 은 source color 까지 덮어 lipsync 안 보이는 부작용.
                # MIXED 는 source/target 둘 다 보존 → lipsync 가시성 + 경계 부드럽게.
                print(f"\n--- [Mouth-only enhance] Poisson MIXED_CLONE + GFPGAN ---")
                _rc = subprocess.run(
                    [_gfpgan_py, _mouth_script,
                     "--lipsync", output_path,
                     "--original", _chunk_final,
                     "--output", _mouth_out,
                     "--blend-mode", "poisson_mixed",   # changed from "poisson"
                     "--feather-sigma", "3",
                     "--color-match",
                     "--temporal-smooth", "5",
                     "--face-diag-min-ratio", "0.10",
                     "--mask-erode-px", "2",
                     "--no-nvenc",
                     "--mux-audio-from", output_path],
                    capture_output=True, text=True, timeout=1800,
                )
                if _rc.returncode == 0 and os.path.isfile(_mouth_out):
                    print(f"[Mouth-Enhance] 완료: {_mouth_out}")
                    output_path = _mouth_out
                else:
                    print(f"[Mouth-Enhance] ⚠️  실패 (rc={_rc.returncode}) — 립싱크 버전 유지")
                    if _rc.stderr:
                        print(f"  stderr 마지막 200: {_rc.stderr[-200:]!r}")
            else:
                print(f"[Mouth-Enhance] skip (script/chunk_final 없음)")
        except Exception as _e_mouth:
            print(f"[Mouth-Enhance] 예외 (계속 진행): {_e_mouth}")
    # ──────────────────────────────────────────────────────────

    # JSON 리포트 저장
    report_path = save_pipeline_report(file_name, tgt_lang, chunk_data, output_path)

    print(f"\n{'='*50}")
    print(f"[Pipeline] 완료!")
    print(f"[Pipeline] Run ID: {ctx.run_id}")
    print(f"[Pipeline] 작업 공간: {ctx.root}")
    print(f"[Pipeline] 출력: {output_path}")
    print(f"[Pipeline] 리포트: {report_path}")
    print(f"{'='*50}\n")

    return output_path


# ─── 실행 진입점 ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 더빙 파이프라인")
    parser.add_argument("--input",   required=True,  help="입력 영상 경로 (예: /data/input/movie.mp4)")
    parser.add_argument("--name",    default=None,   help="파일 이름 (예: movie). 없으면 파일명에서 자동 추출")
    parser.add_argument("--lang",    default="ko",   help="목표 언어 코드 (기본: ko)")
    parser.add_argument("--src",     default=None,   help="원본 언어 코드 (기본: 자동 감지)")
    parser.add_argument("--segment", default=300, type=int, help="청크 길이 초 (기본: 300)")
    parser.add_argument("--speakers", default=None, type=int, help="화자 수 (예: 3). 미지정 시 자동 감지")
    parser.add_argument("--run-id", default=None, dest="run_id",
                        help="실행 ID 직접 지정 (미지정 시 자동 생성). "
                             "재실행/디버깅 시 이전 run과 동일 ID로 돌릴 수 있음")
    parser.add_argument("--content-type", default="auto", dest="content_type",
                        choices=["auto", "lecture", "interview", "news", "movie", "drama"],
                        help="콘텐츠 타입. 감정 처리 정책에 영향 (기본: auto). "
                             "lecture/interview/news는 모든 세그먼트를 Neutral로 고정하여 "
                             "감정 오탐지 및 TTS 반복 환각을 방지함. "
                             "movie/drama/auto는 emotion2vec+ 감지값을 그대로 사용.")
    # 🎬 LatentSync 1.6 립싱크 옵션
    parser.add_argument("--enable-lipsync", action="store_true", dest="enable_lipsync",
                        help="립싱크 적용 (concat 후). 출력은 _lipsync.mp4 별도 파일")
    parser.add_argument("--lipsync-steps", default=10, type=int, dest="lipsync_steps",
                        help="diffusion inference steps (기본 10 with DPMSolver++ 5/11. DDIM 사용 시 20 권장)")
    parser.add_argument("--lipsync-guidance", default=1.5, type=float, dest="lipsync_guidance",
                        help="guidance scale (기본 1.5, 1.0~3.0)")
    parser.add_argument("--lipsync-seed", default=1247, type=int, dest="lipsync_seed",
                        help="diffusion seed (기본 1247)")
    parser.add_argument("--lipsync-config", default="stage2_512_nf16.yaml", dest="lipsync_config",
                        help="LatentSync config: stage2_512_nf16.yaml(v27 default, 16 frames @ 512, 깨끗한 베이스) "
                             "/ stage2_512.yaml(4 frames @ 512) / stage2_efficient.yaml(256, 빠름)")
    parser.add_argument("--lipsync-ckpt", default=None, dest="lipsync_ckpt",
                        help="가중치 경로 (None=베이스 모델 사용. --use-lora 시 lang별 자동)")
    parser.add_argument("--use-lora", action="store_true", dest="use_lora",
                        help="LoRA 자동 인식 활성화 (default: 베이스 모델만 — v27 기본값, 입 울렁거림 방지)")
    parser.add_argument("--lipsync-deepcache", action="store_true", dest="lipsync_deepcache",
                        help="DeepCache 활성화 (interval=7, -10~15%% 시간). ⚠️ sm_120 Blackwell에서 CUDA stream issue 발생 — OFF 권장")
    parser.add_argument("--lipsync-vae-chunk", default=2, type=int, dest="lipsync_vae_chunk",
                        help="VAE chunk_size (기본 2, v42 정확 매칭. 4=10%% 빠름but 미세 quality 저하 가능)")
    parser.add_argument("--enable-postprocess", action="store_true", dest="enable_postprocess",
                        help="GFPGAN 후처리 활성화 (face quality 향상, 입술 sharpen)")
    parser.add_argument("--postprocess-upscale", default=1, type=int, dest="postprocess_upscale",
                        help="GFPGAN upscale (1=유지 default 5/11, 2=2x SR)")
    parser.add_argument("--postprocess-downscale-detect", default=0, type=int,
                        dest="postprocess_downscale_detect",
                        help="GFPGAN v3 only: detection 만 N배 다운스케일 (0=off, 2=540p detect — 시간 -58%%)")
    # 5/11 new: lipsync 가속·품질 옵션
    parser.add_argument("--lipsync-use-trt", action="store_true", dest="lipsync_use_trt",
                        default=True,
                        help="LatentSync TRT FP16 엔진 사용 (기본 ON, 3.18× 가속)")
    parser.add_argument("--no-lipsync-trt", action="store_false", dest="lipsync_use_trt",
                        help="TRT 비활성화 (PyTorch UNet fallback)")
    parser.add_argument("--lipsync-scheduler", default="dpm", choices=["dpm", "ddim"],
                        dest="lipsync_scheduler",
                        help="DPMSolver++ (default) 또는 DDIM. DPM 사용 시 steps=10 권장")
    parser.add_argument("--lipsync-teacache", default=0.1, type=float,
                        dest="lipsync_teacache",
                        help="TeaCache rel_l1 threshold (기본 0.1, 0=off). UNet step 50%% skip")
    # 5/12: 사용자 요청 — "정면만 lipsync 적용". 측면 (yaw>0.35) 은 학습 분포
    # 밖이라 입만 따로 움직이는 artifact 생김 → 다시 enable.
    # ASD bbox + Profile + Audio gender 와 함께 작동 (중복 방어).
    parser.add_argument("--lipsync-profile-threshold", default=0.35, type=float,
                        dest="lipsync_profile_threshold",
                        help="측면 face skip yaw threshold (기본 0.35, 0=off)")
    parser.add_argument("--lipsync-face-min-ratio", default=0.0, type=float,
                        dest="lipsync_face_diag_min_ratio",
                        help="작은 face skip threshold (기본 0=off, ASD 통합 후 비활성)")
    parser.add_argument("--lipsync-chunk-seconds", default=0, type=int,
                        dest="lipsync_chunk_seconds",
                        help=">0 시 chunked inference (장편 영상 메모리 절약, 권장 10)")
    parser.add_argument("--lipsync-face-strict", action="store_true",
                        dest="lipsync_face_strict",
                        help="face detector strict mode (드라마용). det_score≥0.85, w/h≥0.55 (측면 더 강하게 skip), roll±30° 이상 skip")
    parser.add_argument("--fast-separate", action="store_true", dest="fast_separate",
                        help="Vocal sep htdemucs로 (-60초, drama/movie 빼고 추천. SDR 9.5)")
    parser.add_argument("--smart-daemon", action="store_true", dest="smart_daemon",
                        help="더빙 단계만 daemon 사용, lipsync 전 자동 stop (OOM 회피, 단일영상 -45초, batch -3분/영상)")
    args = parser.parse_args()
    # SEP_FAST 환경변수로 separate_audio에 전달
    if args.fast_separate:
        os.environ["SEP_FAST"] = "1"

    file_name = args.name or os.path.basename(args.input).rsplit(".", 1)[0]

    # 파이프라인 실행 (모델은 단계별로 자동 로드/언로드)
    run_pipeline(
        video_path=args.input,
        file_name=file_name,
        tgt_lang=args.lang,
        src_lang=args.src,
        segment_time=args.segment,
        num_speakers=args.speakers,
        run_id=args.run_id,
        content_type=args.content_type,
        enable_lipsync=args.enable_lipsync,
        lipsync_steps=args.lipsync_steps,
        lipsync_guidance=args.lipsync_guidance,
        lipsync_seed=args.lipsync_seed,
        lipsync_config=args.lipsync_config,
        lipsync_ckpt=args.lipsync_ckpt,
        use_lora=args.use_lora,
        lipsync_deepcache=args.lipsync_deepcache,
        lipsync_vae_chunk=args.lipsync_vae_chunk,
        enable_postprocess=args.enable_postprocess,
        postprocess_upscale=args.postprocess_upscale,
        postprocess_downscale_detect=args.postprocess_downscale_detect,  # 5/11
        # 5/11 new: 가속·품질 옵션
        lipsync_use_trt=args.lipsync_use_trt,
        lipsync_scheduler=args.lipsync_scheduler,
        lipsync_teacache=args.lipsync_teacache,
        lipsync_profile_threshold=args.lipsync_profile_threshold,
        lipsync_face_diag_min_ratio=args.lipsync_face_diag_min_ratio,
        lipsync_chunk_seconds=args.lipsync_chunk_seconds,
        lipsync_face_strict=args.lipsync_face_strict,
        smart_daemon=args.smart_daemon,
    )
