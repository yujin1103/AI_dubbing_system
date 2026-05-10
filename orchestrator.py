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
        """감정에 맞는 레퍼런스 반환. 없으면 Neutral 반환."""
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
    use_fast = os.environ.get("SEP_FAST", "0") == "1"
    if use_fast:
        # htdemucs_ft: SDR 9.5, 30초 (BS-Roformer 90초 대비 -60초)
        sep_script = (
            "import os; "
            "from audio_separator.separator import Separator; "
            "sep = Separator(output_dir=os.environ['OUT_DIR'], "
            "model_file_dir='/workspace/media/model_cache/audio_separator', "
            "log_level=30, use_autocast=True); "
            "sep.load_model(model_filename='htdemucs_ft.yaml'); "
            "sep.separate(os.environ['INPUT_PATH'])"
        )
        print(f"[Separate] SEP_FAST=1 → htdemucs_ft (-60초)")
    else:
        # BS-Roformer: SDR 12.97, 90초 (default, drama/movie BGM 강한 영상)
        sep_script = (
            "import os; "
            "from audio_separator.separator import Separator; "
            "sep = Separator(output_dir=os.environ['OUT_DIR'], "
            "model_file_dir='/workspace/media/model_cache/audio_separator', "
            "log_level=30, use_autocast=True); "
            "sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt'); "
            "sep.separate(os.environ['INPUT_PATH'])"
        )

    result = subprocess.run(
        ["/opt/venv_lipsync/bin/python", "-c", sep_script],
        capture_output=True, text=True,
        env={**os.environ,
             "OUT_DIR": out_dir,
             "INPUT_PATH": chunk_path,
             "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}
    )
    if result.returncode != 0:
        raise RuntimeError(f"BS-Roformer 실패:\n{result.stderr}")

    # BS-Roformer 출력: <name>_(Vocals)_<model>.wav, <name>_(Instrumental)_<model>.wav
    vocals_src = None
    bgm_src = None
    for f in os.listdir(out_dir):
        if "(Vocals)" in f:
            vocals_src = os.path.join(out_dir, f)
        elif "(Instrumental)" in f:
            bgm_src = os.path.join(out_dir, f)

    if not vocals_src or not bgm_src:
        raise FileNotFoundError(f"BS-Roformer 출력 누락: {os.listdir(out_dir)}")

    vocals_dst = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
    bgm_dst    = os.path.join(BGM_DIR,    f"{chunk_name}_bgm.wav")

    shutil.move(vocals_src, vocals_dst)
    shutil.move(bgm_src,    bgm_dst)
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

    # === DIARIZE_DAEMON_CLIENT (v28): 데몬 우선 (모델 로딩 20-30초 절감) ===
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

    # === DiariZen 우선 사용 (pyannote보다 DER ~30% 향상) ===
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
                # 마지막 줄이 JSON
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
                    print(f"[Diarize] DiariZen 결과 비어있음: {data} → pyannote fallback")
            else:
                err = (r.stderr or r.stdout)[:300]
                print(f"[Diarize] DiariZen 실패 → pyannote fallback: {err}")
        except Exception as e:
            print(f"[Diarize] DiariZen 호출 실패 → pyannote fallback: {e}")

    # === pyannote fallback ===
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    result = _diarization_model(vocals_path, **kwargs)

    # pyannote 4.x: DiarizeOutput 객체 → Annotation 추출
    if hasattr(result, "speaker_diarization"):
        return result.speaker_diarization
    return result


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
        print("[Diarize] pyannote 없음 → 후처리 스킵")
        return None, {}

    n_pyannote_speakers = len(set(s for _, _, s in diarization.itertracks(yield_label=True)))
    apply_merge = n_pyannote_speakers >= 3
    if not apply_merge:
        print(f"[Diarize] pyannote {n_pyannote_speakers}명 detect → 병합 skip, outlier 감지만 적용")

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
    new_turns.sort(key=lambda t: t[0])
    merged_turns = []
    for s, e, spk in new_turns:
        if merged_turns and merged_turns[-1][2] == spk and (s - merged_turns[-1][1]) < 0.3:
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
    # 각 segment의 embedding이 모든 centroid에서 너무 멀면 새 화자로 분리
    # (pyannote가 1.2s 같은 짧은 다른 화자 발화를 메인 화자에 잘못 합치는 경우 해결)
    OUTLIER_FAR_THRESH = 0.40  # 이하 = 새 화자 (남자 톤 변화 false positive 더 줄이기, 여자(0.29) 유지)
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
) -> List[List[int]]:
    """짧은 문장(<min_dur)은 인접 문장과 병합. 총 길이 cap 초과 금지."""
    if not sentence_word_indices:
        return []

    def dur(idxs):
        return words[idxs[-1]].end - words[idxs[0]].start

    merged = []
    for s in sentence_word_indices:
        if not s:
            continue
        if merged and (dur(s) < min_dur or dur(merged[-1]) < min_dur):
            combined_end = words[s[-1]].end
            combined_start = words[merged[-1][0]].start
            if (combined_end - combined_start) <= cap:
                merged[-1] = merged[-1] + s
                continue
        merged.append(list(s))
    return merged


def _split_groups_by_speaker(
    groups: List[List[int]],
    words: List["WordTiming"],
    diarization,
) -> List[List[int]]:
    """sentence group 내부에서 화자가 바뀌는 word 경계에서 강제 split.

    LLM 구두점 복원이 두 화자 발화를 한 문장으로 묶어버린 경우를 보정.
    1-word 깜빡임(A B A 패턴)은 smoothing으로 무시 → false split 방지.
    """
    if diarization is None:
        return groups
    result = []
    n_split = 0
    for grp in groups:
        if len(grp) < 2:
            result.append(grp)
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
    final_groups = _merge_short_sentences(expanded, words)

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

def extract_emotion(vocals_path: str, start: float, end: float) -> Tuple[str, float]:
    """
    emotion2vec+로 오디오 구간의 감정 추출.

    INPUT:
      vocals_path : str   — /data/vocals/chunk_000_vocals.wav
      start       : float — 구간 시작 (초)
      end         : float — 구간 끝 (초)

    OUTPUT:
      emotion       : str   — "Neutral" / "Angry" / "Sad" / "Happy" / ...
      emotion_score : float — 신뢰도 (0.0 ~ 1.0)
    """
    if _emotion_model is None:
        return "Neutral", 0.0

    try:
        audio, sr = sf.read(vocals_path)
        chunk = audio[int(start * sr):int(end * sr)]

        # 너무 짧은 구간 (0.3초 미만) 은 감정 추출 불가
        if len(chunk) < sr * 0.3:
            return "Neutral", 0.0

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
                best_idx  = scores.index(max(scores))
                raw_label = labels[best_idx].lower()
            
                # [추가] "生气/angry" 처럼 /가 있으면 뒤쪽의 "angry"만 떼어냅니다.
                if "/" in raw_label:
                    raw_label = raw_label.split("/")[-1]
                
                emotion   = EMOTION_LABEL_MAP.get(raw_label, "Neutral")
                return emotion, round(float(scores[best_idx]), 3)

    except Exception as e:
        print(f"[Emotion] 추출 실패: {e}")

    return "Neutral", 0.0


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

    override_count = 0
    for seg in segments:
        raw_emo, raw_score = extract_emotion(vocals_path, seg.start, seg.end)
        seg.raw_emotion = raw_emo
        seg.raw_emotion_score = raw_score

        if policy == "neutral_only":
            seg.emotion = "Neutral"
            seg.emotion_score = 1.0
            if raw_emo != "Neutral":
                override_count += 1
            print(f"[Emotion] seg {seg.id}: {raw_emo} ({raw_score:.2f}) → Neutral [policy]")
        else:  # passthrough
            seg.emotion = raw_emo
            seg.emotion_score = raw_score
            print(f"[Emotion] seg {seg.id}: {raw_emo} ({raw_score:.2f})")

    if policy == "neutral_only" and override_count > 0:
        print(f"[Emotion] 정책 적용으로 {override_count}/{len(segments)}개 세그먼트가 Neutral로 변경됨")
    return segments


# ─── Step 7: Speaker Profile Bank 구성 ───────────────────────

def build_speaker_profiles(
    segments: List[Segment],
    vocals_path: str
) -> Dict[str, SpeakerProfile]:
    """
    화자별 + 감정별 레퍼런스 음성 파일 자동 추출.
    MOS 평가로 가장 깨끗한 구간을 레퍼런스로 선택.
    3~15초 사이 구간만 후보로 사용 (CosyVoice3 제한: 30초).
    """
    # 화자별 감정별로 후보 구간 수집 (3~15초)
    candidates: Dict[str, Dict[str, List[Tuple[float, Segment]]]] = {}

    for seg in segments:
        duration = seg.end - seg.start
        if duration < 3.0 or duration > 15.0:
            continue
        if seg.speaker not in candidates:
            candidates[seg.speaker] = {}
        if seg.emotion not in candidates[seg.speaker]:
            candidates[seg.speaker][seg.emotion] = []
        candidates[seg.speaker][seg.emotion].append((duration, seg))

    # 후보가 없으면 길이 제한 완화 (1~25초)
    if not candidates:
        for seg in segments:
            duration = seg.end - seg.start
            if duration < 1.0 or duration > 25.0:
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

                # MOS 평가로 가장 깨끗한 구간 선택
                if _mos_evaluator is not None:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, clip, sr)
                        mos_score = _mos_evaluator.evaluate(tmp.name)
                        os.unlink(tmp.name)

                    if mos_score > best_mos:
                        best_mos = mos_score
                        best_seg = seg
                        best_duration = duration
                else:
                    # MOS 없으면 가장 긴 구간 선택 (기존 방식)
                    if duration > best_duration:
                        best_seg = seg
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
    use_emotion_desc = (policy == "passthrough")

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

    lines = []
    for batch_local_idx, (i, seg) in enumerate(batch):
        duration = max(0.3, seg.end - seg.start)
        seg_emotion = getattr(seg, 'emotion', 'Neutral') or 'Neutral'
        emo_factor = LLM_EMOTION_FACTOR.get(seg_emotion, 1.0)
        rate = BASE_SYL_RATE * emo_factor
        target_min = max(1, int(duration * (rate - 0.5)))
        target_max = max(target_min + 1, int(duration * (rate + 0.5)))
        if use_emotion_desc:
            raw_e = getattr(seg, 'raw_emotion', seg.emotion) or seg.emotion
            raw_s = getattr(seg, 'raw_emotion_score', seg.emotion_score) or 0.0
            lines.append(
                f"[{batch_local_idx}] (duration: {duration:.1f}s, target: {target_min}~{target_max} syllables, "
                f"emotion2vec_hint: {raw_e} {raw_s:.2f}, speaker: {seg.speaker}) {seg.text}"
            )
        else:
            lines.append(
                f"[{batch_local_idx}] (duration: {duration:.1f}s, target: {target_min}~{target_max} syllables) {seg.text}"
            )
    batch_text = "\n".join(lines)

    # ── 시스템 프롬프트 ──
    if use_emotion_desc:
        system_prompt = (
            f"You are a professional dubbing translator and emotion designer.\n"
            f"For each numbered line, output TWO fields:\n"
            f"  1. korean: natural spoken {lang_name} translation for dubbing\n"
            f"  2. tone: full English imperative phrase combining STYLE + EMOTION + SITUATION.\n"
            f"        FORMAT: starts with 'in a' or 'with', natural English imperative.\n"
            f"        EXAMPLES:\n"
            f"          'in a casual, pleased, lightly confident tone with warm but restrained excitement'\n"
            f"          'in a low, deliberate tone with grim resolve, conveying a quiet warning'\n"
            f"          'with subdued sadness, soft pacing, conveying lingering regret'\n"
            f"          'in a sharp, impatient tone with controlled frustration'\n"
            f"        BAD: 'angry' (too short), '낮은 톤' (Korean — must be English).\n"
            f"\n"
            f"OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\n"
            f"[N]\n"
            f"korean: <translation>\n"
            f"tone: <imperative phrase>\n"
            f"\n"
            f"TRANSLATION RULES:\n"
            f"- Each line specifies a syllable target: (target: MIN~MAX syllables).\n"
            f"- Translation MUST use between MIN and MAX syllables — HARD CONSTRAINT.\n"
            f"- SHORTER than MIN is FAILURE (causes silence). Add natural particles/expansions.\n"
            f"- LONGER than MAX is FAILURE (causes overflow). Compress, use shorter synonyms.\n"
            f"- Aim for the MIDDLE of the range.\n"
            f"- You MUST output translation in {lang_name}. DO NOT output original text.\n"
            f"- Do NOT merge content between lines.\n"
            f"- Make consecutive lines from the same speaker sound natural in sequence.\n"
            f"\n"
            f"TONE RULES:\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\n"
            f"- tone: English imperative phrase (15-30 words) starting with 'in a' or 'with'.\n"
            f"   Combine: STYLE (low/loud/slow/quick) + EMOTION (sad/angry/happy/calm) + SITUATION.\n"
            f"- This becomes TTS prefix: 'You are a helpful assistant. Please say this sentence {{tone}}.<|endofprompt|>'\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."
        )
    else:
        system_prompt = (
            f"You are a professional dubbing translator. Translate each numbered line into "
            f"natural spoken {lang_name} for dubbing with lip-sync.\n"
            f"CRITICAL RULES:\n"
            f"- Keep the same numbering format [0], [1], [2]...\n"
            f"- Each line specifies a syllable target: (target: MIN~MAX syllables).\n"
            f"- Your translation MUST use between MIN and MAX syllables — HARD CONSTRAINT.\n"
            f"- SHORTER than MIN is FAILURE (causes silence). Add natural particles.\n"
            f"- LONGER than MAX is FAILURE (causes overflow). Compress, use shorter synonyms.\n"
            f"- Aim for the MIDDLE of the range.\n"
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

    # === v13: 카테고리 fallback 강한 묘사 (drama 격렬한 감정 대응) ===
    # v12 ("with subdued sadness, soft pacing")는 너무 약함 → 사용자 "감정 부족"
    # 드라마는 톤 변화 큼 (소리지름, 격분, 절규) → 강한 영어 묘사 필요
    # 단 학습 분포 안에 있는 자연스러운 표현 유지 (leak 위험 ↓)
    _ = tts_emotion  # LLM 풍부 묘사는 폐기 (영어 leak 방지)
    if emotion in {"Sad", "Angry", "Happy", "Surprised", "Scared"}:
        cat_imperative = {
            "Sad":       "with deep, anguished sadness, slow pacing, restrained voice",
            "Angry":     "with sharp, raised, confrontational tone, intense urgency",
            "Happy":     "in a bright, energetic tone with lively, exuberant pacing",
            "Surprised": "with sudden, sharp surprise, raised intonation",
            "Scared":    "with raw, tense fear, shaky breathy voice",
        }[emotion]
        prefix = f'You are a helpful assistant. Please say this sentence {cat_imperative}.<|endofprompt|>'
    else:
        prefix = 'You are a helpful assistant.<|endofprompt|>'

    ref_16k = os.path.join(tempfile.gettempdir(), "ref_16k_temp.wav")
    try:
        # 레퍼런스를 16kHz 모노로 변환 (CosyVoice3 요구사항)
        subprocess.run([
            "ffmpeg", "-y", "-i", ref_audio,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", ref_16k
        ], capture_output=True)

        output = []
        # APRIL_28_REVERT: inference_cross_lingual + prefix in tts_text (검증된 방식)
        for result in _cosy_model.inference_cross_lingual(
            tts_text=f'{prefix}{text}',
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

        # CosyVoice3 출력(24000Hz)과 TTS_SAMPLE_RATE가 다르면 리샘플링
        if _cosy_model.sample_rate != TTS_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=_cosy_model.sample_rate, target_sr=TTS_SAMPLE_RATE)

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

    # 길이 조정 파라미터 (LIP_SYNC v27: 입 움직임에 정확히 맞추기)
    MAX_STRETCH = 1.25  # 압축 한도 1.15 → 1.25 (lip-sync 우선, 약간의 음질 trade-off)
    MIN_STRETCH = 0.85  # atempo 늘림 한도 (한국어가 짧을 때 0.85x까지 늘림)
    TOL_LATE = 0.15     # 입 움직임 끝 + 0.15s까지는 자연스러움 (사람 더빙도 약간 어긋남)
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
        # segment 자체 audio을 reference로 사용 → 음색 보존 + 더빙 발화
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
                    seg_dur = last_seg.end - first_seg.start
                    # 너무 짧으면 양쪽 0.5s 패딩 (CosyVoice3에 더 안정적)
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
                        print(f"  ↳ self-ref fallback: {first_seg.speaker} ({seg_dur:.2f}s, "
                              f"+{pad*2:.1f}s padding)")
            except Exception as _e:
                print(f"  ↳ self-ref 실패: {_e}")

        if not ref_path or not os.path.exists(ref_path):
            print(f"  ⚠️ {first_seg.speaker} reference 없음 — segment skip")
            continue

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
        PREDICT_THRESHOLD = MAX_STRETCH  # 1.15 — 압축 한도 초과 예상 시 재번역

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
                if new_syl < combined_syllables:
                    print(f"  ✅ 축약됨: {combined_syllables}음절 → {new_syl}음절")
                    combined_text = shorter
                    group[0].translated = shorter
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
                if is_last_group:
                    print(f"  ⚠️ 마지막 segment ratio={ratio:.2f} — {MAX_STRETCH}x 압축 "
                          f"+ 비디오 끝 맞춰 {trimmed_sec:.2f}s trim")
                else:
                    print(f"  ⚠️ ratio={ratio:.2f} 너무 큼 — {MAX_STRETCH}x 압축 "
                          f"+ 다음 segment 침범 방지 {trimmed_sec:.2f}s trim")
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
    # 🔥 SYNC FIX: duration=first → shortest. 비디오(0:v)/더빙/BGM 중 가장 짧은 길이로 맞춤.
    # 5/7 VOLUME FIX: loudnorm으로 perceptual loudness 매칭 (기계음 + 너무 큰 더빙 fix)
    if use_loudnorm:
        # EBU R128 loudnorm: 모든 컨텐츠가 일관된 perceptual loudness 가짐
        # I=-23 LUFS: 대화 표준 (영화 dialogue)
        # TP=-2 dBTP: 클립 방지
        # LRA=11: 자연스러운 dynamic range
        dubbed_filter = f"loudnorm=I={target_lufs}:TP=-2:LRA=11"
        filter_complex = (
            f"[1:a]aformat=channel_layouts=mono,{dubbed_filter}[dub];"
            f"[2:a]aformat=channel_layouts=mono,volume={bgm_volume}[bgm];"
            "[dub][bgm]amix=inputs=2:duration=shortest:normalize=0[a]"
        )
    else:
        # Legacy: 고정 비율
        filter_complex = (
            f"[1:a]aformat=channel_layouts=mono,volume={dubbed_volume}[dub];"
            f"[2:a]aformat=channel_layouts=mono,volume={bgm_volume}[bgm];"
            "[dub][bgm]amix=inputs=2:duration=shortest:normalize=0[a]"
        )
    cmd = [
        "ffmpeg",
        "-i", chunk_path,
        "-i", dubbed_path,
        "-i", bgm_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
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
    inference_steps: int = 20,   # 5/7 revert: 15→20 (한국어 phoneme 정확도 우선, v42 검증)
    guidance_scale: float = 1.5,
    seed: int = 1247,
    config_name: str = "stage2_512_nf16.yaml",  # v27 default: 16 frames @ 512
    ckpt_path: Optional[str] = None,        # None이면 베이스 (LoRA 안 씀)
    enable_deepcache: bool = False,         # 5/7 sm_120 CUDA stream issue — OFF default
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

    # === DAEMON CLEANUP (defense-in-depth): direct call에서도 안전 ===
    # run_pipeline에서 이미 호출되지만, apply_lipsync()를 단독 사용 시 보호
    # pkill no-op (daemon 없으면 무해), CUDA context fragmentation 방지
    try:
        _stop_daemons()
    except Exception as _e:
        print(f"[Lipsync] daemon cleanup 실패: {_e} (진행)")

    # 16kHz mono audio 추출
    audio_temp = os.path.join("/tmp", f"latentsync_audio_{os.getpid()}.wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", dubbed_video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
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
               "config_name", "ckpt_path", "enable_deepcache"}
    ls_kwargs = {k: v for k, v in kwargs.items() if k in ls_keys}

    # 🌐 lang별 가중치 자동 선택 (latentsync_<lang>.pt 자동 인식)
    if tgt_lang:
        explicit = ls_kwargs.get("ckpt_path")
        ls_kwargs["ckpt_path"] = resolve_lipsync_ckpt(tgt_lang, explicit)

    return apply_latent_sync(dubbed_video_path, output_path, **ls_kwargs)


# === GFPGAN 후처리 (face quality 향상) ===
# 5/7 update: async I/O 버전 default (sequential 대비 -22% 시간, 동일 품질)
GFPGAN_PYTHON = "/opt/venv_gfpgan/bin/python"
GFPGAN_SCRIPT = "/opt/LatentSync/scripts/gfpgan_async_postprocess.py"
GFPGAN_SCRIPT_FALLBACK = "/opt/LatentSync/scripts/gfpgan_postprocess.py"  # async 미존재 시 폴백
GFPGAN_MODEL  = "/opt/gfpgan_models/GFPGANv1.4.pth"

def apply_gfpgan_postprocess(
    lipsync_video_path: str,
    output_path: str,
    upscale: int = 1,
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
    # 5/7: async script 우선, 없으면 fallback
    gfpgan_script = GFPGAN_SCRIPT if os.path.isfile(GFPGAN_SCRIPT) else GFPGAN_SCRIPT_FALLBACK
    if not os.path.isfile(gfpgan_script):
        print(f"[GFPGAN] script 없음: {GFPGAN_SCRIPT} 및 {GFPGAN_SCRIPT_FALLBACK}")
        return None
    print(f"[GFPGAN] using: {os.path.basename(gfpgan_script)}")
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
    """파이프라인 중간 결과를 JSON으로 저장."""
    report = {
        "run_id": CURRENT_RUN_ID,
        "run_root": os.path.join(RUNS_DIR, CURRENT_RUN_ID) if CURRENT_RUN_ID else None,
        "input": file_name,
        "target_lang": tgt_lang,
        "timestamp": datetime.now().isoformat(),
        "output": output_path,
        "chunks": []
    }

    for chunk_name, data in chunk_data.items():
        chunk_report = {
            "name": chunk_name,
            "vocals_path": data.get("vocals_path", ""),
            "bgm_path": data.get("bgm_path", ""),
            "detected_lang": data.get("detected_lang", ""),
            "segments": [],
            "speaker_profiles": {}
        }

        for seg in data.get("segments", []):
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
                "tts_context": getattr(seg, 'tts_context', ''),
                "tts_emotion": getattr(seg, 'tts_emotion', ''),
                "speed": seg.speed,
                "tts_mos": getattr(seg, '_tts_mos', 0.0),
                "tts_retries": getattr(seg, '_tts_retries', 0),
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
    daemon_scripts = [
        "cosyvoice_daemon.py",
        "asr_daemon.py",
        "diarize_daemon.py",
    ]

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
            ["pgrep", "-f", "(cosyvoice_daemon|asr_daemon|diarize_daemon)"],
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
    lipsync_steps: int = 20,             # 5/7 revert: v42 ema_FINAL 품질이 한국어에 최적, 15는 phoneme 정확도 ↓
    lipsync_guidance: float = 1.5,       # guidance scale
    lipsync_seed: int = 1247,
    lipsync_config: str = "stage2_512_nf16.yaml",   # v27 default: 16 frames @ 512
    lipsync_ckpt: Optional[str] = None,        # None=베이스 (v27 default, use_lora 시 lang별 자동)
    use_lora: bool = False,                    # True면 LoRA 자동 인식
    lipsync_deepcache: bool = False,           # 5/7 sm_120 호환성 issue — OFF default
    lipsync_vae_chunk: int = 2,                # 5/7 final: chunk=2 (v42 정확 매칭, chunk=4는 quality 미세 저하 가능)
    enable_postprocess: bool = False,          # GFPGAN 후처리 (face quality, v42 setup)
    postprocess_upscale: int = 2,              # 5/7: 2x default (v42 검증 quality)
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
    chunk_data = {}
    for chunk_path in chunks:
        chunk_name = os.path.basename(chunk_path).replace(".mp4", "")
        print(f"\n--- [Separate] 청크: {chunk_name} ---")
        vocals_path, bgm_path = separate_audio(chunk_path)
        # 🔥 수정 N: 청크 비디오 길이 기록 (마지막 세그먼트 TTS 경계로 사용)
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
        #   pyannote가 같은 사람 톤 변화를 다른 화자로 잘못 인식하는 문제 해결.
        #   centroid 거리가 가까운 (0.65 cosine) 화자 쌍 자동 병합 + 짧은 turn 재할당.
        speaker_centroids: Dict[str, np.ndarray] = {}
        if num_speakers is None or num_speakers > 1:
            # num_speakers=1 이면 단일 화자로 간주 → 후처리 불필요
            diarization, speaker_centroids = post_process_diarization(diarization, data["vocals_path"])
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
                    merge_pairs = detect_face_based_merges(face_count, min_shared_frames=10, min_share_ratio=0.30)
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

                # 결과 저장 (lipsync 단계에서 활용 가능)
                data["av_fusion"] = fusion
                data["asd_result"] = asd_result
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
            refined = refine_segments(
                segments,
                asd_for_refine,
                words_by_seg,
                speaker_centroids=ecapa_centroids,
                vocals_path=data["vocals_path"],
                ecapa_model=_ecapa_model,
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
        data["profiles"] = build_speaker_profiles(
            data["segments"], data["vocals_path"]
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
    load_cosy()
    # OPTIMIZATION: MOS_RESYNTH_ENABLED=False이라 TTS 단계 MOS load 불필요 (~60s 절약)
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [TTS] 청크: {chunk_name} ---")

        # Step 9: 음성 합성 (MOS 기반 자동 재합성, 마지막 세그먼트 비디오 길이 보호)
        dubbed_path = synthesize_chunk(
            data["segments"], data["profiles"], chunk_name, tgt_lang,
            video_duration=data.get("video_duration"),
        )

        # Step 10: 믹싱
        final_path = os.path.join(CHUNKS_DIR, f"{chunk_name}_final.mp4")
        mix_audio(data["chunk_path"], dubbed_path, data["bgm_path"], final_path)
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
        )
        if result:
            print(f"[Pipeline] 립싱크 적용된 영상: {result}")
            print(f"[Pipeline] 원본 (오디오만): {output_path}")
            output_path = result  # 최종 출력은 립싱크 버전
        else:
            print(f"[Pipeline] ⚠️  립싱크 실패 — 오디오만 더빙된 원본 유지")
    # ──────────────────────────────────────────────────────────

    # ─── 🎨 GFPGAN 후처리 (face quality 향상) ──────────────────
    # 5/7: v42 setup (LoRA + smaller + steps=20 + EMA + GFPGAN 2x) = 사용자 검증 quality
    # Color Match는 rectangle 자국 이슈로 제외 (production에서 사용 안 함)
    if enable_lipsync and enable_postprocess:
        gfpgan_out = output_path.replace(".mp4", "_gfpgan.mp4")
        gfp_result = apply_gfpgan_postprocess(
            lipsync_video_path=output_path,
            output_path=gfpgan_out,
            upscale=postprocess_upscale,
        )
        if gfp_result:
            print(f"[Pipeline] GFPGAN 후처리 완료: {gfp_result}")
            output_path = gfp_result
        else:
            print(f"[Pipeline] ⚠️  GFPGAN 실패 — 립싱크 버전 유지")
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
    parser.add_argument("--lipsync-steps", default=20, type=int, dest="lipsync_steps",
                        help="diffusion inference steps (기본 20, 한국어 phoneme 정확도 우선. 영어/단순 영상엔 15도 OK)")
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
                        help="GFPGAN 후처리 활성화 (face quality 향상, 입술 sharpen, ~2분 추가)")
    parser.add_argument("--postprocess-upscale", default=2, type=int, dest="postprocess_upscale",
                        help="GFPGAN upscale (1=유지, 2=2x SR default. 5/7 v42 검증 quality)")
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
        smart_daemon=args.smart_daemon,
    )
