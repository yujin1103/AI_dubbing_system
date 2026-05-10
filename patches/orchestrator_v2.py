"""
AI 더빙 파이프라인 — 단일 Python 오케스트레이터
=====================================================

현재 구조 (단일 Python):
  orchestrator.py — 모든 모델을 직접 로드하여 함수 호출로 파이프라인 실행

실행 방법:
    python orchestrator.py --input /workspace/media/input/test2.mp4 --lang ko --speaker 3

의존성 설치:
  pip install qwen-asr pyannote.audio funasr deep-translator
  pip install soundfile librosa numpy torch demucs
  pip install git+https://github.com/FunAudioLLM/CosyVoice.git
"""

import os
import sys
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

# ─── 환경 설정 ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR  = os.path.join(BASE_DIR, "media")
INPUT_DIR  = os.path.join(MEDIA_DIR, "input")
CHUNKS_DIR = os.path.join(MEDIA_DIR, "chunks")
VOCALS_DIR = os.path.join(MEDIA_DIR, "vocals")
BGM_DIR    = os.path.join(MEDIA_DIR, "bgm")
DUBBED_DIR = os.path.join(MEDIA_DIR, "dubbed")
OUTPUT_DIR = os.path.join(MEDIA_DIR, "output")
REF_DIR    = os.path.join(MEDIA_DIR, "reference")
REPORT_DIR = os.path.join(MEDIA_DIR, "reports")

for d in [CHUNKS_DIR, VOCALS_DIR, BGM_DIR, DUBBED_DIR, OUTPUT_DIR, REF_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# MOS 모델 경로
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
    emotion:       str = "Neutral"   # emotion2vec+ 결과
    emotion_score: float = 0.0
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


# ─── 모델 관리 (순차 로드/언로드) ─────────────────────────────
# GPU 16GB 환경에서 모든 모델을 동시에 올릴 수 없으므로
# 파이프라인 단계별로 필요한 모델만 로드하고 끝나면 해제.

_diarization_model  = None
_emotion_model      = None
_cosy_model         = None   # CosyVoice3
_google_translator  = None
_mos_evaluator      = None   # MOS 품질 평가

# ASR은 venv_asr에서 subprocess로 실행 (transformers 4.57.6 필요)
ASR_VENV_PYTHON = "/opt/venv_asr/bin/python"
ASR_WORKER_PATH = os.path.join(BASE_DIR, "asr_worker.py")


def _unload(name: str):
    """GPU 모델을 메모리에서 해제."""
    global _diarization_model, _emotion_model, _cosy_model, _mos_evaluator

    model_map = {
        "diarization": "_diarization_model",
        "emotion": "_emotion_model",
        "cosy": "_cosy_model",
        "mos": "_mos_evaluator",
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
    Demucs로 청크에서 목소리(vocals)와 배경음(BGM) 분리.

    INPUT:
      chunk_path  : str — /data/chunks/movie_chunk_000.mp4

    OUTPUT:
      vocals_path : str — /data/vocals/movie_chunk_000_vocals.wav
      bgm_path    : str — /data/bgm/movie_chunk_000_bgm.wav
    """
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"청크 파일 없음: {chunk_path}")

    chunk_name = os.path.basename(chunk_path).replace(".mp4", "")
    out_dir = os.path.join(tempfile.gettempdir(), "demucs", chunk_name)
    os.makedirs(out_dir, exist_ok=True)

    result = subprocess.run(
        ["python", "-m", "demucs", "-n", "htdemucs_ft", "--two-stems=vocals",
         "--out", out_dir, chunk_path],
        capture_output=True, text=True,
        env={**os.environ,
             "TORCHAUDIO_USE_BACKEND_DISPATCHER": "0",
             "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}
)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs 실패:\n{result.stderr}")

    demucs_out = os.path.join(out_dir, "htdemucs_ft", chunk_name)
    vocals_src = os.path.join(demucs_out, "vocals.wav")
    bgm_src    = os.path.join(demucs_out, "no_vocals.wav")

    if not os.path.exists(vocals_src):
        raise FileNotFoundError("Demucs 출력 파일 없음")

    vocals_dst = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
    bgm_dst    = os.path.join(BGM_DIR,    f"{chunk_name}_bgm.wav")

    shutil.move(vocals_src, vocals_dst)
    shutil.move(bgm_src,    bgm_dst)
    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"[Separate] vocals: {vocals_dst}")
    print(f"[Separate] bgm:    {bgm_dst}")
    return vocals_dst, bgm_dst


# ─── Step 3: 음성 인식 + 타임스탬프 ─────────────────────────

def transcribe(vocals_path: str, language: Optional[str] = None) -> Tuple[str, List[WordTiming]]:
    """
    Qwen3-ASR + ForcedAligner로 음성을 텍스트로 변환 (subprocess).
    qwen-asr는 transformers 4.57.6 필요 → /opt/venv_asr에서 실행.
    """
    lang_name = LANG_CODE_TO_NAME.get(language, language) if language else None

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

    print(f"[ASR] 감지 언어: {lang_code}, 단어 수: {len(words)}")
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
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    result = _diarization_model(vocals_path, **kwargs)

    # pyannote 4.x: DiarizeOutput 객체 → Annotation 추출
    if hasattr(result, "speaker_diarization"):
        return result.speaker_diarization
    return result


def _get_speaker_at(diarization, time: float) -> str:
    """특정 시간에 말하는 화자 반환. pyannote 없으면 SPEAKER_00."""
    if diarization is None:
        return "SPEAKER_00"
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= time <= turn.end:
            return speaker
    return "SPEAKER_UNK"


# ─── Step 5: 세그먼트 조합 ────────────────────────────────────

def build_segments(
    words: List[WordTiming],
    diarization,
    vocals_path: str,
    max_duration: float = MAX_SEG_DURATION,
    max_words: int = MAX_SEG_WORDS
) -> List[Segment]:
    """
    ASR 단어 타임스탬프 + pyannote 화자 구간을 합쳐서
    세그먼트(문장 단위) 리스트 생성.

    INPUT:
      words        : List[WordTiming] — ASR 결과
      diarization  : pyannote Annotation — 화자 분리 결과
      vocals_path  : str              — 오디오 파일 (길이 계산용)
      max_duration : float            — 세그먼트 최대 길이 (초)
      max_words    : int              — 세그먼트 최대 단어 수

    OUTPUT:
      segments : List[Segment] — speaker, start, end, text, words 채워진 상태
                                  emotion은 아직 "Neutral" (Step 6에서 채움)
    """
    if not words:
        # 단어가 없으면 전체를 하나의 세그먼트로
        duration = sf.info(vocals_path).duration
        speaker  = _get_speaker_at(diarization, duration / 2)
        return [Segment(id=0, speaker=speaker, start=0.0, end=duration, text="")]

    segments  = []
    seg_id    = 0
    seg_words = []
    seg_texts = []
    seg_start = None
    seg_end   = None

    for i, w in enumerate(words):
        if seg_start is None:
            seg_start = w.start

        seg_end = w.end
        seg_texts.append(w.word)
        seg_words.append(w)

        is_last     = (i == len(words) - 1)
        is_punct    = w.word.strip().endswith((".", "?", "!", "。", "？", "！"))
        is_too_long = (seg_end - seg_start) >= max_duration
        is_too_many = len(seg_words) >= max_words

        if is_punct or is_too_long or is_too_many or is_last:
            seg_text = " ".join(seg_texts).strip()
            seg_mid  = (seg_start + seg_end) / 2
            speaker  = _get_speaker_at(diarization, seg_mid)

            # 길이 0인 세그먼트 방지
            if seg_end <= seg_start:
                seg_end = seg_start + 0.1

            segments.append(Segment(
                id=seg_id,
                speaker=speaker,
                start=round(seg_start, 3),
                end=round(seg_end, 3),
                text=seg_text,
                words=list(seg_words)
            ))
            seg_id   += 1
            seg_words = []
            seg_texts = []
            seg_start = None

    print(f"[Segments] {len(segments)}개 세그먼트 생성")
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
                emotion   = EMOTION_LABEL_MAP.get(raw_label, "Neutral")
                return emotion, round(float(scores[best_idx]), 3)

    except Exception as e:
        print(f"[Emotion] 추출 실패: {e}")

    return "Neutral", 0.0


def fill_emotions(segments: List[Segment], vocals_path: str) -> List[Segment]:
    """
    세그먼트 리스트 전체에 감정 정보 채움.

    INPUT:
      segments    : List[Segment] — emotion이 "Neutral"인 상태
      vocals_path : str

    OUTPUT:
      segments : List[Segment] — emotion, emotion_score 채워진 상태
    """
    for seg in segments:
        seg.emotion, seg.emotion_score = extract_emotion(vocals_path, seg.start, seg.end)
        print(f"[Emotion] seg {seg.id}: {seg.emotion} ({seg.emotion_score:.2f})")
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

            sf.write(ref_path, clip, sr)
            profile.references[emotion] = ref_path

            mos_str = f", MOS={best_mos:.2f}" if best_mos > 0 else ""
            print(f"[Profile] {speaker} / {emotion}: {ref_filename} ({best_duration:.1f}s{mos_str})")

        profiles[speaker] = profile

    return profiles


# ─── Step 8: 번역 ─────────────────────────────────────────────

def translate_segments(segments: List[Segment], tgt_lang: str) -> List[Segment]:
    """세그먼트 전체 번역 — LLM은 묶어서 한 번에."""
    if _google_translator == "vectorengine":
        return _translate_segments_llm(segments, tgt_lang)

    for seg in segments:
        seg.translated = _translate_google(seg.text, tgt_lang)
        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
    return segments


def _translate_segments_llm(segments: List[Segment], tgt_lang: str) -> List[Segment]:
    """VectorEngine GPT로 청크 전체 세그먼트를 한 번에 번역."""
    import requests
    import re

    api_key = os.environ.get("VECTORENGINE_API_KEY", "")
    base_url = os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai")
    model = os.environ.get("VECTORENGINE_MODEL", "gpt-5.4-xhigh")

    lang_names = {
        "ko": "한국어", "ja": "일본어", "zh": "중국어", "fr": "프랑스어",
        "de": "독일어", "es": "스페인어", "en": "영어", "ru": "러시아어",
        "pt": "포르투갈어", "it": "이탈리아어", "ar": "아랍어", "nl": "네덜란드어",
    }
    lang_name = lang_names.get(tgt_lang, tgt_lang)

    to_translate = [(i, seg) for i, seg in enumerate(segments) if seg.text.strip()]
    if not to_translate:
        return segments

    lines = []
    for idx, (i, seg) in enumerate(to_translate):
        lines.append(f"[{idx}] {seg.text}")
    batch_text = "\n".join(lines)

    system_prompt = (
        f"You are a professional translator. Translate each numbered line into natural spoken {lang_name} for dubbing. "
        f"CRITICAL RULES:\n"
        f"- Keep the same numbering format [0], [1], [2]...\n"
        f"- Each line MUST be translated SEPARATELY with similar length to its original\n"
        f"- You MUST output in {lang_name}. DO NOT output the original text.\n"
        f"- Do NOT merge content between lines\n"
        f"- Make consecutive lines from the same speaker sound natural when spoken in order\n"
        f"- Output ONLY translations. No thinking, no explanation."
    )

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": batch_text},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=180,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()

        if "<think>" in result:
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

        translations = {}
        for line in result.split("\n"):
            line = line.strip()
            match = re.match(r'\[(\d+)\]\s*(.*)', line)
            if match:
                translations[int(match.group(1))] = match.group(2).strip()

        for idx, (i, seg) in enumerate(to_translate):
            if idx in translations and translations[idx]:
                seg.translated = translations[idx]
                print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
            else:
                seg.translated = _translate_google(seg.text, tgt_lang)
                print(f"[Translate] (폴백) {seg.text[:30]}... → {seg.translated[:30]}...")

        return segments

    except Exception as e:
        print(f"[Translate] LLM 배치 실패: {e} → Deep Translator 폴백")
        for seg in segments:
            if seg.text.strip():
                seg.translated = _translate_google(seg.text, tgt_lang)
                print(f"[Translate] (폴백) {seg.text[:30]}... → {seg.translated[:30]}...")
        return segments


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


# ─── Step 9: 음성 합성 ────────────────────────────────────────

def synthesize_segment_cosy(
    text: str,
    ref_audio: str,
    lang: str,
    speed: float = 1.0,
    emotion: str = "Neutral"
) -> np.ndarray:
    """
    CosyVoice3로 세그먼트 1개 합성.
    감정 지시(instruction)를 텍스트 프리픽스로 포함.
    """
    if _cosy_model is None:
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    if not ref_audio or not os.path.exists(ref_audio):
        print(f"[TTS] 레퍼런스 파일 없음: {ref_audio}")
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    # 감정 지시 텍스트
    emotion_instruction = {
        "Angry": "Speak with anger and intensity.",
        "Sad": "Speak with sadness and sorrow.",
        "Happy": "Speak with joy and excitement.",
        "Surprised": "Speak with surprise and wonder.",
        "Scared": "Speak with fear and anxiety.",
        "Neutral": "",
    }
    instruct = emotion_instruction.get(emotion, "")
    if instruct:
        prefix = f'{instruct}<|endofprompt|>'
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


def synthesize_chunk(
    segments: List[Segment],
    profiles: Dict[str, SpeakerProfile],
    chunk_name: str,
    tgt_lang: str,
) -> str:
    """
    청크 전체 더빙 오디오 생성.
    MOS 평가로 품질이 낮은 세그먼트는 자동 재합성 (최대 3회).
    """
    if not segments:
        raise ValueError("segments가 비어 있습니다")

    total_duration = max(seg.end for seg in segments)
    total_samples  = int(total_duration * TTS_SAMPLE_RATE) + TTS_SAMPLE_RATE
    output_audio   = np.zeros(total_samples, dtype=np.float32)

    for seg in segments:
        if not seg.translated.strip():
            continue

        profile  = profiles.get(seg.speaker)
        ref_path = profile.get_ref(seg.emotion) if profile else ""
        if not ref_path or not os.path.exists(ref_path):
            ref_path = profile.get_ref("Neutral") if profile else ""
        if not ref_path or not os.path.exists(ref_path):
            print(f"[TTS] [{seg.speaker}] 레퍼런스 없음 → 스킵")
            continue

        print(f"[TTS] [{seg.speaker}][{seg.emotion}] {seg.start:.1f}s → '{seg.translated[:40]}'")
        print(f"[TTS] ref: {os.path.basename(ref_path)}")

        # MOS 기반 재합성 루프
        best_audio = None
        best_mos = -1
        max_retries = 3 if _mos_evaluator is not None else 1
        retry_speeds = [seg.speed, seg.speed * 0.9, seg.speed * 1.1]

        for attempt in range(max_retries):
            speed = retry_speeds[attempt] if attempt < len(retry_speeds) else seg.speed

            audio_chunk = synthesize_segment_cosy(
                text=seg.translated,
                ref_audio=ref_path,
                lang=tgt_lang,
                speed=speed,
                emotion=seg.emotion
            )

            if _mos_evaluator is not None and np.max(np.abs(audio_chunk)) > 0.01:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_chunk, TTS_SAMPLE_RATE)
                    mos_score = _mos_evaluator.evaluate(tmp.name)
                    os.unlink(tmp.name)

                print(f"[MOS] attempt {attempt+1}: {mos_score:.2f} (speed={speed:.2f})")

                if mos_score > best_mos:
                    best_mos = mos_score
                    best_audio = audio_chunk.copy()

                if mos_score >= 3.5:
                    break  # 충분히 좋음
            else:
                best_audio = audio_chunk
                best_mos = 0.0
                break

        if best_audio is None:
            best_audio = np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

        # 세그먼트에 MOS 점수 저장 (JSON용)
        seg._tts_mos = best_mos
        seg._tts_retries = min(max_retries, max(0, max_retries - 1)) if max_retries > 1 else 0

        # 원본 duration 슬롯에 맞게 배치
        target_samples = int((seg.end - seg.start) * TTS_SAMPLE_RATE)
        if target_samples <= 0:
            continue
        
        # 너무 길면 리샘플링으로 압축 (잘리지 않게)
        if len(best_audio) > target_samples * 1.1:
            best_audio = librosa.resample(
                best_audio, 
                orig_sr=TTS_SAMPLE_RATE, 
                target_sr=int(TTS_SAMPLE_RATE * len(best_audio) / target_samples)
            )[:target_samples]
        elif len(best_audio) > target_samples:
            best_audio = best_audio[:target_samples]
        else:
            best_audio = np.pad(best_audio, (0, target_samples - len(best_audio)))

        start_sample = int(seg.start * TTS_SAMPLE_RATE)
        end_sample   = min(start_sample + len(best_audio), total_samples)
        copy_len     = end_sample - start_sample
        output_audio[start_sample:end_sample] = best_audio[:copy_len]

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
    dubbed_volume: float = 1.0,
    bgm_volume: float = 0.5
) -> str:
    """
    FFmpeg로 더빙 오디오 + BGM 믹싱 후 원본 영상 트랙에 붙이기.

    INPUT:
      chunk_path    : str   — 원본 영상 청크 (비디오 트랙 사용)
      dubbed_path   : str   — 더빙 오디오
      bgm_path      : str   — 배경음
      output_path   : str   — 출력 파일
      dubbed_volume : float — 더빙 볼륨 (기본 5.0)
      bgm_volume    : float — BGM 볼륨 (기본 0.15)

    OUTPUT:
      output_path : str — /data/chunks/movie_chunk_000_final.mp4
    """
    filter_complex = (
        f"[1:a]aformat=channel_layouts=mono,volume={dubbed_volume}[dub];"
        f"[2:a]aformat=channel_layouts=mono,volume={bgm_volume}[bgm];"
        "[dub][bgm]amix=inputs=2:duration=first:normalize=0[a]"
    )
    cmd = [
        "ffmpeg",
        "-i", chunk_path,
        "-i", dubbed_path,
        "-i", bgm_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        output_path, "-y"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg mix 실패:\n{result.stderr}")

    print(f"[Mix] 완료: {output_path}")
    return output_path


def concat_chunks(file_name: str, tgt_lang: str) -> str:
    """
    모든 _final.mp4 청크를 이어붙여 최종 영상 생성.

    INPUT:
      file_name : str — "movie"
      tgt_lang  : str — "ko"

    OUTPUT:
      output_path : str — /data/output/movie_ko_dubbed.mp4
    """
    finals = sorted([
        f for f in os.listdir(CHUNKS_DIR) if "_final.mp4" in f
    ])
    if not finals:
        raise FileNotFoundError("final 청크가 없습니다")

    concat_list_path = os.path.join(OUTPUT_DIR, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for chunk in finals:
            f.write(f"file '{os.path.join(CHUNKS_DIR, chunk)}'\n")

    output_path = os.path.join(OUTPUT_DIR, f"{file_name}_{tgt_lang}_dubbed.mp4")
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

    report_path = os.path.join(REPORT_DIR, f"{file_name}_{tgt_lang}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Report] 저장: {report_path}")
    return report_path


def run_pipeline(
    video_path: str,
    file_name: str,
    tgt_lang: str = "ko",
    src_lang: Optional[str] = None,
    segment_time: int = 300,
    num_speakers: int = None  
):
    """
    전체 더빙 파이프라인 실행.
    GPU 16GB 환경을 위해 모델을 단계별로 로드/언로드.
    """
    print(f"\n{'='*50}")
    print(f"[Pipeline] 시작: {video_path} → {tgt_lang}")
    print(f"[Pipeline] device: {DEVICE}")
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
        chunk_data[chunk_name] = {
            "chunk_path": chunk_path,
            "vocals_path": vocals_path,
            "bgm_path": bgm_path,
        }

    # GPU 정리 후 ASR
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # 🔥 추가: Demucs가 쓴 VRAM이 완벽히 반환되도록 강제 동기화
    import time; time.sleep(2)

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

        # Step 5: 세그먼트 조합
        segments = build_segments(
            data["words"], diarization, data["vocals_path"]
        )
        data["diarization"] = diarization
        data["segments"] = segments
    _unload("diarization")

    # ── 3단계: 감정 추출 + MOS 레퍼런스 선택 ──────────────
    load_emotion()
    load_mos()  # MOS 모델 로드 (레퍼런스 평가용)
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [Emotion] 청크: {chunk_name} ---")

        # Step 6: 감정 추출
        data["segments"] = fill_emotions(
            data["segments"], data["vocals_path"]
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
        data["segments"] = translate_segments(data["segments"], tgt_lang)

        # 속도 자동 조절 (번역 길이 비율 기반)
        for seg in data["segments"]:
            if seg.text and seg.translated:
                # 음절 기반 비율 (한국어 1글자 ≈ 영어 2~3글자 발화 시간)
                src_len = len(seg.text.split())       # 영어: 단어 수
                tgt_len = len(seg.translated)          # 한국어: 글자 수
                estimated_ratio = (tgt_len * 0.4) / max(src_len, 1)
                seg.speed = max(0.9, min(1.3, estimated_ratio))

    # ── 5단계: TTS + MOS 평가 + 믹싱 ─────────────────────
    load_cosy()
    load_mos()  # MOS 다시 로드 (TTS 출력 평가용)
    for chunk_name, data in chunk_data.items():
        print(f"\n--- [TTS] 청크: {chunk_name} ---")

        # Step 9: 음성 합성 (MOS 기반 자동 재합성)
        dubbed_path = synthesize_chunk(
            data["segments"], data["profiles"], chunk_name, tgt_lang
        )

        # Step 10: 믹싱
        final_path = os.path.join(CHUNKS_DIR, f"{chunk_name}_final.mp4")
        mix_audio(data["chunk_path"], dubbed_path, data["bgm_path"], final_path)
    _unload("cosy")
    _unload("mos")

    # 전체 청크 합치기
    output_path = concat_chunks(file_name, tgt_lang)

    # JSON 리포트 저장
    report_path = save_pipeline_report(file_name, tgt_lang, chunk_data, output_path)

    print(f"\n{'='*50}")
    print(f"[Pipeline] 완료!")
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
    args = parser.parse_args()

    file_name = args.name or os.path.basename(args.input).rsplit(".", 1)[0]

    # 파이프라인 실행 (모델은 단계별로 자동 로드/언로드)
    run_pipeline(
        video_path=args.input,
        file_name=file_name,
        tgt_lang=args.lang,
        src_lang=args.src,
        segment_time=args.segment,
        num_speakers=args.speakers
    )
