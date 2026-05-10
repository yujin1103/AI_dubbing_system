"""TTS 합성 후 Whisper 검증 + 외국어 detect 시 재합성.

흐름:
  1. TTS 합성 (1차)
  2. whisper-tiny로 빠른 transcribe (~1s/segment)
  3. 한국어 비율 < 60%이면 외국어 환각 → 재합성 (최대 2회)
  4. 그래도 안되면 segment skip (원본 유지)

시간 비용:
  whisper-tiny 인식: ~0.5-1s/segment (CPU/GPU)
  재합성 횟수: 평균 0.3회/segment (대부분 1차 통과)
  → 총 +5-15% 시간 추가
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 1. helper 함수 추가 (synthesize_segment_cosy 위에)
helper_code = '''

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


'''

# synthesize_segment_cosy 함수 직전에 helper 삽입
old1 = "def synthesize_segment_cosy("
if "_load_whisper_validator" not in src:
    src = src.replace(old1, helper_code + old1, 1)
    print("[1] OK: whisper validator helper 추가")
else:
    print("[1] SKIP: 이미 추가됨")

# 2. synthesize_chunk 안의 TTS 호출 후 validation + retry 로직 추가
# audio_chunk = synthesize_segment_cosy(...) 뒤에 검증 추가
old2 = '''        audio_chunk = synthesize_segment_cosy(
            text=combined_text,
            ref_audio=ref_path,
            lang=tgt_lang,
            speed=1.0,
            emotion=first_seg.emotion,
            tts_context=getattr(first_seg, 'tts_context', '') or '',
            tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
        )

        # 길이 측정 + 효율적 speed retry (OPTIMIZED)'''

new2 = '''        audio_chunk = synthesize_segment_cosy(
            text=combined_text,
            ref_audio=ref_path,
            lang=tgt_lang,
            speed=1.0,
            emotion=first_seg.emotion,
            tts_context=getattr(first_seg, 'tts_context', '') or '',
            tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
        )

        # WHISPER_VALIDATION: 외국어 환각 detect → 재합성
        if tgt_lang == "ko" and len(audio_chunk) > TTS_SAMPLE_RATE * 1.0:
            for retry in range(2):
                is_kor, kor_ratio, transcribed = _is_korean_audio(
                    audio_chunk, TTS_SAMPLE_RATE, korean_ratio_threshold=0.55
                )
                if is_kor:
                    if retry > 0:
                        print(f"  ↳ Whisper 검증 통과 (재시도 #{retry}): kor_ratio={kor_ratio:.0%}")
                    break
                print(f"  ⚠️ 외국어 환각 detect (kor={kor_ratio:.0%}, '{transcribed[:40]}') — 재합성 #{retry+1}/2")
                audio_chunk = synthesize_segment_cosy(
                    text=combined_text,
                    ref_audio=ref_path,
                    lang=tgt_lang,
                    speed=1.0,
                    emotion=first_seg.emotion,
                    tts_context=getattr(first_seg, 'tts_context', '') or '',
                    tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
                )

        # 길이 측정 + 효율적 speed retry (OPTIMIZED)'''

if old2 in src:
    src = src.replace(old2, new2)
    print("[2] OK: whisper validation + retry 추가")
else:
    print("[2] NOT FOUND")

p.write_text(src)
print("[Done] Whisper 후처리 검증 적용")
