"""LLM 풍부한 감정 묘사 + CosyVoice3 학습 format wrapping.

전략:
  - LLM `tts_emotion` (예: "sharp challenge, impatient edge") 활용
  - 학습 format에 맞춰 wrapping: "You are a helpful assistant. Please say it with {emotion}.<|endofprompt|>"
  - 너무 길면 (>100 chars) 카테고리로 fallback (안정성 보장)
  - tts_context는 사용 안 함 (OOD long-form text → 환각 위험)

학습 분포 매칭:
  CosyVoice3 training data에 있던 영어 instruction:
    "Please say a sentence as loudly as possible."
    "Please say a sentence in a very soft voice."
  우리 wrapping:
    "Please say it with sharp challenge, impatient edge emotion."
  → 같은 imperative 패턴, 모델이 익숙

LLM 묘사 품질 보존 + 환각 방지.
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''        output = []
        # CosyVoice3 instruct2 — 학습 분포에 맞는 format 사용 (common.py instruct_list 참조)
        # emotion 카테고리만 사용. LLM tts_emotion (영어 long-form) 무시 (OOD → 환각).
        EMOTION_TO_INSTRUCT = {
            "Sad":       "请非常伤心地说一句话。",
            "Angry":     "请非常生气地说一句话。",
            "Happy":     "请非常开心地说一句话。",
            "Surprised": "请用非常惊讶、激动的语气说话。",
            "Scared":    "请用非常紧张、害怕、颤抖的语气说话。",
        }
        zh_instruct = EMOTION_TO_INSTRUCT.get(emotion, None)

        if zh_instruct:
            # instruct2 (학습 format 그대로)
            instruct_text = f"You are a helpful assistant. {zh_instruct}<|endofprompt|>"
            inference_iter = _cosy_model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct_text,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )
        else:
            # Neutral → cross_lingual (자연 합성, reference voice 톤)
            inference_iter = _cosy_model.inference_cross_lingual(
                tts_text=text,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )

        for result in inference_iter:
            output.append(result["tts_speech"].squeeze().numpy())'''

new = '''        output = []
        # === CosyVoice3 instruct2 - 학습 format에 LLM 풍부 묘사 wrapping ===
        # LLM tts_emotion (예: "sharp challenge, impatient edge") → wrap in training imperative
        # OOD/긴 묘사는 카테고리 fallback로 안전 처리
        EMOTION_FALLBACK_ZH = {
            "Sad":       "请非常伤心地说一句话",
            "Angry":     "请非常生气地说一句话",
            "Happy":     "请非常开心地说一句话",
            "Surprised": "请用非常惊讶、激动的语气说话",
            "Scared":    "请用非常紧张、害怕、颤抖的语气说话",
        }

        llm_emotion_desc = (tts_emotion or "").strip().rstrip(".,;")
        # 콤마/따옴표 등 정제
        llm_emotion_desc = llm_emotion_desc.replace('"', '').replace("'", "")

        instruct_text = None
        if llm_emotion_desc and len(llm_emotion_desc) <= 80 and len(llm_emotion_desc) >= 3:
            # LLM 풍부 묘사 사용 (학습 imperative 포맷으로 wrap)
            # 학습 데이터: "Please say a sentence as loudly as possible."
            # 우리: "Please say it with {emotion} emotion."
            instruct_text = (
                f"You are a helpful assistant. "
                f"Please say it with {llm_emotion_desc} emotion.<|endofprompt|>"
            )
        elif emotion in EMOTION_FALLBACK_ZH:
            # LLM 묘사 없거나 너무 길면 → 카테고리 중국어 fallback (학습 분포 매칭)
            instruct_text = (
                f"You are a helpful assistant. {EMOTION_FALLBACK_ZH[emotion]}<|endofprompt|>"
            )

        if instruct_text:
            inference_iter = _cosy_model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct_text,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )
        else:
            # Neutral + LLM emotion 없음 → cross_lingual (자연 합성)
            inference_iter = _cosy_model.inference_cross_lingual(
                tts_text=text,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )

        for result in inference_iter:
            output.append(result["tts_speech"].squeeze().numpy())'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: LLM 풍부 묘사 wrapping 적용")
else:
    print("NOT FOUND")
