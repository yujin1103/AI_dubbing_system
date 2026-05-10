"""CosyVoice3 instruct2를 학습 분포에 맞는 정확한 format으로 사용.

이전 실패 원인:
  영어 long-form context (LLM 출력 "The speaker offers alternative...") 를 instruction으로 사용
  → CosyVoice3가 OOD 입력을 받아 그대로 합성 또는 환각

해결:
  공식 training data (common.py instruct_list) 의 format 그대로 사용:
    "You are a helpful assistant. {중국어 짧은 감정 지시}<|endofprompt|>"

  emotion 카테고리 → 중국어 instruction 매핑:
    - Sad      → 请非常伤心地说一句话
    - Angry    → 请非常生气地说一句话
    - Happy    → 请非常开心地说一句话
    - Surprised → 请用非常惊讶、激动的语气说话
    - Scared   → 请用非常紧张、害怕的语气说话
    - Neutral  → instruction 안 씀 (cross_lingual)

  LLM tts_emotion (영어 long form) 무시. 카테고리만 사용.
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 현재 코드 (cross_lingual 통일됨) 을 instruct2 + 올바른 format 으로 변경
old = '''        output = []
        # CosyVoice3 inference_instruct2 사용 폐기 (영어 instruction 누출 + 외국어 환각 발생).
        # cross_lingual 통일 — 감정은 emotion별 reference voice (SPEAKER_X_Sad.wav 등)에서 자연 전이.
        inference_iter = _cosy_model.inference_cross_lingual(
            tts_text=text,
            prompt_wav=ref_16k,
            stream=False,
            speed=speed,
        )

        for result in inference_iter:
            output.append(result["tts_speech"].squeeze().numpy())'''

new = '''        output = []
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

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: 학습 format 그대로 instruct2 적용")
else:
    print("NOT FOUND")
