"""drama에서도 instruction 폐기 - cross_lingual + emotion별 reference voice.

근거:
  4/28 강연 report 검증: tts_context/tts_emotion 모두 ""
  → cross_lingual + Neutral reference로 합성
  → leak 0%, MOS 4.0+

drama에서 leak 발생한 이유:
  passthrough 정책 → LLM이 emotion description 생성
  → instruction으로 CosyVoice3에 전달
  → CosyVoice3가 instruction 텍스트를 그대로 합성/환각

해결:
  drama도 lecture와 같이 instruction 없이 cross_lingual.
  emotion 표현은 emotion별 reference voice (이미 시스템에 있음)에 위임.
  - Sad reference (SPEAKER_X_Sad.wav) → Sad 톤 자동 전이
  - Angry reference (SPEAKER_X_Angry.wav) → Angry 톤 자동 전이

LLM은 여전히 번역(translation) 만 하고, emotion/tone instruction 안 만듦.
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# CosyVoice3 호출 부분: instruction 모두 폐기, cross_lingual로 통일
old = '''        output = []
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

        llm_tone = (tts_emotion or "").strip().rstrip(".,;")
        # 따옴표/대괄호 등 정제
        llm_tone = llm_tone.replace('"', '').replace("'", "").replace("[", "").replace("]", "")
        # LLM이 'in a' / 'with' 로 시작하지 않으면 보정
        # 한국어 tone 직접 사용 (Fun-CosyVoice3 Korean fine-tune에 맞춤)
        # 학습 분포 매칭: "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>"
        # 우리 적용 (Korean):  "You are a helpful assistant. 슬프고 차분한 톤으로...<|endofprompt|>"

        instruct_text = None
        if llm_tone and 5 <= len(llm_tone) <= 200:
            # 한국어 tone 그대로 wrap
            instruct_text = (
                f"You are a helpful assistant. {llm_tone}<|endofprompt|>"
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

new = '''        output = []
        # === CosyVoice3 cross_lingual unified — instruction 폐기 ===
        # 이전 시도들 (v7~v11): inference_instruct2 + 다양한 instruction format → 모두 leak 18-24/60s
        # 4/28 강연 report 검증: instruction 없이 (tts_context="", tts_emotion="")
        #   → cross_lingual로 합성 → MOS 4.0+, leak 0%
        # 결론: instruction이 leak 원인. 모델은 cross_lingual로 충분히 안정.
        # 감정 표현 → emotion별 reference voice (SPEAKER_X_Sad/Angry/Happy.wav)에서 자동 전이.

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
    print("OK: drama도 cross_lingual 통일 (instruction 폐기)")
else:
    print("NOT FOUND - check current state")
