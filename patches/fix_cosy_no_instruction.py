"""CosyVoice3 instruction 사용 폐기 — cross_lingual로 통일.

이유:
  inference_instruct2 + endofprompt 적용해도 instruction(영어/긴 문장)이 음성으로
  합성되거나 외국어(中/日/泰/葡 등) 환각 발생.

해결:
  inference_cross_lingual (instruction 없음) 통일.
  감정 표현은 emotion별 reference voice에서 자연 전이:
    - SPEAKER_X_Sad.wav → 슬픈 톤 한국어
    - SPEAKER_X_Angry.wav → 화난 톤 한국어
    - profile.get_ref(emotion) 이미 emotion 매칭

영어 leak 완전 제거 + 환각 방지.
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''        output = []
        if instruct:
            # 감정 지시 있음 → inference_instruct2
            # CosyVoice3 LLM은 instruct_text에 <|endofprompt|> 토큰 (id 151646) 필수
            instruct_with_eop = f"{instruct}<|endofprompt|>"
            inference_iter = _cosy_model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct_with_eop,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )
        else:
            # Neutral + 무 instruction → cross_lingual (자연 합성)
            inference_iter = _cosy_model.inference_cross_lingual(
                tts_text=text,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )

        for result in inference_iter:
            output.append(result["tts_speech"].squeeze().numpy())'''

new = '''        output = []
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

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: cross_lingual 통일 완료")
else:
    print("NOT FOUND")
