"""CosyVoice3: inference_cross_lingual → inference_instruct2 변경.

원인:
  inference_cross_lingual은 instruction을 별도 처리 안 함.
  prefix를 tts_text에 넣으면 그대로 음성 합성됨 → "speaker currently", "patient edge of" 등 영어가 들림.

해결:
  inference_instruct2(tts_text, instruct_text, prompt_wav, ...) 사용.
  - tts_text: 한국어 합성 텍스트만
  - instruct_text: 감정/스타일 지시 (별도 처리)

조건:
  emotion instruction이 있을 때만 inference_instruct2 사용.
  Neutral + 빈 instruction이면 cross_lingual 유지 (자연 합성).
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''    if instruct:
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
            output.append(result["tts_speech"].squeeze().numpy())'''

new = '''    # CosyVoice3 API:
    #   instruction 있으면 inference_instruct2 (instruction 별도 처리)
    #   instruction 없으면 inference_cross_lingual (자연 합성)
    # 이전 버그: prefix를 tts_text에 넣으면 instruction이 그대로 음성으로 합성됨
    #   → "speaker currently", "patient edge of" 등 영어 leak

    ref_16k = os.path.join(tempfile.gettempdir(), "ref_16k_temp.wav")
    try:
        # 레퍼런스를 16kHz 모노로 변환 (CosyVoice3 요구사항)
        subprocess.run([
            "ffmpeg", "-y", "-i", ref_audio,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", ref_16k
        ], capture_output=True)

        output = []
        if instruct:
            # 감정 지시 있음 → inference_instruct2
            inference_iter = _cosy_model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct,
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

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: inference_instruct2 적용 완료")
else:
    print("NOT FOUND - check current state")
