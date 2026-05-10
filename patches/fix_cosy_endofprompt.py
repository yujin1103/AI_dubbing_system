"""inference_instruct2 호출 시 instruct_text에 <|endofprompt|> 토큰 추가.

이유:
  CosyVoice3 LLM의 assertion 'assert 151646 in text' (151646 = <|endofprompt|> token)
  → instruct_text가 prompt_text로 사용되며 이 토큰을 반드시 포함해야 함.

format (common.py 참조):
  "{instruction}<|endofprompt|>" 또는
  "You are a helpful assistant. {instruction}<|endofprompt|>"
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''        if instruct:
            # 감정 지시 있음 → inference_instruct2
            inference_iter = _cosy_model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )'''

new = '''        if instruct:
            # 감정 지시 있음 → inference_instruct2
            # CosyVoice3 LLM은 instruct_text에 <|endofprompt|> 토큰 (id 151646) 필수
            instruct_with_eop = f"{instruct}<|endofprompt|>"
            inference_iter = _cosy_model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct_with_eop,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: <|endofprompt|> 토큰 추가")
else:
    print("NOT FOUND")
