"""LLM parser에 'tone' 필드 인식 추가.

기존: (korean|context|emotion) 만 인식
새로: (korean|context|emotion|tone) 인식

bug 시나리오 (v9에서 발생):
  LLM 출력:
    [0]
    korean: 해봐
    tone: in a clipped, forceful tone...

  parser:
    'tone' 필드 인식 못함 → 'korean'의 multiline 으로 추가
    korean = "해봐 tone: in a clipped, forceful tone..."

  TTS: 영어 합쳐서 합성 → leak
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''    field_pat = _re.compile(r'^\\s*(korean|context|emotion)\\s*[:：]\\s*(.*)$', _re.IGNORECASE)'''

new = '''    field_pat = _re.compile(r'^\\s*(korean|context|emotion|tone)\\s*[:：]\\s*(.*)$', _re.IGNORECASE)'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: parser에 'tone' 필드 추가")
else:
    print("NOT FOUND")
