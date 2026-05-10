"""non-dialogue segment filter 추가.

목적:
  - 한숨/탄식/짧은 의성어를 ASR이 잘못 텍스트화하는 경우 자동 필터
  - 빈 텍스트는 이미 필터됨, 의성어 단어들도 추가 필터
  - non-dialogue segment는 translation/TTS skip → 원본 audio 유지

추가 위치:
  translation 시작 부분 (line 1721 근처)
  to_translate 필터 강화
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 기존 필터 강화
old = '''    to_translate = [(i, seg) for i, seg in enumerate(segments) if seg.text.strip()]
    if not to_translate:
        return segments'''

new = '''    # DIALOGUE_FILTER: 한숨/탄식/단순 의성어는 dubbing 안 함 (원본 유지)
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
        return segments'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: dialogue filter 추가 완료")
else:
    print("NOT FOUND - check current state")
