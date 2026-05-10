"""LLM에게 한국어 tone instruction 출력 + Korean instruction 사용.

이전 문제:
  영어 instruction (예: "in a low, reflective tone with...")
  → Fun-CosyVoice3-0.5B (한국어 fine-tune)이 영어를 그대로 합성/환각

해결:
  1. LLM이 tone을 한국어로 출력
  2. instruct_text도 한국어로 wrap
  3. Korean fine-tune 모델 + Korean instruction → 학습 분포 일치 → 안정

학습 데이터 (common.py): 중국어 instruction 사용
  "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>"
  → 같은 언어 (LLM 출력 언어와 instruction 언어 일치) 가 핵심

우리 적용:
  "You are a helpful assistant. 슬프고 차분한 톤으로, 깊은 후회와 함께 말해주세요.<|endofprompt|>"
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 1. LLM prompt: tone을 한국어로 요청
old1 = '''            f"  2. tone: full imperative phrase combining STYLE + EMOTION + SITUATION (English).\\n"
            f"        STRUCTURE: 'in a {{tone+style}}, with {{emotion}}, conveying {{situation}}'\\n"
            f"        EXAMPLE GOOD: 'in a low, deliberate tone with grim resolve, conveying a quiet warning before violence'\\n"
            f"        EXAMPLE GOOD: 'in a casual, slightly amused tone with light pride, sharing a small triumph among friends'\\n"
            f"        EXAMPLE GOOD: 'in a hushed, urgent voice with restrained fear, warning of approaching danger'\\n"
            f"        EXAMPLE BAD: 'angry' (too short, missing style/situation)\\n"
            f"        EXAMPLE BAD: 'sharp challenge, impatient edge' (no imperative format)\\n"
            f"        EXAMPLE BAD: 'happy' (no style/situation context)\\n"'''

new1 = '''            f"  2. tone: full imperative phrase in {lang_name} combining STYLE + EMOTION + SITUATION.\\n"
            f"        STRUCTURE (Korean example): '{{스타일+톤}}, {{감정}} 담아, {{상황}} 전달하며 말해주세요'\\n"
            f"        EXAMPLE GOOD: '낮고 신중한 톤으로, 단호한 결의를 담아, 폭력 전 조용한 경고를 전달하며 말해주세요'\\n"
            f"        EXAMPLE GOOD: '편안하고 살짝 즐거운 톤으로, 옅은 자부심과 함께, 친구들과 작은 승리를 나누듯 말해주세요'\\n"
            f"        EXAMPLE GOOD: '낮고 다급한 목소리로, 억눌린 두려움을 담아, 다가오는 위험을 경고하듯 말해주세요'\\n"
            f"        EXAMPLE BAD: '화남' (too short, missing style/situation)\\n"
            f"        EXAMPLE BAD: 'sharp challenge' (영어 안 됨, 한국어 사용)\\n"
            f"        IMPORTANT: tone MUST be in Korean (한국어). NOT English. CosyVoice3 Korean model expects Korean instructions.\\n"'''

if old1 in src:
    src = src.replace(old1, new1)
    print("[1] OK: LLM tone in Korean")
else:
    print("[1] NOT FOUND")

# 2. RULES도 한국어 출력 강제
old2 = '''            f"STYLE/EMOTION/SITUATION RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- tone: imperative phrase combining 3 elements:\\n"
            f"   (a) STYLE: speaker delivery style (low/loud, slow/quick, hushed/booming, casual/formal)\\n"
            f"   (b) EMOTION: emotional state (sadness, grim resolve, light amusement, restrained fear)\\n"
            f"   (c) SITUATION: what is happening (conveying warning, sharing triumph, accusation, etc)\\n"
            f"- Write 15-30 words natural English imperative starting with 'in a' or 'with'.\\n"
            f"- This becomes TTS instruction: 'Please say this sentence {{tone}}.'\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

new2 = '''            f"STYLE/EMOTION/SITUATION RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- tone: 한국어 imperative 한 문장. 3 elements 포함:\\n"
            f"   (a) STYLE: 발화 스타일 (낮은/큰, 느린/빠른, 속삭이는/우렁찬, 편안한/딱딱한)\\n"
            f"   (b) EMOTION: 감정 (슬픔, 단호함, 즐거움, 두려움 등)\\n"
            f"   (c) SITUATION: 상황 (경고를 전달하며, 승리를 나누듯, 비난하며 등)\\n"
            f"- 한국어 자연스러운 imperative, 15-30자, '~톤으로 말해주세요' 등으로 끝남.\\n"
            f"- This becomes TTS instruction: 'You are a helpful assistant. {{tone}}<|endofprompt|>'\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

if old2 in src:
    src = src.replace(old2, new2)
    print("[2] OK: rules - Korean tone")
else:
    print("[2] NOT FOUND")

# 3. CosyVoice3 wrapping: 한국어 tone 직접 사용
old3 = '''        # LLM이 'in a' / 'with' 로 시작하지 않으면 보정
        if llm_tone and not (llm_tone.lower().startswith("in a")
                              or llm_tone.lower().startswith("with")
                              or llm_tone.lower().startswith("in an")):
            llm_tone = f"with {llm_tone}"

        instruct_text = None
        if llm_tone and 5 <= len(llm_tone) <= 300:  # 15-30 단어 여유
            # 팀원 검증 format: "Please say this sentence {tone}."
            # 학습 분포 매칭: "Please say a sentence as loudly as possible."
            instruct_text = (
                f"You are a helpful assistant. "
                f"Please say this sentence {llm_tone}.<|endofprompt|>"
            )'''

new3 = '''        # 한국어 tone 직접 사용 (Fun-CosyVoice3 Korean fine-tune에 맞춤)
        # 학습 분포 매칭: "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>"
        # 우리 적용 (Korean):  "You are a helpful assistant. 슬프고 차분한 톤으로...<|endofprompt|>"

        instruct_text = None
        if llm_tone and 5 <= len(llm_tone) <= 200:
            # 한국어 tone 그대로 wrap
            instruct_text = (
                f"You are a helpful assistant. {llm_tone}<|endofprompt|>"
            )'''

if old3 in src:
    src = src.replace(old3, new3)
    print("[3] OK: Korean instruct wrap")
else:
    print("[3] NOT FOUND")

p.write_text(src)
print("[Done] Korean instruction 적용")
