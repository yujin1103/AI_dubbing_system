"""LLM 지시문에 (원문 스타일 + 감정 + 상황) 모두 포함.

이전 (v8): tone = emotion description만
새로 (v9): tone = "preserving the original speaker's natural style, with {emotion}, in {situation}"

CosyVoice3 처리:
  - prompt_wav가 화자 음색/자연 스타일 보존
  - instruct_text가 LLM 지시문 (스타일 강조 + 감정 + 상황)
  - 학습 imperative 패턴 유지

기존 emotion + context 두 필드 → 하나의 풍부한 tone phrase로 통합
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# LLM prompt: tone 설명을 더 풍부하게 (style + emotion + situation)
old1 = '''            f"You are a professional dubbing translator and emotion designer.\\n"
            f"For each numbered line, output THREE fields:\\n"
            f"  1. korean: natural spoken {lang_name} translation for dubbing\\n"
            f"  2. context: one short sentence describing the speaker's situation/intent\\n"
            f"  3. tone: full imperative phrase describing HOW to say it (English).\\n"
            f"        FORMAT: starts with 'in a' or 'with' — natural English imperative.\\n"
            f"        EXAMPLE GOOD: 'in a casual, pleased, lightly confident tone with warm but restrained excitement'\\n"
            f"        EXAMPLE GOOD: 'with subdued sadness, low pitch, slow pacing, resigned tone'\\n"
            f"        EXAMPLE GOOD: 'in a sharp, impatient tone with quick clipped delivery'\\n"
            f"        EXAMPLE BAD: 'angry' (too short)\\n"
            f"        EXAMPLE BAD: 'sharp challenge, impatient edge' (not imperative format)\\n"
            f"\\n"
            f"OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\\n"
            f"[N]\\n"
            f"korean: <translation>\\n"
            f"context: <context sentence>\\n"
            f"tone: <imperative phrase>\\n"'''

new1 = '''            f"You are a professional dubbing translator and emotion designer.\\n"
            f"For each numbered line, output TWO fields:\\n"
            f"  1. korean: natural spoken {lang_name} translation for dubbing\\n"
            f"  2. tone: full imperative phrase combining STYLE + EMOTION + SITUATION (English).\\n"
            f"        STRUCTURE: 'in a {{tone+style}}, with {{emotion}}, conveying {{situation}}'\\n"
            f"        EXAMPLE GOOD: 'in a low, deliberate tone with grim resolve, conveying a quiet warning before violence'\\n"
            f"        EXAMPLE GOOD: 'in a casual, slightly amused tone with light pride, sharing a small triumph among friends'\\n"
            f"        EXAMPLE GOOD: 'in a hushed, urgent voice with restrained fear, warning of approaching danger'\\n"
            f"        EXAMPLE BAD: 'angry' (too short, missing style/situation)\\n"
            f"        EXAMPLE BAD: 'sharp challenge, impatient edge' (no imperative format)\\n"
            f"        EXAMPLE BAD: 'happy' (no style/situation context)\\n"
            f"\\n"
            f"OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\\n"
            f"[N]\\n"
            f"korean: <translation>\\n"
            f"tone: <imperative phrase combining style+emotion+situation>\\n"'''

if old1 in src:
    src = src.replace(old1, new1)
    print("[1] OK: LLM prompt - tone 풍부화 (style+emotion+situation)")
else:
    print("[1] NOT FOUND")

# 2. EMOTION/CONTEXT RULES → STYLE/EMOTION/SITUATION RULES
old2 = '''            f"EMOTION/TONE RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- context: describe what the speaker is doing/feeling in this specific line.\\n"
            f"- tone: imperative description starting with 'in a' or 'with', describing HOW to speak.\\n"
            f"  Mention: emotion + pacing + pitch + intensity. 10-25 words natural English.\\n"
            f"  This becomes a TTS instruction like 'Please say this sentence {{tone}}.'\\n"
            f"- Write context/tone in English (CosyVoice3 trained on English imperatives).\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

new2 = '''            f"STYLE/EMOTION/SITUATION RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- tone: imperative phrase combining 3 elements:\\n"
            f"   (a) STYLE: speaker delivery style (low/loud, slow/quick, hushed/booming, casual/formal)\\n"
            f"   (b) EMOTION: emotional state (sadness, grim resolve, light amusement, restrained fear)\\n"
            f"   (c) SITUATION: what is happening (conveying warning, sharing triumph, accusation, etc)\\n"
            f"- Write 15-30 words natural English imperative starting with 'in a' or 'with'.\\n"
            f"- This becomes TTS instruction: 'Please say this sentence {{tone}}.'\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

if old2 in src:
    src = src.replace(old2, new2)
    print("[2] OK: rules - style+emotion+situation")
else:
    print("[2] NOT FOUND")

# 3. 파싱: tts_context는 그대로, tts_emotion = tone field
# (이미 v8에서 tone fallback emotion으로 처리. context는 더 이상 LLM이 출력하지 않으니 빈 값)
old3 = '''                    seg.tts_context = entry.get('context', '') or ''
                    # 'tone' 우선, 호환성을 위해 'emotion'도 fallback
                    seg.tts_emotion = entry.get('tone', '') or entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:50]}]")'''

new3 = '''                    # 'tone' 통합 필드 (style+emotion+situation 포함)
                    seg.tts_context = ''  # 더 이상 별도 사용 X (tone에 통합됨)
                    seg.tts_emotion = entry.get('tone', '') or entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:60]}]")'''

if old3 in src:
    src = src.replace(old3, new3)
    print("[3] OK: 파싱 - tone 통합")
else:
    print("[3] NOT FOUND")

# 4. CosyVoice3 wrapping는 그대로 (이미 v8에서 'Please say this sentence {tone}.' 적용)
# tone phrase 자체에 style+emotion+situation이 통합되어 있으므로 wrapping 동일
# 단지 length 한계를 늘려둠 (15-30 단어 여유)
old4 = '''        instruct_text = None
        if llm_tone and 5 <= len(llm_tone) <= 200:'''

new4 = '''        instruct_text = None
        if llm_tone and 5 <= len(llm_tone) <= 300:  # 15-30 단어 여유'''

if old4 in src:
    src = src.replace(old4, new4)
    print("[4] OK: tone length 한계 200 → 300")
else:
    print("[4] NOT FOUND")

p.write_text(src)
print("[Done] LLM 지시문 = 원문 스타일 + 감정 + 상황 통합")
