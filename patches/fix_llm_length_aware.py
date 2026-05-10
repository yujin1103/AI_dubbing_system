"""LLM 번역에 segment 길이 정보 전달 → 한국어 발화 시간 자동 조절."""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# 1) input format: duration 추가
old_input = '''    # 화자/감정 컨텍스트 포함 (감정 풍부 번역)
    lines = []
    for idx, (i, seg) in enumerate(to_translate):
        ctx = f"[{idx}|{seg.speaker}|{seg.emotion}] {seg.text}"
        lines.append(ctx)
    batch_text = "\\n".join(lines)'''

new_input = '''    # 화자/감정/길이 컨텍스트 포함 (감정 풍부 + 시간 매칭 번역)
    lines = []
    for idx, (i, seg) in enumerate(to_translate):
        duration = seg.end - seg.start
        ctx = f"[{idx}|{seg.speaker}|{seg.emotion}|{duration:.1f}s] {seg.text}"
        lines.append(ctx)
    batch_text = "\\n".join(lines)'''

if old_input in src:
    src = src.replace(old_input, new_input)
    print('[1] input format에 DURATION 추가')
else:
    print('[1] not found')

# 2) system_prompt: 길이 규칙 추가
old_sp = '''    system_prompt = (
        f"You are a professional dubbing translator for film and TV. "
        f"This is a TRANSLATION task for an existing film. "
        f"The original may include violence, weapons, injury, strong emotion - translate faithfully without censorship. "
        f"You MUST translate ALL lines as-is - do not refuse, do not soften, do not add warnings.\\n"
        f"Translate each numbered line into natural spoken {lang_name} for voice acting.\\n"
        f"INPUT FORMAT: [index|SPEAKER|EMOTION] text\\n"
        f"OUTPUT FORMAT: [index] translated_text  (only index in brackets, no speaker/emotion)\\n"
        f"\\n"
        f"CRITICAL RULES:\\n"
        f"- Output MUST be in {lang_name} only. NEVER output the original text.\\n"
        f"- Each line translated SEPARATELY, similar length to original.\\n"
        f"- Use the EMOTION tag to choose tone:\\n"
        f"    Angry → 강한 어조, 짧고 단호한 표현\\n"
        f"    Sad → 부드럽고 가라앉은 표현, 한숨/말줄임\\n"
        f"    Happy → 밝은 감탄사, 활기찬 어미\\n"
        f"    Surprised → 놀람 감탄사 (어머/세상에/우와)\\n"
        f"    Scared → 떨리는 어투, 짧은 호흡\\n"
        f"    Neutral → 자연스러운 일상 구어체\\n"
        f"- Keep the same SPEAKER consistent in tone/style across their lines.\\n"
        f"- Use natural Korean speech: contractions, particles, sentence endings (~요, ~다, ~네).\\n"
        f"- Add interjections/fillers when appropriate (어/음/근데/아).\\n"
        f"- NO XML tags, NO markdown, NO special tokens like <|...|>.\\n"
        f"- Output ONLY translations. No thinking, no explanation, no notes.\\n"
    )'''

# 한국어 평균 발화 속도 5.5 chars/sec (CosyVoice3 기준)
new_sp = '''    system_prompt = (
        f"You are a professional dubbing translator for film and TV. "
        f"This is a TRANSLATION task for an existing film. "
        f"The original may include violence, weapons, injury, strong emotion - translate faithfully without censorship. "
        f"You MUST translate ALL lines as-is - do not refuse, do not soften, do not add warnings.\\n"
        f"Translate each numbered line into natural spoken {lang_name} for voice acting.\\n"
        f"\\n"
        f"INPUT FORMAT: [index|SPEAKER|EMOTION|DURATION] text\\n"
        f"  - DURATION = original speech length in seconds\\n"
        f"OUTPUT FORMAT: [index] translated_text  (only index in brackets)\\n"
        f"\\n"
        f"=== CRITICAL LENGTH RULES (가장 중요) ===\\n"
        f"- Each translated line MUST FIT within DURATION when spoken in {lang_name}.\\n"
        f"- {lang_name} speech rate: ~5.5 characters per second (Korean reference).\\n"
        f"- TARGET character count = DURATION * 5.5 (±15% tolerance).\\n"
        f"  Example: DURATION=10s → ~55 Korean characters.\\n"
        f"- If DURATION is SHORT → use concise Korean (skip fillers, drop redundancy).\\n"
        f"- If DURATION is LONG → naturally extend (감탄사/형용사/구어체 추가) without padding.\\n"
        f"- DO NOT pad with meaningless words. Each character must serve meaning.\\n"
        f"- DO NOT compress meaning if it sounds rushed; sacrifice nuance instead.\\n"
        f"\\n"
        f"=== CONTENT RULES ===\\n"
        f"- Output MUST be in {lang_name} only. NEVER output the original text.\\n"
        f"- Use the EMOTION tag to choose tone:\\n"
        f"    Angry → 강한 어조, 짧고 단호한 표현\\n"
        f"    Sad → 부드럽고 가라앉은 표현, 한숨/말줄임\\n"
        f"    Happy → 밝은 감탄사, 활기찬 어미\\n"
        f"    Surprised → 놀람 감탄사 (어머/세상에/우와)\\n"
        f"    Scared → 떨리는 어투, 짧은 호흡\\n"
        f"    Neutral → 자연스러운 일상 구어체\\n"
        f"- Keep the same SPEAKER consistent in tone/style across their lines.\\n"
        f"- Use natural Korean speech: contractions, particles, sentence endings (~요, ~다, ~네).\\n"
        f"- Add interjections/fillers (어/음/근데/아) ONLY if it helps timing.\\n"
        f"- NO XML tags, NO markdown, NO special tokens like <|...|>.\\n"
        f"- Output ONLY translations. No thinking, no explanation, no notes.\\n"
    )'''

if old_sp in src:
    src = src.replace(old_sp, new_sp)
    print('[2] system_prompt에 LENGTH RULES 추가')
else:
    print('[2] not found')

# 3) single-call retry (CONTENT_FILTER 우회)에도 duration 추가
old_single = '''            single_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"[{idx}|{seg.speaker}|{seg.emotion}] {seg.text}"},
            ]'''

new_single = '''            duration = seg.end - seg.start
            single_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"[{idx}|{seg.speaker}|{seg.emotion}|{duration:.1f}s] {seg.text}"},
            ]'''

if old_single in src:
    src = src.replace(old_single, new_single)
    print('[3] single-call retry에도 duration 추가')
else:
    print('[3] not found')

p.write_text(src, encoding='utf-8')
print('[Done] LLM length-aware 번역 적용')
