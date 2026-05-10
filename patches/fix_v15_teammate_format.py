"""v15: cross_lingual 유지 + 팀원 imperative format prefix.

원리:
  - 4월 28일 코드: inference_cross_lingual + prefix in tts_text 그대로
  - prefix만 팀원 검증 format으로 변경:
    "You are a helpful assistant. Please say this sentence {tone}.<|endofprompt|>"

LLM 출력:
  - korean: 한국어 번역
  - tone: 풀 imperative phrase (style+emotion+situation)

prefix 빌드:
  - tone 있으면: "You are a helpful assistant. Please say this sentence {tone}.<|endofprompt|>"
  - 없으면: "You are a helpful assistant.<|endofprompt|>" (4월 28일 default)
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# ============================================================
# 1. LLM prompt: emotion → tone (풀 imperative)
# ============================================================
old1 = '''            f"  2. context: one short sentence describing the speaker's situation/intent\\n"
            f"  3. emotion: 2~5 evocative words capturing the speaking tone\\n"'''

new1 = '''            f"  2. tone: full English imperative phrase combining STYLE + EMOTION + SITUATION.\\n"
            f"        FORMAT: starts with 'in a' or 'with', natural English imperative.\\n"
            f"        EXAMPLES:\\n"
            f"          'in a casual, pleased, lightly confident tone with warm but restrained excitement'\\n"
            f"          'in a low, deliberate tone with grim resolve, conveying a quiet warning'\\n"
            f"          'with subdued sadness, soft pacing, conveying lingering regret'\\n"
            f"          'in a sharp, impatient tone with controlled frustration'\\n"
            f"        BAD: 'angry' (too short), '낮은 톤' (Korean — must be English).\\n"'''

if old1 in src:
    src = src.replace(old1, new1)
    print("[1a] OK: LLM prompt - tone field")
else:
    print("[1a] NOT FOUND")

# OUTPUT FORMAT 변경
old2 = '''OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\\n"
            f"[N]\\n"
            f"korean: <translation>\\n"
            f"context: <context sentence>\\n"
            f"emotion: <emotion words>\\n"'''

new2 = '''OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\\n"
            f"[N]\\n"
            f"korean: <translation>\\n"
            f"tone: <imperative phrase>\\n"'''

if old2 in src:
    src = src.replace(old2, new2)
    print("[1b] OK: OUTPUT FORMAT")
else:
    print("[1b] NOT FOUND")

# RULES 변경
old3 = '''            f"EMOTION/CONTEXT RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- context: describe what the speaker is doing/feeling in this specific line.\\n"
            f"- emotion: 2~5 short evocative words. AVOID single-word labels like 'happy/sad/angry'.\\n"
            f"- Write context/emotion in English (CosyVoice3 understands English instructions best).\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

new3 = '''            f"TONE RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- tone: English imperative phrase (15-30 words) starting with 'in a' or 'with'.\\n"
            f"   Combine: STYLE (low/loud/slow/quick) + EMOTION (sad/angry/happy/calm) + SITUATION.\\n"
            f"- This becomes TTS prefix: 'You are a helpful assistant. Please say this sentence {{tone}}.<|endofprompt|>'\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

if old3 in src:
    src = src.replace(old3, new3)
    print("[1c] OK: RULES")
else:
    print("[1c] NOT FOUND")

# ============================================================
# 2. Parser: 'tone' 필드 인식 추가
# ============================================================
old_parser = '''    field_pat = _re.compile(r'^\\s*(korean|context|emotion)\\s*[:：]\\s*(.*)$', _re.IGNORECASE)'''
new_parser = '''    field_pat = _re.compile(r'^\\s*(korean|context|emotion|tone)\\s*[:：]\\s*(.*)$', _re.IGNORECASE)'''

if old_parser in src:
    src = src.replace(old_parser, new_parser)
    print("[2] OK: parser 'tone' 추가")
else:
    print("[2] NOT FOUND")

# ============================================================
# 3. 파싱 코드: tone field 받기
# ============================================================
old_parse = '''                    seg.tts_context = entry.get('context', '') or ''
                    seg.tts_emotion = entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:40]}]")'''

new_parse = '''                    seg.tts_context = entry.get('context', '') or ''
                    # 'tone' 우선 (v15 새 방식), 없으면 'emotion' fallback
                    seg.tts_emotion = entry.get('tone', '') or entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:60]}]")'''

if old_parse in src:
    src = src.replace(old_parse, new_parse)
    print("[3] OK: 파싱 - tone field")
else:
    print("[3] NOT FOUND")

# ============================================================
# 4. synthesize_segment_cosy: prefix를 팀원 format으로
# ============================================================
old_prefix = '''    # 1순위: LLM 자연어 묘사
    if tts_context or tts_emotion:
        parts = []
        if tts_context:
            parts.append(f"Context: {tts_context}")
        if tts_emotion:
            parts.append(f"Emotion: {tts_emotion}")
        instruct = "\\n".join(parts)
    else:
        # 2순위: 카테고리 폴백
        instruct = emotion_instruction.get(emotion, "")

    if instruct:
        prefix = f'{instruct}<|endofprompt|>'
    else:
        prefix = 'You are a helpful assistant.<|endofprompt|>'

    ref_16k = os.path.join(tempfile.gettempdir(), "ref_16k_temp.wav")'''

new_prefix = '''    # === 팀원 검증 format prefix ===
    # 학습 분포 매칭: "You are a helpful assistant. Please say a sentence as loudly as possible."
    # 우리 적용:     "You are a helpful assistant. Please say this sentence {tone}."
    #
    # tts_emotion (= tone)이 풀 imperative phrase이면 그대로 사용
    # emotion 키워드만 (예: "angry") 또는 빈 문자열이면 카테고리 fallback

    llm_tone = (tts_emotion or "").strip().rstrip(".,;")
    llm_tone = llm_tone.replace('"', '').replace("'", "").replace("[", "").replace("]", "")
    # 'in a' / 'with' 시작 자동 보정
    if llm_tone and not (llm_tone.lower().startswith("in a")
                          or llm_tone.lower().startswith("with")
                          or llm_tone.lower().startswith("in an")):
        llm_tone = f"with {llm_tone}"

    if llm_tone and 5 <= len(llm_tone) <= 300:
        # 팀원 format
        prefix = f'You are a helpful assistant. Please say this sentence {llm_tone}.<|endofprompt|>'
    elif emotion in {"Sad", "Angry", "Happy", "Surprised", "Scared"}:
        # 카테고리 fallback (CosyVoice3 학습 데이터 영어 imperative)
        cat_imperative = {
            "Sad":       "with subdued sadness, soft pacing, conveying lingering regret",
            "Angry":     "in a firm, decisive tone with strong conviction",
            "Happy":     "in a warm, engaging tone with light enthusiasm",
            "Surprised": "with genuine surprise and curious intonation",
            "Scared":    "in a hushed, tense voice with restrained fear",
        }[emotion]
        prefix = f'You are a helpful assistant. Please say this sentence {cat_imperative}.<|endofprompt|>'
    else:
        # Neutral or unknown
        prefix = 'You are a helpful assistant.<|endofprompt|>'

    ref_16k = os.path.join(tempfile.gettempdir(), "ref_16k_temp.wav")'''

if old_prefix in src:
    src = src.replace(old_prefix, new_prefix)
    print("[4] OK: prefix → 팀원 format")
else:
    print("[4] NOT FOUND")

p.write_text(src)
print("[Done] v15: cross_lingual + 팀원 imperative format")
