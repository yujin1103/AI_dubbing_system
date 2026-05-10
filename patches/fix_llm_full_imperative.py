"""LLM prompt 수정 - 풀 imperative tone description 생성.

이전 (v7): LLM이 2-5 단어 emotion ("sharp challenge, impatient edge")
새 방식 (v8): LLM이 풀 imperative phrase
  예: "in a casual, pleased, lightly confident tone with warm but restrained excitement"

CosyVoice3 학습 분포 매칭:
  학습: "Please say a sentence as loudly as possible."
  팀원: "Please say this sentence in a casual, pleased, lightly confident tone..."
  우리 새: "Please say this sentence {LLM_output}."

LLM 출력 → wrap → CosyVoice3 instruct2
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 1. LLM emotion field 설명을 imperative phrase로
old1 = '''            f"You are a professional dubbing translator and emotion designer.\\n"
            f"For each numbered line, output THREE fields:\\n"
            f"  1. korean: natural spoken {lang_name} translation for dubbing\\n"
            f"  2. context: one short sentence describing the speaker's situation/intent\\n"
            f"  3. emotion: 2~5 evocative words capturing the speaking tone\\n"
            f"\\n"
            f"OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\\n"
            f"[N]\\n"
            f"korean: <translation>\\n"
            f"context: <context sentence>\\n"
            f"emotion: <emotion words>\\n"'''

new1 = '''            f"You are a professional dubbing translator and emotion designer.\\n"
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

if old1 in src:
    src = src.replace(old1, new1)
    print("[1] OK: LLM emotion → tone (imperative phrase)")
else:
    print("[1] NOT FOUND")

# 2. EMOTION/CONTEXT RULES 업데이트
old2 = '''            f"EMOTION/CONTEXT RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- context: describe what the speaker is doing/feeling in this specific line.\\n"
            f"- emotion: 2~5 short evocative words. AVOID single-word labels like 'happy/sad/angry'.\\n"
            f"- Write context/emotion in English (CosyVoice3 understands English instructions best).\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

new2 = '''            f"EMOTION/TONE RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- context: describe what the speaker is doing/feeling in this specific line.\\n"
            f"- tone: imperative description starting with 'in a' or 'with', describing HOW to speak.\\n"
            f"  Mention: emotion + pacing + pitch + intensity. 10-25 words natural English.\\n"
            f"  This becomes a TTS instruction like 'Please say this sentence {{tone}}.'\\n"
            f"- Write context/tone in English (CosyVoice3 trained on English imperatives).\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

if old2 in src:
    src = src.replace(old2, new2)
    print("[2] OK: tone rules")
else:
    print("[2] NOT FOUND")

# 3. 파싱 코드: emotion → tone field
old3 = '''                    seg.tts_context = entry.get('context', '') or ''
                    seg.tts_emotion = entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:40]}]")'''

new3 = '''                    seg.tts_context = entry.get('context', '') or ''
                    # 'tone' 우선, 호환성을 위해 'emotion'도 fallback
                    seg.tts_emotion = entry.get('tone', '') or entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:50]}]")'''

if old3 in src:
    src = src.replace(old3, new3)
    print("[3] OK: tone field 파싱")
else:
    print("[3] NOT FOUND")

# 4. CosyVoice3 wrapping: 풀 imperative format
# tts_emotion이 이제 풀 phrase ("in a casual, pleased ... tone")이라
# wrapping format도 그에 맞게 변경
old4 = '''        llm_emotion_desc = (tts_emotion or "").strip().rstrip(".,;")
        # 콤마/따옴표 등 정제
        llm_emotion_desc = llm_emotion_desc.replace('"', '').replace("'", "")

        instruct_text = None
        if llm_emotion_desc and len(llm_emotion_desc) <= 80 and len(llm_emotion_desc) >= 3:
            # LLM 풍부 묘사 사용 (학습 imperative 포맷으로 wrap)
            # 학습 데이터: "Please say a sentence as loudly as possible."
            # 우리: "Please say it with {emotion} emotion."
            instruct_text = (
                f"You are a helpful assistant. "
                f"Please say it with {llm_emotion_desc} emotion.<|endofprompt|>"
            )'''

new4 = '''        llm_tone = (tts_emotion or "").strip().rstrip(".,;")
        # 따옴표/대괄호 등 정제
        llm_tone = llm_tone.replace('"', '').replace("'", "").replace("[", "").replace("]", "")
        # LLM이 'in a' / 'with' 로 시작하지 않으면 보정
        if llm_tone and not (llm_tone.lower().startswith("in a")
                              or llm_tone.lower().startswith("with")
                              or llm_tone.lower().startswith("in an")):
            llm_tone = f"with {llm_tone}"

        instruct_text = None
        if llm_tone and 5 <= len(llm_tone) <= 200:
            # 팀원 검증 format: "Please say this sentence {tone}."
            # 학습 분포 매칭: "Please say a sentence as loudly as possible."
            instruct_text = (
                f"You are a helpful assistant. "
                f"Please say this sentence {llm_tone}.<|endofprompt|>"
            )'''

if old4 in src:
    src = src.replace(old4, new4)
    print("[4] OK: 'Please say this sentence {tone}.' wrapping")
else:
    print("[4] NOT FOUND")

p.write_text(src)
print("[Done] LLM full imperative + 팀원 검증 format 적용")
