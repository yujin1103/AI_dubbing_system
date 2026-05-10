"""4월 28일자 orchestrator.py 그대로 복구 (잘 작동했던 버전).

복구 항목:
  1. synthesize_segment_cosy: inference_cross_lingual + prefix in tts_text
  2. synthesize_chunk: profile.get_ref(emotion) MOS-selected (self ref 제거)
  3. LLM prompt: emotion: 2~5 words 형식 (tone 제거)
  4. Parser: 'tone' 필드 제거 (emotion만)
  5. Whisper validation 제거 (사용자 요청)

유지 (4월 28일 이후 추가된 좋은 fix):
  - BS-Roformer separate_audio
  - dialogue filter (한숨/탄식 skip)
  - post-TTS speed retry (효율화)
  - MOS reload 제거
  - Google fallback 제거
  - AV-Fusion (LightASD)
  - face_skip / brightness check / loop_video valid_mask fix (lipsync용)
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# ============================================================
# 1. synthesize_segment_cosy 함수 복구
# ============================================================
# 현재 코드의 try 블록 전체를 4월 28일 버전으로 교체

# 현재 코드 — inference_instruct2 + 복잡한 wrap
old_synth = '''    ref_16k = os.path.join(tempfile.gettempdir(), "ref_16k_temp.wav")
    try:
        # 레퍼런스를 16kHz 모노로 변환 (CosyVoice3 요구사항)
        subprocess.run([
            "ffmpeg", "-y", "-i", ref_audio,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", ref_16k
        ], capture_output=True)

        output = []
        # === CosyVoice3 instruct2 - 학습 format에 LLM 풍부 묘사 wrapping ===
        # LLM tts_emotion (예: "sharp challenge, impatient edge") → wrap in training imperative
        # OOD/긴 묘사는 카테고리 fallback로 안전 처리
        EMOTION_FALLBACK_ZH = {
            "Sad":       "请非常伤心地说一句话",
            "Angry":     "请非常生气地说一句话",
            "Happy":     "请非常开心地说一句话",
            "Surprised": "请用非常惊讶、激动的语气说话",
            "Scared":    "请用非常紧张、害怕、颤抖的语气说话",
        }

        llm_tone = (tts_emotion or "").strip().rstrip(".,;")
        # 따옴표/대괄호 등 정제
        llm_tone = llm_tone.replace('"', '').replace("'", "").replace("[", "").replace("]", "")
        # 팀원 검증 format: "You are a helpful assistant. Please say this sentence {English tone}.<|endofprompt|>"
        # CosyVoice3 학습 분포: "Please say a sentence as loudly as possible." 등 영어 imperative
        # LLM이 'in a' / 'with' 시작하지 않으면 보정
        if llm_tone and not (llm_tone.lower().startswith("in a")
                              or llm_tone.lower().startswith("with")
                              or llm_tone.lower().startswith("in an")):
            llm_tone = f"with {llm_tone}"

        instruct_text = None
        if llm_tone and 5 <= len(llm_tone) <= 300:
            instruct_text = (
                f"You are a helpful assistant. "
                f"Please say this sentence {llm_tone}.<|endofprompt|>"
            )
        elif emotion in EMOTION_FALLBACK_ZH:
            # LLM 묘사 없거나 너무 길면 → 카테고리 중국어 fallback (학습 분포 매칭)
            instruct_text = (
                f"You are a helpful assistant. {EMOTION_FALLBACK_ZH[emotion]}<|endofprompt|>"
            )

        if instruct_text:
            inference_iter = _cosy_model.inference_instruct2(
                tts_text=text,
                instruct_text=instruct_text,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )
        else:
            # Neutral + LLM emotion 없음 → cross_lingual (자연 합성)
            inference_iter = _cosy_model.inference_cross_lingual(
                tts_text=text,
                prompt_wav=ref_16k,
                stream=False,
                speed=speed,
            )

        for result in inference_iter:
            output.append(result["tts_speech"].squeeze().numpy())'''

# 4월 28일 버전 (검증된)
new_synth = '''    if instruct:
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
        # APRIL_28_REVERT: inference_cross_lingual + prefix in tts_text (검증된 방식)
        for result in _cosy_model.inference_cross_lingual(
            tts_text=f'{prefix}{text}',
            prompt_wav=ref_16k,
            stream=False,
            speed=speed
        ):
            output.append(result["tts_speech"].squeeze().numpy())'''

if old_synth in src:
    src = src.replace(old_synth, new_synth)
    print("[1] OK: synthesize_segment_cosy 복구 (cross_lingual + prefix)")
else:
    print("[1] NOT FOUND - synthesize_segment_cosy")

# ============================================================
# 2. WHISPER VALIDATION 제거
# ============================================================
old_whisper = '''        # WHISPER_VALIDATION: 외국어 환각 detect → 재합성
        if tgt_lang == "ko" and len(audio_chunk) > TTS_SAMPLE_RATE * 1.0:
            for retry in range(2):
                is_kor, kor_ratio, transcribed = _is_korean_audio(
                    audio_chunk, TTS_SAMPLE_RATE, korean_ratio_threshold=0.55
                )
                if is_kor:
                    if retry > 0:
                        print(f"  ↳ Whisper 검증 통과 (재시도 #{retry}): kor_ratio={kor_ratio:.0%}")
                    break
                print(f"  ⚠️ 외국어 환각 detect (kor={kor_ratio:.0%}, '{transcribed[:40]}') — 재합성 #{retry+1}/2")
                audio_chunk = synthesize_segment_cosy(
                    text=combined_text,
                    ref_audio=ref_path,
                    lang=tgt_lang,
                    speed=1.0,
                    emotion=first_seg.emotion,
                    tts_context=getattr(first_seg, 'tts_context', '') or '',
                    tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
                )

        # 길이 측정 + 효율적 speed retry (OPTIMIZED)'''

new_whisper = '''        # 길이 측정 + 효율적 speed retry (OPTIMIZED)'''

if old_whisper in src:
    src = src.replace(old_whisper, new_whisper)
    print("[2] OK: Whisper validation 제거")
else:
    print("[2] NOT FOUND - Whisper validation")

# ============================================================
# 3. SELF REFERENCE 제거 (MOS-selected reference 복구)
# ============================================================
old_self_ref = '''        profile = profiles.get(first_seg.speaker)

        # SELF_REFERENCE_FIX: segment 자체 audio를 reference로 사용 (팀원 검증 방식)
        # 같은 음성 + 자연 emotion 전이 → instruction leak 우회 + voice cloning 정확
        # short segment (<1.5s)는 self reference 너무 짧음 → MOS-selected fallback
        ref_path = ""
        seg_dur = last_seg.end - first_seg.start
        if seg_dur >= 1.5:
            # self reference: 이 segment 시간 구간을 vocals에서 추출
            try:
                # vocals_path는 chunk_data에서 가져와야 함 — 함수 인자에 없음
                # 대신 chunk_name으로 vocals 경로 추정
                vocals_path = os.path.join(VOCALS_DIR, f"{chunk_name}_clean_vocals.wav")
                if not os.path.exists(vocals_path):
                    vocals_path = os.path.join(VOCALS_DIR, f"{chunk_name}_vocals.wav")
                if os.path.exists(vocals_path):
                    self_ref_path = os.path.join(
                        tempfile.gettempdir(),
                        f"selfref_{chunk_name}_{gi}.wav"
                    )
                    ext_result = subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-ss", str(first_seg.start), "-to", str(last_seg.end),
                        "-i", vocals_path,
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                        self_ref_path,
                    ], capture_output=True, text=True)
                    if ext_result.returncode == 0 and os.path.exists(self_ref_path):
                        ref_path = self_ref_path
                        print(f"  ↳ self reference 사용: {seg_dur:.2f}s (segment 자체)")
            except Exception as _e:
                print(f"  ↳ self reference 실패 ({_e}) → MOS-selected fallback")

        # fallback: MOS-selected emotion-별 reference
        if not ref_path or not os.path.exists(ref_path):
            ref_path = profile.get_ref(first_seg.emotion) if profile else ""
            if not ref_path or not os.path.exists(ref_path):
                ref_path = profile.get_ref("Neutral") if profile else ""
            if ref_path:
                print(f"  ↳ MOS-selected fallback: {os.path.basename(ref_path)}")
        if not ref_path or not os.path.exists(ref_path):
            continue'''

new_self_ref = '''        profile = profiles.get(first_seg.speaker)
        ref_path = profile.get_ref(first_seg.emotion) if profile else ""
        if not ref_path or not os.path.exists(ref_path):
            ref_path = profile.get_ref("Neutral") if profile else ""
        if not ref_path or not os.path.exists(ref_path):
            continue'''

if old_self_ref in src:
    src = src.replace(old_self_ref, new_self_ref)
    print("[3] OK: self reference 제거 (MOS-selected 복구)")
else:
    print("[3] NOT FOUND - self reference")

# ============================================================
# 4. LLM prompt: tone 필드 → emotion 필드 복구
# ============================================================
old_llm = '''            f"  2. tone: full English imperative phrase combining STYLE + EMOTION + SITUATION.\\n"
            f"        STRUCTURE: 'in a {{style+tone}} with {{emotion}}, conveying {{situation}}'\\n"
            f"        EXAMPLE GOOD: 'in a low, deliberate tone with grim resolve, conveying a quiet warning before violence'\\n"
            f"        EXAMPLE GOOD: 'in a casual, pleased, lightly confident tone with warm but restrained excitement'\\n"
            f"        EXAMPLE GOOD: 'in a hushed, urgent voice with restrained fear, warning of approaching danger'\\n"
            f"        EXAMPLE BAD: 'angry' (too short)\\n"
            f"        EXAMPLE BAD: '낮은 톤' (Korean — should be English)\\n"
            f"        IMPORTANT: tone MUST be English imperative. CosyVoice3 trained on English+Chinese imperatives.\\n"'''

new_llm = '''            f"  2. context: one short sentence describing the speaker's situation/intent\\n"
            f"  3. emotion: 2~5 evocative words capturing the speaking tone\\n"'''

if old_llm in src:
    src = src.replace(old_llm, new_llm)
    print("[4a] OK: LLM prompt tone → context+emotion 복구")
else:
    print("[4a] NOT FOUND")

# OUTPUT FORMAT 복구
old_format = '''OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\\n"
            f"[N]\\n"
            f"korean: <translation>\\n"
            f"tone: <imperative phrase combining style+emotion+situation>\\n"'''

new_format = '''OUTPUT FORMAT (strict — keep exactly this structure for every numbered line):\\n"
            f"[N]\\n"
            f"korean: <translation>\\n"
            f"context: <context sentence>\\n"
            f"emotion: <emotion words>\\n"'''

if old_format in src:
    src = src.replace(old_format, new_format)
    print("[4b] OK: OUTPUT FORMAT 복구")
else:
    print("[4b] NOT FOUND")

# RULES section 복구
old_rules = '''            f"STYLE/EMOTION/SITUATION RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- tone: English imperative phrase combining 3 elements:\\n"
            f"   (a) STYLE: speaker delivery (low/loud, slow/quick, hushed/booming, casual/formal)\\n"
            f"   (b) EMOTION: emotional state (sadness, grim resolve, light amusement, restrained fear)\\n"
            f"   (c) SITUATION: situation (conveying warning, sharing triumph, accusation, etc)\\n"
            f"- 15-30 words natural English imperative starting with 'in a' or 'with'.\\n"
            f"- This becomes TTS instruction: 'Please say this sentence {{tone}}.'\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

new_rules = '''            f"EMOTION/CONTEXT RULES:\\n"
            f"- emotion2vec_hint is the audio classifier's guess — use as REFERENCE only.\\n"
            f"  If the text content suggests a different tone, prefer the text-based judgment.\\n"
            f"- context: describe what the speaker is doing/feeling in this specific line.\\n"
            f"- emotion: 2~5 short evocative words. AVOID single-word labels like 'happy/sad/angry'.\\n"
            f"- Write context/emotion in English (CosyVoice3 understands English instructions best).\\n"
            f"- Output ONLY the formatted blocks. No thinking, no explanation."'''

if old_rules in src:
    src = src.replace(old_rules, new_rules)
    print("[4c] OK: RULES 복구")
else:
    print("[4c] NOT FOUND")

# ============================================================
# 5. 파싱 코드 복구 (tone → context+emotion)
# ============================================================
old_parse = '''                    # 'tone' 통합 필드 (style+emotion+situation 포함)
                    seg.tts_context = ''  # 더 이상 별도 사용 X (tone에 통합됨)
                    seg.tts_emotion = entry.get('tone', '') or entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:60]}]")'''

new_parse = '''                    seg.tts_context = entry.get('context', '') or ''
                    seg.tts_emotion = entry.get('emotion', '') or ''
                    if seg.tts_emotion:
                        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}... "
                              f"[{seg.tts_emotion[:40]}]")'''

if old_parse in src:
    src = src.replace(old_parse, new_parse)
    print("[5] OK: 파싱 복구")
else:
    print("[5] NOT FOUND")

p.write_text(src)
print("[Done] 4월 28일 setup 복구 완료")
