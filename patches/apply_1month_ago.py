"""1달 전 working 패턴을 cosy_legacy의 orchestrator.py에 적용.

변경:
1. synthesize_segment_cosy: cross_lingual + 16kHz mono + emotion instruct + normalize
2. _translate_segments_llm 함수 추가 (VectorEngine GPT 옵션)
3. translate_segments LLM/google 분기
4. load_translator VectorEngine 분기
5. synthesize_chunk에서 emotion 인자 전달
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# ============================================================
# 1. synthesize_segment_cosy 함수 통째로 교체
# ============================================================
old_func = '''def synthesize_segment_cosy(
    text: str,
    ref_audio: str,
    lang: str,
    speed: float = 1.0
) -> np.ndarray:
    """
    CosyVoice2로 세그먼트 1개 합성.
    화자 프로필에서 감정별로 선택된 ref_audio를 사용하여
    원본 화자의 감정 특성을 그대로 복제.

    INPUT:
      text      : str   — 번역된 텍스트
      ref_audio : str   — profiles[speaker][emotion] 로 감정별 선택된 레퍼런스 경로
                          예) /data/reference/SPEAKER_00_Sad.wav
      lang      : str   — "ko"
      speed     : float — 발화 속도 (0.85 ~ 1.15)

    OUTPUT:
      audio : np.ndarray — 합성된 오디오 (float32, 22050 Hz)
    """
    if _cosy_model is None:
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    if not ref_audio or not os.path.exists(ref_audio):
        print(f"[TTS] 레퍼런스 파일 없음: {ref_audio}")
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    try:
        # COSYVOICE3_FIX: zero_shot + instruct prefix (endofprompt token 필수)
        output = []
        for result in _cosy_model.inference_zero_shot(
            tts_text=text,
            prompt_text='You are a helpful assistant.<|endofprompt|>',
            prompt_wav=ref_audio,
            stream=False,
            speed=speed
        ):
            output.append(result["tts_speech"].squeeze().numpy())

        if not output:
            return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

        wav = np.concatenate(output, axis=0).astype(np.float32)

        # CosyVoice2 출력은 22050Hz
        if _cosy_model.sample_rate != TTS_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=_cosy_model.sample_rate, target_sr=TTS_SAMPLE_RATE)

        return wav

    except Exception as e:
        import traceback
        print(f"[TTS] CosyVoice 합성 실패: {e}", flush=True)
        traceback.print_exc()
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)'''

new_func = '''def synthesize_segment_cosy(
    text: str,
    ref_audio: str,
    lang: str,
    speed: float = 1.0,
    emotion: str = "Neutral"
) -> np.ndarray:
    """CosyVoice3 합성 (1달 전 working 패턴).
    - inference_cross_lingual (영→한 zero-shot voice clone)
    - reference 16kHz mono 변환 후 전달 (frontend 안정화)
    - 감정별 instruct prefix + endofprompt 토큰
    - 음량 normalize (peak 0.9)
    """
    if _cosy_model is None:
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    if not ref_audio or not os.path.exists(ref_audio):
        print(f"[TTS] 레퍼런스 파일 없음: {ref_audio}")
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

    # 감정별 instruct (1달 전 working pattern)
    emotion_instruction = {
        "Angry":     "Speak with anger and intensity.",
        "Sad":       "Speak with sadness and sorrow.",
        "Happy":     "Speak with joy and excitement.",
        "Surprised": "Speak with surprise and wonder.",
        "Scared":    "Speak with fear and anxiety.",
        "Neutral":   "",
    }
    instruct = emotion_instruction.get(emotion, "")
    if instruct:
        prefix = f'{instruct}<|endofprompt|>'
    else:
        prefix = 'You are a helpful assistant.<|endofprompt|>'

    ref_16k = os.path.join(tempfile.gettempdir(), f"ref_16k_{os.getpid()}.wav")
    try:
        # reference를 16kHz mono로 사전 변환 (frontend 안정화)
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
            output.append(result["tts_speech"].squeeze().numpy())

        if not output:
            return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)

        wav = np.concatenate(output, axis=0).astype(np.float32)

        # 음량 normalize (1달 전 working)
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav * (0.9 / peak)

        # CosyVoice3 출력(24kHz)이 TTS_SAMPLE_RATE와 다르면 리샘플
        if _cosy_model.sample_rate != TTS_SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=_cosy_model.sample_rate, target_sr=TTS_SAMPLE_RATE)

        return wav

    except Exception as e:
        import traceback
        print(f"[TTS] CosyVoice3 합성 실패: {e}", flush=True)
        traceback.print_exc()
        return np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)
    finally:
        if os.path.exists(ref_16k):
            try:
                os.unlink(ref_16k)
            except Exception:
                pass'''

if old_func in src:
    src = src.replace(old_func, new_func)
    print("[1] synthesize_segment_cosy 패치 OK")
else:
    print("[1] synthesize_segment_cosy 패턴 못 찾음 (이미 패치된 상태?)")

# ============================================================
# 2. synthesize_chunk에서 emotion 전달 (synthesize_segment_cosy(emotion=) 인자)
# ============================================================
# 호출부 찾아서 emotion 추가
old_call = '''        audio_chunk = synthesize_segment_cosy(
            text=seg.translated,
            ref_audio=ref_path,
            lang=tgt_lang,
            speed=seg.speed
        )'''
new_call = '''        audio_chunk = synthesize_segment_cosy(
            text=seg.translated,
            ref_audio=ref_path,
            lang=tgt_lang,
            speed=seg.speed,
            emotion=seg.emotion,
        )'''
if old_call in src:
    src = src.replace(old_call, new_call)
    print("[2] synthesize_chunk emotion 전달 OK")
else:
    print("[2] synthesize_chunk 호출 패턴 못 찾음")

# ============================================================
# 3. translate_segments + _translate_segments_llm + load_translator 확장
# ============================================================
# 기존 translate_segments 패치
old_translate = '''def translate_segments(segments: List[Segment], tgt_lang: str) -> List[Segment]:'''
new_translate_block = '''def _translate_segments_llm(segments: List[Segment], tgt_lang: str) -> List[Segment]:
    """LLM (VectorEngine GPT)으로 청크 전체 세그먼트 한 번에 번역.
    감정 풍부, 화자 일관성, 자연스러운 한국어 구어체.
    실패 시 Google 폴백.
    """
    import requests, re

    api_key = os.environ.get("VECTORENGINE_API_KEY", "")
    base_url = os.environ.get("VECTORENGINE_BASE_URL", "https://api.vectorengine.ai")
    model = os.environ.get("VECTORENGINE_MODEL", "gpt-5.4-xhigh")

    lang_names = {
        "ko": "한국어", "ja": "일본어", "zh": "중국어", "fr": "프랑스어",
        "de": "독일어", "es": "스페인어", "en": "영어", "ru": "러시아어",
        "pt": "포르투갈어", "it": "이탈리아어", "ar": "아랍어", "nl": "네덜란드어",
    }
    lang_name = lang_names.get(tgt_lang, tgt_lang)

    to_translate = [(i, seg) for i, seg in enumerate(segments) if seg.text.strip()]
    if not to_translate:
        return segments

    # 화자/감정 컨텍스트도 LLM에 같이 제공 (감정 풍부 번역)
    lines = []
    for idx, (i, seg) in enumerate(to_translate):
        ctx = f"[{idx}|{seg.speaker}|{seg.emotion}] {seg.text}"
        lines.append(ctx)
    batch_text = "\\n".join(lines)

    system_prompt = (
        f"You are a professional dubbing translator. Translate each numbered line into natural spoken {lang_name} for voice acting.\\n"
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
    )

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": batch_text},
                ],
                "temperature": 0.4,
                "max_tokens": 4096,
            },
            timeout=180,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()

        if "<think>" in result:
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        translations = {}
        for line in result.split("\\n"):
            line = line.strip()
            match = re.match(r"\\[(\\d+)\\]\\s*(.*)", line)
            if match:
                translations[int(match.group(1))] = match.group(2).strip()

        for idx, (i, seg) in enumerate(to_translate):
            if idx in translations and translations[idx]:
                seg.translated = translations[idx]
                print(f"[Translate-LLM] [{seg.emotion}] {seg.text[:30]}... → {seg.translated[:30]}...")
            else:
                seg.translated = translate_segment(seg.text, tgt_lang)
                print(f"[Translate-Fallback] {seg.text[:30]}... → {seg.translated[:30]}...")

        return segments

    except Exception as e:
        print(f"[Translate] LLM 배치 실패: {e} → Google 폴백")
        for seg in segments:
            if seg.text.strip():
                seg.translated = translate_segment(seg.text, tgt_lang)
                print(f"[Translate-Fallback] {seg.text[:30]}... → {seg.translated[:30]}...")
        return segments


def translate_segments(segments: List[Segment], tgt_lang: str) -> List[Segment]:'''

if old_translate in src:
    src = src.replace(old_translate, new_translate_block)
    print("[3] _translate_segments_llm 추가 OK")
else:
    print("[3] translate_segments 못 찾음")

# translate_segments 본문에 LLM 분기 추가
old_body = '''def translate_segments(segments: List[Segment], tgt_lang: str) -> List[Segment]:
    """세그먼트 전체 번역 (각각 호출)."""
    for seg in segments:
        seg.translated = translate_segment(seg.text, tgt_lang)
        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
    return segments'''

new_body = '''def translate_segments(segments: List[Segment], tgt_lang: str) -> List[Segment]:
    """세그먼트 전체 번역. VectorEngine API 키 있으면 LLM 배치, 없으면 Google."""
    if os.environ.get("VECTORENGINE_API_KEY", ""):
        return _translate_segments_llm(segments, tgt_lang)
    for seg in segments:
        seg.translated = translate_segment(seg.text, tgt_lang)
        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
    return segments'''

if old_body in src:
    src = src.replace(old_body, new_body)
    print("[4] translate_segments LLM 분기 OK")
else:
    print("[4] translate_segments 본문 패턴 못 찾음")

# ============================================================
# 저장
# ============================================================
p.write_text(src, encoding='utf-8')
print("[Done] orchestrator.py 패치 완료")
