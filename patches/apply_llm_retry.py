"""LLM 번역 강제 (Google fallback 금지) + 429 retry 패치."""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# ============================================================
# _translate_segments_llm 통째로 교체:
#  - 429/network 에러 시 exponential backoff retry (60s → 120s → 240s → 480s)
#  - 배치가 큰 경우 절반으로 분할 재시도 (rate limit 회피)
#  - 최종 실패 시 RuntimeError raise (Google fallback 안 함)
# ============================================================
old = '''def _translate_segments_llm(segments: List[Segment], tgt_lang: str) -> List[Segment]:
    """VectorEngine GPT로 청크 전체 세그먼트를 한 번에 번역."""
    import requests
    import re

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

    lines = []
    for idx, (i, seg) in enumerate(to_translate):
        lines.append(f"[{idx}] {seg.text}")
    batch_text = "\\n".join(lines)

    system_prompt = (
        f"You are a professional translator. Translate each numbered line into natural spoken {lang_name} for dubbing. "
        f"CRITICAL RULES:\\n"
        f"- Keep the same numbering format [0], [1], [2]...\\n"
        f"- Each line MUST be translated SEPARATELY with similar length to its original\\n"
        f"- You MUST output in {lang_name}. DO NOT output the original text.\\n"
        f"- Do NOT merge content between lines\\n"
        f"- Make consecutive lines from the same speaker sound natural when spoken in order\\n"
        f"- Output ONLY translations. No thinking, no explanation."
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
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=180,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()

        if "<think>" in result:
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

        translations = {}
        for line in result.split("\\n"):
            line = line.strip()
            match = re.match(r'\\[(\\d+)\\]\\s*(.*)', line)
            if match:
                translations[int(match.group(1))] = match.group(2).strip()

        for idx, (i, seg) in enumerate(to_translate):
            if idx in translations and translations[idx]:
                seg.translated = translations[idx]
                print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
            else:
                seg.translated = _translate_google(seg.text, tgt_lang)
                print(f"[Translate] (폴백) {seg.text[:30]}... → {seg.translated[:30]}...")

        return segments

    except Exception as e:
        print(f"[Translate] LLM 배치 실패: {e} → Deep Translator 폴백")
        for seg in segments:
            if seg.text.strip():
                seg.translated = _translate_google(seg.text, tgt_lang)
                print(f"[Translate] (폴백) {seg.text[:30]}... → {seg.translated[:30]}...")
        return segments'''

new = '''def _llm_call_with_retry(messages, model, base_url, api_key, max_retries=5):
    """LLM 호출 + 429/network 에러 시 exponential backoff retry.
    Google fallback 안 함. 최종 실패 시 RuntimeError.
    """
    import requests, time
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.4,
                    "max_tokens": 4096,
                },
                timeout=180,
            )
            if resp.status_code == 429:
                wait = 60 * (2 ** attempt)  # 60, 120, 240, 480, 960
                print(f"[Translate] 429 rate limit, {wait}s 대기 후 재시도 ({attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait)
                last_err = f"429 attempt {attempt+1}"
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            wait = 30 * (attempt + 1)
            print(f"[Translate] network 오류 {attempt+1}/{max_retries}, {wait}s 후 재시도: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"LLM 호출 {max_retries}회 모두 실패: {last_err}")


def _translate_segments_llm(segments: List[Segment], tgt_lang: str) -> List[Segment]:
    """VectorEngine GPT로 청크 전체 세그먼트 한 번에 번역.
    - 화자/감정 컨텍스트 포함 (감정 풍부)
    - 429/network 에러는 retry (Google fallback 안 함)
    - 최종 실패 시 RuntimeError raise
    """
    import re

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

    # 화자/감정 컨텍스트 포함 (감정 풍부 번역)
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": batch_text},
    ]

    result = _llm_call_with_retry(messages, model, base_url, api_key, max_retries=5)

    if "<think>" in result:
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

    translations = {}
    for line in result.split("\\n"):
        line = line.strip()
        match = re.match(r'\\[(\\d+)\\]\\s*(.*)', line)
        if match:
            translations[int(match.group(1))] = match.group(2).strip()

    # 결과 검증: 모든 segment 번역됐는지 확인
    missing = [idx for idx, _ in enumerate(to_translate) if idx not in translations or not translations[idx]]
    if missing:
        # 누락된 segment 만 다시 single-call (rate limit 분산)
        print(f"[Translate] 누락 {len(missing)}개 → 개별 호출로 재시도", flush=True)
        for idx in missing:
            i, seg = to_translate[idx]
            single_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"[{idx}|{seg.speaker}|{seg.emotion}] {seg.text}"},
            ]
            try:
                single_result = _llm_call_with_retry(single_messages, model, base_url, api_key, max_retries=3)
                m = re.match(r'\\[(\\d+)\\]\\s*(.*)', single_result.strip())
                if m:
                    translations[idx] = m.group(2).strip()
                else:
                    translations[idx] = single_result.strip()
            except Exception as e:
                raise RuntimeError(f"개별 segment {idx} LLM 번역 실패: {e}")

    for idx, (i, seg) in enumerate(to_translate):
        seg.translated = translations[idx]
        print(f"[Translate-LLM] [{seg.emotion}] {seg.text[:30]}... → {seg.translated[:30]}...")

    return segments'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding='utf-8')
    print('OK: LLM retry + Google fallback 제거')
else:
    print('NOT FOUND: _translate_segments_llm 패턴')
