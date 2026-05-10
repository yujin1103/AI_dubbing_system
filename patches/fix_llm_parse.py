"""LLM 응답 파싱 안정화. content 키 누락 등 응답 형식 이상 시 디버그 + 안전 처리."""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

old = '''def _llm_call_with_retry(messages, model, base_url, api_key, max_retries=5):
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
    raise RuntimeError(f"LLM 호출 {max_retries}회 모두 실패: {last_err}")'''

new = '''def _llm_call_with_retry(messages, model, base_url, api_key, max_retries=5):
    """LLM 호출 + 429/network 에러 시 exponential backoff retry.
    응답 형식 이상 시 raw json 디버그 출력.
    """
    import requests, time, json
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
                wait = 60 * (2 ** attempt)
                print(f"[Translate] 429 rate limit, {wait}s 대기 후 재시도 ({attempt+1}/{max_retries})...", flush=True)
                time.sleep(wait)
                last_err = f"429 attempt {attempt+1}"
                continue
            resp.raise_for_status()
            data = resp.json()
            # 안전 파싱: 다양한 응답 구조 시도
            try:
                choices = data.get("choices", [])
                if not choices:
                    raise KeyError("choices 비어있음")
                first = choices[0]
                msg = first.get("message") or first.get("delta") or {}
                content = msg.get("content")
                if content is None:
                    # 일부 model: content_array, text 등 fallback
                    content = first.get("text") or msg.get("reasoning_content") or ""
                if not content:
                    raise KeyError(f"content 비어있음. 응답: {json.dumps(data)[:500]}")
                return content.strip()
            except (KeyError, IndexError, TypeError) as parse_err:
                print(f"[Translate] 응답 파싱 실패 ({parse_err}). raw json:", flush=True)
                print(json.dumps(data, ensure_ascii=False)[:1000], flush=True)
                last_err = f"parse: {parse_err}"
                # 짧은 대기 후 재시도 (응답이 일시적으로 빈 경우)
                time.sleep(10)
                continue
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            wait = 30 * (attempt + 1)
            print(f"[Translate] network 오류 {attempt+1}/{max_retries}, {wait}s 후 재시도: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"LLM 호출 {max_retries}회 모두 실패: {last_err}")'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding='utf-8')
    print('OK: LLM parse 안전화')
else:
    print('NOT FOUND')
