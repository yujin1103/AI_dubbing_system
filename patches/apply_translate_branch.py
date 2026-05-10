"""translate_segments 본문에 LLM 분기 추가 (docstring 큰 버전 대응)."""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

old = '''    OUTPUT:
      segments : List[Segment] — translated 채워진 상태
    """
    for seg in segments:
        seg.translated = translate_segment(seg.text, tgt_lang)
        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
    return segments'''

new = '''    OUTPUT:
      segments : List[Segment] — translated 채워진 상태
    """
    # VectorEngine LLM 키 있으면 배치 번역 (감정 풍부, 화자 일관성)
    if os.environ.get("VECTORENGINE_API_KEY", ""):
        return _translate_segments_llm(segments, tgt_lang)
    for seg in segments:
        seg.translated = translate_segment(seg.text, tgt_lang)
        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
    return segments'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding='utf-8')
    print("OK: translate_segments LLM 분기 적용")
else:
    print("NOT FOUND")
