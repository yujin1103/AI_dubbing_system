"""Google Translate fallback 완전 제거.

이유:
  - LLM이 아니면 사용 안 함 (사용자 요구)
  - Google translate는 감정/맥락 무시 → dubbing 품질 저하
  - 폴백 시 segment 자동 skip → 원본 audio 유지가 더 자연스러움

변경:
  1. line 1706: 비-vectorengine 분기 → LLM 강제
  2. line 1808: 3차 폴백 → 빈 translated (skip)
  3. line 1830: 개별 빈 결과 폴백 → 빈 translated (skip)
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# 1. line 1706 부근: 첫 진입점에서 비-LLM 경로 제거
old1 = '''    if _google_translator == "vectorengine":
        return _translate_segments_llm(segments, tgt_lang, content_type=content_type)

    for seg in segments:
        seg.translated = _translate_google(seg.text, tgt_lang)
        print(f"[Translate] {seg.text[:30]}... → {seg.translated[:30]}...")
    return segments'''

new1 = '''    # LLM 번역 강제 (Google fallback 제거됨 - 사용자 요구)
    if _google_translator != "vectorengine":
        print("[Translate] ⚠️ vectorengine LLM 미설정 — translation skip "
              "(seg.translated 빈 상태 유지, TTS 단계에서 자동 skip)")
        for seg in segments:
            seg.translated = ""
        return segments
    return _translate_segments_llm(segments, tgt_lang, content_type=content_type)'''

if old1 in src:
    src = src.replace(old1, new1)
    print("[1] OK: 첫 진입점 LLM 강제")
else:
    print("[1] NOT FOUND")

# 2. 3차 폴백 (line 1808 부근)
old2 = '''        # 3차 폴백: Google Translator
        if parsed is None:
            print(f"[Translate] {batch_label} LLM 완전 실패 → Google Translator 폴백")
            for batch_local_idx, (i, seg) in enumerate(batch):
                seg.translated = _translate_google(seg.text, tgt_lang)
                print(f"[Translate] (폴백) {seg.text[:30]}... → {seg.translated[:30]}...")
            continue'''

new2 = '''        # 3차 폴백 제거 (Google translate 사용 안 함 - 사용자 요구)
        # LLM 완전 실패 → translated 빈 상태 유지 (TTS 자동 skip)
        if parsed is None:
            print(f"[Translate] ⚠️ {batch_label} LLM 완전 실패 — segments translation skip "
                  f"(원본 audio 유지)")
            for batch_local_idx, (i, seg) in enumerate(batch):
                seg.translated = ""
            continue'''

if old2 in src:
    src = src.replace(old2, new2)
    print("[2] OK: 3차 폴백 제거")
else:
    print("[2] NOT FOUND")

# 3. 개별 빈 결과 폴백 (line 1830 부근)
old3 = '''            else:
                seg.translated = _translate_google(seg.text, tgt_lang)
                print(f"[Translate] (Google 폴백) {seg.text[:30]}... → {seg.translated[:30]}...")'''

new3 = '''            else:
                # LLM이 한국어 미생성 → translation skip (Google fallback 제거)
                seg.translated = ""
                print(f"[Translate] ⚠️ LLM korean 누락 → skip: {seg.text[:30]}...")'''

if old3 in src:
    src = src.replace(old3, new3)
    print("[3] OK: 개별 빈 결과 폴백 제거")
else:
    print("[3] NOT FOUND")

p.write_text(src)
print("[Done] Google fallback 완전 제거")
