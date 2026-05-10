"""TTS 단계의 MOS reload 제거 (MOS_RESYNTH 비활성화 상태이면 사용 안 함).

이유:
  현재 코드: MOS 모델을 reference 선택 후 unload, TTS 단계에서 다시 load
  하지만 MOS_RESYNTH_ENABLED = False이라 TTS 단계에서 사용 안 됨
  → 약 30-60s 절약
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''    # ── 5단계: TTS + MOS 평가 + 믹싱 ─────────────────────
    load_cosy()
    load_mos()  # MOS 다시 로드 (TTS 출력 평가용)
    for chunk_name, data in chunk_data.items():'''

new = '''    # ── 5단계: TTS + 믹싱 (MOS reload 제거 - MOS_RESYNTH 비활성 상태) ───
    load_cosy()
    # OPTIMIZATION: MOS_RESYNTH_ENABLED=False이라 TTS 단계 MOS load 불필요 (~60s 절약)
    for chunk_name, data in chunk_data.items():'''

if old in src:
    src = src.replace(old, new)
    # 이어지는 _unload("mos")도 제거
    old2 = '''    _unload("cosy")
    _unload("mos")

    # 전체 청크 합치기'''
    new2 = '''    _unload("cosy")

    # 전체 청크 합치기'''
    if old2 in src:
        src = src.replace(old2, new2)
        print("[2] OK: TTS 단계 _unload('mos') 제거")
    p.write_text(src)
    print("[1] OK: TTS 단계 load_mos() 제거")
else:
    print("[1] NOT FOUND")
