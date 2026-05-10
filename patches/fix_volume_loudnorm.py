"""음량 조정 + loudnorm (-19 LUFS) 적용.
- dubbed_volume 1.0 → 0.65 (한국어 음성 35% 줄임)
- bgm_volume 0.5 유지
- amix 후 loudnorm I=-19 TP=-1.5 LRA=11 (Netflix/방송 dubbing 표준)
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# default 값 변경
old_def = 'def mix_audio(\n    chunk_path: str,\n    dubbed_path: str,\n    bgm_path: str,\n    output_path: str,\n    dubbed_volume: float = 1.0,\n    bgm_volume: float = 0.5\n) -> str:'
new_def = 'def mix_audio(\n    chunk_path: str,\n    dubbed_path: str,\n    bgm_path: str,\n    output_path: str,\n    dubbed_volume: float = 0.65,\n    bgm_volume: float = 0.5\n) -> str:'
if old_def in src:
    src = src.replace(old_def, new_def)
    print('[1] dubbed_volume default 0.65 적용')
else:
    print('[1] mix_audio signature 못 찾음')

# filter_complex에 loudnorm 추가
old_filter = '''    filter_complex = (
        f"[1:a]aformat=channel_layouts=mono,volume={dubbed_volume}[dub];"
        f"[2:a]aformat=channel_layouts=mono,volume={bgm_volume}[bgm];"
        "[dub][bgm]amix=inputs=2:duration=first:normalize=0[a]"
    )'''
new_filter = '''    # VOLUME_FIX: amix 후 loudnorm으로 자동 음량 정규화 (-19 LUFS = dubbing 표준)
    filter_complex = (
        f"[1:a]aformat=channel_layouts=mono,volume={dubbed_volume}[dub];"
        f"[2:a]aformat=channel_layouts=mono,volume={bgm_volume}[bgm];"
        "[dub][bgm]amix=inputs=2:duration=first:normalize=0,"
        "loudnorm=I=-19:TP=-1.5:LRA=11[a]"
    )'''
if old_filter in src:
    src = src.replace(old_filter, new_filter)
    print('[2] loudnorm 적용')
else:
    print('[2] filter_complex 패턴 못 찾음')

p.write_text(src, encoding='utf-8')
print('[Done] volume + loudnorm 패치 완료')
