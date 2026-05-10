"""짧은 chunk (< 3초) 자동 skip 패치.

원인: 영상 길이가 segment_time의 정확한 배수가 아니면 마지막 짧은 fragment 생성됨.
예: 60.48s 영상 + segment 60 → chunk_000(60s) + chunk_001(0.48s)
0.48s는 demucs가 처리 못함 (오디오 stream length 0)

해결: split_video에서 ffprobe로 각 chunk duration 측정 → 3초 미만은 삭제 + 리스트에서 제외.
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

old = '''    chunks = sorted([
        os.path.join(CHUNKS_DIR, f)
        for f in os.listdir(CHUNKS_DIR)
        if f.startswith(file_name) and f.endswith(".mp4") and "final" not in f
    ])
    print(f"[Split] {len(chunks)}개 청크 생성 완료")
    return chunks'''

new = '''    chunks = sorted([
        os.path.join(CHUNKS_DIR, f)
        for f in os.listdir(CHUNKS_DIR)
        if f.startswith(file_name) and f.endswith(".mp4") and "final" not in f
    ])

    # SHORT_CHUNK_FIX: 너무 짧은 (< 3초) chunk 자동 skip + 삭제
    filtered = []
    for c in chunks:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", c],
                capture_output=True, text=True
            )
            dur = float(r.stdout.strip()) if r.stdout.strip() else 0.0
        except Exception:
            dur = 0.0
        if dur >= 3.0:
            filtered.append(c)
        else:
            print(f"[Split] 짧은 chunk 삭제: {os.path.basename(c)} ({dur:.2f}s)")
            try:
                os.remove(c)
            except Exception:
                pass

    print(f"[Split] {len(filtered)}개 청크 (필터 후)")
    return filtered'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding='utf-8')
    print('OK: short chunk filter 적용')
else:
    print('NOT FOUND')
