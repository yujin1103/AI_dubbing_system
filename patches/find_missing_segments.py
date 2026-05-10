"""silero-vad로 발화 구간 detect → ASR이 빠뜨린 구간 확인."""
import sys
import json
import os

VOCALS = "/workspace/media/runs/20260506_032617_test3v24_c0bf6a/vocals/test3v24_chunk_000_clean_vocals.wav"
REPORT = "/workspace/media/reports/test3v24_ko_20260506_032617_test3v24_c0bf6a.json"

# 우리 ASR segments 로드
with open(REPORT) as f:
    rep = json.load(f)
asr_segments = []
for chunk in rep["chunks"]:
    for seg in chunk["segments"]:
        asr_segments.append((seg["start"], seg["end"], seg["original_text"][:30]))

print(f"=== ASR detect 한 segments ({len(asr_segments)}개) ===")
for s, e, txt in asr_segments:
    print(f"  [{s:5.2f}~{e:5.2f}] {txt}")

# silero-vad로 발화 구간 추출
print("\n=== silero-vad 발화 구간 detect ===")
try:
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
except ImportError:
    # fallback: subprocess로 venv_asr 사용
    import subprocess
    r = subprocess.run([
        "/opt/venv_asr/bin/python", "-c",
        f"""
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
import json
m = load_silero_vad()
w = read_audio('{VOCALS}', sampling_rate=16000)
s = get_speech_timestamps(w, m, sampling_rate=16000, min_silence_duration_ms=500, min_speech_duration_ms=300)
print(json.dumps([(x['start']/16000, x['end']/16000) for x in s]))
"""
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"silero_vad subprocess failed: {r.stderr[:300]}")
        sys.exit(1)
    vad_segments = json.loads(r.stdout.strip().split("\n")[-1])
    print(f"VAD detect 발화 구간 ({len(vad_segments)}개):")
    for s, e in vad_segments:
        print(f"  [{s:5.2f}~{e:5.2f}] dur={e-s:.2f}s")
    # missing detect
    missing = []
    for vs, ve in vad_segments:
        overlapped = False
        for as_, ae, _ in asr_segments:
            if ae < vs or as_ > ve:
                continue
            overlap = min(ae, ve) - max(as_, vs)
            if overlap / (ve - vs) > 0.5:
                overlapped = True
                break
        if not overlapped:
            missing.append((vs, ve))
    print("\n=== ASR이 빠뜨린 발화 구간 ===")
    for vs, ve in missing:
        print(f"  ⚠️ [{vs:5.2f}~{ve:5.2f}] dur={ve-vs:.2f}s")
    print(f"\n총 ASR miss: {len(missing)}개")
    print(f"총 missed duration: {sum(e-s for s,e in missing):.1f}s")
    sys.exit(0)

model = load_silero_vad()
wav = read_audio(VOCALS, sampling_rate=16000)
speech = get_speech_timestamps(
    wav, model, sampling_rate=16000,
    min_silence_duration_ms=500,  # 500ms 이상 silence면 분리
    min_speech_duration_ms=300,   # 300ms 미만 발화는 무시
)
vad_segments = [(s["start"]/16000, s["end"]/16000) for s in speech]

print(f"VAD detect 발화 구간 ({len(vad_segments)}개):")
for s, e in vad_segments:
    print(f"  [{s:5.2f}~{e:5.2f}] dur={e-s:.2f}s")

# ASR과 비교: VAD에서 발견됐지만 ASR에서 빠진 구간
print("\n=== ASR이 빠뜨린 발화 구간 ===")
missing = []
for vs, ve in vad_segments:
    # ASR segment와 겹치는지 확인 (50% 이상)
    overlapped = False
    for as_, ae, _ in asr_segments:
        if ae < vs or as_ > ve:
            continue  # 안 겹침
        overlap = min(ae, ve) - max(as_, vs)
        vad_dur = ve - vs
        if overlap / vad_dur > 0.5:  # VAD의 50% 이상 ASR과 겹침
            overlapped = True
            break
    if not overlapped:
        missing.append((vs, ve))
        print(f"  ⚠️ [{vs:5.2f}~{ve:5.2f}] dur={ve-vs:.2f}s — ASR detect 못함")

print(f"\n총 ASR miss: {len(missing)}개")
print(f"총 missed duration: {sum(e-s for s,e in missing):.1f}s")
