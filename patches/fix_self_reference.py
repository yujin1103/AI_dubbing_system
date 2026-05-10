"""self reference 도입 (팀원 설정과 동일).

이전: profile.get_ref(emotion) - MOS-selected emotion-별 reference
새로: 각 segment가 자기 자신 audio를 reference로 사용

장점:
  - 같은 음성 → 가장 정확한 voice cloning
  - emotion 자동 전이 (segment 본래 emotion)
  - MOS auto-selection 우회 (단, 품질 보장 X)

short segment 처리:
  segment < 1.5s 일 경우 self reference 너무 짧음 → MOS-selected fallback
  가능하면 self, 안 되면 fallback
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

# synthesize_chunk 내 reference 결정 부분 변경
old = '''        profile = profiles.get(first_seg.speaker)
        ref_path = profile.get_ref(first_seg.emotion) if profile else ""
        if not ref_path or not os.path.exists(ref_path):
            ref_path = profile.get_ref("Neutral") if profile else ""
        if not ref_path or not os.path.exists(ref_path):
            continue'''

new = '''        profile = profiles.get(first_seg.speaker)

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

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: self reference 도입 완료")
else:
    print("NOT FOUND - check current state")
