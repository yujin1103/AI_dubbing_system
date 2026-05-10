"""orchestrator에 AV fusion 통합 (Option A).

추가:
  - diarize 후 ASD + AV fusion 호출
  - spurious speaker 자동 제거
  - data["av_fusion"], data["asd_result"] 저장 (lipsync 단계에서 사용 가능)

위치: post_process_diarization 직후 (line ~3005)
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''        # 🔥 Step 4-2: ECAPA centroid 후처리 (over-segmentation 해결)
        #   pyannote가 같은 사람 톤 변화를 다른 화자로 잘못 인식하는 문제 해결.
        #   centroid 거리가 가까운 (0.65 cosine) 화자 쌍 자동 병합 + 짧은 turn 재할당.
        if num_speakers is None or num_speakers > 1:
            # num_speakers=1 이면 단일 화자로 간주 → 후처리 불필요
            diarization = post_process_diarization(diarization, data["vocals_path"])

        # Step 5: 세그먼트 조합 (문장 단위, 수정 P)
        segments = build_segments('''

new = '''        # 🔥 Step 4-2: ECAPA centroid 후처리 (over-segmentation 해결)
        #   pyannote가 같은 사람 톤 변화를 다른 화자로 잘못 인식하는 문제 해결.
        #   centroid 거리가 가까운 (0.65 cosine) 화자 쌍 자동 병합 + 짧은 turn 재할당.
        if num_speakers is None or num_speakers > 1:
            # num_speakers=1 이면 단일 화자로 간주 → 후처리 불필요
            diarization = post_process_diarization(diarization, data["vocals_path"])

        # 🔥 Step 4-3: AV Fusion (LightASD) — 화자 분리 정확도 향상
        #   Visual ASD로 화면 내 발화자를 식별하여 audio diarization 검증.
        #   spurious speaker (한숨/짧은 noise를 별개 화자로 잘못 인식) 자동 제거.
        try:
            import sys as _sys
            if "/workspace/scripts" not in _sys.path:
                _sys.path.insert(0, "/workspace/scripts")
            from asd_runner import run_asd
            from av_fusion import fuse_av_diarization
            print(f"[AV-Fusion] LightASD on {chunk_name}...")
            asd_result = run_asd(data["chunk_path"])
            if asd_result and diarization is not None:
                audio_segments_list = [
                    (turn.start, turn.end, spk)
                    for turn, _, spk in diarization.itertracks(yield_label=True)
                ]
                fusion = fuse_av_diarization(audio_segments_list, asd_result, verbose=False)
                # spurious speaker 제거
                if fusion["spurious_speakers"]:
                    from pyannote.core import Annotation
                    new_diar = Annotation()
                    for turn, track, spk in diarization.itertracks(yield_label=True):
                        if spk not in fusion["spurious_speakers"]:
                            new_diar[turn, track] = spk
                    print(f"[AV-Fusion] spurious 화자 제거: {fusion['spurious_speakers']}")
                    diarization = new_diar
                # 결과 저장 (lipsync 단계에서 활용 가능)
                data["av_fusion"] = fusion
                data["asd_result"] = asd_result
                print(f"[AV-Fusion] face tracks={len(asd_result['tracks'])}, "
                      f"lipsync target frames={sum(1 for t in fusion['per_frame_target'] if t is not None)}/"
                      f"{fusion['n_frames']}")
            else:
                print(f"[AV-Fusion] ASD 결과 없음 → skip")
        except Exception as _e:
            import traceback as _tb
            print(f"[AV-Fusion] 실패 (계속 진행): {_e}")
            _tb.print_exc()

        # Step 5: 세그먼트 조합 (문장 단위, 수정 P)
        segments = build_segments('''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: AV fusion 통합 완료")
else:
    print("NOT FOUND - check current state")
