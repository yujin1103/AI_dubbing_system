"""post-TTS length check + CosyVoice3 speed retry 추가.

목적:
  현재 코드: speed=1.0 고정 → 길면 time_stretch (1.15x 한계)
  개선: TTS 후 length check → 너무 길면 speed 1.1, 1.15로 재합성 → 그래도 길면 time_stretch

CosyVoice3 speed parameter:
  - 1.0 = 자연 속도 (기본)
  - >1.0 = 빠르게 (max ~1.5 허용)
  - <1.0 = 느리게

전략:
  1차: speed=1.0 합성 → 측정
  2차 (overflow 시): speed=1.10 합성 → 측정
  3차 (여전히 overflow): speed=1.15 합성 → 측정
  → 가장 적합한 결과 사용
  → 그래도 overflow면 기존 time_stretch fallback
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''        # speed=1.0 고정: TTS는 자연 속도, 길이 조정은 이후 time_stretch로
        speed = 1.0

        # ─── TTS 합성 (1회만, 더빙본 MOS 재평가 + 재합성은 시간 절약 위해 비활성화) ──
        # 사용자 요청 (2026-05-03): 1분 영상 → 15분+ 소요 문제 해결
        # MOS 모델은 reference 선택에만 사용 (line 1576-1664), 더빙본 재평가 X
        # 재활성화 원하면: MOS_RESYNTH_ENABLED = True 로 변경
        MOS_RESYNTH_ENABLED = False

        mos_score = 0.0
        retry_count = 0

        audio_chunk = synthesize_segment_cosy(
            text=combined_text,
            ref_audio=ref_path,
            lang=tgt_lang,
            speed=speed,
            emotion=first_seg.emotion,
            tts_context=getattr(first_seg, 'tts_context', '') or '',
            tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
        )'''

new = '''        # POST_TTS_LENGTH_CHECK: speed=1.0 합성 후 길이 측정
        # overflow 시 speed=1.10, 1.15로 재합성하여 자연스럽게 줄이기 시도
        # (time_stretch보다 음질 좋음 — TTS는 phoneme duration 자연 조절, time_stretch는 사후 압축)
        MOS_RESYNTH_ENABLED = False
        mos_score = 0.0
        retry_count = 0

        # 1차: speed=1.0
        audio_chunk = synthesize_segment_cosy(
            text=combined_text,
            ref_audio=ref_path,
            lang=tgt_lang,
            speed=1.0,
            emotion=first_seg.emotion,
            tts_context=getattr(first_seg, 'tts_context', '') or '',
            tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
        )

        # 길이 측정
        cur_duration = len(audio_chunk) / TTS_SAMPLE_RATE

        # overflow check: 1.10x 임계값 (10% 초과 시 speed retry)
        if cur_duration > max_allowed_duration * 1.05:
            speed_candidates = [1.10, 1.15, 1.20]
            best_audio = audio_chunk
            best_dur_diff = abs(cur_duration - max_allowed_duration)

            for try_speed in speed_candidates:
                retry_audio = synthesize_segment_cosy(
                    text=combined_text,
                    ref_audio=ref_path,
                    lang=tgt_lang,
                    speed=try_speed,
                    emotion=first_seg.emotion,
                    tts_context=getattr(first_seg, 'tts_context', '') or '',
                    tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
                )
                retry_dur = len(retry_audio) / TTS_SAMPLE_RATE
                # max_allowed보다 짧거나 같으면 즉시 채택
                if retry_dur <= max_allowed_duration:
                    print(f"  ↳ speed={try_speed} 재합성 성공 ({cur_duration:.2f}s → {retry_dur:.2f}s)")
                    best_audio = retry_audio
                    best_dur_diff = max_allowed_duration - retry_dur  # 음수 = OK
                    break
                # 더 가까워졌으면 best 갱신
                d = abs(retry_dur - max_allowed_duration)
                if d < best_dur_diff:
                    best_dur_diff = d
                    best_audio = retry_audio
                    print(f"  ↳ speed={try_speed} 재합성 (best: {retry_dur:.2f}s vs target {max_allowed_duration:.2f}s)")
            audio_chunk = best_audio'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: post-TTS speed retry 추가 완료")
else:
    print("NOT FOUND - check current state")
