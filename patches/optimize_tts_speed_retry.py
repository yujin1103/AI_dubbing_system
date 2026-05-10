"""TTS 속도 최적화 1: speed retry 효율화.

현재 문제:
  speed=1.0 결과가 target과 너무 다르면 (예: 24s vs 4.7s) speed retry 무의미
  → 1.10, 1.15, 1.20 3번 시도 = 약 3분 낭비

해결:
  1. 첫 결과 비율 따라 retry 결정:
     - ratio < 1.05: retry 안 함 (이미 OK 또는 거의 OK)
     - 1.05 <= ratio < 1.40: 1.15만 시도 (1번)
     - ratio >= 1.40: retry 무의미 → 즉시 time_stretch로
  2. 예측 단계에서 너무 긴 거 미리 발견: pre-flight check
"""
from pathlib import Path

p = Path('/workspace/orchestrator.py')
src = p.read_text()

old = '''        # 길이 측정
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

new = '''        # 길이 측정 + 효율적 speed retry (OPTIMIZED)
        cur_duration = len(audio_chunk) / TTS_SAMPLE_RATE
        ratio = cur_duration / max(0.1, max_allowed_duration)

        # ratio 따라 retry 전략 결정 (불필요한 합성 회피)
        if ratio < 1.05:
            pass  # 이미 OK, retry 안 함
        elif ratio < 1.40:
            # 1.15x 한 번만 시도 (speed retry로 해결 가능한 범위)
            retry_audio = synthesize_segment_cosy(
                text=combined_text,
                ref_audio=ref_path,
                lang=tgt_lang,
                speed=1.15,
                emotion=first_seg.emotion,
                tts_context=getattr(first_seg, 'tts_context', '') or '',
                tts_emotion=getattr(first_seg, 'tts_emotion', '') or '',
            )
            retry_dur = len(retry_audio) / TTS_SAMPLE_RATE
            if retry_dur < cur_duration:
                print(f"  ↳ speed=1.15 재합성 ({cur_duration:.2f}s → {retry_dur:.2f}s)")
                audio_chunk = retry_audio
        else:
            # ratio >= 1.40: speed retry로 해결 불가능한 격차 → 즉시 time_stretch fallback
            print(f"  ↳ ratio={ratio:.2f} 너무 큼 → speed retry skip, "
                  f"time_stretch + trim으로 fallback")'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src)
    print("OK: speed retry 최적화 (3 retries → 0~1 retry)")
else:
    print("NOT FOUND")
