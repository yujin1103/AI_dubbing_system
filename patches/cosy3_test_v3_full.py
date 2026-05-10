"""4.51.3 환경에서 본 더빙 segments 전체 재합성 → 비교용.

orchestrator의 synthesize_segment_cosy 함수 직접 호출.
이전 더빙의 7개 segments (각각 emotion 다름) 합성.
"""
import sys, os, subprocess, tempfile
sys.path.insert(0, '/app')
os.environ['HF_HOME'] = '/app/media/model_cache/huggingface'
os.environ['MODELSCOPE_CACHE'] = '/app/media/model_cache/modelscope'

import numpy as np
import soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice3

print('[V3] Loading CosyVoice3 (transformers 4.51.3)...', flush=True)
m = CosyVoice3(
    '/app/media/model_cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    load_trt=False,
)
print(f'[V3] sample_rate={m.sample_rate}', flush=True)

# 본 더빙의 7개 segments (text, emotion)
segments = [
    ('어서 물어보세요 그 상처에', 'Angry'),
    ('저스틴은 구멍 난 것 같아 궁금해', 'Neutral'),
    ('칼로 만든 것일 수도 있지만 작고 둥글고 깨끗하다면 더 작습니다', 'Neutral'),
    ('총알구멍보다 더 깨끗해 마치 화살이 그에게 맞은 것처럼 보이거나', 'Sad'),
    ('볼트 너가 하던 이런 짓을 내가 해야 했을 때', 'Neutral'),
    ('넌 정말 내가 그랬다고 생각하지 아니 근데 다른 사람들은 그렇게 내가 만', 'Sad'),
    ('물론 내가 그 사람을 죽였다면 눈에 띄게 죽였을 거야', 'Neutral'),
]

emotion_instruction = {
    'Angry':     'Speak with anger and intensity.',
    'Sad':       'Speak with sadness and sorrow.',
    'Happy':     'Speak with joy and excitement.',
    'Surprised': 'Speak with surprise and wonder.',
    'Scared':    'Speak with fear and anxiety.',
    'Neutral':   '',
}

REF_DIR = '/app/media/reference'
OUT_DIR = '/workspace/media/output/v3_dubbed_4513'
os.makedirs(OUT_DIR, exist_ok=True)

all_audio = []
for i, (text, emotion) in enumerate(segments):
    ref_audio = os.path.join(REF_DIR, f'SPEAKER_00_{emotion}.wav')
    if not os.path.exists(ref_audio):
        ref_audio = os.path.join(REF_DIR, 'SPEAKER_00_Neutral.wav')

    instruct = emotion_instruction.get(emotion, '')
    prefix = f'{instruct}<|endofprompt|>' if instruct else 'You are a helpful assistant.<|endofprompt|>'

    ref_16k = f'/tmp/ref_16k_v3_{i}.wav'
    subprocess.run([
        'ffmpeg', '-y', '-i', ref_audio,
        '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', ref_16k
    ], capture_output=True)

    print(f'[V3] seg {i} [{emotion}] ref={os.path.basename(ref_audio)} → "{text[:30]}..."', flush=True)

    out = []
    for r in m.inference_cross_lingual(
        tts_text=f'{prefix}{text}',
        prompt_wav=ref_16k,
        stream=False,
        speed=1.0
    ):
        out.append(r['tts_speech'].squeeze().numpy())

    if not out:
        print(f'[V3]   → NO OUTPUT', flush=True)
        continue

    wav = np.concatenate(out, axis=0).astype(np.float32)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav * (0.9 / peak)

    # 개별 segment 저장 (int16)
    out_path = os.path.join(OUT_DIR, f'seg_{i}_{emotion}.wav')
    sf.write(out_path, (wav.clip(-1, 1) * 32767).astype(np.int16), m.sample_rate, subtype='PCM_16')
    print(f'[V3]   → {wav.shape[0]} samples ({wav.shape[0]/m.sample_rate:.2f}s) saved', flush=True)

    # 0.5초 무음 + 다음 segment
    all_audio.append(wav)
    all_audio.append(np.zeros(int(0.5 * m.sample_rate), dtype=np.float32))

    if os.path.exists(ref_16k):
        os.unlink(ref_16k)

# 전체 합쳐서 저장
if all_audio:
    full = np.concatenate(all_audio)
    full_path = os.path.join(OUT_DIR, '_FULL_korean_4513.wav')
    sf.write(full_path, (full.clip(-1, 1) * 32767).astype(np.int16), m.sample_rate, subtype='PCM_16')
    print(f'[V3] FULL: {full.shape[0]} samples ({full.shape[0]/m.sample_rate:.2f}s)', flush=True)
    print(f'[V3] saved: {full_path}', flush=True)

print(f'[V3] Done. Listen to {OUT_DIR}/_FULL_korean_4513.wav', flush=True)
