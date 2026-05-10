"""1달 전 working pattern 검증.
- inference_cross_lingual (zero_shot 아님)
- reference를 16kHz mono로 변환 후 전달
- 감정별 instruct + endofprompt
- 음량 normalize
"""
import sys, os, subprocess, tempfile
sys.path.insert(0, '/app')
os.environ['HF_HOME'] = '/app/media/model_cache/huggingface'
os.environ['MODELSCOPE_CACHE'] = '/app/media/model_cache/modelscope'

import numpy as np
import soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice3

print('[V2] Loading CosyVoice3 (local path)...', flush=True)
m = CosyVoice3('/app/media/model_cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512', load_trt=False)
print(f'[V2] sample_rate={m.sample_rate}', flush=True)

ref_audio = '/app/media/reference/SPEAKER_00_Neutral.wav'
text = '안녕하세요. 한국어 더빙 테스트입니다. 자연스럽게 들리는지 확인해주세요.'

# 1달 전 working: reference를 16kHz mono로 변환
ref_16k = '/tmp/ref_16k_v2.wav'
subprocess.run([
    'ffmpeg', '-y', '-i', ref_audio,
    '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', ref_16k
], capture_output=True)
print(f'[V2] ref converted to 16kHz mono: {ref_16k}', flush=True)

# 1달 전 working: cross_lingual + emotion instruct + endofprompt
emotion_instruction = {
    'Neutral': '',
    'Happy': 'Speak with joy and excitement.',
}
instruct = emotion_instruction.get('Neutral', '')
prefix = f'{instruct}<|endofprompt|>' if instruct else 'You are a helpful assistant.<|endofprompt|>'

print('[V2] inference_cross_lingual + 16kHz + prefix...', flush=True)
out = []
for r in m.inference_cross_lingual(
    tts_text=f'{prefix}{text}',
    prompt_wav=ref_16k,
    stream=False, speed=1.0
):
    out.append(r['tts_speech'].squeeze().numpy())

if not out:
    print('[V2] NO OUTPUT', flush=True)
    sys.exit(1)

wav = np.concatenate(out, axis=0).astype(np.float32)
print(f'[V2] raw wav: shape={wav.shape}, max={np.abs(wav).max():.4f}', flush=True)

# 1달 전 working: 음량 normalize
peak = np.max(np.abs(wav))
if peak > 0:
    wav = wav * (0.9 / peak)
print(f'[V2] normalized: max={np.abs(wav).max():.4f}', flush=True)

# int16 PCM으로 저장 (들리게)
out_path = '/workspace/media/output/cosy_test/_V2_korean_clpattern.wav'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
wav_int16 = (wav.clip(-1, 1) * 32767).astype(np.int16)
sf.write(out_path, wav_int16, m.sample_rate, subtype='PCM_16')
print(f'[V2] saved: {out_path}', flush=True)
print(f'[V2] Done. Listen to {out_path}', flush=True)
