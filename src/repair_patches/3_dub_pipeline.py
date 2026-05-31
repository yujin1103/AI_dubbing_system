"""Stage 3 — Full dubbing: VAD-corrected boundary + duration-aware translation + CosyVoice synth.

Pipeline per segment:
  1. Silero VAD on original audio → auto-correct segment.start (prev-end constrained)
  2. LLM 3-candidate translation (short/normal/long) with phoneme budget
  3. Synth all 3 → pick closest to target duration
  4. Iterative correction: if off > tolerance, retry with adjusted budget (max 2 LLM rounds)
  5. Speed fine-tune (0.85-1.50) to fit corrected window
  6. LLM ultra-compress fallback if speed cap reached
Final: composite into video timeline, mux with original video stream → dubbed.mp4

Concurrency:
  LLM_CONCURRENCY (default 10) — parallel LLM HTTP calls
  COSY_CONCURRENCY (default 3) — parallel CosyVoice HTTP calls (daemon is GPU-bound, low parallelism best)

Requirements:
  - CosyVoice3 HTTP daemon running at COSYVOICE_URL
  - LLM API (OpenAI-compatible): VECTORENGINE_API_KEY, VECTORENGINE_BASE_URL, VECTORENGINE_MODEL
  - Silero VAD (pip install silero-vad)
"""
import argparse, base64, io, json, os, re, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
import scipy.signal as sps
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# === Length control hyperparameters ===
PHONEMES_PER_SEC = 14.0      # Korean jamo/sec at natural pace
MAX_SPEED = 1.35              # CosyVoice speed cap (품질 안전선; 극단 speedup 방지)
MIN_SPEED = 0.85
TOLERANCE = 0.15              # ±15% triggers iterative re-translate
FIT_TOLERANCE = 0.10          # ±100ms ok during speed fit

# === VAD boundary refinement ===
SHIFT_BACK = 2.5              # VAD onset can be up to 2.5s before diar.start
SHIFT_FWD = 3.0               # or up to 3.0s after (catches breath/silence-front errors)
MIN_SHIFT = 0.20              # min |shift| to apply

MIN_SYNTH_DUR = 0.2           # segments shorter than this are SKIPPED (0.5→0.2: 짧은 대사 "Maybe"·"Find another" 살림)
COMPOSITE_SR = 44100

DEFAULT_EMOTION = 'Neutral'
DEFAULT_TONE = 'natural'

# Thread-safe printing
_print_lock = threading.Lock()
def log(msg):
    with _print_lock: print(msg, flush=True)


# === Helpers ===
def count_jamo(text):
    n = 0
    for ch in text:
        c = ord(ch)
        if 0xAC00 <= c <= 0xD7A3:
            jong = (c - 0xAC00) % 28
            n += 2 if jong == 0 else 3
    return n

def phoneme_budget(target_dur):
    return max(int(target_dur * PHONEMES_PER_SEC), 4)


# === LLM client with retry ===
class LLM:
    def __init__(self, api_key, base_url, model, max_retries=3, timeout=60):
        self.k = api_key; self.b = base_url; self.m = model
        self.max_retries = max_retries
        self.timeout = timeout

    def chat(self, prompt, max_tokens=500, json_mode=True):
        kwargs = {'model': self.m, 'messages':[{'role':'user','content':prompt}],
                  'max_tokens': max_tokens, 'temperature': 0.3}
        if json_mode: kwargs['response_format'] = {'type':'json_object'}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f'{self.b}/v1/chat/completions',
                    headers={'Authorization': f'Bearer {self.k}'},
                    json=kwargs, timeout=self.timeout)
                if r.status_code == 429:
                    # rate limited — exponential backoff
                    import time as _t; _t.sleep(2 ** attempt)
                    last_err = f'HTTP 429 (attempt {attempt+1})'
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
                raw = r.json()['choices'][0]['message']['content'].strip()
                if raw.startswith('```'):
                    raw = raw.split('```')[1]
                    if raw.startswith('json'): raw = raw[4:]
                    raw = raw.strip().rstrip('`').strip()
                try:
                    return json.loads(raw)
                except Exception:
                    m = re.search(r'\{.*\}', raw, re.S)
                    if not m: raise
                    return json.loads(m.group(0))
            except (requests.RequestException, RuntimeError) as ex:
                last_err = ex
                if attempt < self.max_retries - 1:
                    import time as _t; _t.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f'LLM exhausted retries: {last_err}')


def llm_translate_multi(llm, en_text, target_dur, speaker_desc, override_budget=None,
                        target_lang='Korean', context=''):
    budget = override_budget if override_budget else phoneme_budget(target_dur)
    short_b = max(int(budget * 0.80), 4); long_b = int(budget * 1.20)
    ctx_block = ""
    if context:
        ctx_block = f"""
**대화 맥락** (이 장면의 앞뒤 대사 — 말투·관계·감정 흐름 파악용. 번역하지 말 것):
{context}

위 맥락을 반영해, '>>>' 로 표시된 대사를 문맥에 맞게(말다툼이면 말다툼 톤, 존댓말/반말 일관) 번역하세요.
"""
    target_line = f">>> {en_text}" if context else en_text
    prompt = f"""English source: "{target_line}"
{ctx_block}
영어 → {target_lang} 번역. 의미와 감정·말투 유지, 자연스러운 흐름.

**목표 시간**: {target_dur:.2f}초 (자모 약 {budget}개 ≈ 음절 약 {budget//2}개)
**화자 어조**: {speaker_desc}

3가지 길이 후보 (의미는 같지만 길이만 다르게):
- SHORT  : 약 {short_b} 자모 (핵심만)
- NORMAL : 약 {budget} 자모 (목표 일치)
- LONG   : 약 {long_b} 자모 (풍부한 표현)

규칙:
1. 자연스러운 구어체 (직역 X), 앞뒤 대사와 말투·호칭 일관
2. 감정 표현 (느낌표, 의문문) 유지
3. '>>>' 대사 하나만 번역 (맥락 대사는 번역 금지)
4. JSON 한 줄로만 출력

출력: {{"short":"...","normal":"...","long":"..."}}"""
    obj = llm.chat(prompt)
    out = {}
    for k in ('short','normal','long'):
        v = str(obj.get(k,'')).strip().strip('"').strip()
        if v: out[k] = v
    return out if out else None


def llm_ultra_compress(llm, en_text, budget, speaker_desc, target_lang='Korean'):
    prompt = f"""English: "{en_text}"

영어 → {target_lang} **압축 번역**, **반드시 {budget} 자모 이내**, 핵심 의미만.
화자: {speaker_desc}
부수절/관형어/조사 최대한 생략 OK. JSON 한 줄로:
{{"ko":"..."}}"""
    return llm.chat(prompt, max_tokens=200).get('ko','').strip()


# === CosyVoice HTTP client with semaphore-based concurrency limit ===
class CosyVoice:
    def __init__(self, base_url, concurrency=3, timeout=180):
        self.b = base_url
        self.sem = threading.Semaphore(max(1, concurrency))
        self.timeout = timeout

    def synth(self, text, ref_path, emotion='Neutral', tone='natural', speed=1.0):
        with self.sem:
            try:
                r = requests.post(f'{self.b}/synthesize',
                    json={'text':text,'ref_audio_path':ref_path,
                          'emotion':emotion,'tone':tone,'speed':speed},
                    timeout=self.timeout).json()
            except requests.RequestException as ex:
                return None, f'request_error: {ex}'
        if not r.get('success'): return None, r.get('error')
        return base64.b64decode(r['audio_b64']), None

    def synth_to_dur(self, text, ref_path, emotion='Neutral', tone='natural', speed=1.0):
        """Synth and return (audio_bytes, dur_seconds, sr) or (None, None, None) on failure."""
        audio_b, err = self.synth(text, ref_path, emotion, tone, speed)
        if err: return None, None, None
        arr, sr = sf.read(io.BytesIO(audio_b))
        if arr.ndim > 1: arr = np.mean(arr, axis=1)
        return audio_b, len(arr) / sr, sr


# === VAD boundary refinement (sequential by default — silero_vad model is thread-unsafe) ===
def vad_refine_boundaries(audio_full, sr, segs, vad_model, max_workers=1):
    """Returns dict {idx: (orig_start, new_start)} for segments whose VAD onset differs > MIN_SHIFT.

    Silero VAD model is NOT thread-safe — concurrent get_speech_timestamps() calls cause
    silent native abort (free(): corrupted unsorted chunks). Stays sequential by default.
    """
    # Precompute prev_end per seg
    prev_ends = []
    last_end = 0.0
    for s in segs:
        prev_ends.append(last_end)
        if (s['end'] - s['start']) >= MIN_SYNTH_DUR:
            last_end = s['end']

    def detect_one(i_seg):
        i, seg = i_seg
        s, e = seg['start'], seg['end']
        if (e-s) < MIN_SYNTH_DUR: return None
        search_lo = max(prev_ends[i] + 0.05, s - SHIFT_BACK)
        search_hi = min(s + SHIFT_FWD, e)
        if search_hi <= search_lo: return None
        w_lo = max(search_lo - 0.2, 0.0); w_hi = min(search_hi + 0.3, e)
        sa = audio_full[int(w_lo*sr):int(w_hi*sr)]
        if len(sa) < 0.3*sr: return None
        try:
            ts = get_speech_timestamps(torch.from_numpy(sa).float(), vad_model,
                                      sampling_rate=sr, threshold=0.5,
                                      min_silence_duration_ms=100, min_speech_duration_ms=150)
        except Exception:
            return None
        for t in ts:
            ao = w_lo + t['start']/sr
            if search_lo <= ao <= search_hi:
                if abs(ao - s) >= MIN_SHIFT:
                    return (i, s, ao)
                return None
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(detect_one, list(enumerate(segs))))
    return {i: (orig, new) for r in results if r for (i, orig, new) in [r]}


# === Per-segment iterative fit ===
def synth_and_fit(cosy, llm, seg, ref, emo, tone, spk_desc, avail, target_lang='Korean', context=''):
    """Returns (final_dur, final_ko, audio_bytes) or (None, None, None) on failure."""
    en = seg['text'].strip()
    target_dur = seg['end'] - seg['start']
    # no-lipsync: 번역 예산을 '다음 발화까지의 여유(avail)' 기준으로 — 원본 구간보다 넉넉히(상한 있음).
    # 압축 왜곡 제거. avail 이 작으면(촘촘) 자동으로 작게 유지 → 다음 발화 침범 안 함.
    budget_dur = min(max(avail, target_dur), max(target_dur * 1.8, target_dur + 1.0))

    best = None; used_budget = None
    # Iterative loop (max 2 LLM rounds)
    for attempt in range(2):
        try:
            cands = llm_translate_multi(llm, en, budget_dur, spk_desc,
                                        override_budget=used_budget if attempt > 0 else None,
                                        target_lang=target_lang, context=context)
        except Exception as ex:
            log(f'      LLM error (A{attempt+1}): {ex}')
            break
        if not cands: break

        # Synth all 3 candidates at speed=1.0
        results = []
        for label, ko in cands.items():
            audio_b, dur, _sr = cosy.synth_to_dur(ko, ref, emo, tone, speed=1.0)
            if audio_b is None: continue
            results.append({'label':label,'ko':ko,'dur':dur,'audio':audio_b,
                           'jamo':count_jamo(ko)})
        if not results: break

        pick = min(results, key=lambda x: abs(x['dur'] - budget_dur))
        if best is None or abs(pick['dur']-budget_dur) < abs(best['dur']-budget_dur):
            best = pick
        off = abs(pick['dur'] - budget_dur) / budget_dur
        if off <= TOLERANCE: break
        measured_rate = pick['jamo'] / pick['dur']
        used_budget = max(int(budget_dur * measured_rate), 4)
    if best is None: return None, None, None

    final_ko = best['ko']; final_audio = best['audio']; final_dur = best['dur']

    # Speed fine-tune to fit avail window (iterative)
    cur_speed = 1.0
    for _ in range(4):
        if final_dur <= avail + FIT_TOLERANCE: break
        next_speed = min(cur_speed * (final_dur/avail) * 1.05, MAX_SPEED)
        if abs(next_speed - cur_speed) < 0.02:
            # Cap reached → LLM ultra-compress
            budget = int(avail * MAX_SPEED * PHONEMES_PER_SEC * 0.85)
            try:
                ko2 = llm_ultra_compress(llm, en, budget, spk_desc, target_lang=target_lang)
            except Exception:
                break
            if ko2 and count_jamo(ko2) < count_jamo(final_ko):
                final_ko = ko2; cur_speed = 1.0
                audio_b, dur, _ = cosy.synth_to_dur(final_ko, ref, emo, tone, speed=cur_speed)
                if audio_b is not None:
                    final_audio = audio_b; final_dur = dur
                continue
            break
        cur_speed = next_speed
        audio_b, dur, _ = cosy.synth_to_dur(final_ko, ref, emo, tone, speed=cur_speed)
        if audio_b is None: break
        final_audio = audio_b; final_dur = dur

    # no-lipsync: 짧은 한국어를 원본 길이에 맞추려 인위적으로 늦추지 않음(자연 속도 유지, 뒤는 침묵/배경).
    return final_dur, final_ko, final_audio


# === Main pipeline ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--diarize-json', required=True)
    ap.add_argument('--refs-manifest', required=True, help='manifest.json from stage 2')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--cosyvoice-url', default=os.environ.get('COSYVOICE_URL','http://127.0.0.1:8901'))
    ap.add_argument('--llm-key', default=os.environ.get('VECTORENGINE_API_KEY'))
    ap.add_argument('--llm-base', default=os.environ.get('VECTORENGINE_BASE_URL','https://api.vectorengine.ai'))
    ap.add_argument('--llm-model', default=os.environ.get('VECTORENGINE_MODEL','gpt-5.4'))
    ap.add_argument('--target-lang', default='Korean')
    ap.add_argument('--speaker-config', help='Optional JSON: {"SPEAKER_00":{"desc":...,"emotion":...,"tone":...}, ...}')
    ap.add_argument('--video-duration', type=float, default=None, help='Override (default: probe via ffprobe)')
    ap.add_argument('--bg-segments', default=None,
                    help='JSON of BG segments [{start,end}] to fill with ORIGINAL audio (no translation)')
    ap.add_argument('--bg-stem', default=None,
                    help='보컬 제거된 배경음(no_vocals) wav — 전 구간 연속 bed 로 깔고 그 위에 합성 overlay')
    ap.add_argument('--bg-gain', type=float, default=0.5, help='배경음 bed 볼륨 (default 0.5)')
    ap.add_argument('--synth-gain', type=float, default=0.75, help='한국어 합성 볼륨 (default 0.75, 낮출수록 작아짐)')
    ap.add_argument('--orig-gain', type=float, default=0.7, help='BG passthrough 원본 오디오 볼륨 (default 0.7)')
    ap.add_argument('--duck', type=float, default=0.5, help='합성이 나올 때 배경 bed 를 이 배율로 낮춤(ducking). 1.0=끔')
    ap.add_argument('--llm-concurrency', type=int, default=int(os.environ.get('LLM_CONCURRENCY','10')),
                    help='Parallel segment workers (each may do multiple LLM calls). Default 10.')
    ap.add_argument('--cosy-concurrency', type=int, default=int(os.environ.get('COSY_CONCURRENCY','3')),
                    help='Parallel CosyVoice synth requests. Default 3 (daemon GPU-bound).')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Probe video duration
    if args.video_duration:
        video_dur = args.video_duration
    else:
        r = subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_format',args.video],
                          capture_output=True, text=True, check=True)
        video_dur = float(json.loads(r.stdout)['format']['duration'])
    log(f'Video duration: {video_dur:.2f}s')

    # Load inputs
    with open(args.diarize_json) as f: data = json.load(f)
    segs = data['segments']
    with open(args.refs_manifest) as f: refs_data = json.load(f)
    refs = {r['speaker']: r['ref_path'] for r in refs_data['refs']}
    log(f'Segments: {len(segs)}, Speakers: {len(refs)}')

    # Speaker config
    spk_cfg = {}
    if args.speaker_config and Path(args.speaker_config).exists():
        with open(args.speaker_config) as f: spk_cfg = json.load(f)

    # Extract 16kHz audio for VAD
    audio_16k = out_dir / '_orig_16k.wav'
    if not audio_16k.exists():
        subprocess.run(['ffmpeg','-y','-i',args.video,'-vn','-ac','1','-ar','16000',
                       str(audio_16k)], check=True, capture_output=True)
    audio_full, sr_vad = sf.read(audio_16k)
    assert sr_vad == 16000

    # Stage 1: VAD-correct boundaries (parallel, prev-end constrained)
    vad_model = load_silero_vad()
    corrections = vad_refine_boundaries(audio_full, sr_vad, segs, vad_model)
    log(f'\n=== VAD boundary corrections: {len(corrections)} ===')
    for i, (orig, new) in sorted(corrections.items()):
        log(f'  seg[{i+1}] {segs[i]["speaker"]}: {orig:.2f}→{new:.2f} ({new-orig:+.2f}s)')

    def corrected_start(i): return corrections[i][1] if i in corrections else segs[i]['start']
    def next_non_skip_start(i):
        for j in range(i+1, len(segs)):
            if (segs[j]['end']-segs[j]['start']) >= MIN_SYNTH_DUR:
                return corrected_start(j)
        return video_dur

    # Stage 2-6: per-segment translate + synth + fit (parallel)
    cosy = CosyVoice(args.cosyvoice_url, concurrency=args.cosy_concurrency)
    llm = LLM(args.llm_key, args.llm_base, args.llm_model)
    log(f'\n=== Translate + synth ({args.target_lang}) llm_workers={args.llm_concurrency} cosy_workers={args.cosy_concurrency} ===')

    def process_seg(i_seg):
        i, seg = i_seg
        # BG 화자: 번역/합성 안 함. 최종 조립 시 원본 오디오로 채움(passthrough).
        if seg['speaker'].startswith('SPEAKER_BG'):
            log(f'  [{i+1}] BG passthrough (no translate) "{seg["text"][:30]}"')
            return None
        target_dur = seg['end'] - seg['start']
        if target_dur < MIN_SYNTH_DUR:
            log(f'  [{i+1}] SKIP ({target_dur:.2f}s) "{seg["text"][:30]}"')
            return None
        spk = seg['speaker']
        if spk not in refs:
            log(f'  [{i+1}] no ref for {spk} → skip'); return None
        ref = refs[spk]
        cfg = spk_cfg.get(spk, {})
        emo = cfg.get('emotion', DEFAULT_EMOTION)
        tone = cfg.get('tone', DEFAULT_TONE)
        spk_desc = cfg.get('desc', spk)
        new_st = corrected_start(i)
        next_st = next_non_skip_start(i)
        avail = next_st - new_st - 0.03
        # 문맥 인식 번역: 앞 3 + 뒤 2 대사(화자+영어)를 맥락으로 전달 → 말투/흐름 일관(말다툼 톤 등).
        ctx_lines = []
        for j in range(max(0, i - 3), min(len(segs), i + 3)):
            if j == i:
                continue
            t = segs[j].get('text', '').strip()
            if t:
                ctx_lines.append(f"  ({segs[j]['speaker']}) {t}")
        context = "\n".join(ctx_lines)
        log(f'  [{i+1}] {spk} t={target_dur:.2f}s avail={avail:.2f}s EN="{seg["text"][:50]}"')
        try:
            fdur, fko, faudio = synth_and_fit(cosy, llm, seg, ref, emo, tone, spk_desc,
                                              avail, target_lang=args.target_lang, context=context)
        except Exception as ex:
            log(f'  [{i+1}] FAIL: {ex}'); return None
        if fdur is None:
            log(f'  [{i+1}] no candidate produced'); return None
        clone_path = out_dir / f'seg_{i:02d}_{spk}.wav'
        with open(clone_path,'wb') as f: f.write(faudio)
        log(f'  [{i+1}] ✓ {fdur:.2f}s "{fko[:50]}"')
        return {'clone':{'idx':i,'speaker':spk,'start':new_st,'end':seg['end'],
                         'next_start':next_st,'path':str(clone_path)},
                'log':{'idx':i,'speaker':spk,'en':seg['text'],'ko':fko,
                       'target':target_dur,'final':fdur,'corrected_start':new_st}}

    with ThreadPoolExecutor(max_workers=max(1, args.llm_concurrency)) as ex:
        results = list(ex.map(process_seg, list(enumerate(segs))))
    clones = []; translations_log = []
    for r in results:
        if r is None: continue
        clones.append(r['clone']); translations_log.append(r['log'])
    clones.sort(key=lambda c: c['idx'])
    translations_log.sort(key=lambda t: t['idx'])

    # Stage 7: composite + mux — 배경음 bed(연속) + 한국어 합성 overlay(ducking) + BG 원본 passthrough
    log(f'\n=== Composite ({len(clones)} clips) ===')
    N = int(video_dur * COMPOSITE_SR)

    # (1) 배경음 bed: 보컬 제거 stem 을 전 구간 연속으로 깔기 (없으면 무음 bed = 기존 동작)
    bed = np.zeros(N, dtype=np.float32)
    if args.bg_stem and os.path.exists(args.bg_stem):
        b, sr_ = sf.read(args.bg_stem)
        if b.ndim > 1: b = np.mean(b, axis=1)
        if sr_ != COMPOSITE_SR:
            b = sps.resample_poly(b, COMPOSITE_SR, sr_).astype(np.float32)
        m = min(N, len(b)); bed[:m] = b[:m].astype(np.float32) * args.bg_gain
        log(f'  background bed: {os.path.basename(args.bg_stem)} x{args.bg_gain}')
    else:
        log(f'  background bed: (none — bg_stem 미지정, 합성 구간 무음)')

    # (2) 한국어 합성 overlay (볼륨 args.synth_gain) + 합성 존재 마스크(ducking 용)
    synth = np.zeros(N, dtype=np.float32)
    voice_mask = np.zeros(N, dtype=np.float32)
    cuts = []
    for c in clones:
        a, sr_ = sf.read(c['path'])
        if a.ndim > 1: a = np.mean(a, axis=1)
        if sr_ != COMPOSITE_SR:
            a = sps.resample_poly(a, COMPOSITE_SR, sr_).astype(np.float32)
        s_idx = int(c['start'] * COMPOSITE_SR); e_idx = s_idx + len(a)
        max_e = int(c['next_start'] * COMPOSITE_SR) - int(0.03*COMPOSITE_SR)
        if e_idx > max_e:
            cut_len = max_e - s_idx
            if cut_len >= int(0.3*COMPOSITE_SR):
                cuts.append((c['idx'], (e_idx-max_e)/COMPOSITE_SR))
                a = a[:cut_len].copy()
                fn = int(0.05*COMPOSITE_SR); a[-fn:] *= np.linspace(1,0,fn)
                e_idx = s_idx + len(a)
        if e_idx > N: e_idx = N; a = a[:e_idx-s_idx]
        synth[s_idx:e_idx] += a.astype(np.float32) * args.synth_gain
        voice_mask[s_idx:e_idx] = 1.0

    # (3) ducking: 합성이 나오는 구간은 배경 bed 를 args.duck 배율로 낮춰 대사 명료도 확보
    if args.duck < 1.0:
        # 마스크를 살짝 부드럽게(50ms) → 급격한 볼륨 점프 방지
        w = max(1, int(0.05 * COMPOSITE_SR))
        sm = np.convolve(voice_mask, np.ones(w)/w, mode='same')
        bed = bed * (1.0 - (1.0 - args.duck) * np.clip(sm, 0, 1))

    composite = bed + synth

    # (4) BG passthrough: BG 구간(+--bg-segments)을 원본 오디오로 덮어쓰기(배경+원본음성 그대로)
    bg_ranges = [(float(s['start']), float(s['end'])) for s in segs
                 if s['speaker'].startswith('SPEAKER_BG')]
    if args.bg_segments:
        try:
            extra = json.load(open(args.bg_segments))
            extra = extra.get('segments', extra) if isinstance(extra, dict) else extra
            bg_ranges += [(float(b['start']), float(b['end'])) for b in extra]
        except Exception as ex:
            log(f'  ⚠ --bg-segments load fail: {ex}')
    if bg_ranges:
        oa = sps.resample_poly(audio_full, COMPOSITE_SR, sr_vad).astype(np.float32)
        for bs, be in bg_ranges:
            s_idx = int(bs * COMPOSITE_SR); e_idx = min(int(be * COMPOSITE_SR), N, len(oa))
            if e_idx > s_idx:
                composite[s_idx:e_idx] = oa[s_idx:e_idx] * args.orig_gain  # 원본(배경+음성) 그대로
        log(f'  BG passthrough: {len(bg_ranges)} ranges → 원본 오디오 x{args.orig_gain}')

    peak = float(np.max(np.abs(composite)) + 1e-9)
    if peak > 0.95: composite *= 0.95/peak
    if cuts:
        log(f'  ⚠ residual cuts: {cuts}')
    else:
        log(f'  ✓ no cuts')

    dub_wav = out_dir / 'dub_audio.wav'
    sf.write(dub_wav, composite, COMPOSITE_SR)
    out_video = out_dir / 'dubbed.mp4'
    subprocess.run(['ffmpeg','-y','-i',args.video,'-i',str(dub_wav),
                   '-c:v','copy','-c:a','aac','-b:a','192k',
                   '-map','0:v:0','-map','1:a:0','-shortest', str(out_video)],
                  check=True, capture_output=True)

    # Save translations + corrected baseline
    with open(out_dir/'translations.json','w') as f:
        json.dump({'translations': translations_log}, f, indent=2, ensure_ascii=False)
    nd = {'segments': []}
    for i, s in enumerate(segs):
        if i in corrections:
            s2 = dict(s); s2['start'] = corrections[i][1]
            s2['original_start'] = corrections[i][0]; s2['vad_corrected'] = True
            nd['segments'].append(s2)
        else: nd['segments'].append(s)
    with open(out_dir/'baseline_vad_corrected.json','w') as f:
        json.dump(nd, f, indent=2, ensure_ascii=False)

    log(f'\n✓ {out_video}')
    log(f'✓ {out_dir/"translations.json"}')
    log(f'✓ {out_dir/"baseline_vad_corrected.json"}')

if __name__ == '__main__':
    main()
