"""화자 분리 강화 패치 (1달 전 orchestrator.py 기반).

변경:
1. pyannote 모델: community-1 → speaker-diarization-3.1
2. ECAPA-TDNN centroid clustering 후처리 (같은 화자 합치기)
3. _get_speaker_at SPEAKER_UNK fallback 개선 (가장 가까운 turn 화자)
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# ============================================================
# 1. pyannote 모델 변경
# ============================================================
old_model = '"pyannote/speaker-diarization-community-1"'
new_model = '"pyannote/speaker-diarization-3.1"'
if old_model in src:
    src = src.replace(old_model, new_model)
    print('[1] pyannote 3.1 적용 OK')
else:
    print(f'[1] {old_model} 못 찾음')

# ============================================================
# 2. _get_speaker_at fallback 개선 (SPEAKER_UNK 대신 nearest)
# ============================================================
old_fn = '''def _get_speaker_at(diarization, time: float) -> str:
    """특정 시간에 말하는 화자 반환. pyannote 없으면 SPEAKER_00."""
    if diarization is None:
        return "SPEAKER_00"
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= time <= turn.end:
            return speaker
    return "SPEAKER_UNK"'''

new_fn = '''def _get_speaker_at(diarization, time: float) -> str:
    """특정 시간에 말하는 화자 반환. 직접 매칭 안 되면 가장 가까운 turn의 화자."""
    if diarization is None:
        return "SPEAKER_00"
    # 1) 정확한 match
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= time <= turn.end:
            return speaker
    # 2) 가장 가까운 turn 의 화자 (UNK 대신)
    nearest_speaker = None
    nearest_dist = float("inf")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dist = min(abs(time - turn.start), abs(time - turn.end))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_speaker = speaker
    return nearest_speaker if nearest_speaker else "SPEAKER_00"'''

if old_fn in src:
    src = src.replace(old_fn, new_fn)
    print('[2] _get_speaker_at fallback 개선 OK')
else:
    print('[2] _get_speaker_at 패턴 못 찾음')

# ============================================================
# 3. ECAPA-TDNN 후처리 함수 추가 + diarize 함수에 통합
# ============================================================
ecapa_fn = '''
# ─── ECAPA-TDNN centroid clustering 후처리 ───────────────
_ecapa_model = None

def _load_ecapa():
    """ECAPA-TDNN 화자 임베딩 모델 로드 (1회만)."""
    global _ecapa_model
    if _ecapa_model is not None:
        return _ecapa_model
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        _ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/ecapa_savedir",
            run_opts={"device": DEVICE},
        )
        print("[Diarize] ECAPA-TDNN 로드 완료 ✅")
    except Exception as e:
        print(f"[Diarize] ECAPA 로드 실패 (후처리 스킵): {e}")
        _ecapa_model = None
    return _ecapa_model


def _refine_speakers_with_ecapa(diarization, vocals_path, threshold=0.5):
    """pyannote 결과를 ECAPA-TDNN embedding 기반으로 후처리.
    같은 음성 특성 가진 화자끼리 cosine similarity > threshold 면 병합.
    """
    ecapa = _load_ecapa()
    if ecapa is None:
        return diarization

    try:
        audio, sr = sf.read(vocals_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000

        # 화자별 embedding 모으기 (각 turn마다 1개)
        speaker_embeddings = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration < 1.0:  # 1초 미만 무시 (너무 짧으면 noisy)
                continue
            start_s = max(0, int(turn.start * sr))
            end_s = min(len(audio), int(turn.end * sr))
            clip = audio[start_s:end_s]
            if len(clip) < sr * 0.5:
                continue
            clip_t = torch.from_numpy(clip).float().unsqueeze(0)
            with torch.no_grad():
                emb = ecapa.encode_batch(clip_t).squeeze().cpu().numpy()
            speaker_embeddings.setdefault(speaker, []).append(emb)

        if not speaker_embeddings:
            print("[Diarize] ECAPA: embedding 추출 실패 → 원본 유지")
            return diarization

        # 화자별 centroid
        centroids = {sp: np.mean(embs, axis=0) for sp, embs in speaker_embeddings.items()}
        speakers = list(centroids.keys())

        # Union-Find로 cosine sim > threshold 합치기
        parent = {sp: sp for sp in speakers}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(speakers)):
            for j in range(i+1, len(speakers)):
                ci, cj = centroids[speakers[i]], centroids[speakers[j]]
                sim = float(np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-9))
                if sim > threshold:
                    union(speakers[i], speakers[j])
                    print(f"[Diarize] ECAPA 병합: {speakers[i]} ↔ {speakers[j]} (sim={sim:.3f})")

        # 합친 결과로 SPEAKER_00, SPEAKER_01, ... 새로 매핑
        unique_roots = sorted(set(find(sp) for sp in speakers))
        rename_map = {root: f"SPEAKER_{i:02d}" for i, root in enumerate(unique_roots)}
        final_map = {sp: rename_map[find(sp)] for sp in speakers}

        n_before = len(speakers)
        n_after = len(unique_roots)
        print(f"[Diarize] ECAPA 후처리: {n_before} → {n_after} 화자")

        # pyannote rename_labels
        if hasattr(diarization, "rename_labels"):
            return diarization.rename_labels(final_map)
        else:
            # 호환성: 새 Annotation 만들기
            from pyannote.core import Annotation
            new_anno = Annotation()
            for turn, track_id, speaker in diarization.itertracks(yield_label=True):
                new_anno[turn, track_id] = final_map.get(speaker, speaker)
            return new_anno

    except Exception as e:
        print(f"[Diarize] ECAPA 후처리 실패 (원본 유지): {e}")
        return diarization


'''

# 함수를 diarize 함수 직전에 삽입
diarize_marker = "def diarize(vocals_path: str, num_speakers: int = None) -> list:"
if diarize_marker in src and "_refine_speakers_with_ecapa" not in src:
    src = src.replace(diarize_marker, ecapa_fn + diarize_marker)
    print('[3] ECAPA 함수 추가 OK')
else:
    print('[3] ECAPA 추가 위치 못 찾음 (이미 있거나)')

# ============================================================
# 4. diarize 함수에서 ECAPA 후처리 호출
# ============================================================
old_diarize = '''    # pyannote 4.x: DiarizeOutput 객체 → Annotation 추출
    if hasattr(result, "speaker_diarization"):
        return result.speaker_diarization
    return result'''

new_diarize = '''    # pyannote 4.x: DiarizeOutput 객체 → Annotation 추출
    if hasattr(result, "speaker_diarization"):
        diarization = result.speaker_diarization
    else:
        diarization = result

    # ECAPA-TDNN centroid clustering 후처리 (같은 화자 병합)
    diarization = _refine_speakers_with_ecapa(diarization, vocals_path, threshold=0.5)
    return diarization'''

if old_diarize in src:
    src = src.replace(old_diarize, new_diarize)
    print('[4] diarize 함수에 ECAPA 통합 OK')
else:
    print('[4] diarize 패턴 못 찾음')

# ============================================================
# 저장
# ============================================================
p.write_text(src, encoding='utf-8')
print('[Done] orchestrator.py 패치 완료')
