"""화자 분리 강화: Spectral Clustering + Eigengap heuristic.

논문 표준: Normalized Laplacian eigenvalue gap에서 화자 수 자동 결정.
ECAPA-TDNN embedding을 affinity matrix로 사용.

기존 _refine_speakers_with_ecapa (단순 cosine threshold)를
_refine_speakers_with_spectral (Spectral + eigengap)로 교체.
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

# 기존 ECAPA 함수 본문 통째로 교체
old = '''def _refine_speakers_with_ecapa(diarization, vocals_path, threshold=0.5):
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
        return diarization'''

new = '''def _estimate_n_speakers_eigengap(embeddings: np.ndarray, max_k: int = 8) -> int:
    """Normalized Laplacian eigenvalue gap heuristic.
    speaker diarization 표준 방법 (Ng-Jordan-Weiss 2002).

    L_norm = I - D^(-1/2) A D^(-1/2)
    eigenvalue 정렬 후 가장 큰 gap의 인덱스가 화자 수.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    A = (embeddings @ embeddings.T) / (norms @ norms.T + 1e-9)
    A = np.clip(A, 0.0, 1.0)
    np.fill_diagonal(A, 0.0)

    d = A.sum(axis=1)
    d[d == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt

    try:
        eigvals = np.sort(np.linalg.eigvalsh(L_norm))
    except np.linalg.LinAlgError:
        return 1

    eigvals = eigvals[: max_k + 1]
    if len(eigvals) < 2:
        return 1

    gaps = np.diff(eigvals)
    n_speakers = int(np.argmax(gaps)) + 1
    return max(1, min(n_speakers, max_k))


def _refine_speakers_with_ecapa(diarization, vocals_path, threshold=0.5, max_speakers=8):
    """Spectral Clustering + Eigengap heuristic 후처리.
    (이름은 호환 위해 유지, 내부는 spectral 사용)

    1. ECAPA-TDNN으로 turn별 embedding 추출
    2. Normalized Laplacian eigenvalue gap → 화자 수 N 자동 결정
    3. SpectralClustering(N)으로 turn → cluster 할당
    4. cluster를 SPEAKER_00, SPEAKER_01 ... 로 재명명
    """
    ecapa = _load_ecapa()
    if ecapa is None:
        return diarization

    try:
        audio, sr = sf.read(vocals_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000

        # turn별 embedding 추출 (1초 이상만)
        turns = []
        embeddings = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.end - turn.start < 1.0:
                continue
            start_s = max(0, int(turn.start * sr))
            end_s = min(len(audio), int(turn.end * sr))
            clip = audio[start_s:end_s]
            if len(clip) < sr * 0.5:
                continue
            clip_t = torch.from_numpy(clip).float().unsqueeze(0)
            with torch.no_grad():
                emb = ecapa.encode_batch(clip_t).squeeze().cpu().numpy()
            turns.append(turn)
            embeddings.append(emb)

        if len(embeddings) < 2:
            print("[Diarize] Spectral: turn 부족 → 원본 유지")
            return diarization

        embeddings = np.stack(embeddings)
        n_before = len(set(s for _, _, s in diarization.itertracks(yield_label=True)))

        # Eigengap heuristic으로 화자 수 자동 결정
        max_k_eff = min(max_speakers, len(embeddings) - 1)
        n_speakers = _estimate_n_speakers_eigengap(embeddings, max_k=max_k_eff)
        print(f"[Diarize] Spectral eigengap 추정: {n_speakers}명 (max_k={max_k_eff})")

        # 1명이면 모두 SPEAKER_00
        from pyannote.core import Annotation, Segment
        if n_speakers == 1:
            new_anno = Annotation()
            for turn, _, _ in diarization.itertracks(yield_label=True):
                new_anno[turn] = "SPEAKER_00"
            print(f"[Diarize] Spectral 후처리: {n_before} → 1 화자")
            return new_anno

        # SpectralClustering 적용
        from sklearn.cluster import SpectralClustering
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        affinity = (embeddings @ embeddings.T) / (norms @ norms.T + 1e-9)
        affinity = np.clip(affinity, 0.0, 1.0)

        sc = SpectralClustering(
            n_clusters=n_speakers,
            affinity="precomputed",
            random_state=42,
            assign_labels="kmeans",
        )
        labels = sc.fit_predict(affinity)

        # turn → cluster label
        turn_label_map = {(t.start, t.end): int(lab) for t, lab in zip(turns, labels)}

        new_anno = Annotation()
        for turn, _, _ in diarization.itertracks(yield_label=True):
            key = (turn.start, turn.end)
            if key in turn_label_map:
                lab = turn_label_map[key]
            else:
                # 짧은 turn → 시작 시간 가장 가까운 큰 turn의 label
                nearest = min(turn_label_map.keys(), key=lambda k: abs(k[0] - turn.start))
                lab = turn_label_map[nearest]
            new_anno[turn] = f"SPEAKER_{lab:02d}"

        print(f"[Diarize] Spectral 후처리: {n_before} → {n_speakers} 화자")
        return new_anno

    except Exception as e:
        import traceback
        print(f"[Diarize] Spectral 후처리 실패 (원본 유지): {e}")
        traceback.print_exc()
        return diarization'''

if old in src:
    src = src.replace(old, new)
    p.write_text(src, encoding='utf-8')
    print('OK: Spectral + eigengap 적용')
else:
    print('NOT FOUND')
