"""화자 수 자동 감지 fix: eigengap → silhouette score.

eigengap의 한계:
  - 작은 sample (<10 turns)에서 부정확
  - 첫 번째 gap이 가장 클 때 무조건 1명으로 결정 (3 → 1처럼 과도)

silhouette score:
  - 각 k=2..max_k 마다 SpectralClustering 시도
  - silhouette score (cluster 내부 응집 vs 외부 분리) 계산
  - score 가장 높은 k 선택
  - 모든 k의 score < threshold (0.1) 이면 1명으로 결정
"""
from pathlib import Path

p = Path('/app/orchestrator.py')
src = p.read_text(encoding='utf-8')

old = '''def _estimate_n_speakers_eigengap(embeddings: np.ndarray, max_k: int = 8) -> int:
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
    return max(1, min(n_speakers, max_k))'''

new = '''def _estimate_n_speakers_silhouette(embeddings: np.ndarray, max_k: int = 8, score_threshold: float = 0.1) -> int:
    """Silhouette score로 화자 수 자동 결정 (eigengap보다 작은 sample에서 robust).

    각 k=2..max_k 마다 SpectralClustering 시도 → silhouette score 계산 →
    score 가장 높은 k 선택. 모든 score < threshold면 1명으로 판단.
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < 4:
        return 1

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    affinity = (embeddings @ embeddings.T) / (norms @ norms.T + 1e-9)
    affinity = np.clip(affinity, 0.0, 1.0)
    distance = 1.0 - affinity  # silhouette는 distance 입력
    np.fill_diagonal(distance, 0.0)

    best_k = 1
    best_score = -2.0
    score_log = []
    for k in range(2, min(max_k + 1, n)):
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )
            labels = sc.fit_predict(affinity)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(distance, labels, metric="precomputed"))
            score_log.append(f"k={k}: {score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    print(f"[Diarize] silhouette scores: {{{', '.join(score_log)}}}, best={best_k} (score={best_score:.3f})")

    # 모든 k의 score가 너무 낮으면 (cluster 분리 어려움) 1명으로 결정
    if best_score < score_threshold:
        return 1
    return best_k


def _estimate_n_speakers_eigengap(embeddings: np.ndarray, max_k: int = 8) -> int:
    """eigengap heuristic (silhouette과 cross-check 용도로만 유지).
    실제 호출은 _estimate_n_speakers_silhouette 사용.
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
    return max(1, min(int(np.argmax(gaps)) + 1, max_k))'''

if old in src:
    src = src.replace(old, new)
    print('[1] silhouette 함수 추가 + eigengap 보존')
else:
    print('[1] eigengap 함수 못 찾음')

# diarize 호출 변경: eigengap → silhouette
old_call = '''        n_speakers = _estimate_n_speakers_eigengap(embeddings, max_k=max_k_eff)
        print(f"[Diarize] Spectral eigengap 추정: {n_speakers}명 (max_k={max_k_eff})")'''
new_call = '''        n_speakers = _estimate_n_speakers_silhouette(embeddings, max_k=max_k_eff, score_threshold=0.1)
        print(f"[Diarize] Spectral silhouette 추정: {n_speakers}명 (max_k={max_k_eff})")'''
if old_call in src:
    src = src.replace(old_call, new_call)
    print('[2] diarize 호출 변경 OK')
else:
    print('[2] diarize 호출 패턴 못 찾음')

p.write_text(src, encoding='utf-8')
print('[Done] silhouette 패치 완료')
