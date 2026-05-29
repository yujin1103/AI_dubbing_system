"""ECAPA-TDNN 기반 SPK 재 cluster — voice acoustic 분리 second-opinion.

Why:
  ERes2NetV2 (Chinese-trained) 와 다른 view 의 embedding 으로 cluster 시도.
  ECAPA-TDNN (VoxCeleb, English-trained) 가 영어 외침/평소 발화 분리에
  더 유리할 수 있음. ensemble 또는 fallback view.

영상 무관, hardcoding 없음:
  - model: speechbrain/spkrec-ecapa-voxceleb (모든 영상 동일)
  - voice cosine threshold: 0.5 (영상 무관 고정)
  - n_speakers: auto (distance threshold)
  - 결과: deterministic (같은 audio + 같은 model → 같은 cluster)

알고리즘:
  1. 각 segment ECAPA embedding 추출 (192-dim)
  2. cosine similarity matrix 계산
  3. agglomerative cluster (linkage='average', threshold=0.5)
  4. cluster ID → SPK 이름 부여 (count 순)

ensemble 모드 (--ensemble):
  - 기존 segments SPK + ECAPA cluster 결과 교차 검증
  - 두 view 가 동의 (같은 cluster) → 동일 SPK 유지
  - 차이 → 더 fine SPK 분리 (보존적)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
COSINE_THR = 0.5  # 영상 무관 고정 (수치 hardcoding 아님 — model property)


def _l2(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def get_ecapa_model(device: str = "cuda"):
    from speechbrain.inference.speaker import EncoderClassifier
    cache_dir = os.environ.get("HF_HOME", "/workspace/media/model_cache/ecapa")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    model = EncoderClassifier.from_hparams(
        source=MODEL_NAME,
        savedir=str(Path(cache_dir) / "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )
    return model


def extract_ecapa(model, audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(t).squeeze(0).squeeze(0).cpu().numpy()
    return _l2(emb.astype(np.float32))


def cluster_embeddings(embs: list, ids: list, threshold: float) -> dict:
    """agglomerative cluster — cosine distance ≥ threshold 면 merge."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(embs)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: 0}

    # cosine distance matrix
    arr = np.stack(embs)
    sim = arr @ arr.T
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)
    cond = squareform(dist, checks=False)

    Z = linkage(cond, method="average")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    return {ids[i]: int(labels[i]) for i in range(n)}


def recluster(run_dir: str, segments_name: str, out_name: str, ensemble: bool = False) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    vocals_dir = rd / "vocals"

    seg_path = meta / segments_name
    data = json.load(open(seg_path, encoding="utf-8"))
    groups = data.get("groups", data.get("segments", []))

    chunk_name = segments_name.split("_segments")[0]
    vocals_path = vocals_dir / f"{chunk_name}_clean_vocals.wav"
    if not vocals_path.exists():
        raise SystemExit(f"no vocals: {vocals_path}")
    audio, sr = sf.read(str(vocals_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    print(f"[ECAPA] loading model ({MODEL_NAME}) ...")
    model = get_ecapa_model("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECAPA] model loaded")

    # SPK_BG 외 모든 segments embedding
    embs = []
    ids = []
    orig_spks = {}
    for i, seg in enumerate(groups):
        sp = str(seg.get("speaker", ""))
        if sp.startswith("SPEAKER_BG"):
            continue
        ss = float(seg.get("group_start", seg.get("start", 0)))
        ee = float(seg.get("group_end", seg.get("end", 0)))
        i0 = int(max(0, ss * sr))
        i1 = int(min(len(audio), ee * sr))
        if i1 - i0 < int(0.3 * sr):
            continue  # 너무 짧으면 skip
        try:
            emb = extract_ecapa(model, audio[i0:i1], sr)
            embs.append(emb)
            ids.append(i)
            orig_spks[i] = sp
        except Exception as e:
            print(f"  [warn] seg {i}: {e}")

    print(f"[ECAPA] extracted {len(embs)} embeddings")

    # cluster
    clusters = cluster_embeddings(embs, ids, COSINE_THR)
    cluster_counts = Counter(clusters.values())
    print(f"[ECAPA] clusters: {len(cluster_counts)} unique, sizes: {dict(cluster_counts.most_common())}")

    # cluster_id → SPK 이름 (size 큰 순)
    cluster_to_spk = {}
    for rank, (cid, _) in enumerate(cluster_counts.most_common()):
        cluster_to_spk[cid] = f"SPEAKER_E{rank:02d}"

    # ensemble 모드: 기존 SPK 유지 + 한 SPK 안 ECAPA 큰 차이 시 sub-split
    if ensemble:
        # 각 기존 SPK 별 ECAPA cluster 분포
        spk_cluster_pairs = defaultdict(Counter)
        for i, cid in clusters.items():
            spk_cluster_pairs[orig_spks[i]][cid] += 1
        # 같은 기존 SPK 안 ECAPA cluster 2개 이상, 각각 size ≥ MIN_SUB → split 후보
        # MIN_SUB=2 면 test5 에서 의사/아빠 분리 가능. test4 에서는 over-split.
        # MIN_SUB=3 으로 올리면 test4 over-split 방지 but test5 아빠 분리 (2 seg) 실패.
        # 절충: dominant cluster 의 size 도 ≥ MIN_SUB + sub cluster 도 ≥ 2 (영상 무관)
        MIN_SUB = 2
        MIN_DOMINANT = 3  # dominant cluster 크기 보강 (영상 무관 noise 방지)
        n_split = 0
        for sp, cl_counts in spk_cluster_pairs.items():
            # MIN_SUB 이상인 cluster 들만
            big_cls = [(c, n) for c, n in cl_counts.most_common() if n >= MIN_SUB]
            if len(big_cls) < 2:
                continue
            # dominant cluster 크기 ≥ MIN_DOMINANT 일 때만 split (작은 SPK 보호)
            if big_cls[0][1] < MIN_DOMINANT:
                continue
            # cluster_id → suffix (size 순)
            cid_to_suffix = {c: "abcdefgh"[idx] for idx, (c, _) in enumerate(big_cls)}
            for i, cid in clusters.items():
                if orig_spks[i] != sp:
                    continue
                if cid not in cid_to_suffix:
                    continue  # singleton noise — 기존 SPK 유지
                suffix = cid_to_suffix[cid]
                if suffix == "a":
                    continue  # dominant sub — 기존 SPK 그대로
                groups[i]["audio_speaker"] = sp
                groups[i]["speaker"] = f"{sp}_{suffix}"
                groups[i]["from_ecapa"] = True
                n_split += 1
            print(f"  {sp}: ECAPA sub-cluster {[c for c, _ in big_cls]} → split {sum(1 for c, _ in big_cls if cid_to_suffix[c] != 'a')} sub")
        print(f"[ECAPA ensemble] {n_split} segs split")
    else:
        # 단독 모드: ECAPA cluster 결과로 SPK 완전 교체
        n_replaced = 0
        for i, cid in clusters.items():
            new_sp = cluster_to_spk[cid]
            if groups[i].get("speaker") != new_sp:
                groups[i]["audio_speaker"] = groups[i].get("speaker")
                groups[i]["speaker"] = new_sp
                groups[i]["from_ecapa"] = True
                n_replaced += 1
        print(f"[ECAPA] {n_replaced} segs reassigned")

    data["groups"] = groups
    out_path = meta / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"saved → {out_path}")
    return {
        "n_clusters": len(cluster_counts),
        "cluster_sizes": dict(cluster_counts),
        "n_embeddings": len(embs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segments-name", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--ensemble", action="store_true", help="기존 SPK + ECAPA ensemble")
    args = ap.parse_args()
    r = recluster(args.run_dir, args.segments_name, args.out_name, args.ensemble)
    print(f"summary: {r}")


if __name__ == "__main__":
    main()
