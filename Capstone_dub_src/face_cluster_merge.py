"""face cluster auto-merge — 같은 SPK 의 여러 face cluster 를 dominant 와 자동 합치기.

원리:
  - speaker_face_map 의 dominant_face_cluster + alt_clusters 활용
  - 같은 audio SPK 가 매핑된 face cluster 들 (dominant + alt) → 같은 사람의 다른 angle
  - 그 cluster 들을 dominant cluster ID 로 통합 (face_clusters dict 업데이트)
  - 통합 후 cluster 수 감소 → SPK ↔ face 매핑 강화

영상 무관 default:
  - merge_alt_threshold: dominant_confidence < 0.6 면 alt 도 같은 사람으로 가정 (영상 무관)
  - 즉 한 SPK 가 face 다양하게 분산되면 그 alt 들 다 합침
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


def merge_face_clusters(
    run_dir: str,
    *,
    merge_alt_threshold: float = 0.6,
) -> dict:
    rd = Path(run_dir)
    meta = rd / "meta"
    fc_path = meta / "face_clusters.json"
    if not fc_path.exists():
        raise SystemExit(f"face_clusters.json not found: {fc_path}")
    fc = json.load(open(fc_path, encoding="utf-8"))

    face_clusters = {str(k): int(v) for k, v in fc.get("face_clusters", {}).items()}
    spk_face_map = fc.get("speaker_face_map", {})
    cluster_thumbnails = fc.get("cluster_thumbnails", {})

    # 각 SPK 의 dominant + alt cluster 들을 한 그룹으로
    cluster_remap: dict[int, int] = {}  # old_cid → new_cid (dominant)
    n_merged = 0
    for spk, info in spk_face_map.items():
        dom_cid = int(info.get("dominant_face_cluster", -1))
        if dom_cid < 0:
            continue
        conf = float(info.get("dominant_confidence", 1.0))
        if conf >= merge_alt_threshold:
            # 단독 dominant 이미 강함 → merge 불필요
            continue
        alt_clusters = info.get("alt_clusters", {})
        if not alt_clusters:
            continue
        # dominant + alt 모두 dominant 로 합침 (같은 SPK 의 다른 face angle)
        for alt_cid_str in alt_clusters.keys():
            alt_cid = int(alt_cid_str)
            if alt_cid == dom_cid:
                continue
            # 이미 다른 SPK 의 dominant 면 합치지 않음 (충돌)
            alt_is_other_dominant = any(
                int(other_info.get("dominant_face_cluster", -1)) == alt_cid
                for other_spk, other_info in spk_face_map.items()
                if other_spk != spk
            )
            if alt_is_other_dominant:
                print(f"  skip merge {alt_cid} → {dom_cid} (다른 SPK 의 dominant)")
                continue
            cluster_remap[alt_cid] = dom_cid
            n_merged += 1
            print(f"  merge cluster {alt_cid} → {dom_cid} (SPK {spk})")

    # face_clusters dict 업데이트 (remap 적용)
    new_face_clusters = {}
    for tid_str, cid in face_clusters.items():
        new_cid = cluster_remap.get(int(cid), cid)
        new_face_clusters[tid_str] = int(new_cid)

    # speaker_face_count 도 (옛 dict 구조면 보존 schema 유지)
    sfc = fc.get("speaker_face_count", {})

    # n_clusters 재계산
    new_unique_clusters = set(new_face_clusters.values())

    # cluster_thumbnails — merged cluster 의 alt thumbnail 들은 dominant 로 alias
    new_thumbnails = dict(cluster_thumbnails)
    # alt thumbnail 도 dominant 와 동일하게
    for alt_cid, dom_cid in cluster_remap.items():
        # alt cluster 의 jpg 가 있으면 그대로 두고, dominant 의 jpg 가 없으면 alt 로 fallback
        alt_path = new_thumbnails.get(str(alt_cid))
        dom_path = new_thumbnails.get(str(dom_cid))
        if alt_path and not dom_path:
            new_thumbnails[str(dom_cid)] = alt_path

    # spk_face_map 도 dominant 통합 (alt clusters 비움 — 이제 다 dominant)
    new_spk_face_map = {}
    for spk, info in spk_face_map.items():
        dom_cid = int(info.get("dominant_face_cluster", -1))
        new_spk_face_map[spk] = {
            "dominant_face_cluster": dom_cid,
            "dominant_confidence": info.get("dominant_confidence"),
            "face_thumbnail": new_thumbnails.get(str(dom_cid)) or info.get("face_thumbnail"),
            "alt_clusters": {},  # 모두 dominant 로 흡수
            "alt_thumbnails": [],
            "merge_applied": True,
        }

    fc["face_clusters"] = new_face_clusters
    fc["cluster_thumbnails"] = new_thumbnails
    fc["speaker_face_map"] = new_spk_face_map
    fc["n_clusters"] = len(new_unique_clusters)
    fc["n_merged"] = n_merged

    out_path = meta / "face_clusters_merged.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] merged {n_merged} clusters, {len(set(face_clusters.values()))} → {len(new_unique_clusters)} unique")
    print(f"    saved → {out_path}")
    return {"merged": n_merged, "before": len(set(face_clusters.values())), "after": len(new_unique_clusters)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--merge-alt-threshold", type=float, default=0.6)
    args = ap.parse_args()
    s = merge_face_clusters(args.run_dir, merge_alt_threshold=args.merge_alt_threshold)
    print(f"summary: {s}")


if __name__ == "__main__":
    main()
