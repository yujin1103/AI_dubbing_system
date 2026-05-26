# DiariZen 화자분리 결과(speaker_chunks)를 실패유형별 모듈로 교정하는 스테이지
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any, Callable

from common import get_logger, load_json, resolve_project_path, save_json

logger = get_logger("repair_diarization")

Chunk = dict[str, Any]
RepairModule = Callable[..., "list[Chunk]"]

def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _speaker_centroids(chunks: list[Chunk], embs: dict[str, list[float]]) -> dict[str, list[float]]:
    groups: dict[str, list[list[float]]] = defaultdict(list)
    for ch in chunks:
        cid = ch["chunk_id"]
        if cid in embs:
            groups[str(ch["speaker"])].append(embs[cid])
    centroids: dict[str, list[float]] = {}
    for spk, vecs in groups.items():
        dim = len(vecs[0])
        mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        centroids[spk] = _l2(mean)
    return centroids


def _embed_reassign(chunks: list[Chunk], *, embeddings_json: str, margin: float = 0.1,
                    min_sim: float = 0.5, passes: int = 3) -> list[Chunk]:
    # 청크 임베딩을 화자 centroid와 비교해 더 가까운 화자로 재배정한다. 자기 화자와의
    # 비교는 leave-one-out(자기 청크 제외)으로 계산 — 작은 클러스터에서 self-bias로
    # 못 옮기는 문제를 막는다. 다중 패스로 centroid를 갱신하며 오염을 제거한다.
    embs = load_json(embeddings_json)
    chunks = [dict(c) for c in chunks]
    for _ in range(int(passes)):
        members: dict[str, list[int]] = defaultdict(list)
        for idx, ch in enumerate(chunks):
            if ch["chunk_id"] in embs:
                members[str(ch["speaker"])].append(idx)
        sums: dict[str, list[float]] = {}
        centroids: dict[str, list[float]] = {}
        for spk, idxs in members.items():
            vecs = [embs[chunks[i]["chunk_id"]] for i in idxs]
            dim = len(vecs[0])
            total = [sum(v[d] for v in vecs) for d in range(dim)]
            sums[spk] = total
            centroids[spk] = _l2(total)
        changed = 0
        for idx, ch in enumerate(chunks):
            cid = ch["chunk_id"]
            if cid not in embs:
                continue
            e = embs[cid]
            cur_spk = str(ch["speaker"])
            if len(members[cur_spk]) >= 2:
                total = sums[cur_spk]
                loo = _l2([total[d] - e[d] for d in range(len(e))])
                cur_sim = _dot(e, loo)
            else:
                cur_sim = 1.0  # 단독 화자는 해체하지 않는다
            best_spk, best_sim = cur_spk, cur_sim
            for spk, cen in centroids.items():
                if spk == cur_spk:
                    continue
                sim = _dot(e, cen)
                if sim > best_sim:
                    best_sim, best_spk = sim, spk
            if best_spk != cur_spk and best_sim > cur_sim + margin and best_sim >= min_sim:
                ch.setdefault("audio_speaker", cur_spk)
                ch["speaker"] = best_spk
                ch["reassigned_by"] = "embed"
                ch["reassign_sim"] = round(best_sim, 3)
                ch["reassign_prev_sim"] = round(cur_sim, 3)
                changed += 1
        logger.info("embed_reassign pass: %s chunks reassigned", changed)
        if changed == 0:
            break
    return chunks


def _embed_split(chunks: list[Chunk], *, embeddings_json: str, link_threshold: float = 0.5) -> list[Chunk]:
    # 한 화자 클러스터 안에서 임베딩을 단일연결 부분군집화. 서로 먼(링크 임계값 미만)
    # 2개 이상 목소리 그룹으로 갈리면 새 화자 라벨로 분리한다 (두 인물이 한 라벨로
    # 병합된 과병합 오류 교정). 가장 긴 그룹이 원래 라벨을 유지한다.
    embs = load_json(embeddings_json)
    chunks = [dict(c) for c in chunks]
    by_spk: dict[str, list[int]] = defaultdict(list)
    for idx, ch in enumerate(chunks):
        if ch["chunk_id"] in embs:
            by_spk[str(ch["speaker"])].append(idx)

    for spk, idxs in by_spk.items():
        if len(idxs) < 2:
            continue
        parent = {i: i for i in idxs}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if _dot(embs[chunks[i]["chunk_id"]], embs[chunks[j]["chunk_id"]]) >= link_threshold:
                    parent[find(i)] = find(j)

        comps: dict[int, list[int]] = defaultdict(list)
        for i in idxs:
            comps[find(i)].append(i)
        if len(comps) < 2:
            continue

        groups = sorted(comps.values(), key=lambda g: -sum(float(chunks[i].get("duration", 0.0)) for i in g))
        for k, group in enumerate(groups):
            if k == 0:
                continue
            new_label = f"{spk}_{k}"
            for i in group:
                chunks[i]["split_from"] = spk
                chunks[i]["speaker"] = new_label
        logger.info("embed_split: speaker '%s' -> %s sub-speakers", spk, len(groups))
    return chunks


def _cluster_face_tracks(tracks: list[dict], sim_threshold: float) -> dict[int, int]:
    # 얼굴 트랙 임베딩(이미 단위벡터)을 코사인 그리디 군집화해 시각 인물 ID를 부여.
    centroids: list[list[float]] = []
    members: list[list[list[float]]] = []
    face_of: dict[int, int] = {}
    for t in tracks:
        emb = t["embedding"]
        if centroids:
            sims = [_dot(emb, c) for c in centroids]
            best = max(range(len(sims)), key=lambda i: sims[i])
            best_sim = sims[best]
        else:
            best, best_sim = -1, -1.0
        if best_sim >= sim_threshold:
            members[best].append(emb)
            dim = len(emb)
            mean = [sum(v[d] for v in members[best]) / len(members[best]) for d in range(dim)]
            centroids[best] = _l2(mean)
            face_of[t["track_id"]] = best
        else:
            centroids.append(_l2(emb))
            members.append([emb])
            face_of[t["track_id"]] = len(centroids) - 1
    return face_of


def _visual_reconcile(chunks: list[Chunk], *, asd_tracks_json: str, sim_threshold: float = 0.4,
                      min_evidence: float = 5.0, min_split_chunks: int = 2) -> list[Chunk]:
    # 얼굴 신원으로 오디오 화자 클러스터를 교정. 보수적 '분할 전용' — 한 오디오 화자의
    # 청크가 강한 발화증거로 2개 이상 얼굴에 갈릴 때만 얼굴별로 분리한다. 얼굴 증거 없는
    # 청크는 오디오 라벨 유지(폴백). 라벨 공간 혼합 안 함(예전 per-chunk 덮어쓰기 폐기).
    asd = load_json(asd_tracks_json)
    tracks = asd.get("tracks", [])
    out = [dict(c) for c in chunks]
    if not tracks:
        logger.info("visual_reconcile: 얼굴 트랙 없음 — 변경 없음")
        return out

    face_of = _cluster_face_tracks(tracks, sim_threshold)
    chunk_face: dict[str, int] = {}
    for c in chunks:
        start, end = float(c["start"]), float(c["end"])
        ev: dict[int, float] = defaultdict(float)
        for t in tracks:
            fid = face_of[t["track_id"]]
            for f in t["frames"]:
                if start <= f["t"] < end and f["score"] > 0:
                    ev[fid] += f["score"]
        if ev:
            best = max(ev, key=ev.get)
            if ev[best] >= min_evidence:
                chunk_face[str(c["chunk_id"])] = best

    spk_faces: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for c in chunks:
        cid = str(c["chunk_id"])
        if cid in chunk_face:
            spk_faces[str(c["speaker"])][chunk_face[cid]] += 1

    n_split = 0
    for spk, counter in spk_faces.items():
        strong_faces = [fid for fid, n in counter.items() if n >= min_split_chunks]
        if len(strong_faces) >= 2:
            for c in out:
                cid = str(c["chunk_id"])
                if str(c["speaker"]) == spk and chunk_face.get(cid) in strong_faces:
                    c.setdefault("audio_speaker", spk)
                    c["speaker"] = f"{spk}_v{chunk_face[cid]}"
                    c["visual_split"] = True
                    n_split += 1
    logger.info("visual_reconcile: %s개 얼굴 인물 클러스터, %s청크 얼굴분할", len(set(face_of.values())), n_split)
    return out


def _preserved_repair_patches(
    chunks: list[Chunk],
    *,
    run_dir: str,
    main_merge: float = 0.45,
    bg_merge: float = 0.30,
    sim_match: float = 0.45,
    pad: float = 0.5,
    skip: list[str] | None = None,
    venv_python: str | None = None,
) -> list[Chunk]:
    # E:\TTS_capstone 에서 검증된 8 patches 일괄 호출 wrapper.
    # subprocess 로 src/apply_repair_patches.py 실행 → 결과 segments_gapfilled.json
    # 로드 → chunks 변환. run_dir 안에 meta/ 폴더가 있어야 함.
    #
    # test4 best: main_merge=0.99 bg_merge=0.30 sim_match=0.10 pad=0.5
    # test5 best: main_merge=0.40 bg_merge=0.30 sim_match=0.45 pad=0.5
    import os
    import subprocess
    from pathlib import Path

    run_path = Path(run_dir)
    if not (run_path / "meta").exists():
        logger.warning("preserved_repair_patches: %s/meta not found, skipping", run_dir)
        return chunks

    here = Path(__file__).parent
    apply_script = here / "apply_repair_patches.py"
    if not apply_script.exists():
        logger.warning("apply_repair_patches.py not found at %s", apply_script)
        return chunks

    python = venv_python or os.environ.get(
        "PATCHES_VENV_PYTHON", "/opt/venv_diarizen/bin/python"
    )
    cmd = [
        python, str(apply_script), str(run_path),
        "--main-merge", str(main_merge),
        "--bg-merge", str(bg_merge),
        "--sim-match", str(sim_match),
        "--pad", str(pad),
    ]
    if skip:
        cmd += ["--skip", *skip]
    logger.info("preserved_repair_patches: running %s", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        logger.error("preserved_repair_patches failed (rc=%s)", rc)
        return chunks

    # gapfilled 결과 로드 → chunks 변환
    gapfilled_files = sorted((run_path / "meta").glob("*_segments_gapfilled.json"))
    if not gapfilled_files:
        logger.warning("preserved_repair_patches: no segments_gapfilled.json produced")
        return chunks
    merged: list[Chunk] = []
    import json
    for gp in gapfilled_files:
        with open(gp, encoding="utf-8") as f:
            data = json.load(f)
        for seg in data.get("groups", []):
            merged.append({
                "chunk_id": seg.get("group_id") or f"{gp.stem}_g{len(merged):04d}",
                "speaker": seg.get("speaker", ""),
                "start": float(seg.get("group_start", 0.0)),
                "end": float(seg.get("group_end", 0.0)),
                "duration": float(seg.get("group_end", 0.0)) - float(seg.get("group_start", 0.0)),
                "text": seg.get("text", ""),
                "preserved_repair": True,
            })
    logger.info("preserved_repair_patches: %s chunks -> %s gapfilled segments", len(chunks), len(merged))
    return merged


# 실패유형별 교정 모듈 등록부. Phase가 진행되며 채워진다.
#   embed_reassign            — 화자 임베딩으로 짧은/오염 청크를 올바른 화자에 재배정 (기존 화자 간 이동)
#   embed_split               — 한 화자에 두 인물이 병합된 경우 임베딩 부분군집으로 분리 (과병합 교정)
#   visual_reconcile          — 얼굴 보이는(실사) 클립에서 audio 화자를 얼굴 신원으로 분할 (분할 전용·증거 게이트)
#   preserved_repair_patches  — E:\TTS_capstone 검증된 8 patches 일괄 호출 (word_split + focused_nemo
#                               + visual_asd + face_cluster_match + gap_fill + postprocess_reassign_text).
#                               test4/test5 sweep best config 검증 완료 (score 0.998 / 1.167).
REPAIR_MODULES: dict[str, RepairModule] = {
    "embed_reassign": _embed_reassign,
    "embed_split": _embed_split,
    "visual_reconcile": _visual_reconcile,
    "preserved_repair_patches": _preserved_repair_patches,
}


def repair_chunks(chunks: list[Chunk], *, modules: list[str], params: dict[str, dict]) -> list[Chunk]:
    # 등록된 모듈을 주어진 순서대로 적용한다. 모듈이 없으면 입력을 그대로 돌려준다.
    for name in modules:
        if name not in REPAIR_MODULES:
            raise ValueError(f"Unknown repair module: {name} (available: {sorted(REPAIR_MODULES)})")
        before = len(chunks)
        chunks = REPAIR_MODULES[name](chunks, **(params.get(name) or {}))
        logger.info("repair module '%s': %s -> %s chunks", name, before, len(chunks))
    return chunks


def repair_diarization_file(
    input_json: str,
    output_json: str,
    *,
    modules: list[str] | None = None,
    params: dict[str, dict] | None = None,
) -> list[Chunk]:
    modules = list(modules or [])
    params = dict(params or {})
    chunks = load_json(input_json)
    if not modules:
        logger.info("repair_diarization: 활성 모듈 없음, %s chunks 그대로 통과", len(chunks))
    repaired = repair_chunks(chunks, modules=modules, params=params)
    save_json(repaired, output_json)
    logger.info("Wrote repaired chunks to %s (%s chunks)", resolve_project_path(output_json), len(repaired))
    return repaired


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="speaker_chunks를 실패유형별 모듈로 교정한다.")
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    parser.add_argument("--modules", nargs="*", default=[], help="적용할 교정 모듈 이름(순서대로)")
    parser.add_argument("--params-json", help="모듈별 파라미터 JSON 파일(선택)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    params = load_json(args.params_json) if args.params_json else {}
    repair_diarization_file(args.input_json, args.output_json, modules=args.modules, params=params)


if __name__ == "__main__":
    main()
