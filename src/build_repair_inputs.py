#!/usr/bin/env python3
"""모듈 파이프라인 브리지 — preserved_repair.run_dir 를 모듈 stage 출력으로 채운다.

목적(2026-06-01): 팀원 webUI 가 쓰는 모듈 pipeline.py 안에서 검증된 4-way fusion + 8 repair
patch 가 그대로 돌도록 연결. 기존엔 apply_preserved_repair 가 사전 존재하는 orchestrator run_dir
(meta/*_segments.json + vocals/ + faces.json + words.json)을 요구해 모듈 단독 실행 불가했음.

이 브리지가 모듈 stage 출력으로 그 run_dir 를 생성:
  - vocals (separate_audio)            -> run_dir/vocals/<chunk>_clean_vocals.wav
  - 전체영상 ASR (asr_daemon /transcribe) -> run_dir/meta/<chunk>_words.json   (패치 word_level_split/gap_fill 입력)
  - fusion diarization (diarization_json) -> run_dir/meta/<chunk>_segments.json (groups 형식, text 빈 채로)
  - face_clustering faces.json          -> run_dir/meta/faces.json            (face_identity_split 입력)
patch 들은 단일 chunk(=전체영상, <chunk>=<stem>_chunk_000) 단위로 동작.

호출: pipeline.py step_build_repair_inputs(config). idempotent(있으면 덮어씀).
"""
from __future__ import annotations
import json
import os
import shutil
import urllib.request
from pathlib import Path


def _http_post_json(url: str, payload: dict, timeout: int = 1200) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_diar_segments(diar_json_path: str) -> list[dict]:
    """diarization_json (list 또는 {segments:[...]}) -> [{start,end,speaker}] 정렬."""
    d = json.load(open(diar_json_path, encoding="utf-8"))
    segs = d.get("segments", d) if isinstance(d, dict) else d
    out = []
    for s in segs:
        st = s.get("start", s.get("group_start"))
        en = s.get("end", s.get("group_end"))
        sp = s.get("speaker", s.get("label", "SPEAKER_00"))
        if st is None or en is None:
            continue
        out.append({"start": float(st), "end": float(en), "speaker": str(sp)})
    return sorted(out, key=lambda z: z["start"])


def _to_groups(diar_segs: list[dict]) -> dict:
    """fusion diar -> repair 패치 raw _segments.json 형식 (groups, text 빈 채로)."""
    groups = []
    for i, s in enumerate(diar_segs):
        groups.append({
            "group_idx": i,
            "speaker": s["speaker"],
            "group_start": round(s["start"], 3),
            "group_end": round(s["end"], 3),
            "segment_ids": [i],
            "text": "",
        })
    return {"groups": groups}


def build_repair_inputs(
    run_dir: str,
    *,
    stem: str,
    vocals_src: str,
    diarization_json: str,
    faces_src: str | None,
    video_src: str | None = None,
    asr_url: str = "http://127.0.0.1:8902",
    asr_language: str = "English",
    asr_context: str = "",
) -> str:
    """run_dir 에 패치 입력(segments/words/vocals/faces/chunk-video) 생성. chunk 이름 반환."""
    rd = Path(run_dir)
    meta = rd / "meta"
    voc = rd / "vocals"
    chunks = rd / "chunks"
    meta.mkdir(parents=True, exist_ok=True)
    voc.mkdir(parents=True, exist_ok=True)
    chunks.mkdir(parents=True, exist_ok=True)
    chunk = f"{stem}_chunk_000"

    # 1) vocals 복사
    voc_dst = voc / f"{chunk}_clean_vocals.wav"
    if not os.path.exists(vocals_src):
        raise SystemExit(f"[build_repair_inputs] vocals not found: {vocals_src}")
    shutil.copyfile(vocals_src, voc_dst)
    print(f"[build_repair_inputs] vocals -> {voc_dst}")

    # 1b) chunk 비디오(전체영상) 복사 — gap_fill 의 gap re-ASR mp4 슬라이스용 + 더빙 단계 참조용.
    if video_src and os.path.exists(video_src):
        shutil.copyfile(video_src, chunks / f"{chunk}.mp4")
        print(f"[build_repair_inputs] chunk video -> {chunks / (chunk + '.mp4')}")
    else:
        print(f"[build_repair_inputs] WARN: video_src 없음({video_src}) — gap_fill gap re-ASR skip 가능")

    # 2) fusion diar -> raw segments.json (groups)
    diar_segs = _load_diar_segments(diarization_json)
    seg_path = meta / f"{chunk}_segments.json"
    json.dump(_to_groups(diar_segs), open(seg_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[build_repair_inputs] {len(diar_segs)} fusion segments -> {seg_path}")

    # 3) 전체영상 ASR -> words.json (패치 입력)
    words_path = meta / f"{chunk}_words.json"
    try:
        resp = _http_post_json(f"{asr_url}/transcribe",
                               {"audio_path": str(voc_dst), "language": asr_language,
                                "context": asr_context})
        words = resp.get("words", []) if resp.get("success", True) else []
        json.dump({"chunk_name": chunk, "detected_lang": resp.get("detected_language", ""),
                   "words": words}, open(words_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[build_repair_inputs] ASR {len(words)} words -> {words_path}")
    except Exception as ex:
        # words 없으면 word_level_split/gap_fill 가 skip → 4-way 약화. 명시적 실패.
        raise SystemExit(f"[build_repair_inputs] ASR daemon 호출 실패({asr_url}): {ex}")

    # 4) faces.json 복사 (face_clustering 출력)
    if faces_src and os.path.exists(faces_src):
        shutil.copyfile(faces_src, meta / "faces.json")
        print(f"[build_repair_inputs] faces.json -> {meta / 'faces.json'}")
    else:
        print(f"[build_repair_inputs] WARN: faces.json 없음({faces_src}) — face_identity_split skip")

    return chunk
