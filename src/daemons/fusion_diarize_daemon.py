"""DiariZen + NeMo Fusion daemon — 두 화자 분리 모델의 결과를 ArcFace로 결합.

원리:
  1. DiariZen subprocess 호출 → 5명 결과
  2. NeMo subprocess 호출 → 4명 결과
  3. 같은 face_id cluster (시간대별 face)로 두 결과 정렬
  4. 두 모델이 일치한 segment = 높은 confidence
  5. 불일치 segment = ArcFace face owner로 결정

단점: 두 모델 모두 load (메모리/시간 2배)
장점: 두 다른 algorithm 합의 → robustness

사용:
  /opt/venv_diarizen/bin/python fusion_diarize_daemon.py --port 8903

NB: 실제 daemon은 두 sub-daemon (DiariZen + NeMo) 별도 port에 있다고 가정하고
    HTTP로 호출. 또는 single-process 안에 두 pipeline 다 load.
    여기서는 두 sub-daemon HTTP 호출 방식 (간단).
"""
import argparse
import os
import time
from typing import List, Optional

import requests

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class DiarizeRequest(BaseModel):
    vocals_wav: str
    num_speakers: Optional[int] = None
    min_duration: float = 0.3


class DiarizeResponse(BaseModel):
    segments: List[dict]
    n_speakers: int
    success: bool
    error: Optional[str] = None


# 4개 sub-daemon URL (별도 port)
DIARIZEN_URL = os.environ.get("FUSION_DIARIZEN_URL", "http://127.0.0.1:8913")
NEMO_URL = os.environ.get("FUSION_NEMO_URL", "http://127.0.0.1:8923")
PYANNOTE_URL = os.environ.get("FUSION_PYANNOTE_URL", "")     # 3rd model (community-1)
PYANNOTE2_URL = os.environ.get("FUSION_PYANNOTE2_URL", "")   # 4th model (3.1)
VBX_URL = os.environ.get("FUSION_VBX_URL", "")               # 5th model (BUT-FIT VBx)


def _call_daemon(url: str, vocals_wav: str, num_speakers, min_duration: float,
                 timeout: int = int(os.environ.get("FUSION_SUBDAEMON_TIMEOUT", "600"))):
    # DETERMINISM FIX (2026-06-03): timeout 300→600. DiariZen(WavLM-large)이 이 입력에서
    # ~288s 걸려 300s 경계를 가끔 넘겨 drop→fusion 비결정+품질저하. 600s 여유로 항상 완료.
    try:
        r = requests.post(
            f"{url}/diarize",
            json={"vocals_wav": vocals_wav, "num_speakers": num_speakers,
                  "min_duration": min_duration},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("success"):
            return data["segments"]
    except Exception as e:
        print(f"[Fusion] {url} call failed: {e}", flush=True)
    return None


def _fuse(diarizen_segs, nemo_segs, fps: float = 25.0):
    """두 결과 fusion — frame-level voting.

    각 frame에서:
      - DiariZen이 가리키는 speaker
      - NeMo가 가리키는 speaker
      - 두 모델 ID 다르므로 단순 일치 불가
      - 대신 각 frame에 (diarizen_spk, nemo_spk) tuple 부여 → 가장 빈번한 pair = 같은 화자

    그 후:
      - tuple → canonical speaker ID 매핑
      - 같은 tuple의 모든 frame은 같은 canonical speaker
    """
    if not diarizen_segs:
        return nemo_segs or []
    if not nemo_segs:
        return diarizen_segs

    # 각 segment의 끝나는 시간 max
    max_t = max(
        max(s["end"] for s in diarizen_segs),
        max(s["end"] for s in nemo_segs),
    )
    n_frames = int(max_t * fps) + 1
    dz_at_frame = [None] * n_frames
    nm_at_frame = [None] * n_frames
    for s in diarizen_segs:
        f1 = int(s["start"] * fps)
        f2 = min(n_frames, int(s["end"] * fps) + 1)
        for f in range(f1, f2):
            dz_at_frame[f] = s["speaker"]
    for s in nemo_segs:
        f1 = int(s["start"] * fps)
        f2 = min(n_frames, int(s["end"] * fps) + 1)
        for f in range(f1, f2):
            nm_at_frame[f] = s["speaker"]

    # tuple counting
    from collections import Counter
    pair_counts = Counter()
    for f in range(n_frames):
        if dz_at_frame[f] and nm_at_frame[f]:
            pair_counts[(dz_at_frame[f], nm_at_frame[f])] += 1

    # 각 nemo_spk마다 가장 자주 같이 등장하는 dz_spk = canonical pair
    nm_to_dz = {}
    for (dz, nm), n in sorted(pair_counts.items(), key=lambda x: -x[1]):
        if nm not in nm_to_dz:
            nm_to_dz[nm] = dz

    # v60+: NM speaker가 어떤 dz와도 동시 등장 안 한 경우 (frame overlap 0),
    # 시간적으로 가장 가까운 dz speaker로 fallback 매핑 (unmapped → 새 화자 방지)
    nm_speakers_all = set(s["speaker"] for s in nemo_segs)
    for nm in nm_speakers_all:
        if nm in nm_to_dz:
            continue
        # 이 NM speaker의 모든 frame 중간점 → 가장 가까운 DZ frame의 dz_spk
        nm_mid = None
        for s in nemo_segs:
            if s["speaker"] == nm:
                nm_mid = (s["start"] + s["end"]) / 2
                break
        if nm_mid is None:
            continue
        # 가장 가까운 DZ frame 찾기
        nm_f = int(nm_mid * fps)
        best_dz = None
        best_d = float("inf")
        for f in range(n_frames):
            if dz_at_frame[f]:
                d = abs(f - nm_f)
                if d < best_d:
                    best_d = d
                    best_dz = dz_at_frame[f]
        if best_dz:
            nm_to_dz[nm] = best_dz
            print(f"[Fusion] unmapped NM '{nm}' → temporal-nearest DZ '{best_dz}' (frame dist={best_d})", flush=True)

    # 각 frame의 canonical speaker = DiariZen 우선 (더 많은 화자 detect 가정)
    # 단 DiariZen이 None이고 NeMo만 있으면 NeMo → mapped DiariZen
    canon_at_frame = [None] * n_frames
    for f in range(n_frames):
        if dz_at_frame[f]:
            canon_at_frame[f] = dz_at_frame[f]
        elif nm_at_frame[f]:
            canon_at_frame[f] = nm_to_dz.get(nm_at_frame[f], nm_at_frame[f])

    # 연속 같은 speaker frame → segment 묶기
    raw_segs = []
    cur_spk = None
    cur_start = None
    for f in range(n_frames):
        spk = canon_at_frame[f]
        if spk != cur_spk:
            if cur_spk and cur_start is not None:
                raw_segs.append({
                    "start": round(cur_start / fps, 3),
                    "end": round(f / fps, 3),
                    "speaker": cur_spk,
                })
            cur_spk = spk
            cur_start = f
    if cur_spk:
        raw_segs.append({
            "start": round(cur_start / fps, 3),
            "end": round(n_frames / fps, 3),
            "speaker": cur_spk,
        })

    # === Minority absorb (v60+): frame count 매우 적은 speaker는 spurious ===
    # 각 raw_spk의 total frame count
    from collections import Counter as _Cnt
    spk_frame_count = _Cnt()
    for s in raw_segs:
        n = int((s["end"] - s["start"]) * fps)
        spk_frame_count[s["speaker"]] += n
    total_frames = sum(spk_frame_count.values()) or 1
    # 임계: 환경변수 또는 default
    min_ratio = float(os.environ.get("FUSION_MIN_SPEAKER_RATIO", "0.03"))  # < 3%면 minor
    min_abs = int(os.environ.get("FUSION_MIN_SPEAKER_FRAMES", "30"))  # < 30 frames (1.2s @25fps)
    minor_spks = set()
    for spk, c in spk_frame_count.items():
        if c < min_abs or (c / total_frames) < min_ratio:
            minor_spks.add(spk)
    if minor_spks:
        print(f"[Fusion] minority speakers detected: {minor_spks} (total speakers: {len(spk_frame_count)})", flush=True)
        # raw_segs를 다시 보고, minor speaker의 frame을 인접 (시간순 이전/이후) non-minor speaker로 reassign
        major_segs = [s for s in raw_segs if s["speaker"] not in minor_spks]
        # 각 minor segment를 시간적으로 가장 가까운 major segment의 speaker로 reassign
        for s in raw_segs:
            if s["speaker"] not in minor_spks:
                continue
            best = None
            best_dist = float("inf")
            mid = (s["start"] + s["end"]) / 2
            for m in major_segs:
                m_mid = (m["start"] + m["end"]) / 2
                d = abs(m_mid - mid)
                if d < best_dist:
                    best_dist = d
                    best = m
            if best:
                s["speaker"] = best["speaker"]
        # minor 흡수 후 인접 같은 speaker 병합
        merged = []
        for s in raw_segs:
            if merged and merged[-1]["speaker"] == s["speaker"]:
                merged[-1]["end"] = s["end"]
            else:
                merged.append(dict(s))
        raw_segs = merged

    # === label normalize: 다양한 형식 (SPEAKER_XX, speaker_X) → SPEAKER_XX 통일 ===
    # 각 unique speaker → integer ID → "SPEAKER_NN"
    all_spks = sorted(set(s["speaker"] for s in raw_segs))
    spk_to_canon = {}
    for i, spk in enumerate(all_spks):
        spk_to_canon[spk] = f"SPEAKER_{i:02d}"
    for s in raw_segs:
        s["speaker"] = spk_to_canon[s["speaker"]]

    # === 짧은 segment (<0.3s) 흡수 — fusion noise 줄임 ===
    min_seg_dur = 0.3
    cleaned = []
    for s in raw_segs:
        dur = s["end"] - s["start"]
        if dur < min_seg_dur and cleaned:
            # 인접 segment에 흡수 (앞 segment 끝 연장)
            cleaned[-1]["end"] = s["end"]
        else:
            cleaned.append(s)
    return cleaned


@app.on_event("startup")
async def startup():
    print(f"[Fusion] DiariZen URL: {DIARIZEN_URL}", flush=True)
    print(f"[Fusion] NeMo URL: {NEMO_URL}", flush=True)


@app.get("/health")
def health():
    # 두 sub-daemon 모두 ready여야
    try:
        dz_ok = requests.get(f"{DIARIZEN_URL}/health", timeout=2).json().get("model_loaded", False)
        nm_ok = requests.get(f"{NEMO_URL}/health", timeout=2).json().get("model_loaded", False)
        return {"status": "ok", "model_loaded": dz_ok and nm_ok,
                "diarizen": dz_ok, "nemo": nm_ok}
    except Exception as e:
        return {"status": "ok", "model_loaded": False, "error": str(e)}


@app.post("/diarize", response_model=DiarizeResponse)
def diarize(req: DiarizeRequest):
    # PARALLEL_FUSION_PATCH: call all sub-daemons concurrently (was sequential).
    # Each daemon is its own process/GPU, so concurrent calls just overlap HTTP latency.
    # Reduces total wait from sum(times) to max(times).
    backends = [("diarizen", DIARIZEN_URL, True), ("nemo", NEMO_URL, True)]
    if PYANNOTE_URL:  backends.append(("pyannote_c1", PYANNOTE_URL, False))
    if PYANNOTE2_URL: backends.append(("pyannote_3_1", PYANNOTE2_URL, False))
    if VBX_URL:       backends.append(("vbx", VBX_URL, False))
    # SPEED+DETERMINISM (2026-06-03): PARALLEL sub-daemon calls (wall = max, not sum) WITH
    # _call_daemon timeout=600s. The non-determinism was the 300s timeout-DROP of DiariZen
    # (~288s, slower under GPU contention) → dz=0 silently → fusion 비결정+품질저하. NOT
    # GPU-math (sub-daemons deterministic in isolation). 600s 여유로 DiariZen이 경합에도
    # 항상 완료 → drop 없음 → 결정적 + 항상 full N-way + 빠름. (3회 동일 검증)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=len(backends)) as ex:
        futs = {ex.submit(_call_daemon, url, req.vocals_wav, req.num_speakers, req.min_duration): name
                for name, url, _ in backends}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    diarizen_segs   = results.get("diarizen")
    nemo_segs       = results.get("nemo")
    pyannote_segs   = results.get("pyannote_c1")
    pyannote2_segs  = results.get("pyannote_3_1")
    vbx_segs        = results.get("vbx")
    if diarizen_segs is None and nemo_segs is None and pyannote_segs is None and pyannote2_segs is None and vbx_segs is None:
        return DiarizeResponse(segments=[], n_speakers=0, success=False,
                               error="all sub-daemons failed")
    # 2-way fuse: DiariZen + NeMo
    fused_12 = _fuse(diarizen_segs or [], nemo_segs or [])
    # 3-way: pyannote community-1
    if pyannote_segs:
        fused_123 = _fuse(fused_12, pyannote_segs)
    else:
        fused_123 = fused_12
    # 4-way: pyannote 3.1
    if pyannote2_segs:
        fused_1234 = _fuse(fused_123, pyannote2_segs)
    else:
        fused_1234 = fused_123
    # 5-way: VBx (BUT-FIT ResNet101 + AHC)
    if vbx_segs:
        fused = _fuse(fused_1234, vbx_segs)
        print(f"[Fusion] 5-model: dz={len(diarizen_segs or [])}, nm={len(nemo_segs or [])}, pyann_c1={len(pyannote_segs or [])}, pyann_3.1={len(pyannote2_segs or [])}, vbx={len(vbx_segs or [])} → fused={len(fused)}", flush=True)
    else:
        fused = fused_1234
        if pyannote2_segs:
            print(f"[Fusion] 4-model: dz={len(diarizen_segs or [])}, nm={len(nemo_segs or [])}, pyann_c1={len(pyannote_segs or [])}, pyann_3.1={len(pyannote2_segs or [])} → fused={len(fused)}", flush=True)
        else:
            print(f"[Fusion] 3-model: dz={len(diarizen_segs or [])}, nm={len(nemo_segs or [])}, pyann={len(pyannote_segs or [])} → fused={len(fused)}", flush=True)
    speakers = set(s["speaker"] for s in fused)
    return DiarizeResponse(segments=fused, n_speakers=len(speakers), success=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8903)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"[Fusion] starting on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
