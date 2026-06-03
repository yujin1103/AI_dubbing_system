from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from audio_features import analyze_chunk_records, compact_feature_summary, ensure_chunk_features
from common import get_logger, load_json, resolve_project_path, save_json
from quality_gate import assess_asr_row, attach_stage_quality

logger = get_logger("run_asr")


def _prepare_qwen_asr_imports() -> None:
    repo_path = resolve_project_path("third_party/Qwen3-ASR")
    if repo_path.exists():
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)


def _resolve_torch_dtype(name: str) -> object:
    import torch

    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def _get_result_attr(result: object, attr: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(attr, default)
    return getattr(result, attr, default)


def _boost_subchunk_chunks(
    chunk_records: list[dict[str, Any]],
    *,
    volume: float,
    win_sec: float,
    hop_sec: float,
    boost_area: tuple[float, float] | None,
    workdir: Path,
) -> list[dict[str, Any]]:
    # E:\TTS_capstone 의 boost_subchunk_asr.py 검증 로직:
    #   짧은 외침 (Adam! 같은) 을 잡기 위해 volume 3x boost + sub-chunk (4s window,
    #   3s hop, 1s overlap) 로 분할 → 보강 transcripts 생성.
    # test5: 141 → 156 words (+15 fresh, Adam x2 detect).
    #
    # 여기서는 boost된 wav를 새 chunk_records로 추가만 한다 — 메인 transcribe
    # 호출이 그 위에서 그대로 동작. 다운스트림은 chunk_id (suffix _boost)로 구분.
    import subprocess
    if not boost_area:
        return list(chunk_records)
    area_start, area_end = boost_area
    workdir.mkdir(parents=True, exist_ok=True)

    extra: list[dict[str, Any]] = []
    for rec in chunk_records:
        start = float(rec.get("start", 0.0))
        end = float(rec.get("end", 0.0))
        if end <= area_start or start >= area_end:
            continue  # outside boost area
        wav = resolve_project_path(rec["wav"])
        # 4s window, 3s hop sub-chunks
        t = max(start, area_start)
        idx = 0
        while t < min(end, area_end):
            sub_start = t
            sub_end = min(t + win_sec, end, area_end)
            if sub_end - sub_start < 0.5:
                break
            sub_wav = workdir / f"{Path(wav).stem}_boost_{idx:03d}.wav"
            # ffmpeg: cut + boost
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{sub_start - start:.3f}",
                "-t", f"{sub_end - sub_start:.3f}",
                "-i", str(wav),
                "-filter:a", f"volume={volume}",
                str(sub_wav),
            ]
            rc = subprocess.run(cmd).returncode
            if rc == 0 and sub_wav.exists():
                extra.append({
                    **rec,
                    "chunk_id": f"{rec['chunk_id']}_boost{idx:03d}",
                    "wav": str(sub_wav),
                    "start": sub_start,
                    "end": sub_end,
                    "duration": sub_end - sub_start,
                    "from_boost": True,
                })
            idx += 1
            t += hop_sec
    logger.info("boost_subchunk: produced %s extra sub-chunks (volume=%s, win=%s, hop=%s, area=%s)",
                len(extra), volume, win_sec, hop_sec, boost_area)
    return list(chunk_records) + extra


def transcribe_chunks(
    chunk_json: str | Path,
    output_json: str | Path,
    *,
    model_dir: str | Path,
    device: str = "cuda:0",
    dtype: str = "float16",
    max_inference_batch_size: int = 8,
    max_new_tokens: int = 256,
    language: str | None = None,
    features_json: str | Path | None = None,
    boost_subchunk: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    _prepare_qwen_asr_imports()

    # ASR 데몬 라우팅: 파이프라인 venv(venv_lipsync)에 qwen_asr가 없으면(멀티-venv: ASR=venv_asr)
    # ASR 데몬(8902, venv_asr)으로 전사. env ASR_DAEMON_URL 로 강제 가능. build_repair_inputs 동일 패턴.
    import os as _os
    _asr_daemon = _os.environ.get("ASR_DAEMON_URL", "").strip()
    Qwen3ASRModel = None
    if not _asr_daemon:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            _asr_daemon = "http://127.0.0.1:8902"
            logger.info("qwen_asr 미설치 — ASR 데몬(%s)으로 전사", _asr_daemon)

    chunk_records = load_json(chunk_json)
    if not chunk_records:
        save_json([], output_json)
        return []

    if boost_subchunk:
        # boost area=(0,30), volume=3.0, win=4s, hop=3s (1s overlap)
        chunk_records = _boost_subchunk_chunks(
            chunk_records,
            volume=float(boost_subchunk.get("volume", 3.0)),
            win_sec=float(boost_subchunk.get("win_sec", 4.0)),
            hop_sec=float(boost_subchunk.get("hop_sec", 3.0)),
            boost_area=(
                float(boost_subchunk.get("area_start", 0.0)),
                float(boost_subchunk.get("area_end", 30.0)),
            ) if boost_subchunk.get("enabled", True) else None,
            workdir=resolve_project_path(boost_subchunk.get("workdir", "audio/boost_subchunks")),
        )

    if features_json:
        chunk_features = analyze_chunk_records(chunk_records, output_json=features_json)
    else:
        chunk_features = ensure_chunk_features(chunk_records, reference_path=output_json)
    feature_map = {str(item.get("chunk_id", "")).strip(): item for item in chunk_features if item.get("chunk_id")}

    audio_paths = [str(resolve_project_path(item["wav"])) for item in chunk_records]
    language_arg: list[str | None] | None
    if language is None:
        language_arg = None
    else:
        language_arg = [language] * len(audio_paths)

    if _asr_daemon:
        import requests as _rq
        results = []
        for _wav in audio_paths:
            try:
                _resp = _rq.post(
                    f"{_asr_daemon}/transcribe",
                    json={"audio_path": _wav, "language": (language or "English")},
                    timeout=300,
                ).json()
                _ok = _resp.get("success", True)
                results.append({
                    "text": (_resp.get("text", "") if _ok else ""),
                    "language": _resp.get("detected_language"),
                })
            except Exception as _e:
                logger.warning("ASR 데몬 전사 실패 (%s): %s", _wav, _e)
                results.append({"text": "", "language": None})
    else:
        model = Qwen3ASRModel.from_pretrained(
            str(resolve_project_path(model_dir)),
            dtype=_resolve_torch_dtype(dtype),
            device_map=device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        results = model.transcribe(audio=audio_paths, language=language_arg)

    asr_rows: list[dict[str, Any]] = []
    for item, result in zip(chunk_records, results):
        duration = item.get("duration")
        if duration is None and item.get("start") is not None and item.get("end") is not None:
            duration = max(0.0, float(item["end"]) - float(item["start"]))

        chunk_id = str(item["chunk_id"])
        feature = feature_map.get(chunk_id)
        row = {
            "chunk_id": item["chunk_id"],
            "speaker": item["speaker"],
            "wav": item.get("wav"),
            "start": item.get("start"),
            "end": item.get("end"),
            "duration": duration,
            "language": _get_result_attr(result, "language"),
            "text_src": (_get_result_attr(result, "text", "") or "").strip(),
        }
        if feature:
            row["source_audio_summary"] = compact_feature_summary(feature)
        attach_stage_quality(row, "asr", assess_asr_row(row, chunk_feature=feature))
        asr_rows.append(row)

    save_json(asr_rows, output_json)
    logger.info("Saved ASR for %s chunks to %s", len(asr_rows), resolve_project_path(output_json))
    return asr_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR over chunk wav files.")
    parser.add_argument("chunk_json")
    parser.add_argument("output_json")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-inference-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--language")
    parser.add_argument("--features-json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    transcribe_chunks(
        args.chunk_json,
        args.output_json,
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        features_json=args.features_json,
    )


if __name__ == "__main__":
    main()
