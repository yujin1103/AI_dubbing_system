from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import get_logger, load_json, load_json_if_exists, project_relative, resolve_project_path, save_json

logger = get_logger("extract_emotion")

EMOTION_HINTS = {
    "angry": "angry, tense, strong delivery",
    "disgusted": "disgusted, cold, restrained delivery",
    "disgust": "disgusted, cold, restrained delivery",
    "fearful": "fearful, shaky, tense delivery",
    "fear": "fearful, shaky, tense delivery",
    "happy": "happy, bright, energetic delivery",
    "neutral": "neutral, natural delivery",
    "other": "neutral, natural delivery",
    "sad": "sad, low energy, soft delivery",
    "surprised": "surprised, raised pitch, alert delivery",
    "surprise": "surprised, raised pitch, alert delivery",
    "unknown": "neutral, natural delivery",
}


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip()
    if "/" in label:
        label = label.split("/")[-1]
    label = label.strip().lower().replace("<unk>", "unknown")
    label = label.replace(" ", "_")
    if label == "angry":
        return "angry"
    if label in {"disgust", "disgusted"}:
        return "disgusted"
    if label in {"fear", "fearful"}:
        return "fearful"
    if label == "happiness":
        return "happy"
    if label == "sadness":
        return "sad"
    if label in {"surprise", "surprised"}:
        return "surprised"
    if label in {"", "unk"}:
        return "unknown"
    return label


def _to_float(value: Any) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _first_result(raw_result: Any) -> dict[str, Any]:
    raw_result = _jsonable(raw_result)
    if isinstance(raw_result, list):
        if not raw_result:
            return {}
        first = raw_result[0]
        return first if isinstance(first, dict) else {"result": first}
    if isinstance(raw_result, dict):
        return raw_result
    return {"result": raw_result}


def _parse_emotion_result(raw_result: Any) -> dict[str, Any]:
    item = _first_result(raw_result)
    labels = item.get("labels") or item.get("label") or []
    scores = item.get("scores") or item.get("score") or []
    if isinstance(labels, str):
        labels = [labels]
    if not isinstance(scores, list):
        scores = [scores]

    pairs: list[tuple[str, float]] = []
    for label, score in zip(labels, scores):
        pairs.append((_normalize_label(label), _to_float(score)))

    scores_by_label: dict[str, float] = {}
    for label, score in pairs:
        scores_by_label[label] = max(score, scores_by_label.get(label, 0.0))

    if scores_by_label:
        label, confidence = max(scores_by_label.items(), key=lambda pair: pair[1])
    else:
        label, confidence = "unknown", 0.0

    return {
        "label": label,
        "confidence": round(confidence, 6),
        "scores": {key: round(value, 6) for key, value in sorted(scores_by_label.items())},
        "raw": item,
    }


def _build_emotion_input_signature(row: dict[str, Any], model_ref: str, backend: str) -> str:
    wav_value = row.get("wav")
    wav_path = resolve_project_path(wav_value) if wav_value else None
    stat = wav_path.stat() if wav_path and wav_path.exists() else None
    payload = {
        "backend": backend,
        "chunk_id": str(row.get("chunk_id", "") or ""),
        "model": model_ref,
        "wav": str(wav_value or ""),
        "file_size": int(stat.st_size) if stat is not None else None,
        "file_mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class Emotion2VecClassifier:
    def __init__(self, *, model_ref: str, backend: str, device: str) -> None:
        model_path = resolve_project_path(model_ref)
        resolved_model_ref = str(model_path) if model_path.exists() else model_ref
        self.model_ref = resolved_model_ref
        self.backend = backend
        self.device = device
        if backend == "modelscope":
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            self.model = pipeline(task=Tasks.emotion_recognition, model=resolved_model_ref)
        elif backend == "funasr":
            from funasr import AutoModel

            kwargs: dict[str, Any] = {"model": resolved_model_ref, "disable_update": True}
            if device:
                kwargs["device"] = device
            try:
                self.model = AutoModel(**kwargs)
            except TypeError:
                kwargs.pop("disable_update", None)
                self.model = AutoModel(**kwargs)
        else:
            raise ValueError(f"Unsupported emotion backend: {backend}")

    def predict(self, wav_path: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
        if self.backend == "modelscope":
            raw_result = self.model(str(wav_path), granularity="utterance", extract_embedding=False)
        else:
            kwargs: dict[str, Any] = {"granularity": "utterance", "extract_embedding": False}
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                kwargs["output_dir"] = str(output_dir)
            raw_result = self.model.generate(str(wav_path), **kwargs)
        return _parse_emotion_result(raw_result)


def extract_chunk_emotions(
    speaker_chunks_json: str | Path,
    output_json: str | Path,
    *,
    model_ref: str = "iic/emotion2vec_plus_large",
    backend: str = "funasr",
    device: str = "cuda:0",
    skip_existing: bool = True,
    funasr_output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = load_json(speaker_chunks_json)
    existing = load_json_if_exists(output_json, default=[])
    existing_map = {
        str(item.get("chunk_id", "")).strip(): item
        for item in existing
        if isinstance(item, dict) and item.get("chunk_id")
    }
    output_dir = resolve_project_path(funasr_output_dir) if funasr_output_dir else resolve_project_path(output_json).with_suffix("").with_name("emotion2vec_outputs")
    classifier: Emotion2VecClassifier | None = None
    results: list[dict[str, Any]] = []

    for row in rows:
        chunk_id = str(row.get("chunk_id", "")).strip()
        wav_value = row.get("wav")
        if not chunk_id or not wav_value:
            continue
        signature = _build_emotion_input_signature(row, model_ref, backend)
        existing_row = existing_map.get(chunk_id, {})
        if skip_existing and existing_row.get("emotion_input_signature") == signature:
            results.append(existing_row)
            continue

        wav_path = resolve_project_path(wav_value)
        result_row: dict[str, Any] = {
            "chunk_id": chunk_id,
            "speaker": row.get("speaker"),
            "start": row.get("start"),
            "end": row.get("end"),
            "duration": row.get("duration"),
            "wav": project_relative(wav_path),
            "emotion_model": model_ref,
            "emotion_backend": backend,
            "emotion_input_signature": signature,
        }
        if not wav_path.exists():
            result_row["emotion_error"] = f"Wav not found: {wav_path}"
            results.append(result_row)
            logger.warning("Emotion input wav missing for %s: %s", chunk_id, wav_path)
            continue

        if classifier is None:
            logger.info("Loading emotion model: %s via %s", model_ref, backend)
            classifier = Emotion2VecClassifier(model_ref=model_ref, backend=backend, device=device)

        try:
            emotion = classifier.predict(wav_path, output_dir=output_dir)
        except Exception as exc:
            result_row["emotion_error"] = str(exc)
            logger.warning("Emotion extraction failed for %s: %s", chunk_id, exc)
        else:
            result_row["source_emotion"] = {
                "label": emotion["label"],
                "confidence": emotion["confidence"],
                "scores": emotion["scores"],
            }
            result_row["tts_emotion_hint"] = EMOTION_HINTS.get(emotion["label"], EMOTION_HINTS["unknown"])
            logger.info(
                "Emotion %s: %s %.3f",
                chunk_id,
                emotion["label"],
                emotion["confidence"],
            )
        results.append(result_row)

    save_json(results, output_json)
    logger.info("Saved emotion results for %s chunks to %s", len(results), resolve_project_path(output_json))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract source speech emotions with emotion2vec+.")
    parser.add_argument("speaker_chunks_json")
    parser.add_argument("output_json")
    parser.add_argument("--model-ref", default="iic/emotion2vec_plus_large")
    parser.add_argument("--backend", default="funasr", choices=["funasr", "modelscope"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--funasr-output-dir")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extract_chunk_emotions(
        args.speaker_chunks_json,
        args.output_json,
        model_ref=args.model_ref,
        backend=args.backend,
        device=args.device,
        skip_existing=not args.no_skip_existing,
        funasr_output_dir=args.funasr_output_dir,
    )


if __name__ == "__main__":
    main()
