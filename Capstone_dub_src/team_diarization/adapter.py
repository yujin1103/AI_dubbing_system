from __future__ import annotations

import shutil
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from common import get_logger, load_json_if_exists, project_relative, resolve_project_path, save_json
from diarize import diarize_audio, write_rttm
from rttm_to_json import parse_rttm_line

from . import diarize_v195 as team_v195

logger = get_logger("team_diarization")


def _config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _enabled(config: dict[str, Any], *, default: bool = True) -> bool:
    if "enabled" not in config:
        return default
    return bool(config["enabled"])


def _float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(default if value is None else value)


def _int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return int(default if value is None else value)


def _segments_from_rttm(path: Path) -> list[team_v195.Segment]:
    segments: list[team_v195.Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = parse_rttm_line(line)
        if row is None:
            continue
        duration = float(row["duration"])
        if duration < 0.3:
            continue
        segments.append(
            team_v195.Segment(
                speaker=str(row["speaker"]),
                start=round(float(row["start"]), 3),
                end=round(float(row["end"]), 3),
            )
        )
    segments.sort(key=lambda item: (item.start, item.end, item.speaker))
    return segments


def _normalize_words(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("words") or raw.get("segments") or []
    if not isinstance(raw, list):
        return []

    words: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        text = item.get("word", item.get("text", ""))
        words.append({"start": start_f, "end": end_f, "word": str(text).strip()})
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return words


def _load_words(
    input_audio: Path,
    *,
    refiner_config: dict[str, Any],
) -> list[dict[str, Any]]:
    words_json = refiner_config.get("words_json") or refiner_config.get("word_timestamps_json")
    if words_json:
        loaded = load_json_if_exists(words_json, default=None)
        if loaded is None:
            message = f"Configured word timestamp JSON not found: {resolve_project_path(words_json)}"
            if bool(refiner_config.get("require_word_timestamps", False)):
                raise FileNotFoundError(message)
            logger.warning("%s; word-level split will be skipped", message)
            return []
        words = _normalize_words(loaded)
        logger.info("Loaded %s word timestamps from %s", len(words), resolve_project_path(words_json))
        return words

    if not bool(refiner_config.get("run_whisperx", False)):
        return []

    language = str(refiner_config.get("language", "en"))
    try:
        words = _run_whisperx_words(input_audio, refiner_config=refiner_config, language=language)
    except Exception as exc:
        if bool(refiner_config.get("require_word_timestamps", False)):
            raise
        logger.warning("WhisperX word timestamp extraction failed; word-level split skipped: %s", exc)
        return []
    logger.info("WhisperX produced %s word timestamps for team refiner", len(words))
    return words


def _run_whisperx_words(
    input_audio: Path,
    *,
    refiner_config: dict[str, Any],
    language: str,
) -> list[dict[str, Any]]:
    import torch
    import whisperx

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    class FullAudioVAD:
        def __call__(self, payload: dict[str, Any]):
            from pyannote.core import SlidingWindow, SlidingWindowFeature

            waveform = payload["waveform"]
            sample_rate = int(payload["sample_rate"])
            n_samples = int(waveform.shape[-1])
            duration_sec = n_samples / float(sample_rate)
            frame_step = 0.1
            n_frames = max(2, int(np.ceil(duration_sec / frame_step)) + 1)
            scores = np.ones((n_frames, 1), dtype=np.float32)
            window = SlidingWindow(start=-frame_step / 2.0, duration=frame_step, step=frame_step)
            return SlidingWindowFeature(scores, window, labels=["speech"])

    download_dir = resolve_project_path(refiner_config.get("whisperx_download_dir") or "models/whisperx")
    download_dir.mkdir(parents=True, exist_ok=True)
    torch_home = resolve_project_path(refiner_config.get("torch_cache_dir") or "models/torch")
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    device = str(refiner_config.get("whisperx_device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    compute_type = str(refiner_config.get("whisperx_compute_type") or ("float16" if device == "cuda" else "int8"))
    batch_size = int(refiner_config.get("whisperx_batch_size") or 1)
    model_name = str(refiner_config.get("whisperx_model") or "large-v3")

    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        vad_model=FullAudioVAD(),
        download_root=str(download_dir),
    )
    audio = whisperx.load_audio(str(input_audio))
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    words: list[dict[str, Any]] = []
    for segment in aligned.get("segments", []):
        for word in segment.get("words", []):
            if "start" in word and "end" in word:
                words.append(word)
    return words


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path))
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def _load_eres2(refiner_config: dict[str, Any]):
    embedding_config = _config_section(refiner_config, "eres2netv2")
    if not _enabled(embedding_config, default=True):
        logger.info("Skipping ERes2NetV2 because team_refiner.eres2netv2.enabled=false")
        return None
    cache_dir = embedding_config.get("cache_dir") or "models/modelscope"
    cache_path = resolve_project_path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_path))
    try:
        return team_v195.load_eres2()
    except Exception as exc:
        if bool(embedding_config.get("required", False)):
            raise
        logger.warning("ERes2NetV2 unavailable; voice-embedding refiner passes skipped: %s", exc)
        return None


def _assign_words(segments: list[team_v195.Segment], words: list[dict[str, Any]]) -> None:
    if words:
        team_v195.assign_text(segments, words)


def _speaker_counts(segments: list[team_v195.Segment]) -> dict[str, int]:
    return dict(sorted(Counter(seg.speaker for seg in segments).items()))


def _write_report(
    *,
    report_json: str | Path | None,
    input_audio: Path,
    output_rttm: Path,
    base_rttm: Path,
    initial_segments: list[team_v195.Segment],
    final_segments: list[team_v195.Segment],
    steps: list[dict[str, Any]],
    words_count: int,
    embedding_enabled: bool,
) -> None:
    if not report_json:
        return
    save_json(
        {
            "schema_version": 1,
            "engine": "team_refiner_v195",
            "input_audio": project_relative(input_audio),
            "base_rttm": project_relative(base_rttm),
            "output_rttm": project_relative(output_rttm),
            "word_count": words_count,
            "embedding_enabled": embedding_enabled,
            "initial": {
                "n_segments": len(initial_segments),
                "speaker_counts": _speaker_counts(initial_segments),
            },
            "final": {
                "n_segments": len(final_segments),
                "speaker_counts": _speaker_counts(final_segments),
                "segments": [asdict(seg) for seg in final_segments],
            },
            "steps": steps,
        },
        report_json,
    )


def diarize_with_team_refiner(
    input_audio: str | Path,
    output_rttm: str | Path,
    *,
    model_dir: str | Path,
    embedding_model_dir: str | Path,
    device: str = "cuda:0",
    report_json: str | Path | None = None,
    refiner_config: dict[str, Any] | None = None,
    max_speakers: int | None = None,
    min_speakers: int | None = None,
    ahc_threshold: float | None = None,
    ahc_criterion: str | None = None,
    fa: float | None = None,
    fb: float | None = None,
    lda_dim: int | None = None,
    max_iters: int | None = None,
    method: str | None = None,
    min_cluster_size: int | None = None,
    seg_duration: float | None = None,
    segmentation_step: float | None = None,
    batch_size: int | None = None,
    apply_median_filtering: bool | None = None,
) -> Path:
    refiner_config = dict(refiner_config or {})
    input_path = resolve_project_path(input_audio)
    output_path = resolve_project_path(output_rttm)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_rttm_value = refiner_config.get("base_rttm")
    if base_rttm_value:
        base_rttm = resolve_project_path(base_rttm_value)
    else:
        base_rttm = output_path.with_name(f"{output_path.stem}.base.rttm")

    diarize_audio(
        input_path,
        base_rttm,
        model_dir=model_dir,
        embedding_model_dir=embedding_model_dir,
        device=device,
        max_speakers=max_speakers,
        min_speakers=min_speakers,
        ahc_threshold=ahc_threshold,
        ahc_criterion=ahc_criterion,
        fa=fa,
        fb=fb,
        lda_dim=lda_dim,
        max_iters=max_iters,
        method=method,
        min_cluster_size=min_cluster_size,
        seg_duration=seg_duration,
        segmentation_step=segmentation_step,
        batch_size=batch_size,
        apply_median_filtering=apply_median_filtering,
    )

    initial_segments = _segments_from_rttm(base_rttm)
    segments = [
        team_v195.Segment(seg.speaker, seg.start, seg.end, seg.text, seg.word_count)
        for seg in initial_segments
    ]
    steps: list[dict[str, Any]] = [
        {
            "name": "base_diarizen",
            "n_segments": len(segments),
            "speaker_counts": _speaker_counts(segments),
        }
    ]

    if not segments:
        write_rttm([], output_path, input_path.stem)
        _write_report(
            report_json=report_json,
            input_audio=input_path,
            output_rttm=output_path,
            base_rttm=base_rttm,
            initial_segments=initial_segments,
            final_segments=segments,
            steps=steps,
            words_count=0,
            embedding_enabled=False,
        )
        return output_path

    fallback_to_base = bool(refiner_config.get("fallback_to_base_on_error", True))
    try:
        words = _load_words(input_path, refiner_config=refiner_config)
        audio, sample_rate = _load_audio(input_path)
        extract_emb = _load_eres2(refiner_config)

        time_gap_config = _config_section(refiner_config, "time_gap_split")
        if extract_emb is not None and _enabled(time_gap_config, default=True):
            before = _speaker_counts(segments)
            changed = team_v195.time_gap_split(
                segments,
                audio,
                sample_rate,
                extract_emb,
                gap_th=_float(time_gap_config, "gap_sec", 5.0),
                voice_diff_th=_float(time_gap_config, "voice_diff_threshold", 0.15),
                min_target_sim=_float(time_gap_config, "min_target_sim", 0.5),
            )
            steps.append(
                {
                    "name": "time_gap_split",
                    "changed": int(changed),
                    "speaker_counts_before": before,
                    "speaker_counts_after": _speaker_counts(segments),
                }
            )

        intra_config = _config_section(refiner_config, "intra_spk_split")
        if extract_emb is not None and _enabled(intra_config, default=True):
            before = _speaker_counts(segments)
            changed = team_v195.intra_spk_split(
                segments,
                audio,
                sample_rate,
                extract_emb,
                min_sim=_float(intra_config, "min_sim", 0.5),
                passes=_int(intra_config, "passes", 3),
            )
            steps.append(
                {
                    "name": "intra_spk_split",
                    "changed": int(changed),
                    "speaker_counts_before": before,
                    "speaker_counts_after": _speaker_counts(segments),
                }
            )

        sandwich_config = _config_section(refiner_config, "sandwich_override")
        if _enabled(sandwich_config, default=True):
            before = _speaker_counts(segments)
            changed = team_v195.sandwich_override(
                segments,
                gap_th=_float(sandwich_config, "gap_sec", 0.3),
            )
            steps.append(
                {
                    "name": "sandwich_override",
                    "changed": int(changed),
                    "speaker_counts_before": before,
                    "speaker_counts_after": _speaker_counts(segments),
                }
            )

        _assign_words(segments, words)

        word_split_config = _config_section(refiner_config, "word_level_intra_split")
        if extract_emb is not None and words and _enabled(word_split_config, default=True):
            before_n = len(segments)
            before = _speaker_counts(segments)
            segments = team_v195.word_level_intra_split(
                segments,
                words,
                audio,
                sample_rate,
                extract_emb,
                margin=_float(word_split_config, "margin", 0.1),
            )
            steps.append(
                {
                    "name": "word_level_intra_split",
                    "segments_before": before_n,
                    "segments_after": len(segments),
                    "speaker_counts_before": before,
                    "speaker_counts_after": _speaker_counts(segments),
                }
            )
            _assign_words(segments, words)

    except Exception:
        if not fallback_to_base:
            raise
        logger.exception("Team diarization refiner failed; falling back to base DiariZen RTTM")
        shutil.copyfile(base_rttm, output_path)
        return output_path

    records = ((seg.speaker, seg.start, seg.end) for seg in segments)
    write_rttm(records, output_path, input_path.stem)
    _write_report(
        report_json=report_json,
        input_audio=input_path,
        output_rttm=output_path,
        base_rttm=base_rttm,
        initial_segments=initial_segments,
        final_segments=segments,
        steps=steps,
        words_count=len(words),
        embedding_enabled=extract_emb is not None,
    )
    logger.info(
        "Team diarization refiner wrote %s segments across %s speakers to %s",
        len(segments),
        len(_speaker_counts(segments)),
        output_path,
    )
    return output_path
