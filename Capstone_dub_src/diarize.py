# DiariZen-Large-s80-v2 로 화자분리 후 RTTM 출력.
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from common import get_logger, resolve_project_path

logger = get_logger("diarize")


def _iter_segments(annotation: object) -> Iterable[tuple[str, float, float]]:
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            yield str(speaker), float(turn.start), float(turn.end)
        return

    for item in annotation:
        if len(item) != 2:
            continue
        turn, speaker = item
        start = getattr(turn, "start", None)
        end = getattr(turn, "end", None)
        if start is None or end is None:
            continue
        yield str(speaker), float(start), float(end)


def write_rttm(records: Iterable[tuple[str, float, float]], output_rttm: str | Path, file_id: str) -> Path:
    output_path = resolve_project_path(output_rttm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for speaker, start, end in sorted(records, key=lambda item: item[1]):
        duration = max(0.0, end - start)
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info("Wrote RTTM to %s", output_path)
    return output_path


def _build_clustering_override(
    *,
    model_path: Path,
    max_speakers: int | None,
    min_speakers: int | None,
    ahc_threshold: float | None,
    fa: float | None,
    fb: float | None,
    lda_dim: int | None = None,
    max_iters: int | None = None,
    method: str | None = None,
    min_cluster_size: int | None = None,
    ahc_criterion: str | None = None,
    seg_duration: float | None = None,
    segmentation_step: float | None = None,
    batch_size: int | None = None,
    apply_median_filtering: bool | None = None,
) -> dict | None:
    if (
        max_speakers is None
        and min_speakers is None
        and ahc_threshold is None
        and fa is None
        and fb is None
        and lda_dim is None
        and max_iters is None
        and method is None
        and min_cluster_size is None
        and ahc_criterion is None
        and seg_duration is None
        and segmentation_step is None
        and batch_size is None
        and apply_median_filtering is None
    ):
        return None
    try:
        import toml
    except ImportError as exc:
        raise RuntimeError("toml package is required to override DiariZen clustering settings.") from exc
    base_path = model_path / "config.toml"
    if not base_path.exists():
        raise FileNotFoundError(f"DiariZen config.toml not found: {base_path}")
    base = toml.load(base_path.as_posix())
    inference_args = dict(base.get("inference", {}).get("args", {}))
    clustering_args = dict(base.get("clustering", {}).get("args", {}))
    if max_speakers is not None:
        clustering_args["max_speakers"] = int(max_speakers)
    if min_speakers is not None:
        clustering_args["min_speakers"] = int(min_speakers)
    if ahc_threshold is not None:
        clustering_args["ahc_threshold"] = float(ahc_threshold)
    if fa is not None:
        clustering_args["Fa"] = float(fa)
    if fb is not None:
        clustering_args["Fb"] = float(fb)
    if lda_dim is not None:
        clustering_args["lda_dim"] = int(lda_dim)
    if max_iters is not None:
        clustering_args["max_iters"] = int(max_iters)
    if method is not None:
        clustering_args["method"] = str(method)
    if min_cluster_size is not None:
        clustering_args["min_cluster_size"] = int(min_cluster_size)
    if ahc_criterion is not None:
        clustering_args["ahc_criterion"] = str(ahc_criterion)
    if seg_duration is not None:
        inference_args["seg_duration"] = float(seg_duration)
    if segmentation_step is not None:
        inference_args["segmentation_step"] = float(segmentation_step)
    if batch_size is not None:
        inference_args["batch_size"] = int(batch_size)
    if apply_median_filtering is not None:
        inference_args["apply_median_filtering"] = bool(apply_median_filtering)
    return {
        "inference": {"args": inference_args},
        "clustering": {"args": clustering_args},
    }


def diarize_audio(
    input_audio: str | Path,
    output_rttm: str | Path,
    *,
    model_dir: str | Path,
    embedding_model_dir: str | Path,
    device: str = "cuda:0",
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
    try:
        import torch
        from diarizen.pipelines.inference import DiariZenPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diarizen is not installed in the active environment."
        ) from exc

    audio_path = resolve_project_path(input_audio)
    model_path = resolve_project_path(model_dir)
    embedding_path = resolve_project_path(embedding_model_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"DiariZen model directory not found: {model_path}")
    embedding_weights = embedding_path / "speaker-embedding.onnx"
    if not embedding_weights.exists():
        raise FileNotFoundError(f"Embedding model weights not found: {embedding_weights}")

    config_parse = _build_clustering_override(
        model_path=model_path,
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
    if config_parse is not None:
        inference_log_keys = {"seg_duration", "segmentation_step", "batch_size", "apply_median_filtering"}
        clustering_log_keys = {"max_speakers", "min_speakers", "ahc_threshold", "ahc_criterion", "Fa", "Fb", "lda_dim", "max_iters", "method", "min_cluster_size"}
        logger.info(
            "Overriding DiariZen config: %s",
            {
                "inference": {k: v for k, v in config_parse["inference"]["args"].items() if k in inference_log_keys},
                "clustering": {k: v for k, v in config_parse["clustering"]["args"].items() if k in clustering_log_keys},
            },
        )

    pipeline = DiariZenPipeline(
        diarizen_hub=model_path.absolute(),
        embedding_model=str(embedding_weights),
        config_parse=config_parse,
    )
    if hasattr(pipeline, "to"):
        pipeline.to(torch.device(device))

    annotation = pipeline(str(audio_path))
    return write_rttm(_iter_segments(annotation), output_rttm, audio_path.stem)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DiariZen speaker diarization and save RTTM.")
    parser.add_argument("input_audio")
    parser.add_argument("output_rttm")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--embedding-model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--ahc-threshold", type=float)
    parser.add_argument("--ahc-criterion")
    parser.add_argument("--fa", type=float)
    parser.add_argument("--fb", type=float)
    parser.add_argument("--lda-dim", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--method")
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument("--seg-duration", type=float)
    parser.add_argument("--segmentation-step", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--apply-median-filtering", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    diarize_audio(
        args.input_audio,
        args.output_rttm,
        model_dir=args.model_dir,
        embedding_model_dir=args.embedding_model_dir,
        device=args.device,
        max_speakers=args.max_speakers,
        min_speakers=args.min_speakers,
        ahc_threshold=args.ahc_threshold,
        ahc_criterion=args.ahc_criterion,
        fa=args.fa,
        fb=args.fb,
        lda_dim=args.lda_dim,
        max_iters=args.max_iters,
        method=args.method,
        min_cluster_size=args.min_cluster_size,
        seg_duration=args.seg_duration,
        segmentation_step=args.segmentation_step,
        batch_size=args.batch_size,
        apply_median_filtering=args.apply_median_filtering,
    )


if __name__ == "__main__":
    main()
