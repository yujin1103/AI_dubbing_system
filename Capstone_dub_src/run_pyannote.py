# pyannote.audio 4.0 community-1 파이프라인으로 화자분리해 RTTM을 출력 (DiariZen과 별개 경로)
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="pyannote community-1 화자분리 → RTTM")
    parser.add_argument("input_audio")
    parser.add_argument("output_rttm")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    import torch
    from pyannote.audio import Pipeline

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    if not token:
        print("경고: HF_TOKEN 없음 — 게이트 모델 다운로드 실패할 수 있음", file=sys.stderr)

    pipeline = Pipeline.from_pretrained(args.model, token=token)
    pipeline.to(torch.device(args.device))

    kwargs = {}
    if args.num_speakers is not None:
        kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        kwargs["max_speakers"] = args.max_speakers

    result = pipeline(args.input_audio, **kwargs)

    # pyannote.audio 4.0은 DiarizeOutput을 반환 — 실제 Annotation 속성을 찾는다.
    annotation = result if hasattr(result, "itertracks") else None
    if annotation is None:
        for attr in ("speaker_diarization", "exclusive_speaker_diarization", "diarization", "prediction", "annotation"):
            cand = getattr(result, attr, None)
            if cand is not None and hasattr(cand, "itertracks"):
                annotation = cand
                break
    if annotation is None:
        attrs = [a for a in dir(result) if not a.startswith("_")]
        raise SystemExit(f"itertracks 가능한 속성 없음. DiarizeOutput 속성: {attrs}")

    file_id = Path(args.input_audio).stem
    lines = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        duration = max(0.0, float(turn.end) - float(turn.start))
        if duration <= 0:
            continue
        lines.append(f"SPEAKER {file_id} 1 {turn.start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>")

    out = Path(args.output_rttm)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    n_spk = len({ln.split()[7] for ln in lines})
    print(f"wrote {args.output_rttm} ({len(lines)} segments, {n_spk} speakers)")


if __name__ == "__main__":
    main()
