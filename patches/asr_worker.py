#!/usr/bin/env python3
"""
ASR Worker — /opt/venv_asr/bin/python 으로 실행
transformers 4.57.6 환경에서 Qwen3-ASR 구동

orchestrator.py / audio_to_korean.py가 subprocess로 호출하며,
결과는 JSON으로 stdout에 출력.
모델을 1회만 로드하고 여러 청크를 순서대로 처리.
"""
import argparse
import json
import sys
import os
import torch
import gc

def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Worker")
    parser.add_argument("audio", nargs="+", help="Path to audio file(s)")
    parser.add_argument("--language", default=None,
                        help="Force language name (e.g., 'English', 'Korean')")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B",
                        help="ASR model name or local path")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B",
                        help="Forced aligner model name or local path")
    parser.add_argument("--max-tokens", default=512, type=int,
                        help="Max new tokens for generation")
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Max inference batch size") # 🔥 VRAM 방어막 (기본값 1 유지)
    args = parser.parse_args()

    print(f"[ASR Worker] Loading: {args.model}", file=sys.stderr)
    print(f"[ASR Worker] Aligner: {args.aligner}", file=sys.stderr)

    from qwen_asr import Qwen3ASRModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # 🔥 최고 속도를 위해 Flash Attention 명시적 사용
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        forced_aligner=args.aligner,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=device,
        ),
    )
    print(f"[ASR Worker] Model loaded ✅", file=sys.stderr)

    output = []
    for audio_path in args.audio:
        print(f"[ASR Worker] Transcribing: {audio_path}", file=sys.stderr)
        results = model.transcribe(
            audio=audio_path,
            language=args.language,
            return_time_stamps=True,
        )

        for r in results:
            entry = {
                "text": r.text if hasattr(r, 'text') else str(r),
                "language": r.language if hasattr(r, 'language') else None,
                "words": [],
            }

            timestamps = getattr(r, 'time_stamps', None) or []
            for ts in timestamps:
                if hasattr(ts, 'text') and hasattr(ts, 'start_time'):
                    entry["words"].append({
                        "word": ts.text,
                        "start": float(ts.start_time),
                        "end": float(ts.end_time),
                    })
                elif isinstance(ts, (list, tuple)) and len(ts) >= 3:
                    entry["words"].append({
                        "word": ts[0],
                        "start": float(ts[1]),
                        "end": float(ts[2]),
                    })
                elif hasattr(ts, 'word'):
                    entry["words"].append({
                        "word": ts.word,
                        "start": float(ts.start),
                        "end": float(ts.end),
                    })

            output.append(entry)

    # stdout으로 JSON 출력
    json.dump(output, sys.stdout, ensure_ascii=False)

    # GPU 메모리 안전 해제
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"\n[ASR Worker] Done, GPU memory released", file=sys.stderr)

if __name__ == "__main__":
    main()