"""dubbed audio 검증 — qwen-asr 사용 (whisper 대신).

검증 metric:
  1. 한국어 글자 비율 (Korean characters / total transcribed characters)
  2. 외국어/한국어 단어 분류
  3. segment-level 분석

Output:
  - JSON report
  - 시간대별 한국어/외국어 표시
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def transcribe_with_qwen(audio_path: str) -> dict:
    """qwen-asr subprocess로 transcribe."""
    result = subprocess.run([
        "/opt/venv_asr/bin/python", "/workspace/asr_worker.py",
        audio_path, "--language", "Korean",
    ], capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        return {"error": result.stderr[:500]}
    try:
        data = json.loads(result.stdout)
        return data[0] if data else {}
    except Exception as e:
        return {"error": f"parse fail: {e}"}


def analyze_text(text: str) -> dict:
    """text 분석 — 한국어/영어/기타 글자 비율."""
    total = sum(1 for c in text if c.isalpha() or 0xAC00 <= ord(c) <= 0xD7A3)
    if total == 0:
        return {"total": 0, "korean_ratio": 1.0, "foreign_chars": 0}
    korean = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7A3)
    english = sum(1 for c in text if c.isascii() and c.isalpha())
    other = total - korean - english
    return {
        "total": total,
        "korean": korean,
        "english": english,
        "other": other,
        "korean_ratio": korean / total,
        "foreign_chars": english + other,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dubbed_wav", help="dubbed audio file")
    parser.add_argument("--out-json", default=None, help="output JSON report")
    args = parser.parse_args()

    print(f"=== Validating {args.dubbed_wav} ===")

    if not os.path.exists(args.dubbed_wav):
        print(f"FAIL: file not found: {args.dubbed_wav}")
        return 1

    info = sf.info(args.dubbed_wav)
    print(f"Duration: {info.duration:.1f}s, sr={info.samplerate}, ch={info.channels}")

    # 전체 transcribe
    print("\n[1/2] qwen-asr transcribe (전체)...")
    result = transcribe_with_qwen(args.dubbed_wav)
    if "error" in result:
        print(f"transcribe error: {result['error']}")
        return 1
    full_text = result.get("text", "")
    full_analysis = analyze_text(full_text)

    print(f"\n[Full text] {full_text[:300]}")
    print(f"\n[Analysis]")
    print(f"  Total chars: {full_analysis['total']}")
    print(f"  Korean: {full_analysis['korean']} ({full_analysis['korean_ratio']*100:.1f}%)")
    print(f"  English: {full_analysis['english']}")
    print(f"  Other: {full_analysis['other']}")
    print(f"  Foreign total: {full_analysis['foreign_chars']}")

    # word-level (만약 있으면)
    words = result.get("words", [])
    if words:
        print(f"\n[Word-level: {len(words)}] (first 30):")
        for w in words[:30]:
            wtxt = w.get("word", "")
            ws = w.get("start", 0)
            we = w.get("end", 0)
            wa = analyze_text(wtxt)
            kor_pct = wa["korean_ratio"] * 100 if wa["total"] > 0 else 0
            mark = "" if kor_pct >= 50 else " ⚠️"
            print(f"  [{ws:.2f}~{we:.2f}] {wtxt:30s} kor={kor_pct:.0f}%{mark}")

    # JSON report
    out_json = args.out_json or args.dubbed_wav + ".validation.json"
    report = {
        "audio_path": args.dubbed_wav,
        "duration": info.duration,
        "full_text": full_text,
        "analysis": full_analysis,
        "words": words[:200],  # 처음 200개만
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Report saved] {out_json}")

    # verdict
    if full_analysis["korean_ratio"] >= 0.85:
        print(f"\n[VERDICT] ✅ PASS (korean {full_analysis['korean_ratio']*100:.0f}% >= 85%)")
        return 0
    elif full_analysis["korean_ratio"] >= 0.70:
        print(f"\n[VERDICT] ⚠️ MARGINAL (korean {full_analysis['korean_ratio']*100:.0f}%)")
        return 0
    else:
        print(f"\n[VERDICT] ❌ FAIL (korean {full_analysis['korean_ratio']*100:.0f}% < 70%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
