"""Redub — 청크 단위 부분 재합성 CLI (팀원 PPT 형식).

사용:
  # JSON 직접 수정 후 자동 감지 (또는 강제 group 지정)
  python redub.py --run-id <id> --groups 5,7
  python redub.py --run-id <id> --groups 5 \\
      --override '{"5": {"text": "새 한국어 번역", "tts_emotion": "with a calm tone"}}'

흐름:
  1. run_dir/meta/<chunk>_segments.json 로드 (synthesize_chunk가 저장)
  2. override 적용 (text/tts_emotion/speed 수정)
  3. 변경된 group 만 CosyVoice3 daemon 호출 → 새 wav
  4. dubbed_segments/group_*.wav 갱신
  5. 전체 dubbed wav 재생성 (timing 그대로, segment wav 교체)
  6. mix_audio 재실행 → 새 output mp4 (_redub.mp4)
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Cosy daemon URL (orchestrator와 동일)
COSY_DAEMON_URL = os.environ.get("COSY_DAEMON_URL", "http://127.0.0.1:8901")
TTS_SAMPLE_RATE = 24000


def _call_cosy_daemon(text: str, ref_path: str, speed: float,
                     tts_emotion: str, emotion: str) -> np.ndarray:
    """CosyVoice3 daemon HTTP 호출. v22 형식 (instruct2 + endofprompt)."""
    import requests
    r = requests.post(f"{COSY_DAEMON_URL}/synthesize", json={
        "text": text,
        "ref_audio_path": ref_path,
        "speed": speed,
        "tone": tts_emotion or "",
        "emotion": emotion or "Neutral",
    }, timeout=600)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"cosy daemon failed: {data.get('error')}")
    wav_bytes = base64.b64decode(data["audio_b64"])
    audio, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == TTS_SAMPLE_RATE, f"sr mismatch: {sr} != {TTS_SAMPLE_RATE}"
    return audio.astype(np.float32)


def redub_chunk(run_dir: Path, chunk_name: str, group_ids: list[int],
                overrides: dict) -> Path:
    """단일 chunk 재합성. Returns: 새 dubbed wav 경로."""
    meta_path = run_dir / "meta" / f"{chunk_name}_segments.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"메타 없음: {meta_path}. 처음 측정 시 synthesize_chunk가 저장.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    video_duration = meta["video_duration"]
    sr = meta.get("tts_sample_rate", TTS_SAMPLE_RATE)
    groups = meta["groups"]

    # override 적용
    if overrides:
        for gi_str, mods in overrides.items():
            gi = int(gi_str)
            for g in groups:
                if g["group_idx"] == gi:
                    for k, v in mods.items():
                        g[k] = v
                    print(f"[Redub] group {gi} override: {mods}")
                    break

    # group 재합성
    seg_dir = run_dir / "dubbed" / f"{chunk_name}_segments"
    seg_dir.mkdir(exist_ok=True)
    for g in groups:
        gi = g["group_idx"]
        if gi not in group_ids:
            continue  # 변경 안 한 group skip
        text = g["text"]
        ref_path = g["ref_path"]
        speed = float(g.get("speed", 1.0))
        tts_emotion = g.get("tts_emotion", "")
        emotion = g.get("emotion", "Neutral")

        if not text.strip():
            print(f"[Redub] group {gi} text 비어있음 → skip")
            continue
        if not os.path.exists(ref_path):
            print(f"[Redub] group {gi} ref 없음: {ref_path} → skip")
            continue

        print(f"[Redub] group {gi} 재합성: '{text[:40]}...' (speaker={g['speaker']})")
        audio = _call_cosy_daemon(text, ref_path, speed, tts_emotion, emotion)
        # 길이 조정 — 원래 batch와 동일하게 max_allowed 안에 맞춤
        max_allowed = g["max_allowed_duration"]
        cur_dur = len(audio) / sr
        if cur_dur > max_allowed:
            # 1.25x 압축 후 trim
            from scipy.signal import resample_poly
            stretch = min(1.25, cur_dur / max_allowed)
            audio = resample_poly(audio, 1000, int(1000 * stretch))
            if len(audio) / sr > max_allowed:
                audio = audio[:int(max_allowed * sr)]
            print(f"  ↳ 압축: {cur_dur:.2f}s → {len(audio)/sr:.2f}s")
        # peak normalize
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio * (0.9 / peak)
        # 저장
        seg_wav = seg_dir / f"group_{gi:03d}_{g['speaker']}.wav"
        sf.write(str(seg_wav), audio, sr)
        g["wav_path"] = str(seg_wav)

    # 전체 dubbed wav 재생성 (group wav 다 concat by timing)
    total_samples = int(video_duration * sr)
    output_audio = np.zeros(total_samples, dtype=np.float32)
    for g in groups:
        wav_path = g["wav_path"]
        if not os.path.exists(wav_path):
            continue
        audio, _ = sf.read(wav_path)
        start_sample = int(g["group_start"] * sr)
        end_sample = min(start_sample + len(audio), total_samples)
        copy_len = end_sample - start_sample
        if copy_len > 0:
            output_audio[start_sample:end_sample] = audio[:copy_len]

    dubbed_path = run_dir / "dubbed" / f"{chunk_name}_dubbed_redub.wav"
    sf.write(str(dubbed_path), output_audio, sr)
    print(f"[Redub] dubbed wav 재생성: {dubbed_path}")

    # 변경 사항 segments.json 에 반영
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dubbed_path


def remix_chunk(run_dir: Path, chunk_name: str, dubbed_path: Path) -> Path:
    """mix_audio 재실행 → 새 chunk_final.mp4."""
    chunk_mp4 = run_dir / "chunks" / f"{chunk_name}.mp4"  # 원본 청크
    if not chunk_mp4.exists():
        # 원본 video chunk 위치 다를 수 있음 — _final 제외하고 찾기
        for f in (run_dir / "chunks").iterdir():
            if chunk_name in f.name and "final" not in f.name and f.suffix == ".mp4":
                chunk_mp4 = f
                break
    if not chunk_mp4.exists():
        raise FileNotFoundError(f"원본 chunk mp4 없음: {chunk_mp4}")
    bgm_path = run_dir / "bgm" / f"{chunk_name}_bgm.wav"
    if not bgm_path.exists():
        raise FileNotFoundError(f"bgm 없음: {bgm_path}")

    # 원본 sr 감지
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(chunk_mp4),
        ], text=True).strip()
        orig_sr = int(out) if out else 44100
    except Exception:
        orig_sr = 44100

    output_path = run_dir / "chunks" / f"{chunk_name}_final_redub.mp4"
    filter_complex = (
        f"[1:a]aformat=channel_layouts=stereo:sample_rates={orig_sr},"
        f"loudnorm=I=-23:TP=-2:LRA=11[dub];"
        f"[2:a]aformat=channel_layouts=stereo:sample_rates={orig_sr},volume=0.9[bgm];"
        "[dub][bgm]amix=inputs=2:duration=shortest:normalize=0[a]"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(chunk_mp4),
        "-i", str(dubbed_path),
        "-i", str(bgm_path),
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-profile:a", "aac_low",
        "-b:a", "192k", "-ar", str(orig_sr), "-ac", "2",
        "-shortest", str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[Redub] chunk mix 재실행: {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Redub — 청크 단위 부분 재합성")
    ap.add_argument("--run-id", required=True, help="대상 run ID")
    ap.add_argument("--groups", required=True,
                    help="재합성할 group_idx (콤마 구분, 예: 5,7)")
    ap.add_argument("--override", default=None,
                    help='JSON: {"5": {"text": "...", "tts_emotion": "..."}}')
    ap.add_argument("--runs-dir", default="/workspace/media/runs")
    ap.add_argument("--output-dir", default="/workspace/media/output")
    args = ap.parse_args()

    run_dir = Path(args.runs_dir) / args.run_id
    if not run_dir.exists():
        print(f"❌ run 없음: {run_dir}")
        return 1

    group_ids = [int(x) for x in args.groups.split(",")]
    overrides = json.loads(args.override) if args.override else {}

    # 모든 chunk 처리 (단일 chunk 가정 — 멀티 chunk 면 확장)
    chunks_dir = run_dir / "chunks"
    chunk_names = sorted({
        f.stem.replace("_final", "")
        for f in chunks_dir.iterdir()
        if f.suffix == ".mp4" and "final" not in f.name
    })
    if not chunk_names:
        print(f"❌ chunk 없음")
        return 1

    new_chunk_finals = []
    for chunk_name in chunk_names:
        print(f"\n=== Redub chunk: {chunk_name} ===")
        try:
            new_dubbed = redub_chunk(run_dir, chunk_name, group_ids, overrides)
            new_final = remix_chunk(run_dir, chunk_name, new_dubbed)
            new_chunk_finals.append(new_final)
        except Exception as e:
            print(f"❌ chunk {chunk_name} redub 실패: {e}")
            continue

    if not new_chunk_finals:
        print(f"❌ redub 결과 없음")
        return 1

    # 전체 영상 concat (단일 chunk면 그대로 copy)
    output_path = Path(args.output_dir) / f"{run_dir.name}_redub.mp4"
    if len(new_chunk_finals) == 1:
        subprocess.run(["cp", str(new_chunk_finals[0]), str(output_path)], check=True)
    else:
        concat_list = run_dir / "chunks" / "redub_concat.txt"
        with open(concat_list, "w") as f:
            for cf in new_chunk_finals:
                f.write(f"file '{cf}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", str(output_path),
        ], check=True)

    print(f"\n✅ Redub 완료: {output_path}")
    print(f"   호스트: E:\\TTS_capstone\\media\\output\\{output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
