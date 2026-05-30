#!/usr/bin/env python3
"""윈도우 ASR — 오디오를 짧은 창으로 잘라 ASR 후 시간순 병합 (메인 한방 ASR 대체).

배경(2026-05-31 확정): Qwen3-ASR 은 긴 오디오를 한 번에 처리하면
  (a) 약한/짧은 발화(예: paramedic "Good")를 누락하고
  (b) 일부 단어(Get, lawyer)를 같은 시각에 중복 반환한다.
짧은 창으로 잘라 ASR 하면 둘 다 해결된다(boost 가 증거).

방식:
  1. vocals 를 WIN 초 창, HOP 초 간격(overlap=WIN-HOP)으로 잘라 각각 ASR.
  2. 각 창 단어를 global time 으로 보정.
  3. 시간순 정렬 후 중복 제거: 같은 단어가 시각 차 < DEDUP 이면 하나만.
     (overlap 영역에서 두 창이 같은 단어를 잡으므로 필수)
  → 어순 = 시간순(자연 보존), 약한 발화 회수, 중복 0.

vol 은 1.0 기본(증폭 없음 — 일반 구간 왜곡 방지). 약한 발화가 안 잡히면 1.5~2.0.

사용:
  python window_asr.py --audio <vocals.wav> --out <words.json> [--win 6 --hop 4 --vol 1.0]
"""
import argparse, json, subprocess, tempfile, os
from pathlib import Path
import requests

ASR_URL = "http://127.0.0.1:8902/transcribe"
LANG = "English"
DEDUP = 0.35  # 같은 단어가 이 시간차 내면 중복


def norm(w):
    return w.lower().rstrip(".!?,'\" ")


def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nk=1:nw=1", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--hop", type=float, default=4.0)
    ap.add_argument("--vol", type=float, default=1.0)
    args = ap.parse_args()

    tmp = Path(tempfile.gettempdir())
    src = Path(args.audio)
    if args.vol != 1.0:
        boosted = tmp / "wasr_boost.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ar", "16000", "-ac", "1", "-af", f"volume={args.vol}", str(boosted)], check=True)
        src = boosted
    total = probe_dur(args.audio)
    print(f"윈도우 ASR: {total:.1f}s, win={args.win}s hop={args.hop}s vol={args.vol}")

    # commit-region 방식: 각 window 는 WIN 초 ASR(맥락 확보)하되, 단어는 그 window 의
    # '소유 구간' [t, t+HOP) 에 start 가 떨어지는 것만 채택. 각 시간대를 정확히 한 window 가
    # 소유 → 겹침 구간 중복/어순섞임 원천 차단. 뒤쪽 look-ahead(WIN-HOP)는 맥락용일 뿐 다음
    # window 가 채택. 마지막 window 는 끝까지 채택.
    merged = []
    t = 0.0
    while t < total:
        ln = min(args.win, total - t)
        sub = tmp / f"wasr_{int(t*100)}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ss", str(t), "-t", str(ln), str(sub)], check=True)
        try:
            r = requests.post(ASR_URL, json={"audio_path": str(sub), "language": LANG}, timeout=90).json()
        except Exception as ex:
            print(f"  ASR fail @{t:.1f}s: {ex}"); t += args.hop; continue
        is_last = (t + args.win) >= total
        commit_end = total if is_last else (t + args.hop)
        for w in (r.get("words", []) or []):
            gs = float(w["start"]) + t
            ge = float(w["end"]) + t
            # 이 window 의 소유 구간 [t, commit_end) 에 단어 시작이 있을 때만 채택
            if t - 0.01 <= gs < commit_end:
                merged.append({"word": w["word"], "start": gs, "end": ge})
        try:
            os.remove(sub)
        except OSError:
            pass
        t += args.hop

    merged.sort(key=lambda x: x["start"])
    # 안전 dedup: commit-region 으로 이미 없지만, 경계 0.01s 오차로 인한 동일단어만 제거
    out = []
    for w in merged:
        if out and abs(w["start"] - out[-1]["start"]) < 0.05 and norm(w["word"]) == norm(out[-1]["word"]):
            continue
        out.append(w)
    merged = out

    json.dump({"words": merged, "n_words": len(merged), "method": "window_asr_commit"},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[✓] {len(merged)} words (commit-region, 어순보존) → {args.out}")


if __name__ == "__main__":
    main()
