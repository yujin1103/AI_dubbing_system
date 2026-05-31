#!/usr/bin/env python3
"""gap 한정 boost ASR — 메인 ASR이 비운 침묵 구간(gap)에만 boost 적용.

배경(2026-05-30 발견):
  메인 ASR(풀 오디오 한 번에)은 짧은 overlap 발화(예: test4 60.5s "Good")를 놓친다.
  같은 오디오를 짧게 잘라 증폭(boost)하면 인식된다. 단 전체 구간 boost는
  (a) 단어 중복 오염("gonna going to"), (b) 빠른 진짜 반복("no no no")을 가짜 중복으로
  오인 제거할 위험이 있다.

해결(gap 한정):
  메인 ASR 단어 사이 GAP_MIN(기본 0.8s)+ 의 빈 구간만 골라 boost ASR.
  → 메인이 이미 촘촘히 잡은 구간은 절대 안 건드림 → 가짜 중복·진짜 반복 손실 모두 회피.
  → 메인이 비운 곳(짧은 발화가 숨은 곳)만 보강.

처리:
  1. 메인 words.json 로드 → 인접 단어 간 gap 계산
  2. gap >= GAP_MIN 인 구간 목록 (영상 시작/끝 여백도 포함)
  3. 각 gap 을 ±PAD 확장해 잘라 boost(volume) → ASR
  4. gap 안에 떨어지는 새 단어만 채택 (경계 메인 단어와 0.3s+같은단어면 skip)
  5. 합쳐 words_gapboost.json 저장

사용:
  python gap_boost_asr.py --chunk <mp4> --words <main_words.json> --out <out.json>
                          [--gap-min 0.8] [--vol 3.0] [--pad 0.3]
"""
import argparse, json, subprocess, tempfile, os
from pathlib import Path
import requests

ASR_URL = "http://127.0.0.1:8902/transcribe"
LANG = "English"


def norm(w):
    return w.lower().rstrip(".!?,'\" ")


def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nk=1:nw=1", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def find_gaps(words, total, gap_min, edge_min=0.6):
    """메인 단어 사이 빈 구간 (>= gap_min). 시작/끝 여백은 edge_min."""
    iv = sorted(((float(w["start"]), float(w["end"])) for w in words), key=lambda x: x[0])
    gaps = []
    # 시작 여백
    if iv and iv[0][0] >= edge_min:
        gaps.append((0.0, iv[0][0]))
    elif not iv:
        return [(0.0, total)]
    # 단어 사이
    for (s0, e0), (s1, e1) in zip(iv, iv[1:]):
        if s1 - e0 >= gap_min:
            gaps.append((e0, s1))
    # 끝 여백
    if total - iv[-1][1] >= edge_min:
        gaps.append((iv[-1][1], total))
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vol", type=float, default=3.0)
    ap.add_argument("--pad", type=float, default=0.3, help="gap 양쪽 확장 (경계 발화 포착)")
    ap.add_argument("--gap-min", dest="gap_min", type=float, default=0.6,
                    help="메인 ASR 빈 구간이 이 이상이면 boost 대상")
    ap.add_argument("--silence-db", type=float, default=-55.0,
                    help="boost 후 mean_volume 이 이하면 침묵으로 보고 skip(환각 방지)")
    ap.add_argument("--max-word-dur", dest="max_word_dur", type=float, default=2.0,
                    help="단일 단어 span 이 이보다 길면 환각으로 보고 거부('Hello' 9.5s 등)")
    ap.add_argument("--min-word-dur", dest="min_word_dur", type=float, default=0.06,
                    help="단어 span 이 이보다 짧으면(zero-duration) 거부")
    args = ap.parse_args()

    d = json.load(open(args.words, encoding="utf-8"))
    existing = d.get("words", d) if isinstance(d, dict) else d
    existing = [w for w in existing if "start" in w and "end" in w]
    print(f"기존 words: {len(existing)}")

    tmp = Path(tempfile.gettempdir())
    raw = tmp / "gb_raw.wav"
    boost = tmp / "gb_boost.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.chunk,
                    "-ar", "16000", "-ac", "1", str(raw)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                    "-af", f"volume={args.vol}", str(boost)], check=True)
    total = probe_dur(raw)

    gaps = find_gaps(existing, total, args.gap_min)
    gap_secs = sum(e - s for s, e in gaps)
    print(f"전체 {total:.1f}s | gap>={args.gap_min}s 구간 {len(gaps)}개 ({gap_secs:.1f}s, "
          f"전체의 {100*gap_secs/max(total,1):.0f}%) boost x{args.vol}")

    def mean_db(path):
        # ffmpeg volumedetect mean_volume (dBFS). 침묵일수록 매우 낮음(-90 근처).
        r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path),
                            "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True)
        for line in (r.stderr or "").splitlines():
            if "mean_volume" in line:
                try:
                    return float(line.split("mean_volume:")[1].split("dB")[0].strip())
                except (ValueError, IndexError):
                    pass
        return -99.0

    fresh = []
    skipped_silent = 0
    rejected_hallu = []  # (start, end, word, reason) — 환각 필터에 걸린 단어
    for gs, ge in gaps:
        a = max(0.0, gs - args.pad)
        b = min(total, ge + args.pad)
        if b - a < 0.2:
            continue
        sub = tmp / f"gb_{int(a*100)}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(boost),
                        "-ss", str(a), "-t", str(b - a), str(sub)], check=True)
        # 에너지 게이트: boost 후에도 너무 조용하면(순수 침묵) ASR 환각 방지 위해 skip.
        # boost x3 적용된 클립 기준 -55dB 미만이면 발화 없음으로 간주.
        if mean_db(sub) < args.silence_db:
            skipped_silent += 1
            try:
                os.remove(sub)
            except OSError:
                pass
            continue
        try:
            r = requests.post(ASR_URL, json={"audio_path": str(sub), "language": LANG}, timeout=90).json()
        except Exception as ex:
            print(f"  ASR fail @{a:.1f}s: {ex}")
            continue
        for w in (r.get("words", []) or []):
            ws = float(w["start"]) + a
            we = float(w["end"]) + a
            wc = (ws + we) / 2.0
            # 단어 중심이 실제 gap 안(원래 gs~ge)에 있을 때만 — pad 영역 메인단어 중복 방지
            if not (gs - 0.05 <= wc <= ge + 0.05):
                continue
            # 환각 필터: 단일 단어가 비현실적으로 길거나(boost된 노이즈/음악을 한 단어로 환각)
            # zero-duration(타임스탬프 깨짐)이면 거부. 더빙 소스 텍스트 오염 방지.
            dur = we - ws
            if dur > args.max_word_dur:
                rejected_hallu.append((ws, we, w["word"], f"dur {dur:.1f}s>{args.max_word_dur}"))
                continue
            if dur < args.min_word_dur:
                rejected_hallu.append((ws, we, w["word"], f"dur {dur:.2f}s<{args.min_word_dur}"))
                continue
            # 경계 메인 단어와 가깝고 같은 단어면 skip (가짜 중복)
            if any(abs(ws - e["start"]) < 0.3 and norm(w["word"]) == norm(e["word"]) for e in existing):
                continue
            if any(abs(ws - f["start"]) < 0.25 and norm(w["word"]) == norm(f["word"]) for f in fresh):
                continue
            fresh.append({"word": w["word"], "start": ws, "end": we})
        try:
            os.remove(sub)
        except OSError:
            pass

    if rejected_hallu:
        print(f"\n환각 필터: {len(rejected_hallu)}개 단어 거부 (비현실적 span/zero-duration):")
        for s, e, word, reason in rejected_hallu:
            print(f"  ✗ {s:6.2f}-{e:6.2f}  {word!r}  [{reason}]")

    print(f"\ngap-boost: {skipped_silent}개 gap 침묵으로 skip, 새 단어 {len(fresh)}개:")
    for w in fresh:
        print(f"  {w['start']:6.2f}-{w['end']:6.2f}  {w['word']!r}")

    combined = list(existing) + fresh
    combined.sort(key=lambda x: float(x["start"]))
    json.dump({"words": combined, "gapboost_added": len(fresh)},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[✓] {len(combined)} words (+{len(fresh)} gap-boost) → {args.out}")


if __name__ == "__main__":
    main()
