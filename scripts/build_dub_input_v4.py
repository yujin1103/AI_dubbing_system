#!/usr/bin/env python3
"""dub 입력 v4 — 문장 분할(침묵 기준)과 화자 배정(dominant)을 분리.

사용자 설계(2026-05-31):
  문제: 화자분리 구간으로 문장을 자르면 문장이 토막나고, LLM 문장복원은 과의존.
  해결:
    1) segment = 문장 단위. ASR 단어를 '침묵(gap)' 기준으로 묶는다 (LLM 불필요).
       단어 간 gap > SENT_GAP 면 새 문장. 너무 길면(MAX_DUR) 분할.
    2) 화자 = 그 문장 시간대에 '가장 많이 말한 화자'(dominant by overlap).
       = "이 화자가 (그 문장을) 끝까지 발언함". 검증된 gapfilled 화자분리는 배정에만 사용.
  → 문장 안 토막남 + 화자 겹침 없음(문장은 시간순) + 화자분리 결과(1.1167) 보존 + LLM 0 의존.

BG 화자: --include-bg 면 BG 문장도 포함(원본 유지/번역은 dub 단계 정책). 기본은 포함.

사용:
  python build_dub_input_v4.py --gapfilled <gapfilled.json> --words <words.json> --out <dub.json>
        [--sent-gap 0.55] [--max-dur 12] [--no-bg]
"""
import argparse, json
from collections import Counter, defaultdict


def load_words(path):
    d = json.load(open(path, encoding="utf-8"))
    w = d.get("words", d) if isinstance(d, dict) else d
    out = []
    for x in w:
        if not isinstance(x, dict):
            continue
        word = (x.get("word") or x.get("text") or "").strip()
        s, e = x.get("start"), x.get("end")
        if not word or s is None or e is None:
            continue
        out.append({"word": word, "start": float(s), "end": float(e),
                    "src": x.get("src", "main")})
    return sorted(out, key=lambda z: z["start"])


def load_groups(path):
    d = json.load(open(path, encoding="utf-8"))
    g = d.get("groups", d if isinstance(d, list) else [])
    out = []
    for x in g:
        out.append({
            "speaker": x.get("speaker", "SPEAKER_00"),
            "start": float(x.get("group_start", x.get("start", 0))),
            "end": float(x.get("group_end", x.get("end", 0))),
        })
    return sorted(out, key=lambda z: z["start"])


def dominant_speaker(groups, s, e):
    """[s,e] 와 overlap 시간이 가장 큰 화자 = '끝까지 발언한 화자'."""
    acc = defaultdict(float)
    for g in groups:
        ov = min(e, g["end"]) - max(s, g["start"])
        if ov > 0:
            acc[g["speaker"]] += ov
    if acc:
        return max(acc.items(), key=lambda kv: kv[1])[0]
    # overlap 전혀 없으면(드묾) 중심 최근접 화자
    c = (s + e) / 2.0
    return min(groups, key=lambda g: min(abs(g["start"] - c), abs(g["end"] - c)))["speaker"]


def split_sentences(words, sent_gap, max_dur, groups):
    """단어를 (1)단어별 dominant 화자 (2)침묵 기준으로 묶기.
    새 문장 조건: 화자 바뀜 OR 침묵>sent_gap OR 길이>max_dur.
    → 한 문장에 한 화자만 → 화자 걸침/오배정 없음 (사용자 설계)."""
    # 단어별 화자: 단어 '시작 시각'을 포함하는 화자(중심 대신 시작 사용 —
    # 경계를 걸친 단어가 다음 화자로 잘못 넘어가 연속 구절이 쪼개지는 것 방지).
    # 예) "Find another school"의 'school'이 엄마/아빠 경계를 걸쳐도 시작이 엄마면 엄마로.
    for w in words:
        st = float(w["start"])
        spk = None
        for g in groups:
            if g["start"] <= st <= g["end"]:
                spk = g["speaker"]
                break
        if spk is None:
            c = (w["start"] + w["end"]) / 2.0
            spk = min(groups, key=lambda g: min(abs(g["start"] - c), abs(g["end"] - c)))["speaker"]
        w["_spk"] = spk

    sents = []
    cur = None
    for w in words:
        if cur is None:
            cur = [w]
            continue
        gap = w["start"] - cur[-1]["end"]
        too_long = (w["end"] - cur[0]["start"]) > max_dur
        spk_change = w["_spk"] != cur[-1]["_spk"]
        if spk_change or gap > sent_gap or too_long:
            sents.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        sents.append(cur)
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gapfilled", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sent-gap", type=float, default=0.55, help="단어 간 침묵 이상이면 새 문장")
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--no-bg", action="store_true", help="BG 화자 문장 제외")
    ap.add_argument("--speaker-split", action="store_true",
                    help="한 문장 안에서 dominant 화자가 연속으로 바뀌면 화자 경계로 추가 분할")
    ap.add_argument("--bg-short-max", type=float, default=0.0,
                    help="이보다 짧은 세그먼트는 BG로(원본오디오 유지·번역X). 0이면 끔(기본). 인식된 짧은 대사는 번역 유지.")
    ap.add_argument("--merge-short", type=float, default=0.6,
                    help="이보다 짧은 fragment 를 인접 같은-화자 세그먼트에 병합(LLM 번역 단위 일관성). 0이면 끔.")
    ap.add_argument("--merge-gap", type=float, default=1.2,
                    help="짧은 fragment 병합 시 인접 세그먼트와의 최대 gap")
    ap.add_argument("--bg-empty-min", type=float, default=0.3,
                    help="화자분리는 발화인데 ASR 단어가 없는 구간이 이 이상이면 BG 세그먼트 신규 추가. 0이면 끔.")
    ap.add_argument("--vocals", default=None,
                    help="보컬 stem wav. 주면 BG-empty 구간을 vocals 에너지로 게이트(진짜 놓친 발화만 BG, 침묵 제외)")
    ap.add_argument("--bg-energy-rms", type=float, default=0.015,
                    help="BG-empty 구간의 vocals RMS 가 이 이상일 때만 BG(=실제 발화). 침묵 갭 제외용")
    ap.add_argument("--raw-diar", default=None,
                    help="raw(freshraw) 화자분리 json. diar_gap-only 세그먼트가 raw 에서 ≥2 클러스터와 "
                         "겹치면(겹침 외침) BG로. 단일 클러스터(예 Good)는 번역 유지.")
    args = ap.parse_args()

    groups = load_groups(args.gapfilled)
    words = load_words(args.words)
    if not groups or not words:
        raise SystemExit("[v4] ERROR: empty groups or words")

    # raw 화자분리 (겹침 외침 판정용): [s,e] 와 겹치는 distinct 화자 수
    raw_groups = load_groups(args.raw_diar) if args.raw_diar else []
    def raw_cluster_count(s, e):
        spks = set()
        for g in raw_groups:
            if min(e, g["end"]) - max(s, g["start"]) > 0.05:
                spks.add(g["speaker"])
        return len(spks)

    sents = split_sentences(words, args.sent_gap, args.max_dur, groups)

    segs = []
    for sent in sents:
        s = sent[0]["start"]
        e = sent[-1]["end"]
        # 화자: 문장 내 단어는 모두 같은 _spk (split_sentences가 화자 바뀜에 분할).
        # 안전을 위해 overlap dominant 로 재확인.
        spk = sent[0].get("_spk") or dominant_speaker(groups, s, e)
        text = " ".join(w["word"] for w in sent).strip()
        # 메인 ASR 단어가 없고 diar_gap 회수로만 된 세그먼트 = 신뢰 낮은 추측.
        has_main = any(w.get("src", "main") != "diar_gap" for w in sent)
        segs.append({"speaker": spk, "start": round(s, 3), "end": round(e, 3), "text": text,
                     "_rec": not has_main})

    # 외침 버스트 판정: recovery-only 세그먼트가 ±2s 안에 다른 recovery-only 와 몰려있으면
    # (빠른 겹침 외침) → BG 원본 passthrough. 고립된 recovery-only(예 'Good')는 번역 유지.
    rec_idx = [i for i, s in enumerate(segs) if s["_rec"]]
    for i in rec_idx:
        si = segs[i]
        cluster = any(j != i and abs(segs[j]["start"] - si["start"]) <= 2.0 for j in rec_idx)
        if cluster:
            si["speaker"] = "SPEAKER_BG_auto"
    for s in segs:
        s.pop("_rec", None)

    # BG 필터
    if args.no_bg:
        segs = [s for s in segs if not s["speaker"].startswith("SPEAKER_BG")]

    # 겹침 안전장치: 정렬 후 인접 segment 가 겹치면 경계 조정(앞 end 를 뒤 start 로)
    segs.sort(key=lambda x: (x["start"], x["end"]))
    for a, b in zip(segs, segs[1:]):
        if a["end"] > b["start"]:
            a["end"] = round(b["start"], 3)
    # 0초/초단문(<0.12s) 정리: 직전 같은 화자 segment 에 텍스트 병합, 없으면 제거.
    cleaned = []
    for s in segs:
        if s["end"] - s["start"] >= 0.12:
            cleaned.append(s)
        elif cleaned and cleaned[-1]["speaker"] == s["speaker"] and (s["start"] - cleaned[-1]["end"]) < 0.6:
            cleaned[-1]["text"] = (cleaned[-1]["text"] + " " + s["text"]).strip()
            cleaned[-1]["end"] = max(cleaned[-1]["end"], s["end"])
        # else: 0초 단독 → 버림 (합성 불가)
    segs = cleaned

    # 짧은 fragment 를 인접 같은-화자 세그먼트에 병합 (LLM 번역 단위 일관성 — 사용자 우려 2026-05-31)
    merged_log = []
    if args.merge_short > 0:
        changed = True
        while changed:
            changed = False
            for i, s in enumerate(segs):
                if (s["end"] - s["start"]) >= args.merge_short:
                    continue
                cands = []
                if i > 0 and segs[i - 1]["speaker"] == s["speaker"]:
                    cands.append((s["start"] - segs[i - 1]["end"], i - 1))
                if i < len(segs) - 1 and segs[i + 1]["speaker"] == s["speaker"]:
                    cands.append((segs[i + 1]["start"] - s["end"], i + 1))
                cands = [(g, j) for g, j in cands if g <= args.merge_gap]
                if not cands:
                    continue
                _, j = min(cands)
                tgt = segs[j]
                merged_log.append((round(s["start"], 1), s["text"][:18]))
                tgt["start"] = min(tgt["start"], s["start"])
                tgt["end"] = max(tgt["end"], s["end"])
                tgt["text"] = ((tgt["text"] + " " + s["text"]) if j < i
                               else (s["text"] + " " + tgt["text"])).strip()
                segs.pop(i)
                changed = True
                break

    # === BG 라우팅 (사용자 기준 2026-05-31): 텍스트빈 구간 + 짧은(<bg_short_max) → BG passthrough ===
    bg_short = []
    if args.bg_short_max > 0:
        for s in segs:
            if (s["end"] - s["start"]) < args.bg_short_max and not s["speaker"].startswith("SPEAKER_BG"):
                bg_short.append((round(s["start"], 2), round(s["end"], 2), s["text"]))
                s["speaker"] = "SPEAKER_BG_auto"  # 번역X, 원본 오디오 유지
    # vocals 에너지 게이트용 로더 (있으면 침묵 갭을 BG에서 제외)
    voc = None; voc_sr = 16000
    if args.vocals:
        try:
            import soundfile as _sf
            voc, voc_sr = _sf.read(args.vocals)
            if getattr(voc, "ndim", 1) > 1:
                import numpy as _np; voc = _np.mean(voc, axis=1)
        except Exception as ex:
            print(f"  [warn] vocals 로드 실패({ex}) → 에너지 게이트 생략")
            voc = None

    def voc_rms(s, e):
        if voc is None:
            return 1.0  # 게이트 없음 → 항상 통과
        import numpy as _np
        a = voc[int(s * voc_sr):int(e * voc_sr)]
        return float(_np.sqrt(_np.mean(_np.square(a)) + 1e-12)) if len(a) else 0.0

    # 화자분리는 발화라는데 단어가 안 잡힌(=ASR 빈) 구간 → BG 세그먼트 신규 추가
    bg_empty = []
    if args.bg_empty_min > 0 and segs:
        RES = 0.05
        total = max(max(g["end"] for g in groups), max(s["end"] for s in segs))
        nb = int(total / RES) + 1
        covered = [False] * nb
        spoken = [None] * nb  # 그 bin 의 diarization 화자(있으면)
        for s in segs:
            for b in range(int(s["start"] / RES), min(int(s["end"] / RES) + 1, nb)):
                covered[b] = True
        for g in groups:
            for b in range(int(g["start"] / RES), min(int(g["end"] / RES) + 1, nb)):
                if spoken[b] is None:
                    spoken[b] = g["speaker"]
        # 화자는 있는데(spoken) covered 안 된 연속 구간 추출
        b = 0
        while b < nb:
            if spoken[b] is not None and not covered[b]:
                b0 = b
                while b < nb and spoken[b] is not None and not covered[b]:
                    b += 1
                hs, he = b0 * RES, b * RES
                # 에너지 게이트: 그 구간 vocals RMS 가 충분할 때만(=진짜 놓친 발화) BG. 침묵 갭 제외.
                if he - hs >= args.bg_empty_min and voc_rms(hs, he) >= args.bg_energy_rms:
                    bg_empty.append((round(hs, 3), round(he, 3)))
                    segs.append({"speaker": "SPEAKER_BG_auto", "start": round(hs, 3),
                                 "end": round(he, 3), "text": ""})
            else:
                b += 1
        segs.sort(key=lambda x: (x["start"], x["end"]))

    n_spk = len({s["speaker"] for s in segs})
    win = len(words)
    wout = sum(len(s["text"].split()) for s in segs)
    json.dump({"segments": segs, "n_speakers": n_spk},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[v4] {len(segs)} 문장 segment, {n_spk} speakers | words in={win} out={wout}")
    print(f"  speakers: {dict(Counter(s['speaker'] for s in segs))}")
    if merged_log:
        print(f"  [짧은파편 병합] {len(merged_log)}개 → 인접 같은화자: " +
              ", ".join("%.1fs:%r" % (s, t) for s, t in merged_log))
    if bg_short:
        print(f"  [BG←짧음<{args.bg_short_max}s] {len(bg_short)}개 (원본유지·번역X): " +
              ", ".join("%.1f-%.1fs:%r" % (s, e, t[:18]) for s, e, t in bg_short))
    if bg_empty:
        print(f"  [BG←ASR빈] {len(bg_empty)}개 (화자있음·단어없음→원본유지): " +
              ", ".join("%.1f-%.1fs" % (s, e) for s, e in bg_empty))
    # 짧은(<0.4s) segment 경고
    sh = [s for s in segs if s["end"] - s["start"] < 0.4]
    if sh:
        print(f"  [note] {len(sh)} segment <0.4s (짧은 발화): " +
              ", ".join("%.1fs:%r" % (x["end"] - x["start"], x["text"][:15]) for x in sh[:6]))


if __name__ == "__main__":
    main()
