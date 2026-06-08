"""전략 S2 = 베이스라인 v1 additive face-distinct split(S1) + (a)word-split 병합 + (b)경계조각 반환.

베이스 _integrate.py 의 v1-extra (강ASD distinct 얼굴 → 새 FACE_ 화자) 를 S1 으로 그대로 사용
(test4 paramedic=FACE_6 → 6/6 유지). 그 위에:
  (a) '<spk>__sN' (word_level_split 과분할) → 베이스 '<spk>' 로 병합.
      단 그 __sN 세그의 발화얼굴 identity 가 베이스 dom 얼굴과 명확히 다르면 유지.
  (b) 짧은(<FRAG_DUR) 경계조각이 catch-all 라벨인데, 그 발화얼굴 identity 가
      다른 깨끗한 voice 클러스터의 dom 얼굴과 cosine>=RETURN_COS 로 강하게 일치 → 반환.
      (catch-all = 비-BG 중 서로 다른 발화얼굴 identity 가 가장 많은 voice 클러스터)

모든 임계 공통(per-video 하드코딩/화자수강제 없음). 데이터(거리임계/얼굴 identity)로 자동결정.
사용: /opt/venv_diarizen/bin/python _sep_S2.py <test4|test5|test6>
"""
import sys, json, numpy as np, os
from collections import defaultdict, Counter

V = sys.argv[1]
RUN = {"test4": "20260528_090246_test4fresh_74d347/meta/test4_fresh_chunk_000",
       "test5": "20260528_092242_test5fresh_86fb30/meta/test5_fresh_chunk_000",
       "test6": "test6_manual/meta/test6_chunk_000"}[V]
GAP = f"/workspace/media/runs/{RUN}_segments_gapfilled.json"
FACES = f"/workspace/_asd_full/demo/{V}full/faces.json"
GTP = f"/workspace/media/gt/{V}_gt.json"

# ---- 공통 임계 (베이스 _integrate.py 와 동일값 유지) ----
SPEAK = float(os.environ.get("SPEAK", "0.5"))
FACE_CLUS_COS = float(os.environ.get("FACE_CLUS_COS", "0.45"))
V1_ASD = float(os.environ.get("V1_ASD", "1.5"))     # v1 강ASD (additive split)
SPLIT = float(os.environ.get("SPLIT_TH", "0.35"))   # voice 대표얼굴과 distinct 판정
# ---- S2 추가 임계 ----
RETURN_COS = float(os.environ.get("RETURN_COS", "0.50"))  # 경계조각 반환
FRAG_DUR = float(os.environ.get("FRAG_DUR", "1.2"))       # 짧은 경계조각 상한


def load(p):
    d = json.load(open(p)); return d.get("groups", d)
def s0(x): return float(x.get("start", x.get("group_start", 0)))
def e0(x): return float(x.get("end", x.get("group_end", 0)))
def ov(a0, a1, b0, b1): return max(0.0, min(a1, b1) - max(a0, b0))
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


voice = [{"s": s0(x), "e": e0(x), "spk": x["speaker"]} for x in load(GAP)]
tracks = [t for t in json.load(open(FACES)) if t["asd"] >= SPEAK]
for t in tracks:
    t["emb"] = np.array(t["emb"])

# ---- face identity 클러스터 (큰 얼굴부터 그리디) ----
cl = []
for it in sorted(range(len(tracks)), key=lambda i: -tracks[i]["sz"]):
    best = None
    for ci, c in enumerate(cl):
        m = np.mean([cos(tracks[it]["emb"], tracks[j]["emb"]) for j in c])
        if m >= FACE_CLUS_COS and (best is None or m > best[1]):
            best = (ci, m)
    if best:
        cl[best[0]].append(it)
    else:
        cl.append([it])
fclus = [{"emb": np.mean([tracks[j]["emb"] for j in c], axis=0),
          "spans": [(tracks[j]["t0"], tracks[j]["t1"]) for j in c],
          "asd": np.mean([tracks[j]["asd"] for j in c]),
          "dur": sum(tracks[j]["t1"] - tracks[j]["t0"] for j in c),
          "id": i} for i, c in enumerate(cl)]

# voice 화자의 대표 얼굴 (overlap 최대 클러스터)
spk_face = {}
for spk in set(v["spk"] for v in voice):
    segs = [v for v in voice if v["spk"] == spk]; best = None
    for c in fclus:
        o = sum(ov(a, b, v["s"], v["e"]) for a, b in c["spans"] for v in segs)
        if o > 0 and (best is None or o > best[0]):
            best = (o, c["emb"])
    spk_face[spk] = best[1] if best else None

# ---- v1 additive: 강ASD distinct 얼굴 → 새 FACE_ 화자 (= S1) ----
extra = []
for c in fclus:
    if c["asd"] < V1_ASD or c["dur"] < 0.4:
        continue
    vb = None
    for spk in set(v["spk"] for v in voice):
        o = sum(ov(a, b, v["s"], v["e"]) for a, b in c["spans"] for v in voice if v["spk"] == spk)
        if o > 0 and (vb is None or o > vb[0]):
            vb = (o, spk)
    vf = spk_face.get(vb[1]) if vb else None
    if vf is None or cos(c["emb"], vf) < SPLIT:
        extra.append(c)


def dom_extra(s, e):
    best = None
    for c in extra:
        o = sum(ov(s, e, a, b) for a, b in c["spans"])
        if o > 0.3 * (e - s) and (best is None or o > best[0]):
            best = (o, c["id"])
    return best[1] if best else None


# ---- seg 의 발화얼굴 emb (overlap 최대 클러스터) ----
def seg_face_emb(s, e):
    best = None
    for c in fclus:
        o = sum(ov(s, e, a, b) for a, b in c["spans"])
        if o > 0 and (best is None or o > best[0]):
            best = (o, c["emb"])
    return best[1] if best else None


def is_bg(spk): return "BG" in spk
def base_label(spk): return spk.split("__")[0]
def is_wordsplit(spk): return "__" in spk

# ---- 초기 라벨 = v1 적용 결과 ----
for v in voice:
    fe = dom_extra(v["s"], v["e"])
    v["lab"] = f"FACE_{fe}" if fe is not None else v["spk"]

# ---- (a) S2 word-split 병합 ----
spks0 = set(v["spk"] for v in voice)
for ws in [s for s in spks0 if is_wordsplit(s)]:
    base = base_label(ws)
    bemb = spk_face.get(base)
    for v in voice:
        if v["spk"] != ws:
            continue
        if v["lab"].startswith("FACE_"):
            continue  # v1 이 이미 새화자로 분리한 건 건드리지 않음
        if bemb is None:
            v["lab"] = base
            continue
        fe = seg_face_emb(v["s"], v["e"])
        if fe is None or cos(fe, bemb) >= FACE_CLUS_COS:
            v["lab"] = base
        # 명확히 다른 identity 면 유지

# ---- catch-all 식별 (robust): 발화얼굴이 '자기 소유'보다 '타인 소유'가 많은 voice 클러스터 ----
# 각 face identity 의 소유주 = 그 얼굴 spans 와 가장 많이 겹치는 voice 클러스터(원본 spk).
# catch-all = (타인소유 발화얼굴 duration) > (자기소유) 이고 othR>=CATCH_OTH 인 비-BG 클러스터.
# 실제 화자(man1 등)는 reverse-shot 으로 타인얼굴이 보여도 othR 가 낮음 → 안전.
def seg_face_id(s, e):
    best = None
    for c in fclus:
        o = sum(ov(s, e, a, b) for a, b in c["spans"])
        if o > 0 and (best is None or o > best[0]):
            best = (o, c["id"])
    return best[1] if best else None

CATCH_OTH = float(os.environ.get("CATCH_OTH", "0.50"))
orig_spks = set(v["spk"] for v in voice)
face_owner = {}
for c in fclus:
    best = None
    for spk in orig_spks:
        o = sum(ov(a, b, v["s"], v["e"]) for a, b in c["spans"] for v in voice if v["spk"] == spk)
        if o > 0 and (best is None or o > best[0]):
            best = (o, spk)
    face_owner[c["id"]] = best[1] if best else None

cur_spks = set(v["lab"] for v in voice)
nonbg = [s for s in cur_spks if not is_bg(s) and not s.startswith("FACE_")]
def oth_ratio(spk):
    selfd = otherd = noned = 0.0
    for v in voice:
        if v["spk"] != spk:
            continue
        dur = v["e"] - v["s"]
        fid = seg_face_id(v["s"], v["e"])
        if fid is None:
            noned += dur
        elif face_owner.get(fid) == spk:
            selfd += dur
        else:
            otherd += dur
    tot = selfd + otherd + noned
    return otherd / max(tot, 1e-6), otherd, selfd
catchall = None
best_oth = CATCH_OTH
for s in nonbg:
    r, od, sd = oth_ratio(s)
    if r >= best_oth and od > sd:
        best_oth = r; catchall = s

# 반환 대상 = catch-all 아닌 모든 화자(voice + FACE_ additive)로 대표얼굴 존재.
# FACE_ 새화자도 대표얼굴을 가지므로 mom/dad 조각이 그쪽으로 통합됨.
return_targets = {}
for s in cur_spks:
    if s == catchall or is_bg(s):
        continue
    if s.startswith("FACE_"):
        fid = int(s.split("_")[1])
        return_targets[s] = fclus[fid]["emb"]
    elif spk_face.get(s) is not None:
        return_targets[s] = spk_face[s]

# ---- (b) S2 경계조각 반환 (얼굴 identity 가 강 일치할 때) ----
# 매우 강한(>=STRONG_RETURN) 일치는 길이 무관, 약한 일치는 짧은 조각(<FRAG_DUR)만.
STRONG_RETURN = float(os.environ.get("STRONG_RETURN", "0.80"))
if catchall is not None and return_targets:
    for v in voice:
        if v["lab"] != catchall:
            continue
        fe = seg_face_emb(v["s"], v["e"])
        if fe is None:
            continue
        bestc = None
        for s, emb in return_targets.items():
            c = cos(fe, emb)
            if (bestc is None or c > bestc[0]):
                bestc = (c, s)
        if bestc is None:
            continue
        cval, s = bestc
        short = (v["e"] - v["s"]) <= FRAG_DUR
        if cval >= STRONG_RETURN or (short and cval >= RETURN_COS):
            v["lab"] = s

final = [{"s": v["s"], "e": v["e"], "spk": v["lab"]} for v in voice]

print(f"[{V}] v1-extra={len(extra)} catch-all={catchall} return_targets={list(return_targets.keys())}")
print("    최종화자:", sorted(set(f["spk"] for f in final)))

if os.path.exists(GTP):
    gt = json.load(open(GTP))["segments"]
    m = defaultdict(lambda: defaultdict(int))
    for g in gt:
        gs, ge = float(g["start"]), float(g["end"]); bb = None
        for f in final:
            o = ov(gs, ge, f["s"], f["e"])
            if o > 0 and (bb is None or o > bb[0]):
                bb = (o, f["spk"])
        m[g["speaker"]][bb[1] if bb else "∅"] += 1
    prim = {g: max(d, key=d.get) for g, d in m.items()}
    lc = Counter(prim.values()); c = 0
    for g, d in m.items():
        l = max(d, key=d.get)
        if sum(d.values()) == d[l] and lc[l] == 1:
            c += 1
    print(f"=== GT clean 1:1 = {c}/{len(m)} ===")
    for g in m:
        print("  %-8s -> %s" % (g, dict(m[g])))
