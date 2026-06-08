"""발화얼굴 클러스터링: LightASD tracks+scores → ASD-gated 발화 트랙 fullframe임베딩 → cosine greedy cluster
→ distinct 발화 화자(시간span). GT 대조로 어느 GT화자에 해당하는지 검증.
사용: python _build_speakface.py <lightasd_dir> <gt_json> [offset=0] [fps=25]"""
import sys, pickle, numpy as np, json
sys.path.insert(0,"/workspace/src")
import face_clustering as fc
from insightface.app import FaceAnalysis
DIR=sys.argv[1]; GTP=sys.argv[2] if len(sys.argv)>2 else None
OFF=float(sys.argv[3]) if len(sys.argv)>3 else 0.0; FPS=float(sys.argv[4]) if len(sys.argv)>4 else 25.0
GATE=float(__import__("os").environ.get("ASD_GATE","0.3"))
SIM=float(__import__("os").environ.get("FACE_SIM","0.5"))
VID=f"{DIR}/pyavi/video.avi"
tr=pickle.load(open(f"{DIR}/pywork/tracks.pckl","rb")); sc=pickle.load(open(f"{DIR}/pywork/scores.pckl","rb"))
app=FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"]); app.prepare(ctx_id=0, det_size=(640,640))
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
# ASD-gated speaking tracks
items=[]
for i,t in enumerate(tr):
    fr=list(t['track']['frame']); bb=[list(b) for b in t['track']['bbox']]; s=np.array(sc[i])
    n=min(len(fr),len(s));
    if n<4: continue
    asd=float(np.mean(s[:n]))
    if asd<=GATE: continue
    emb=fc._track_avg_embedding(app, VID, fr, bb)
    if emb is None: continue
    # gender vote: top-3 largest-bbox 프레임에서 InsightFace .sex, faceSz 가중 다수결
    import cv2 as _cv2
    areas=[(b[2]-b[0])*(b[3]-b[1]) for b in bb]
    gv={}; cap=_cv2.VideoCapture(VID)
    for fi in sorted(range(len(bb)),key=lambda z:-areas[z])[:3]:
        cap.set(_cv2.CAP_PROP_POS_FRAMES,int(fr[fi])); ok,frm=cap.read()
        if not ok or frm is None: continue
        fs=app.get(frm)
        if not fs: continue
        bx=bb[fi]; bcx,bcy=(bx[0]+bx[2])/2,(bx[1]+bx[3])/2
        bf=min(fs,key=lambda f:abs((f.bbox[0]+f.bbox[2])/2-bcx)+abs((f.bbox[1]+f.bbox[3])/2-bcy))
        sx=(bf.bbox[2]-bf.bbox[0])*(bf.bbox[3]-bf.bbox[1])
        gv[bf.sex]=gv.get(bf.sex,0)+sx
    cap.release()
    gender=max(gv,key=gv.get) if gv else None
    items.append(dict(tid=i,asd=asd,emb=emb,gender=gender,t0=OFF+fr[0]/FPS,t1=OFF+fr[-1]/FPS,
        sz=float(np.mean([(b[2]-b[0])*(b[3]-b[1]) for b in bb]))))
print(f"ASD-gated speaking tracks: {len(items)} (gate>{GATE})")
# save ASD-gated tracks (with embeddings) for integration
import json as _json
_json.dump([{"tid":it["tid"],"asd":it["asd"],"gender":it.get("gender"),"t0":it["t0"],"t1":it["t1"],"sz":it["sz"],"emb":it["emb"].tolist()} for it in items],
           open(f"{DIR}/faces.json","w"))
print("saved", f"{DIR}/faces.json")
# greedy cosine cluster
clusters=[]  # each: list of item idx
for k,it in enumerate(items):
    best=None
    for ci,cl in enumerate(clusters):
        m=np.mean([cos(it['emb'],items[j]['emb']) for j in cl])
        if m>=SIM and (best is None or m>best[1]): best=(ci,m)
    if best: clusters[best[0]].append(k)
    else: clusters.append([k])
# GT
gt=json.load(open(GTP))["segments"] if GTP else None
def gtov(t0,t1):
    if not gt: return {}
    d={}
    for g in gt:
        o=max(0.0,min(float(g["end"]),t1)-max(float(g["start"]),t0))
        if o>0: d[g["speaker"]]=d.get(g["speaker"],0)+o
    return d
print(f"\n=== {len(clusters)} 발화얼굴 클러스터 (SIM>{SIM}) ===")
for ci,cl in enumerate(clusters):
    spans=sorted((items[j]['t0'],items[j]['t1']) for j in cl)
    asd=np.mean([items[j]['asd'] for j in cl]); sz=np.mean([items[j]['sz'] for j in cl])
    gtacc={}
    for j in cl:
        for sp,o in gtov(items[j]['t0'],items[j]['t1']).items(): gtacc[sp]=gtacc.get(sp,0)+o
    gtstr=", ".join("%s:%.1fs"%(k,v) for k,v in sorted(gtacc.items(),key=lambda x:-x[1])[:3])
    print("  C%d: %d tracks asd=%.2f faceSz=%.0f | spans %s | GT[%s]"%(
        ci,len(cl),asd,sz,",".join("%.1f-%.1f"%s for s in spans[:6]),gtstr))
