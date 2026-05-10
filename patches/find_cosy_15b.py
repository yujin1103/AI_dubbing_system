"""CosyVoice 1.5B 모델 가용성 정확히 조사 (ModelScope + HF)."""
import urllib.request
import json

# 1. ModelScope: 다양한 검색 query
queries = [
    ("ModelScope iic", "https://www.modelscope.cn/api/v1/models?Owner=iic&Search=cosyvoice&PageSize=50"),
    ("ModelScope iic v2", "https://modelscope.cn/api/v1/models?Owner=iic&Search=cosyvoice&PageSize=50"),
    ("ModelScope no owner", "https://modelscope.cn/api/v1/models?Search=CosyVoice3&PageSize=50"),
]

for label, url in queries:
    print(f"\n=== {label} ===")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        models = data.get("Data", {}).get("Models") or data.get("data", {}).get("models") or []
        print(f"  found {len(models)} models")
        for m in models[:30]:
            name = m.get("Name") or m.get("name") or "?"
            owner = m.get("Owner") or m.get("owner") or "?"
            mid = m.get("Path") or m.get("path") or m.get("ModelId") or "?"
            print(f"  {owner}/{name} (id: {mid})")
    except Exception as e:
        print(f"  failed: {e}")

# 2. HF API: 모든 author + cosyvoice
print("\n\n=== HF: full search 'cosyvoice' all authors ===")
try:
    from huggingface_hub import HfApi
    api = HfApi()
    res = api.list_models(search="cosyvoice", limit=50)
    for m in res:
        if "cosy" in m.id.lower() or "fun" in m.id.lower():
            tags = ", ".join(m.tags[:5]) if m.tags else "(no tags)"
            print(f"  {m.id} - {tags[:80]}")
except Exception as e:
    print(f"  failed: {e}")

# 3. CosyVoice3 paper에서 언급된 model 크기 확인
print("\n\n=== Paper info reminder ===")
print("  CosyVoice3 paper (arxiv 2505.17589): 0.5B, 1.5B 변종 모두 학습됨")
print("  공개 여부 = 별도 확인 필요")
