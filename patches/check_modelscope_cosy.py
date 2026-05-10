"""ModelScope에서 CosyVoice3 1.5B 같은 모델 직접 검색."""
import urllib.request
import json

# ModelScope API: /api/v1/models?Owner={author}
queries = [
    "https://modelscope.cn/api/v1/models?PageSize=30&Owner=iic&SearchKey=cosyvoice",
    "https://modelscope.cn/api/v1/models?PageSize=30&Owner=FunAudioLLM&SearchKey=cosyvoice",
]

for url in queries:
    print(f"=== {url} ===")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("Data", {}).get("Models", [])
        for m in models[:30]:
            mid = m.get("Path", m.get("Name", "?"))
            owner = m.get("Owner", "?")
            print(f"  {owner}/{mid}")
    except Exception as e:
        print(f"  failed: {e}")
    print()
