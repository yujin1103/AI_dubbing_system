"""사용 가능한 CosyVoice 모델 변종 조사."""
from huggingface_hub import HfApi
api = HfApi()

print("=== FunAudioLLM models on HuggingFace ===")
try:
    models = list(api.list_models(author="FunAudioLLM", limit=30))
    for m in sorted(models, key=lambda x: x.id):
        print(f"  {m.id}")
except Exception as e:
    print(f"  HF list failed: {e}")

print("\n=== iic models on HuggingFace ===")
try:
    models = list(api.list_models(author="iic", limit=30, search="cosyvoice"))
    for m in sorted(models, key=lambda x: x.id):
        if "cosy" in m.id.lower():
            print(f"  {m.id}")
except Exception as e:
    print(f"  HF list failed: {e}")

print("\n=== ModelScope에서 직접 ===")
try:
    from modelscope.hub.api import HubApi
    msapi = HubApi()
    res = msapi.list_models(filter={"name": "cosyvoice"})
    for r in res[:20]:
        print(f"  {r}")
except Exception as e:
    print(f"  modelscope list failed: {e}")
