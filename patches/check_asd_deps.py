"""LightASD 의존성 확인 (venv_lipsync에서)."""
import sys
print(f"Python: {sys.version}")

deps = [
    "python_speech_features", "scenedetect", "scipy", "sklearn",
    "cv2", "torch", "torchaudio", "tqdm", "pickle"
]
for d in deps:
    try:
        if d == "sklearn":
            import sklearn
            mod = sklearn
        elif d == "cv2":
            import cv2
            mod = cv2
        else:
            mod = __import__(d)
        v = getattr(mod, "__version__", "?")
        print(f"  [OK] {d}: {v}")
    except ImportError as e:
        print(f"  [FAIL] {d}: {e}")
