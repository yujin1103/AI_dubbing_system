"""GFPGAN + 의존성 확인 (Python 3.13 venv_lipsync에서 import 가능 여부)."""
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import torch
    print(f"torch: {torch.__version__}, cuda={torch.cuda.is_available()}")
except Exception as e:
    print(f"torch FAIL: {e}")

try:
    import basicsr
    print(f"basicsr: {basicsr.__version__ if hasattr(basicsr, '__version__') else 'OK'}")
except Exception as e:
    print(f"basicsr FAIL: {e}")

try:
    import facexlib
    print(f"facexlib: OK")
except Exception as e:
    print(f"facexlib FAIL: {e}")

try:
    import gfpgan
    from gfpgan import GFPGANer
    print(f"gfpgan: OK - GFPGANer importable")
except Exception as e:
    print(f"gfpgan FAIL: {e}")
