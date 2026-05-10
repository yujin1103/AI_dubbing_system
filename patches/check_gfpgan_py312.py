"""py312 backup venv에 GFPGAN 사용 가능 여부."""
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
    print(f"basicsr OK")
except Exception as e:
    print(f"basicsr FAIL: {e}")

try:
    import facexlib
    print(f"facexlib OK")
except Exception as e:
    print(f"facexlib FAIL: {e}")

try:
    from gfpgan import GFPGANer
    print(f"gfpgan: GFPGANer OK")
except Exception as e:
    print(f"gfpgan FAIL: {e}")
