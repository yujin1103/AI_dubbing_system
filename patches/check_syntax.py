import py_compile
try:
    py_compile.compile("/workspace/orchestrator.py", doraise=True)
    print("SYNTAX OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
