"""LatentSync inference.py에 DPMSolver++ scheduler 옵션 추가.

환경변수 LIPSYNC_SCHEDULER:
  - "ddim" (default): DDIMScheduler (현재)
  - "dpm": DPMSolverMultistepScheduler (algorithm_type="dpmsolver++")

DPMSolver++ 사용 시 inference_steps를 절반(예: 20→10)으로 줄여도 비슷한 품질.
학습 X — scheduler 객체만 swap.

멱등(중복 실행 안전).
"""
from pathlib import Path
import re

INFERENCE_PY = Path("/opt/LatentSync/scripts/inference.py")
MARKER_BEGIN = "# === DPM_SCHEDULER_PATCH_BEGIN ==="
MARKER_END = "# === DPM_SCHEDULER_PATCH_END ==="

PATCH_BLOCK = """
    # === DPM_SCHEDULER_PATCH_BEGIN ===
    # 환경변수 LIPSYNC_SCHEDULER=dpm 이면 DPMSolverMultistepScheduler로 교체
    # (DDIM 20 step ≈ DPM 10 step, 학습 X)
    import os as _dpm_os
    if _dpm_os.getenv("LIPSYNC_SCHEDULER", "ddim").lower() == "dpm":
        try:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                solver_order=2,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=False,
            )
            print("[SCHEDULER_PATCH] DPMSolver++ (dpmsolver++ algorithm, solver_order=2)")
        except Exception as _dpm_e:
            print(f"[SCHEDULER_PATCH] DPMSolver++ init failed: {_dpm_e}, fallback DDIM")
    # === DPM_SCHEDULER_PATCH_END ===
"""


def main():
    src = INFERENCE_PY.read_text()
    if MARKER_BEGIN in src:
        # 기존 patch 제거 (재적용)
        src = re.sub(
            re.escape(MARKER_BEGIN) + r".*?" + re.escape(MARKER_END) + r"\n?",
            "",
            src, count=1, flags=re.DOTALL,
        )
    # scheduler = DDIMScheduler(...) 부분 찾아 그 다음에 patch 삽입
    # 보통 패턴: "scheduler = DDIMScheduler(\n        num_train_timesteps=1000,\n        ..."
    pat = re.search(
        r"(scheduler\s*=\s*DDIMScheduler\([^)]+\))",
        src, re.DOTALL,
    )
    if not pat:
        print("[apply_dpmsolver] DDIMScheduler init not found — abort")
        return 1
    insert_at = pat.end()
    src_new = src[:insert_at] + "\n" + PATCH_BLOCK + src[insert_at:]
    INFERENCE_PY.write_text(src_new)
    print(f"[apply_dpmsolver] DPMSolver++ option installed (env var: LIPSYNC_SCHEDULER=dpm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
