"""DPMSolver from_pretrained → 직접 인자로 교체.

`DPMSolverMultistepScheduler.from_pretrained("configs")` 가 DDIM scheduler_config.json
(`steps_offset=1`) 을 그대로 읽어와 sigmas 인덱스 충돌 → IndexError.

Fix: 명시적 인자로 생성, `euler_at_final=True` 로 마지막 step 안전하게 처리.
"""
from pathlib import Path

INFERENCE_PY = Path("/opt/LatentSync/scripts/inference.py")

OLD = '''    # === v27.8 SCHEDULER_PATCH: DPMSolverMultistepScheduler 옵션 ===
    # LATENTSYNC_SCHEDULER=dpm 으로 같은 품질에 더 빠른 수렴 (steps 줄여도 품질 유지)
    scheduler_type = os.environ.get("LATENTSYNC_SCHEDULER", "ddim").lower()
    if scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_pretrained("configs")
        print("[SCHEDULER_PATCH] DPMSolverMultistep (faster convergence)")
    else:
        scheduler = DDIMScheduler.from_pretrained("configs")
        print("[SCHEDULER_PATCH] DDIM (default)")
'''

NEW = '''    # === v27.8 SCHEDULER_PATCH: DPMSolverMultistepScheduler 옵션 ===
    # LATENTSYNC_SCHEDULER=dpm 으로 같은 품질에 더 빠른 수렴 (steps 줄여도 품질 유지)
    scheduler_type = os.environ.get("LATENTSYNC_SCHEDULER", "ddim").lower()
    if scheduler_type == "dpm":
        # 직접 인자 — from_pretrained 가 DDIM config 의 steps_offset=1 을 그대로
        # 읽어 마지막 step 에서 sigmas IndexError 발생. euler_at_final=True 로 회피.
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            solver_order=2,
            algorithm_type="dpmsolver++",
            lower_order_final=True,
            euler_at_final=True,
            final_sigmas_type="zero",
            use_karras_sigmas=False,
        )
        print("[SCHEDULER_PATCH] DPMSolverMultistep (faster convergence, euler_at_final=True)")
    else:
        scheduler = DDIMScheduler.from_pretrained("configs")
        print("[SCHEDULER_PATCH] DDIM (default)")
'''


def main():
    src = INFERENCE_PY.read_text()
    if OLD not in src:
        print("[fix_dpmsolver_args] OLD block not found — already patched?")
        return 1
    src_new = src.replace(OLD, NEW)
    INFERENCE_PY.write_text(src_new)
    print("[fix_dpmsolver_args] OK — DPMSolverMultistepScheduler now uses explicit args")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
