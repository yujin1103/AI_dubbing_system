"""LatentSync chunk loop 마다 DPMSolver step_index reset.

원인:
  - LatentSync 는 비디오를 num_frames(16) 단위 chunk 로 나눠 각 chunk 마다 N-step 디노이징.
  - DDIM 은 stateless (timestep 기반) 이라 chunk 간 무관.
  - DPMSolver 는 `_step_index` 가 누적되어 두 번째 chunk 첫 호출에서 sigmas[step_index+1] 가
    범위를 벗어남 → IndexError.

Fix:
  - chunk loop 안 (`for i in tqdm.tqdm(range(num_inferences))`) 매 iter 시작에서
    scheduler 의 timesteps + step_index 재설정.

멱등.
"""
from pathlib import Path

PIPELINE_PY = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")

ANCHOR = "            # 9. Denoising loop\n            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order\n"
INJECT = """            # === DPM_RESET_PATCH: chunk 마다 scheduler state 재설정 ===
            # DPMSolver 는 _step_index 가 chunk 간 누적 → IndexError 방지
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            # === DPM_RESET_PATCH end ===
"""
MARKER = "# === DPM_RESET_PATCH"


def main():
    src = PIPELINE_PY.read_text()
    if MARKER in src:
        print("[fix_dpm_chunk_reset] already patched (idempotent)")
        return 0
    if ANCHOR not in src:
        print("[fix_dpm_chunk_reset] anchor not found — abort")
        return 1
    src_new = src.replace(ANCHOR, INJECT + ANCHOR, 1)
    PIPELINE_PY.write_text(src_new)
    print("[fix_dpm_chunk_reset] OK — scheduler reset added before denoising loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
