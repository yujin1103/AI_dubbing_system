"""LatentSync `_chunked_call` 경로에도 DPMSolver scheduler reset 추가.

기존 `fix_dpm_chunk_reset.py` 는 메인 `__call__` 의 chunk loop 만 처리했는데
`_chunked_call` 의 inner inference loop 도 동일한 DPMSolver step_index 누적
문제를 일으킨다. 같은 위치에 reset 코드 삽입.
"""
from pathlib import Path

PIPELINE_PY = Path("/opt/LatentSync/latentsync/pipelines/lipsync_pipeline.py")

ANCHOR = "                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order\n                with self.progress_bar(total=num_inference_steps) as progress_bar:"
INJECT = """                # === DPM_CHUNKED_RESET_PATCH: chunked 경로에서도 chunk 마다 scheduler reset ===
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                if hasattr(self.unet, "reset_cache"):
                    self.unet.reset_cache()
                # === DPM_CHUNKED_RESET_PATCH end ===
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:"""

MARKER = "# === DPM_CHUNKED_RESET_PATCH"


def main():
    src = PIPELINE_PY.read_text()
    if MARKER in src:
        print("[fix_dpm_chunked_call_reset] already patched")
        return 0
    if ANCHOR not in src:
        print("[fix_dpm_chunked_call_reset] anchor not found — abort")
        return 1
    PIPELINE_PY.write_text(src.replace(ANCHOR, INJECT, 1))
    print("[fix_dpm_chunked_call_reset] OK — _chunked_call now resets scheduler per chunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
