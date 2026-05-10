"""학습 진행 실시간 모니터 — CLI + Library dual-purpose.

CLI 사용:
  /opt/venv_lipsync/bin/python /workspace/scripts/monitor_train.py \
      --log /workspace/logs/lora_full_train.log

Library 사용 (UI에서):
  from monitor_train import (
      parse_train_log_tail, get_gpu_status, get_disk_status,
      compute_eta, format_dashboard
  )
  status = parse_train_log_tail("/path/to.log")
  gpu   = get_gpu_status()
  ...

기능:
  - 학습 로그에서 step/loss/it/s 파싱
  - nvidia-smi → VRAM 사용량/peak/util/temp
  - 디스크 여유 (학습 결과 저장용)
  - ETA 자동 계산 (남은 step × 평균 it 시간)
  - 단편화 위험 경고 (peak가 초기보다 너무 높을 때)
  - 학습 완료 자동 감지 + 알림 (텍스트 파일)
  - JSON 출력 모드 (UI 연동용)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─── Data structures ───────────────────────────────────────

@dataclass
class TrainStatus:
    step: int = 0
    total_steps: int = 0
    progress_pct: float = 0.0
    last_loss: float = 0.0
    avg_loss_last_n: float = 0.0           # 최근 N step 평균
    speed_it_per_sec: float = 0.0
    eta_seconds: float = 0.0
    eta_human: str = "?"
    completed: bool = False
    crashed: bool = False
    error_msg: Optional[str] = None


@dataclass
class GPUStatus:
    used_mb: int = 0
    total_mb: int = 0
    util_pct: int = 0
    temp_c: int = 0
    peak_mb: int = 0  # 누적 peak (process-level)


@dataclass
class DiskStatus:
    free_gb: float = 0.0
    used_gb: float = 0.0
    total_gb: float = 0.0
    used_pct: float = 0.0


# ─── Train log parsing ─────────────────────────────────────

# tqdm 형식 매칭: "Steps:  35%|████      | 350/1000 [01:23<02:34,  5.43it/s, ..., step_loss=0.0123]"
_STEP_RE = re.compile(
    r"Steps:\s+\d+%\|[^|]*\|\s+(\d+)/(\d+)\s+\[\S+,\s+([\d.]+)\s*(?:s/it|it/s)[^\]]*step_loss=([\d.eE+-]+)"
)


def parse_train_log_tail(log_path: str, n_recent: int = 100) -> TrainStatus:
    """학습 로그 끝부분에서 진행 상황 추출.

    Args:
      log_path : 학습 stdout 로그 경로
      n_recent : 평균 loss 계산할 최근 step 수

    Returns:
      TrainStatus
    """
    status = TrainStatus()
    if not Path(log_path).is_file():
        status.error_msg = f"log file not found: {log_path}"
        return status

    try:
        # tqdm은 \r로 한 줄에 계속 덮어쓰므로 binary로 읽고 \r 분리
        with open(log_path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        # 마지막 N KB만 (큰 파일 효율)
        text = text[-200_000:]

        # \r과 \n 모두 line break로
        lines = re.split(r"[\r\n]", text)
        # 거꾸로 검색해서 마지막 step 정보
        recent_losses = []
        for line in reversed(lines):
            m = _STEP_RE.search(line)
            if m:
                if status.step == 0:  # 가장 최근만 step/total 채움
                    cur, total, time_per_it, loss = m.groups()
                    status.step = int(cur)
                    status.total_steps = int(total)
                    status.progress_pct = 100 * status.step / max(1, status.total_steps)
                    status.last_loss = float(loss)
                    # tqdm 형식이 it/s 또는 s/it 자동 결정
                    if "it/s" in line:
                        status.speed_it_per_sec = float(time_per_it)
                    else:  # s/it
                        status.speed_it_per_sec = 1.0 / float(time_per_it) if float(time_per_it) > 0 else 0
                recent_losses.append(float(m.group(4)))
                if len(recent_losses) >= n_recent:
                    break

        if recent_losses:
            status.avg_loss_last_n = sum(recent_losses) / len(recent_losses)

        # ETA 계산
        remaining = max(0, status.total_steps - status.step)
        if status.speed_it_per_sec > 0 and remaining > 0:
            status.eta_seconds = remaining / status.speed_it_per_sec
            status.eta_human = format_seconds(status.eta_seconds)

        # 완료 감지 (step == total or "Saved checkpoint" 등)
        if status.total_steps > 0 and status.step >= status.total_steps:
            status.completed = True

        # crash 감지
        if "RuntimeError" in text or "Error occurred" in text or "FAILED" in text:
            # 단, fp16 dtype 충돌 + onnxruntime CUDA fallback (validation 중 false alarm) 무시
            crash_lines = [l for l in lines if "RuntimeError" in l or "FAILED" in l]
            crash_lines = [l for l in crash_lines if "fp16" not in l.lower() and "dtype" not in l.lower()]
            crash_lines = [l for l in crash_lines if "onnxruntime" not in l.lower() and "providerlibrary" not in l.lower() and "cublaslt" not in l.lower() and "tryget" not in l.lower()]
            # 추가: 학습 step이 최근 5분 내 진행됐으면 crash 아님 (false alarm 회피)
            import time as _time
            if status.step > 0 and crash_lines:
                # 마지막 step 진행 시간 추정: log file mtime
                try:
                    log_age = _time.time() - os.path.getmtime(log_path)
                    if log_age < 300:  # 5분 이내 갱신 = 살아있음
                        crash_lines = []
                except Exception:
                    pass
            if crash_lines:
                status.crashed = True
                status.error_msg = crash_lines[-1][:200]

    except Exception as e:
        status.error_msg = f"parse error: {type(e).__name__}: {e}"

    return status


def format_seconds(s: float) -> str:
    """3661.5 → '1h 1min 2s'"""
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}h {m}min"
    if m > 0:
        return f"{m}min {s}s"
    return f"{s}s"


# ─── GPU / disk status ─────────────────────────────────────

def get_gpu_status(prev_peak_mb: int = 0) -> GPUStatus:
    """nvidia-smi → GPU 메모리/util/temp."""
    status = GPUStatus(peak_mb=prev_peak_mb)
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        used, total, util, temp = [x.strip() for x in out.split(",")]
        status.used_mb = int(used)
        status.total_mb = int(total)
        status.util_pct = int(util)
        status.temp_c = int(temp)
        if status.used_mb > status.peak_mb:
            status.peak_mb = status.used_mb
    except Exception:
        pass
    return status


def get_disk_status(path: str = ".") -> DiskStatus:
    """호스트 디스크 여유 (학습 결과 저장 가능 공간)."""
    status = DiskStatus()
    try:
        total, used, free = subprocess.check_output(
            ["df", "-B1", path], timeout=5
        ).decode().strip().splitlines()[-1].split()[1:4]
        status.total_gb = int(total) / (1024 ** 3)
        status.used_gb = int(used) / (1024 ** 3)
        status.free_gb = int(free) / (1024 ** 3)
        status.used_pct = 100 * status.used_gb / max(1e-9, status.total_gb)
    except Exception:
        # Windows fallback (in container we have df, but...)
        try:
            import shutil as _sh
            t, u, f = _sh.disk_usage(path)
            status.total_gb = t / (1024 ** 3)
            status.used_gb = u / (1024 ** 3)
            status.free_gb = f / (1024 ** 3)
            status.used_pct = 100 * status.used_gb / max(1e-9, status.total_gb)
        except Exception:
            pass
    return status


# ─── Dashboard formatting ──────────────────────────────────

def format_dashboard(
    train: TrainStatus,
    gpu: GPUStatus,
    disk: DiskStatus,
    log_path: str = "",
    warnings: Optional[list] = None,
) -> str:
    """터미널 dashboard 포맷."""
    lines = []
    lines.append("─" * 60)
    lines.append("  📊 LoRA 학습 모니터")
    if log_path:
        lines.append(f"     log: {log_path}")
    lines.append("─" * 60)

    # Train
    if train.crashed:
        lines.append(f"  ❌ CRASHED: {train.error_msg}")
    elif train.completed:
        lines.append(f"  ✅ COMPLETED — {train.step}/{train.total_steps} steps")
    elif train.step == 0:
        lines.append(f"  ⏳ Waiting for first step... ({train.error_msg or 'log empty'})")
    else:
        bar_w = 30
        filled = int(bar_w * train.progress_pct / 100)
        bar = "█" * filled + "░" * (bar_w - filled)
        lines.append(f"  Step:    {train.step:>6,} / {train.total_steps:>6,}  ({train.progress_pct:5.1f}%)")
        lines.append(f"           [{bar}]")
        lines.append(f"  Loss:    {train.last_loss:.5f}    (avg last 100: {train.avg_loss_last_n:.5f})")
        lines.append(f"  Speed:   {train.speed_it_per_sec:.2f} it/s")
        lines.append(f"  ETA:     {train.eta_human}")

    lines.append("")

    # GPU
    if gpu.total_mb > 0:
        bar_w = 30
        gpu_pct = 100 * gpu.used_mb / max(1, gpu.total_mb)
        filled = int(bar_w * gpu_pct / 100)
        bar = "█" * filled + "░" * (bar_w - filled)
        lines.append(f"  GPU mem: {gpu.used_mb/1024:.2f}GB / {gpu.total_mb/1024:.2f}GB  "
                     f"({gpu_pct:.0f}%, peak {gpu.peak_mb/1024:.2f}GB)")
        lines.append(f"           [{bar}]")
        lines.append(f"  GPU util:{gpu.util_pct:>3}%   temp: {gpu.temp_c}°C")

    lines.append("")

    # Disk
    if disk.total_gb > 0:
        lines.append(f"  Disk:    {disk.free_gb:.0f}GB free  ({disk.used_pct:.0f}% used of {disk.total_gb:.0f}GB)")

    # Warnings
    if warnings:
        lines.append("")
        for w in warnings:
            lines.append(f"  ⚠️  {w}")

    lines.append("─" * 60)
    return "\n".join(lines)


# ─── Warning detection ─────────────────────────────────────

def detect_warnings(
    train: TrainStatus,
    gpu: GPUStatus,
    disk: DiskStatus,
    history: list,
) -> list:
    """위험 신호 감지."""
    warnings = []

    # GPU peak 위험 (16GB GPU 기준 14.5GB+)
    if gpu.total_mb > 0 and gpu.peak_mb > 0.92 * gpu.total_mb:
        warnings.append(f"GPU peak {gpu.peak_mb/1024:.1f}GB가 한도 92% 초과 — OOM 위험")

    # 단편화 누적 (peak가 step 진행하면서 계속 올라가면)
    if len(history) >= 5:
        old_peak = history[-5].get("gpu_peak_mb", 0)
        if old_peak > 0 and gpu.peak_mb > old_peak * 1.1:
            warnings.append(f"VRAM peak 누적 증가 감지 ({old_peak/1024:.1f}→{gpu.peak_mb/1024:.1f}GB)")

    # 디스크 부족
    if disk.free_gb > 0 and disk.free_gb < 50:
        warnings.append(f"디스크 여유 {disk.free_gb:.0f}GB — 체크포인트 저장 위험 (5GB/회 권장)")

    # Loss 발산
    if train.avg_loss_last_n > 1.0 and train.step > 100:
        warnings.append(f"Loss 너무 큼 ({train.avg_loss_last_n:.3f}) — 학습 발산 가능")

    # 멈춤 감지 (step이 안 올라감)
    if len(history) >= 3:
        recent_steps = [h.get("step", 0) for h in history[-3:]]
        if len(set(recent_steps)) == 1 and recent_steps[0] > 0:
            warnings.append(f"step {recent_steps[0]}에서 멈춤 — process 확인 필요")

    return warnings


# ─── Main loop ─────────────────────────────────────────────

def watch(
    log_path: str,
    interval: int = 30,
    json_output: Optional[str] = None,
    notify_completion: Optional[str] = None,
    disk_path: str = "/workspace/media",
):
    """학습 로그를 주기적으로 파싱 + dashboard 갱신."""
    history = []
    prev_peak = 0
    last_status = None

    print(f"[Monitor] 시작 — log: {log_path}, interval: {interval}s")
    while True:
        train = parse_train_log_tail(log_path)
        gpu = get_gpu_status(prev_peak_mb=prev_peak)
        prev_peak = gpu.peak_mb
        disk = get_disk_status(disk_path)

        snapshot = {
            "ts": time.time(),
            "step": train.step,
            "loss": train.last_loss,
            "gpu_used_mb": gpu.used_mb,
            "gpu_peak_mb": gpu.peak_mb,
        }
        history.append(snapshot)
        if len(history) > 100:
            history.pop(0)

        warnings = detect_warnings(train, gpu, disk, history)

        # 터미널 dashboard
        os.system("clear" if sys.platform != "win32" else "cls")
        print(format_dashboard(train, gpu, disk, log_path, warnings))
        print(f"  ({time.strftime('%H:%M:%S')} — Ctrl+C로 종료, {interval}s 간격)")

        # JSON 출력 (UI 연동용)
        if json_output:
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump({
                    "train": asdict(train),
                    "gpu": asdict(gpu),
                    "disk": asdict(disk),
                    "warnings": warnings,
                    "ts": snapshot["ts"],
                }, f, indent=2, ensure_ascii=False)

        # 학습 완료 / crash 알림
        if train.completed and last_status != "completed":
            msg = f"✅ 학습 완료: {train.step} steps, final loss {train.last_loss:.5f}"
            print(f"\n[Monitor] {msg}")
            if notify_completion:
                with open(notify_completion, "w", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")
            last_status = "completed"
            break  # 자동 종료
        if train.crashed and last_status != "crashed":
            msg = f"❌ 학습 실패: {train.error_msg}"
            print(f"\n[Monitor] {msg}")
            if notify_completion:
                with open(notify_completion, "w", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")
            last_status = "crashed"
            break

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[Monitor] 종료")
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA 학습 모니터 (CLI + Library dual-purpose)")
    parser.add_argument("--log", required=True, help="학습 stdout 로그 파일 경로")
    parser.add_argument("--interval", type=int, default=30, help="갱신 주기 (초, 기본 30)")
    parser.add_argument("--json-output", default=None, help="JSON 상태 저장 경로 (UI 연동용)")
    parser.add_argument("--notify", default=None,
                        help="완료/실패 시 알림 텍스트 파일 경로")
    parser.add_argument("--disk-path", default="/workspace/media",
                        help="디스크 여유 체크 경로 (기본 /workspace/media)")
    parser.add_argument("--once", action="store_true",
                        help="1회만 출력 후 종료 (스크립팅용)")
    args = parser.parse_args()

    if args.once:
        train = parse_train_log_tail(args.log)
        gpu = get_gpu_status()
        disk = get_disk_status(args.disk_path)
        warnings = detect_warnings(train, gpu, disk, [])
        if args.json_output:
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump({"train": asdict(train), "gpu": asdict(gpu),
                           "disk": asdict(disk), "warnings": warnings},
                          f, indent=2, ensure_ascii=False)
        else:
            print(format_dashboard(train, gpu, disk, args.log, warnings))
        return 0

    watch(args.log, args.interval, args.json_output, args.notify, args.disk_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
