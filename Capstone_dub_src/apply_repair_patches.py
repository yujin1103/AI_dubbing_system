"""Apply 8 preserved repair patches to a run directory.

Each patch operates on `meta/{chunk}_segments_*.json` files in a run dir
and chains stage outputs (raw → v190 → v194 → face_matched → gapfilled).

Final output: `meta/{chunk}_segments_gapfilled.json` (the canonical result
consumed by downstream stages).

Patch order matches the validated baseline (test4 v305f, test5 v305f):
    word_level_split (v194)
    focused_nemo_split (v190)
    visual_asd_reassign
    face_cluster_match
    gap_fill (main_merge + bg_merge + sim_match)
    postprocess_reassign_text (word-level text reassign)

Per-video best config (sweep_gt_match.py):
    test4: main_merge=0.99 bg_merge=0.30 sim_match=0.10 pad=0.5
    test5: main_merge=0.40 bg_merge=0.30 sim_match=0.45 pad=0.5

Usage:
    python apply_repair_patches.py <run_dir> [--main-merge 0.45] ...

Environment:
    PATCHES_VENV_PYTHON: override the venv python path used to invoke
        repair_patches/*.py (default: /opt/venv_diarizen/bin/python).
        Each patch loads numpy/soundfile/torch and must run in venv_diarizen.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PATCHES_ROOT = Path(__file__).parent / "repair_patches"
DEFAULT_VENV_PYTHON = os.environ.get(
    "PATCHES_VENV_PYTHON", "/opt/venv_diarizen/bin/python"
)

PATCH_ORDER: list[tuple[str, str, bool]] = [
    ("word_level_split", "word_level_split.py", False),
    ("focused_nemo_split", "focused_nemo_split.py", False),
    ("visual_asd_reassign", "visual_asd_reassign.py", False),
    ("face_cluster_match", "face_cluster_match.py", False),
    ("gap_fill", "gap_fill.py", True),
    ("ecapa_recluster", "ecapa_recluster.py", False),
    ("postprocess_reassign_text", "postprocess_reassign_text.py", False),
]
# ecapa_recluster lives in Capstone_dub_src/, not repair_patches/ —
# special-cased path in apply_all().
ECAPA_SCRIPT_PATH = Path(__file__).parent / "ecapa_recluster.py"


def apply_all(
    run_dir: str,
    *,
    gap_fill_args: dict | None = None,
    skip: Iterable[str] = (),
    venv_python: str = DEFAULT_VENV_PYTHON,
) -> None:
    skip_set = set(skip)
    for name, script, takes_args in PATCH_ORDER:
        if name in skip_set:
            print(f"[skip] {name}")
            continue
        if name == "ecapa_recluster":
            # ensemble sub-split on gap_fill output; conditional on gap_fill main SPK count.
            # 조건 (영상 무관 logic):
            #   gap_fill 후 main SPK ≤ 3 → 적용 (under-merge 가능성, ECAPA sub-cluster 검출)
            #   gap_fill 후 main SPK ≥ 4 → skip (이미 분리 적정, ECAPA over-split 부작용)
            # gap_fill 결과의 main 수 기준 (raw 가 아님). gap_fill 가 적극 merge 한 경우에도
            # ECAPA 가 sub-split 로 적정 main 수 회복.
            import json as _json
            meta_dir = Path(run_dir) / "meta"
            chunks = sorted(meta_dir.glob("*_chunk_*_segments_gapfilled.json"))
            for gf in chunks:
                chunk_stem = gf.name.replace("_segments_gapfilled.json", "")
                gf_data = _json.load(open(gf, encoding="utf-8"))
                gf_segs = gf_data.get("groups", gf_data.get("segments", []))
                gf_main_spk = {
                    str(s.get("speaker", ""))
                    for s in gf_segs
                    if str(s.get("speaker", "")) and
                       not str(s.get("speaker", "")).startswith("SPEAKER_BG") and
                       not str(s.get("speaker", "")).startswith("SPEAKER_9")
                }
                if len(gf_main_spk) > 3:
                    print(f"\n[skip] {name} ({chunk_stem}): gap_fill main SPK={len(gf_main_spk)} > 3 — already adequate")
                    continue
                ecapa_out = f"{chunk_stem}_segments_ecapa.json"
                cmd = [venv_python, str(ECAPA_SCRIPT_PATH), run_dir,
                       "--segments-name", gf.name,
                       "--out-name", ecapa_out,
                       "--ensemble"]
                print(f"\n[run] {name}: {' '.join(cmd)}")
                rc = subprocess.run(cmd).returncode
                if rc != 0:
                    raise SystemExit(f"patch {name!r} failed (rc={rc})")
                # overwrite gap_fill output so downstream stages pick ECAPA result
                ecapa_path = meta_dir / ecapa_out
                if ecapa_path.exists():
                    import shutil as _sh
                    _sh.copy(str(ecapa_path), str(gf))
            continue
        cmd = [venv_python, str(PATCHES_ROOT / script), run_dir]
        if takes_args and gap_fill_args:
            for key, value in gap_fill_args.items():
                cmd += [f"--{key.replace('_', '-')}", str(value)]
        print(f"\n[run] {name}: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f"patch {name!r} failed (rc={rc})")
    print("\n[done] all patches applied")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("run_dir")
    ap.add_argument("--main-merge", type=float, default=0.45,
                    help="gap_fill main cluster merge threshold (cosine)")
    ap.add_argument("--bg-merge", type=float, default=0.30,
                    help="gap_fill BG cluster merge threshold (cosine)")
    ap.add_argument("--sim-match", type=float, default=0.45,
                    help="gap_fill main match threshold (cosine)")
    ap.add_argument("--pad", type=float, default=0.5,
                    help="gap_fill audio pad (sec)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="patch names to skip")
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PYTHON,
                    help=f"python interpreter for patches (default: {DEFAULT_VENV_PYTHON})")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_all(
        args.run_dir,
        gap_fill_args={
            "main_merge": args.main_merge,
            "bg_merge": args.bg_merge,
            "sim_match": args.sim_match,
            "pad": args.pad,
        },
        skip=args.skip,
        venv_python=args.venv_python,
    )


if __name__ == "__main__":
    main()
