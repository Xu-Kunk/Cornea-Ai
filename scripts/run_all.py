#!/usr/bin/env python
"""Corneal-FM: Full pipeline orchestrator.

Runs the complete fine-tuning → inference → benchmark pipeline sequentially:

    Stage 1:  protocol_06_finetune_gf.py         — Geneformer fine-tuning
    Stage 2:  protocol_07_finetune_scgpt.py      — scGPT fine-tuning
    Stage 3:  protocol_08_inference_sclong.py    — scLong zero-shot inference
    Stage 4:  protocol_09_benchmark.py         — Full benchmark suite (E1–E5)

All console output is mirrored to a timestamped log file under ``save/``.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --dry_run
    python scripts/run_all.py --skip_finetune
    python scripts/run_all.py --skip_gf --skip_scgpt
    python scripts/run_all.py --skip_finetune --skip_scgpt
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import click

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
SAVE_ROOT = PROJECT_ROOT / "save"

SCRIPT_06 = SCRIPTS_DIR / "protocol_06_finetune_gf.py"
SCRIPT_07 = SCRIPTS_DIR / "protocol_07_finetune_scgpt.py"
SCRIPT_08 = SCRIPTS_DIR / "protocol_08_inference_sclong.py"
SCRIPT_09 = SCRIPTS_DIR / "protocol_09_benchmark.py"

PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

StageResult = dict[str, Any]


def _run_stage(
    label: str,
    script: Path,
    args: list[str],
    log_fh: TextIO | None,
    dry_run: bool,
) -> StageResult:
    """Execute one pipeline stage and return its result dict."""
    cmd = [PYTHON, str(script), *args]
    cmd_str = " ".join(str(c) for c in cmd)
    indent = "  "

    print(f"\n{'=' * 64}")
    print(f"  [{label}]")
    print(f"  CMD: {cmd_str}")
    print(f"{'=' * 64}")

    if log_fh is not None:
        log_fh.write(f"\n{'=' * 64}\n")
        log_fh.write(f"  [{label}]\n")
        log_fh.write(f"  CMD: {cmd_str}\n")
        log_fh.write(f"{'=' * 64}\n")

    if dry_run:
        print(f"{indent}DRY-RUN — skipped")
        return {
            "stage": label,
            "script": str(script),
            "success": True,
            "elapsed_s": 0.0,
            "returncode": 0,
            "dry_run": True,
        }

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,  # let output flow to terminal (and log)
            text=True,
        )
        elapsed = time.time() - t0
        success = proc.returncode == 0

        if success:
            print(f"\n{indent}✅ [{label}] completed in {elapsed:.1f}s")
        else:
            print(f"\n{indent}❌ [{label}] FAILED (rc={proc.returncode}) in {elapsed:.1f}s")

        return {
            "stage": label,
            "script": str(script),
            "success": success,
            "elapsed_s": elapsed,
            "returncode": proc.returncode,
            "dry_run": False,
        }

    except FileNotFoundError:
        elapsed = time.time() - t0
        print(f"\n{indent}❌ [{label}] SCRIPT NOT FOUND: {script}")
        return {
            "stage": label,
            "script": str(script),
            "success": False,
            "elapsed_s": elapsed,
            "returncode": -1,
            "error": "Script not found",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--dry_run",
    is_flag=True,
    help="Print the execution plan without running anything.",
)
@click.option(
    "--skip_finetune",
    is_flag=True,
    help="Skip both fine-tuning stages (06 Geneformer + 07 scGPT).",
)
@click.option(
    "--skip_gf",
    is_flag=True,
    help="Skip Geneformer fine-tuning (stage 06) only.",
)
@click.option(
    "--skip_scgpt",
    is_flag=True,
    help="Skip scGPT fine-tuning (stage 07) only.",
)
@click.option(
    "--config",
    default=None,
    help="YAML config file passed to all stages.",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed (default: 42).",
)
def main(
    dry_run: bool,
    skip_finetune: bool,
    skip_gf: bool,
    skip_scgpt: bool,
    config: str | None,
    seed: int,
) -> None:
    """Run the Corneal-FM pipeline: finetune → inference → benchmark.

    Stages are executed sequentially.  If any stage fails, subsequent stages
    still run unless the failure was a missing script.
    """
    # Resolve skip logic
    skip_gf = skip_gf or skip_finetune
    skip_scgpt = skip_scgpt or skip_finetune

    # Build config arg for downstream scripts
    config_args = ["--config", config] if config else []

    # ------------------------------------------------------------------
    # Log file
    # ------------------------------------------------------------------
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = SAVE_ROOT / f"run_{timestamp}.log"

    log_fh: TextIO | None = None
    if not dry_run:
        log_fh = open(log_path, "w", encoding="utf-8")
        log_fh.write(f"Corneal-FM Pipeline Run — {timestamp}\n")
        log_fh.write(f"Args: dry_run={dry_run} skip_finetune={skip_finetune} ")
        log_fh.write(f"skip_gf={skip_gf} skip_scgpt={skip_scgpt}\n")

    # ------------------------------------------------------------------
    # Stage definitions
    # ------------------------------------------------------------------
    stages: list[tuple[str, Path, list[str], bool]] = [
        (
            "Stage 1: Geneformer Finetune",
            SCRIPT_06,
            config_args + ["--seed", str(seed)],
            skip_gf,
        ),
        (
            "Stage 2: scGPT Finetune",
            SCRIPT_07,
            config_args + ["--seed", str(seed)],
            skip_scgpt,
        ),
        (
            "Stage 3: scLong Zero-Shot",
            SCRIPT_08,
            config_args + ["--seed", str(seed)],
            False,
        ),
        (
            "Stage 4: Benchmark",
            SCRIPT_09,
            config_args + ["--seed", str(seed)],
            False,
        ),
    ]

    # ------------------------------------------------------------------
    # Print plan
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Corneal-FM Pipeline Orchestrator")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python  : {PYTHON}")
    print(f"  CWD     : {PROJECT_ROOT}")
    print(f"  Dry run : {dry_run}")
    print(f"  Log     : {log_path if not dry_run else '(dry-run)'}")
    print("=" * 64)

    print("\nExecution plan:")
    for label, script, args, skipped in stages:
        status = "⏭ SKIP" if skipped else "  → RUN"
        extra = f" (--config {config})" if config else ""
        print(f"  {status}  {label}")
        print(f"         {script.name} {' '.join(args)}{extra}")

    if dry_run:
        print("\n✅ Dry-run complete.  Remove --dry_run to execute.")
        if log_fh is not None:
            log_fh.close()
        return

    # ------------------------------------------------------------------
    # Execute stages
    # ------------------------------------------------------------------
    results: list[StageResult] = []
    pipeline_ok = True

    for label, script, args, skipped in stages:
        if skipped:
            print(f"\n  ⏭  [{label}] skipped")
            results.append({
                "stage": label,
                "script": str(script),
                "success": True,
                "elapsed_s": 0.0,
                "returncode": 0,
                "skipped": True,
            })
            continue

        result = _run_stage(label, script, args, log_fh, dry_run=False)
        results.append(result)

        if not result["success"]:
            pipeline_ok = False
            print(f"\n  ⚠  Pipeline continuing after failure in [{label}]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = sum(r["elapsed_s"] for r in results)

    print()
    print("=" * 64)
    print("  Pipeline Summary")
    print("=" * 64)
    print(f"  {'Stage':<40s} {'Status':<10s} {'Time':<10s}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 10}")

    final_status = "✅ ALL PASSED" if pipeline_ok else "❌ SOME FAILED"

    for r in results:
        label_short = r["stage"]
        if r.get("skipped"):
            status = "⏭ skip"
        elif r["success"]:
            status = "✅ pass"
        else:
            status = "❌ fail"
        elapsed_str = f"{r['elapsed_s']:.0f}s" if r["elapsed_s"] > 0 else "-"
        print(f"  {label_short:<40s} {status:<10s} {elapsed_str:<10s}")

    print(f"  {'-' * 40} {'-' * 10} {'-' * 10}")
    print(f"  {'Total':<40s} {'':<10s} {total_elapsed:.0f}s")
    print()
    print(f"  {final_status}")
    print(f"  Log: {log_path}")
    print("=" * 64)

    # Write final summary to log
    if log_fh is not None:
        log_fh.write("\n" + "=" * 64 + "\n")
        log_fh.write("  Pipeline Summary\n")
        log_fh.write("=" * 64 + "\n")
        for r in results:
            log_fh.write(
                f"  {r['stage']:<45s} "
                f"{'pass' if r['success'] else 'FAIL':<8s} "
                f"{r['elapsed_s']:.1f}s\n"
            )
        log_fh.write(f"  {'Total':<45s} {total_elapsed:.0f}s\n")
        log_fh.write(f"  {final_status}\n")
        log_fh.write("=" * 64 + "\n")
        log_fh.close()

    sys.exit(0 if pipeline_ok else 1)


if __name__ == "__main__":
    main()
