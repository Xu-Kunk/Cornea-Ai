#!/usr/bin/env python
"""Corneal-FM: Full benchmark suite (E1–E4).

Executes the four evaluation tasks defined in corneal_fm.benchmark:
  E1 — Cell type classification (macro F1)
  E2 — Perturbation prediction (Pearson R)
  E3 — Cross-species generalisation (accuracy)
  E4 — Embedding quality (silhouette score, cluster preservation)

Compares all models (GF, scGPT, scLong) on a unified metrics table.

Usage:
    python scripts/08_benchmark.py
    python scripts/08_benchmark.py --tasks E1 E2
    python scripts/08_benchmark.py --skip_sclong --output report.csv
    python scripts/08_benchmark.py --dry_run
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from corneal_fm.config import load_config, set_seed, get_save_dir, archive_script
from corneal_fm.benchmark import BenchmarkRunner
from corneal_fm.utils import get_logger, get_device

logger = get_logger(__name__)


@click.command()
@click.option("--config", default=None, help="YAML config override")
@click.option(
    "--tasks",
    multiple=True,
    type=click.Choice(["E1", "E2", "E3", "E4"], case_sensitive=False),
    help="Tasks to run (default: all)",
)
@click.option("--skip_gf", is_flag=True, help="Skip Geneformer evaluation")
@click.option("--skip_scgpt", is_flag=True, help="Skip scGPT evaluation")
@click.option("--skip_sclong", is_flag=True, help="Skip scLong evaluation")
@click.option("--output", default=None, help="Save metrics table to CSV")
@click.option("--dry_run", is_flag=True, help="List tasks without executing")
@click.option("--seed", default=42, type=int)
def main(config, tasks, skip_gf, skip_scgpt, skip_sclong, output, dry_run, seed):
    """Run the full Corneal-FM benchmark suite (E1–E4)."""
    # Load config
    cfg = load_config(config or "config/default_train_args.yml")

    set_seed(seed)
    device = get_device()
    logger.info(f"Device: {device}")

    # Setup save directory
    save_dir = get_save_dir("benchmark", "E1-E4")
    save_dir.mkdir(parents=True, exist_ok=True)
    archive_script(__file__, save_dir)

    # Determine task list
    all_tasks = ["E1", "E2", "E3", "E4"]
    task_list = list(tasks) if tasks else all_tasks

    if dry_run:
        logger.info("Dry-run mode — tasks to execute:")
        for t in task_list:
            models = []
            if not skip_gf:
                models.append("Geneformer")
            if not skip_scgpt:
                models.append("scGPT")
            if not skip_sclong:
                models.append("scLong")
            logger.info(f"  {t}: {', '.join(models)}")
        return

    # Initialize benchmark runner
    runner = BenchmarkRunner(cfg, save_dir=save_dir)
    runner.add_models(
        geneformer=not skip_gf,
        scgpt=not skip_scgpt,
        sclong=not skip_sclong,
    )

    # Run
    metrics = runner.run(tasks=task_list)
    logger.info(f"Benchmark complete — {len(metrics)} metric rows")

    # Save
    if output:
        import pandas as pd

        path = save_dir / output
        pd.DataFrame(metrics).to_csv(path, index=False)
        logger.info(f"Metrics table saved to {path}")
    else:
        runner.print_summary(metrics)


if __name__ == "__main__":
    main()
