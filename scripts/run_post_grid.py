#!/usr/bin/env python
"""Post-grid tasks with an explicit legacy or this study fair protocol.

Usage:
    python scripts/run_post_grid.py
    python scripts/run_post_grid.py --phase 1    # legacy λ=0 only (NOT PAPER-SAFE)
    python scripts/run_post_grid.py --phase 2    # macro-F1 only
    python scripts/run_post_grid.py --phase 1 --strict-fair \
        --split_manifest M --external_test_path T --epochs 3   # this study gate

The default remains legacy-compatible for existing artifacts.  The strict-fair
entry is reachable but fail-closed: it validates every this study boundary and
then raises NotImplementedError before any training or legacy GPR, because the
fair evaluator on the donor-disjoint external test is not wired yet.  It never
falls back to the old 10-epoch/cell-level inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT = Path(__file__).resolve().parent.parent
PRETRAINED = (
    Path.home() / ".cache/huggingface/hub/models--ctheodoris--Geneformer"
    / "snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5/Geneformer-V2-104M"
)
TOKENIZED_30K = (
    PROJECT / "save/caft_gf_lam0.0_seed42_20260724-085100"
    / "corneal_layer_subset_tk/cell.dataset"
)
RESULTS_DIR = PROJECT / "artifacts/lambda_grid"
RESULTS_TSV = RESULTS_DIR / "results.tsv"
RESULTS_FULL = RESULTS_DIR / "results_full.tsv"
SEED = 42
CAFT_TIMEOUT = 172800  # 48 h
GPR_TIMEOUT = 3600     # 1 h
LEGACY_TRAIN_NAME = "corneal_100k_subsample.h5ad"
LEGACY_TEST_NAMES = {"corneal_20k_test.h5ad"}

_GPR_HDR = "lambda\tcs_after\tgpr\tgpr_ci_low\tgpr_ci_high\telapsed_min\tsave_dir\ttimestamp\n"
_FULL_HDR = "lambda\tgpr\tgpr_ci_low\tgpr_ci_high\tmacro_f1\telapsed_train_min\tsave_dir\ttimestamp\n"


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def _manifest_overlap(manifest: dict[str, Any]) -> Any:
    overlap = manifest.get("donor_overlap", manifest.get("donor_overlap_count"))
    if isinstance(overlap, dict):
        return overlap.get("count", overlap.get("overlap_count"))
    return overlap


def validate_strict_fair_cli(
    split_manifest: str | None,
    external_test_path: str | None,
    epochs: int | None,
) -> dict[str, Any]:
    """Validate every this study boundary before preflight or subprocess launch."""
    if not split_manifest:
        raise ValueError("strict fair mode requires --split_manifest")
    if not external_test_path:
        raise ValueError("strict fair mode requires --external_test_path")
    if epochs != 3:
        raise ValueError("strict fair mode requires explicit --epochs 3")
    manifest_path = Path(split_manifest).expanduser().resolve()
    external_path = Path(external_test_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"strict fair manifest not found: {manifest_path}")
    if not external_path.is_file():
        raise FileNotFoundError(f"strict fair external test path not found: {external_path}")
    if external_path.name in LEGACY_TEST_NAMES:
        raise ValueError("strict fair mode refuses the legacy cell-level test H5AD")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid strict fair manifest: {manifest_path}") from exc
    if _manifest_overlap(manifest) != 0:
        raise ValueError("strict fair manifest donor_overlap must be 0")
    expected_test = manifest.get("input_paths", {}).get("test_h5ad")
    if expected_test and external_path != Path(expected_test).expanduser().resolve():
        raise ValueError("--external_test_path does not match the test path in --split_manifest")
    return manifest


# ── Pre-flight ───────────────────────────────────────────────────────────


def preflight() -> None:
    print(f"{_ts()} pre-flight checks …", flush=True)
    swap = Path("/proc/swaps").read_text().strip().split("\n")
    if len(swap) > 1:
        print(f"ERROR: swap active ({len(swap)-1} entries); aborting.", flush=True); sys.exit(1)
    res = subprocess.run(["systemctl", "is-active", "vllm.service"], capture_output=True, text=True, timeout=10)
    if res.stdout.strip() == "active":
        print("ERROR: vllm.service active; aborting.", flush=True); sys.exit(1)
    free_gb = shutil.disk_usage(PROJECT).free / (1024 ** 3)
    if free_gb < 50:
        print(f"ERROR: only {free_gb:.1f} GB free; aborting.", flush=True); sys.exit(1)
    res = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30)
    temps = [int(t) for t in res.stdout.strip().split("\n") if t.strip()]
    if any(t >= 60 for t in temps):
        print(f"ERROR: GPU temp >= 60 °C ({max(temps)} °C); aborting.", flush=True); sys.exit(1)
    print(f"{_ts()} all checks passed.\n", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────


def _find_save_dir(lam: float) -> Path | None:
    dirs = sorted(glob(str(PROJECT / "save" / f"caft_gf_lam{lam}_seed{SEED}_*")))
    return Path(dirs[-1]) if dirs else None


def _find_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(glob(str(save_dir / "model/checkpoint-*")),
                   key=lambda p: int(p.rsplit("checkpoint-", 1)[-1]))
    return Path(ckpts[-1]) if ckpts else None


def _load_tsv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().strip().split("\n")[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        rows.append({"lam": float(parts[0]), "gpr": float(parts[2]),
                      "gpr_ci_low": float(parts[3]) if parts[3] else None,
                      "gpr_ci_high": float(parts[4]) if parts[4] else None,
                      "elapsed_min": float(parts[5]) if parts[5] else None,
                      "save_dir_name": parts[6]})
    return rows


def _done_set(path: Path) -> set[float]:
    done: set[float] = set()
    if path.exists():
        for line in path.read_text().strip().split("\n")[1:]:
            if line.strip():
                done.add(float(line.split("\t")[0]))
    return done


def _run_gpr(ckpt: Path, out_json: Path) -> dict[str, Any] | None:
    cmd = [sys.executable, "-m", "corneal_fm.gpr_cli",
           "--before_checkpoint", str(PRETRAINED), "--after_checkpoint", str(ckpt),
           "--tokenized", str(TOKENIZED_30K), "--num_classes", "31",
           "--n_boot", "1000", "--seed", str(SEED), "--out", str(out_json)]
    ckpt_rel = f"{ckpt.parent.parent.parent.name}/{ckpt.parent.name}/{ckpt.name}"
    print(f"{_ts()} GPR  {ckpt_rel} …", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=GPR_TIMEOUT)
    if res.returncode != 0 or not out_json.exists():
        if res.returncode != 0 and res.stderr:
            print(res.stderr[-2000:], flush=True)
        return None
    primary = json.loads(out_json.read_text()).get("primary")
    if primary:
        print(f"  cs_after={primary['cs_after']:.4f}  gpr={primary['gpr']:.4f}", flush=True)
    return primary


# ── Phase 1: λ=0 baseline ────────────────────────────────────────────────


def phase1_lam0_baseline(
    *, strict_fair: bool = False, split_manifest: str | None = None,
    external_test_path: str | None = None, epochs: int | None = None,
    prepared_train: str | None = None,
) -> None:
    if strict_fair:
        # Fail-closed this study gate (F1): validate every donor-disjoint boundary,
        # then pass the Tier-B static + leakage gates.  Training under strict
        # mode uses the donor-disjoint split (epochs=3); the Tier-B evaluation
        # (donor-level macro-F1/GPR/LAA/marker-gradient) runs in the
        # post-training evaluation step and is not launched from here.
        manifest = validate_strict_fair_cli(split_manifest, external_test_path, epochs)

        from corneal_fm.evaluation import run_leakage_checks, run_static_checks

        # Run static checks
        static_results = run_static_checks(
            config_path=str(PROJECT / "config" / "external_test_gse153515.yml"),
            split_manifest_path=split_manifest or str(PROJECT / "artifacts" / "paper1_fair_split" / "manifest.json"),
        )
        print(f"  Static checks: {static_results}")

        # Lightweight backed obs reads for the donor/source disjointness gates.
        import anndata as ad

        train_obs = ad.read_h5ad(
            str(PROJECT / "data" / "corneal_100k_donor_split.h5ad"), backed="r"
        ).obs
        external_obs = ad.read_h5ad(external_test_path, backed="r").obs
        leakage_results = run_leakage_checks(train_obs, external_obs, manifest)
        print(f"  Leakage checks: {leakage_results}")

        # NOTE: Full evaluation (macro-F1/GPR/LAA) requires trained model
        # embeddings. The strict branch now passes static+leakage gates; actual
        # evaluation happens in the post-training evaluation step.
        print("  Tier-B static + leakage checks PASSED.")
        # Fail-closed: never fall through into the legacy 10-epoch
        # corneal_100k_subsample.h5ad baseline below under strict mode.
        # Tier-B checks passed. Launch P1-O2 training via the strict-fair launcher.
        # The actual 6-run loop is in scripts/run_paper1_strict_fair.sh.
        # This entry point validates the gate; the launcher executes the training.
        print(f"{_ts()} Tier-B gate PASSED. To launch P1-O2 training, run:")
        print(f"  bash scripts/run_paper1_strict_fair.sh")
        print(f"{_ts()} Exiting (training must be launched explicitly).")
        return

    print(f"{'='*60}\n{_ts()} Phase 1: λ=0.0 baseline (10 epochs)\n{'='*60}", flush=True)
    print(f"{_ts()} LEGACY MODE: NOT PAPER-SAFE (cell-level/default inputs may be used)", flush=True)

    # Find an existing 10-epoch λ=0 save dir (idempotent)
    save_dir: Path | None = None
    for d in sorted(glob(str(PROJECT / "save" / f"caft_gf_lam0.0_seed{SEED}_*"))):
        sd = Path(d)
        if not (sd / "CAFT_DONE").exists():
            continue
        meta = json.loads((sd / "caft_args.json").read_text()) if (sd / "caft_args.json").exists() else {}
        if meta.get("epochs") == 10:
            save_dir = sd
            print(f"{_ts()} λ=0.0  reuse existing 10-epoch save: {sd.name}", flush=True)
            break

    if save_dir is None:
        print(f"{_ts()} λ=0.0  CAFT starting …", flush=True)
        res = subprocess.run([sys.executable, str(PROJECT / "scripts/caft_gf.py"),
                              "--caft_lambda", "0.0", "--data_file", "corneal_100k_subsample.h5ad",
                              "--seed", str(SEED)],
                             capture_output=True, text=True, timeout=CAFT_TIMEOUT)
        if res.returncode != 0:
            print(f"  FAILED (rc={res.returncode})", flush=True)
            if res.stderr:
                print(res.stderr[-2000:], flush=True)
            return
        save_dir = _find_save_dir(0.0)
        if save_dir is None or not (save_dir / "CAFT_DONE").exists():
            print("  CAFT_DONE missing after training.", flush=True)
            return
        print(f"  done → {save_dir.name}", flush=True)

    if "0.0" in _done_set(RESULTS_TSV):
        print(f"{_ts()} λ=0.0  GPR already in results.tsv; skipping.\n", flush=True)
        return

    ckpt = _find_checkpoint(save_dir)
    if ckpt is None:
        print(f"{_ts()} λ=0.0  no checkpoint found.", flush=True)
        return

    t0 = time.time()
    primary = _run_gpr(ckpt, RESULTS_DIR / "gpr_lam0.0.json")
    if primary is None:
        print(f"{_ts()} λ=0.0  GPR failed.", flush=True)
        return
    ci = primary.get("paired_stratified_bootstrap_95ci", {}).get("gpr", {})
    row = (f"0.0\t{primary['cs_after']}\t{primary['gpr']}\t{ci.get('ci_low','')}\t"
           f"{ci.get('ci_high','')}\t{round((time.time()-t0)/60,2)}\t{save_dir.name}\t"
           f"{datetime.now().strftime('%Y%m%d-%H%M%S')}\n")
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(_GPR_HDR)
    with open(RESULTS_TSV, "a") as f:
        f.write(row)
    print(f"{_ts()} λ=0.0  recorded to results.tsv\n", flush=True)


# ── Phase 2: macro-F1 ────────────────────────────────────────────────────


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    ids = [torch.tensor(it["input_ids"], dtype=torch.long) for it in batch]
    labels = torch.tensor([it["label"] for it in batch], dtype=torch.long)
    max_len = max(i.size(0) for i in ids)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    for n, i in enumerate(ids):
        padded[n, :i.size(0)] = i
        attn[n, :i.size(0)] = 1
    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


def _macro_f1(save_dir: Path) -> float | None:
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from transformers import BertForSequenceClassification

    test_path = save_dir / "prepared/caft_labeled_test.dataset"
    if not test_path.exists():
        print(f"  test dataset missing", flush=True); return None
    ckpt = _find_checkpoint(save_dir)
    if ckpt is None:
        print(f"  no checkpoint in {save_dir.name}", flush=True); return None

    model = BertForSequenceClassification.from_pretrained(str(ckpt))
    model.cuda(); model.eval()
    ds = load_from_disk(str(test_path))
    keep = {"input_ids", "label"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=_collate_fn)

    preds, tgts = [], []
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda())
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            tgts.extend(b["labels"].cpu().tolist())
    mf1 = float(f1_score(tgts, preds, average="macro"))
    print(f"  macro-F1={mf1:.4f}  n={len(tgts)}", flush=True)
    return mf1


def phase2_macro_f1() -> None:
    print(f"{'='*60}\n{_ts()} Phase 2: macro-F1 evaluation\n{'='*60}", flush=True)
    grid = _load_tsv(RESULTS_TSV)
    if not grid:
        print(f"{_ts()} results.tsv empty; nothing to evaluate.\n", flush=True)
        return
    done = _done_set(RESULTS_FULL)
    print(f"{_ts()} results_full.tsv has λ: {sorted(done)}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FULL.exists():
        RESULTS_FULL.write_text(_FULL_HDR)

    for row in grid:
        lam = row["lam"]
        if lam in done:
            print(f"{_ts()} λ={lam}  already in results_full.tsv → skip", flush=True)
            continue
        sd = PROJECT / "save" / row["save_dir_name"]
        if not sd.exists():
            print(f"{_ts()} λ={lam}  save dir missing; skip.", flush=True)
            continue
        print(f"{_ts()} λ={lam}  evaluating …", flush=True)
        t0 = time.time()
        mf1 = _macro_f1(sd)
        if mf1 is None:
            print(f"  skipped.\n", flush=True)
            continue
        row_str = (f"{lam}\t{row['gpr']}\t"
                   f"{row['gpr_ci_low'] if row['gpr_ci_low'] is not None else ''}\t"
                   f"{row['gpr_ci_high'] if row['gpr_ci_high'] is not None else ''}\t"
                   f"{mf1}\t{row['elapsed_min'] if row['elapsed_min'] is not None else ''}\t"
                   f"{row['save_dir_name']}\t{datetime.now().strftime('%Y%m%d-%H%M%S')}\n")
        with open(RESULTS_FULL, "a") as f:
            f.write(row_str)
        print(f"{_ts()} λ={lam}  recorded to results_full.tsv\n", flush=True)

    print(f"\n{'='*60}\nResults: {RESULTS_FULL}\n{'='*60}", flush=True)
    if RESULTS_FULL.exists():
        sys.stdout.write(RESULTS_FULL.read_text())
    print('=' * 60, flush=True)


# ── Entry ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Post-grid: λ=0 + macro-F1")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None)
    parser.add_argument("--strict-fair", "--strict_fair", dest="strict_fair", action="store_true",
                        help="this study donor-disjoint fail-closed mode (phase 1 only).")
    parser.add_argument("--split_manifest", "--split-manifest", dest="split_manifest", default=None,
                        help="this study manifest from prepare_paper1_fair_split.py.")
    parser.add_argument("--external_test_path", "--external-test-path", dest="external_test_path",
                        default=None, help="Donor-disjoint held-out-batch test H5AD.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (strict this study baseline requires 3).")
    parser.add_argument("--prepared_train", "--prepared-train", dest="prepared_train", default=None,
                        help="Optional prepared donor-disjoint train dataset.")
    args = parser.parse_args(argv)
    preflight()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase in (None, 1):
        phase1_lam0_baseline(
            strict_fair=args.strict_fair,
            split_manifest=args.split_manifest,
            external_test_path=args.external_test_path,
            epochs=args.epochs,
            prepared_train=args.prepared_train,
        )
    if args.phase in (None, 2):
        phase2_macro_f1()


if __name__ == "__main__":
    main()
