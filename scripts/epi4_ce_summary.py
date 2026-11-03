#!/usr/bin/env python
"""Aggregate both epi4-CE variants and tabulate them against the reference arms.

Layout (results/paper1_vanzyl/epi4_ce/):

    README.md                  top-level record (primary comparison, step counts)
    summary.json               combined record over both variants
    drop_nonepi/seed{42,123,456}/   eval_gse153515.json + percell_epi4_{seed}.npz
    drop_nonepi/summary.json
    mask_nonepi/seed{42,123,456}/   eval_gse153515.json + percell_epi4_{seed}.npz
    mask_nonepi/summary.json
    paired_drho_reference_arms.json (frozen-naive, frozen-ordinal, naive-ordinal)

Variants:

* ``drop_nonepi`` — epithelial cells only in the dataset (non-epithelial rows
  dropped), plain 4-class CE; 2,616 optimizer steps / 3 epochs.
  Arm: ``method_arm_gf.py --arm ce --label-space epi4`` (save prefix method_ce_epi4).
* ``mask_nonepi`` — all 79,684 rows kept, 4-class CE on epithelial rows only;
  7,473 optimizer steps / 3 epochs (step-count matched to the ordinal and naive
  arms). Arm: ``--arm axis_ft --lambda-arm 0.0`` (save prefix method_axis_ft).

Every Δρ uses ``scripts/paired_drho_bootstrap.paired_delta``: cells resampled
with replacement within each donor, one index vector per replicate applied to
both arms, each arm's centroid axis re-fitted inside the replicate, 2,000
replicates, seed 42.

Usage:
    python scripts/epi4_ce_summary.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import paired_drho_bootstrap as pdb  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "paper1_vanzyl" / "epi4_ce"
NAIVE_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
ORDINAL_DIR = PROJECT_ROOT / "artifacts" / "paper1_ordinal_eval"
SEEDS = (42, 123, 456)
METRICS = ("macro_f1", "laa", "rho", "ece")

VARIANTS = {
    "drop_nonepi": {
        "arm_tag": "method_ce_epi4",
        "epochs": 3,
        "train_rows": 27878,
        "train_cmd": "method_arm_gf.py --arm ce --label-space epi4",
        "loss": "plain 4-class CE on epithelial cells only; non-epithelial rows "
                "dropped from the train dataset (not merely loss-masked)",
        "steps_per_epoch": 872,
        "steps_total": 2616,
        "primary": False,
    },
    "mask_nonepi": {
        "arm_tag": "method_axis_ft",
        "epochs": 3,
        "train_rows": 79684,
        "train_cmd": "method_arm_gf.py --arm axis_ft --lambda-arm 0.0",
        "loss": "4-class CE on epithelial rows only (axis_ft at lambda=0); "
                "non-epithelial rows stay in the batches and contribute no loss",
        "steps_per_epoch": 2491,
        "steps_total": 7473,
        "primary": True,
    },
}
PRIMARY = "mask_nonepi"


def _row_from_eval(doc: dict) -> dict:
    pe, se = doc["primary_endpoint"], doc["secondary_endpoints"]
    return {
        "macro_f1": pe["macro_f1"],
        "macro_f1_ci_lower": pe["ci_lower"],
        "macro_f1_ci_upper": pe["ci_upper"],
        "per_donor_macro_f1": pe["per_donor"],
        "per_class_f1": {k: v["f1"] for k, v in doc["per_class"].items()},
        "laa": se["layer_adjacency_accuracy"],
        "rho": se["marker_gradient_rank_correlation"],
        "ece": se["expected_calibration_error"],
        "checkpoint": doc.get("model", {}).get("checkpoint"),
        "checkpoint_step": doc.get("model", {}).get("step"),
        "num_labels": doc.get("model", {}).get("num_labels"),
    }


def _agg(values: dict) -> dict:
    xs = [v for v in values.values() if v is not None]
    if not xs:
        return {"mean": None, "sd": None, "n": 0, "per_seed": values}
    return {
        "mean": float(st.mean(xs)),
        "sd": float(st.stdev(xs)) if len(xs) > 1 else None,
        "sd_kind": "sample SD (ddof=1)",
        "n": len(xs),
        "per_seed": values,
    }


def _ref_rows(directory: Path, tag: str) -> dict:
    rows = {}
    for seed in SEEDS:
        path = directory / f"results_0.0_{seed}.json"
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        pe, se = doc["primary_endpoint"], doc["secondary_endpoints"]
        rows[str(seed)] = {
            "macro_f1": pe["macro_f1"],
            "laa": se["layer_adjacency_accuracy"],
            "rho": se["marker_gradient_rank_correlation"],
            "ece": se["expected_calibration_error"],
            "source": str(path.relative_to(PROJECT_ROOT)),
        }
    out = {"arm": tag, "per_seed": rows}
    out["aggregate"] = {
        m: _agg({s: row[m] for s, row in rows.items()}) for m in METRICS
    }
    return out


def variant_record(name: str, spec: dict, ordinal: dict) -> dict:
    vdir = OUT_DIR / name
    per_seed = {}
    for seed in SEEDS:
        path = vdir / f"seed{seed}" / "eval_gse153515.json"
        if not path.is_file():
            print(f"variant {name}: seed {seed} record not present yet -> marked pending")
            return {
                "variant": name,
                "primary_comparison": name == PRIMARY,
                "status": "pending",
                "reason": f"no evaluation record for seed {seed} yet ({path})",
                "arm": spec["arm_tag"],
                "train_cmd": spec["train_cmd"],
                "loss": spec["loss"],
                "training": {
                    "train_dataset_rows": spec["train_rows"],
                    "epochs": spec["epochs"],
                    "steps_per_epoch": spec["steps_per_epoch"],
                    "optimizer_steps_total": spec["steps_total"],
                },
            }
        per_seed[str(seed)] = _row_from_eval(json.loads(path.read_text()))
    aggregate = {
        m: _agg({s: row[m] for s, row in per_seed.items()}) for m in METRICS
    }
    per_class = {
        cls: _agg({s: row["per_class_f1"][cls] for s, row in per_seed.items()})
        for cls in per_seed[str(SEEDS[0])]["per_class_f1"]
    }
    steps = sorted({row["checkpoint_step"] for row in per_seed.values()})
    record = {
        "variant": name,
        "primary_comparison": name == PRIMARY,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "arm": spec["arm_tag"],
        "train_cmd": spec["train_cmd"],
        "loss": spec["loss"],
        "label_space": {
            "classes": ["LSC", "TAC", "basal", "differentiated"],
            "source": "corneal_fm.ordinal.class_to_ordinal_map "
                      "(wing+squamous -> differentiated)",
        },
        "training": {
            "train_dataset_rows": spec["train_rows"],
            "rows_contributing_loss": spec["train_rows"],
            "epochs": 3,
            "steps_per_epoch": spec["steps_per_epoch"],
            "optimizer_steps_total": spec["steps_total"],
            "checkpoint_steps_observed": steps,
            "lr": 5e-5,
            "lr_scheduler": "linear",
            "warmup_steps": 500,
            "weight_decay": 0.001,
            "max_grad_norm": 1.0,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 32,
            "precision": "bf16 weights (bf16=True)",
            "checkpoint": "final epoch",
            "split_manifest_sha256": "f4393ae16959ed8e1923661a5e7849bb93d220bb202cb77c6f1dcb85549b8cd6",
        },
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "aggregate_per_class_f1": per_class,
        "reference_arms": {"ordinal_only": ordinal},
        "paired_delta_rho_vs_ordinal": variant_paired(name, ordinal),
    }
    return record


def variant_paired(name: str, ordinal: dict) -> dict:
    pre = OUT_DIR / f"paired_drho_{name}.json"
    if pre.is_file():
        doc = json.loads(pre.read_text())
        out = {"status": "available", "source": str(pre.relative_to(PROJECT_ROOT)),
               "n_boot": doc["pairs"][0]["n_boot"], "boot_seed": doc["pairs"][0]["boot_seed"],
               "pairs": {p["pair"]: p for p in doc["pairs"]}}
        for p in doc["pairs"]:
            if p["pair"].endswith("- ordinal"):
                out["mean_delta_rho"] = p.get("mean_delta")
                out["sd_delta_rho"] = p.get("sd_delta")
                out["sd_kind"] = "sample SD (ddof=1) over seeds"
                out["per_seed"] = p["per_seed"]
        return out
    rows = {}
    for seed in SEEDS:
        a = pdb.load_arm("epi4_ce", seed, variant=name)
        b = pdb.load_arm("ordinal", seed)
        if not pdb.check_pairing(a, b):
            return {"status": "not_computable",
                    "reason": f"seed {seed}: labels/donors differ; pairing not exact"}
        res = pdb.paired_delta(a["emb"], b["emb"], a["labels"], a["donors"])
        rows[str(seed)] = {
            "n_cells": int(a["emb"].shape[0]),
            "rho_a": res["rho_a"],
            "rho_b": res["rho_b"],
            "delta_rho": res["delta"],
            "delta_ci_lower": res["ci_lower"],
            "delta_ci_upper": res["ci_upper"],
            "frac_boot_positive": res["frac_boot_positive"],
            "n_boot": res["n_boot"],
            "boot_seed": res["boot_seed"],
        }
    ds = [v["delta_rho"] for v in rows.values()]
    return {
        "status": "available",
        "comparison": f"{name} - ordinal_only   (delta = rho_{name} - rho_ordinal)",
        "method": "donor-stratified paired bootstrap, axis refit per replicate "
                  "(scripts/paired_drho_bootstrap.paired_delta)",
        "n_boot": pdb.DEFAULT_N_BOOT,
        "boot_seed": pdb.DEFAULT_BOOT_SEED,
        "per_seed": rows,
        "mean_delta_rho": float(st.mean(ds)),
        "sd_delta_rho": float(st.stdev(ds)) if len(ds) > 1 else None,
        "sd_kind": "sample SD (ddof=1) over seeds",
    }


def fmt(agg) -> str:
    if agg["mean"] is None:
        return "n/a"
    sd = f" ± {agg['sd']:.4f}" if agg["sd"] is not None else ""
    return f"{agg['mean']:.4f}{sd}"


def main() -> int:
    naive = _ref_rows(NAIVE_DIR, "naive 31-class CE (3 epochs, lambda=0)")
    ordinal = _ref_rows(ORDINAL_DIR, "ordinal-only CORN (3 epochs, lambda=0, 4 stages)")

    ref_paired_path = OUT_DIR / "paired_drho_reference_arms.json"
    ref_paired = (json.loads(ref_paired_path.read_text())
                  if ref_paired_path.is_file() else None)

    records = {}
    for name, spec in VARIANTS.items():
        rec = variant_record(name, spec, ordinal)
        rec["paired_delta_rho_reference_arms"] = ref_paired
        records[name] = rec
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / name / "summary.json").write_text(json.dumps(rec, indent=2))

    combined = {
        "primary_comparison": PRIMARY,
        "primary_reason": "mask_nonepi supervises the same epithelial cells as "
                          "the ordinal arm AND takes the same number of "
                          "optimizer steps (7,473); drop_nonepi additionally "
                          "removes the non-epithelial rows (2,616 steps), so it "
                          "confounds label space with row composition and step "
                          "count. drop_nonepi is kept as the secondary control.",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "step_counts": {
            "drop_nonepi": 2616,
            "mask_nonepi": 7473,
            "ordinal_only": 7473,
            "naive_author31_ce": 7473,
        },
        "variants": records,
        "reference_arms": {"naive_author31_ce": naive, "ordinal_only": ordinal},
        "paired_delta_rho_reference_arms": ref_paired,
        "paired_delta_rho_variants_vs_ordinal": {
            name: rec.get("paired_delta_rho_vs_ordinal", {"status": "pending"})
            for name, rec in records.items()
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(combined, indent=2))

    print(f"{'arm':<48}{'rho':>20}{'macro-F1':>20}")
    for name, rec in records.items():
        tag = f"{name}{' (primary)' if name == PRIMARY else ''}"
        if rec.get("status") == "pending":
            print(f"{tag:<48}{'pending':>20}{'pending':>20}")
            continue
        print(f"{tag:<48}{fmt(rec['aggregate']['rho']):>20}"
              f"{fmt(rec['aggregate']['macro_f1']):>20}")
    print(f"{'naive 31-class CE (lambda=0)':<48}{fmt(naive['aggregate']['rho']):>20}"
          f"{fmt(naive['aggregate']['macro_f1']):>20}")
    print(f"{'ordinal-only CORN (lambda=0)':<48}{fmt(ordinal['aggregate']['rho']):>20}"
          f"{fmt(ordinal['aggregate']['macro_f1']):>20}")

    print("\npaired delta rho (donor-stratified, 2000 reps, seed 42):")
    if ref_paired:
        for p in ref_paired["pairs"]:
            if "mean_delta" in p:
                print(f"  {p['pair']:<26} {p['mean_delta']:+.4f}")
    for name, rec in records.items():
        pd = rec.get("paired_delta_rho_vs_ordinal")
        if pd is None:
            continue
        if pd["status"] == "available":
            print(f"  {name + ' - ordinal':<26} {pd['mean_delta_rho']:+.4f}")
    print(f"\nwrote {OUT_DIR / 'summary.json'} (+ one per variant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
