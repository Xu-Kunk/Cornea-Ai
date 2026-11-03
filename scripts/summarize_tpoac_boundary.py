#!/usr/bin/env python
"""Task 3 — GSE299152 (true out-of-corpus) vs GSE153515 (same-corpus) summary.

Per-arm rho/F1: frozen / CE-naive (3 seeds) / ordinal lam0.0 (3 seeds) /
ordinal lam0.5 (3 seeds), on both external tests, to complete the
"TP-OAC is same-domain specific" boundary picture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/home/dq/Codex/Ai")


def read(path: str) -> dict:
    p = ROOT / path
    return json.load(open(p)) if p.is_file() else {}


def row(arm, lam, seed, src):
    d = read(f"{src}/{path_name(arm, lam, seed)}")
    pe = d.get("primary_endpoint", {})
    se = d.get("secondary_endpoints", {})
    return {
        "arm": arm, "lambda": lam, "seed": seed,
        "rho_L12": se.get("marker_gradient_rank_correlation"),
        "macro_f1": pe.get("macro_f1"),
        "LAA": se.get("layer_adjacency_accuracy"),
        "ECE": se.get("expected_calibration_error"),
        "n_eval_cells": (d.get("provenance") or {}).get("n_eval_cells"),
    }


def path_name(arm, lam, seed):
    if arm == "frozen":
        return "results_frozen_gf.json"
    if arm == "caft_gf":
        return f"results_{lam}_{seed}.json"
    return f"results_{arm}_{lam}_{seed}.json"


def main() -> None:
    # same-corpus: GSE153515
    # out-of-corpus: GSE299152

    rows = []
    for seed in (42, 123, 456):
        rows.append(row("frozen", 0.0, 42, "artifacts/paper1_tier_b_eval"))
        rows.append(row("caft_gf", 0.0, seed, "artifacts/paper1_tier_b_eval"))
        rows.append(row("ordinal_gf", 0.0, seed, "artifacts/paper1_ordinal_eval"))
        rows.append(row("ordinal_gf", 0.5, seed, "artifacts/paper1_ordinal_eval"))
    df153 = pd.DataFrame(rows).assign(test="GSE153515 (same-corpus)")
    df153 = df153[df153["arm"] != "frozen"]  # frozen only once

    rows2 = []
    for seed in (42, 123, 456):
        rows2.append(row("caft_gf", 0.0, seed, "artifacts/paper1_tier_b_eval_gse299152"))
        rows2.append(row("ordinal_gf", 0.0, seed, "artifacts/paper1_tier_b_eval_gse299152"))
        rows2.append(row("ordinal_gf", 0.5, seed, "artifacts/paper1_tier_b_eval_gse299152"))
    rows2.append(row("frozen", 0.0, 42, "artifacts/paper1_tier_b_eval_gse299152"))
    df299 = pd.DataFrame(rows2).assign(test="GSE299152 (out-of-corpus)")

    df = pd.concat([df153, df299], ignore_index=True)
    df = df.dropna(subset=["rho_L12"])
    print(df[["test", "arm", "lambda", "seed", "rho_L12", "macro_f1", "LAA", "ECE"]].to_string(index=False))

    # 3-seed means
    pivot = df.groupby(["test", "arm", "lambda"])[["rho_L12", "macro_f1", "LAA"]].agg(
        ["mean", "std"]
    ).round(4)
    print("\n=== 3-seed means (frozen is a single run) ===")
    print(pivot.to_string())

    out = ROOT / "artifacts/paper1_fdr_nulls/gse299152_vs_gse153515_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
