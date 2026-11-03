#!/usr/bin/env python
"""van Zyl 6-run evaluation (PREREG-2026-08-11 §8 + G0 revision, LOCKED).

Eval set = test donor sanes2022_Hu0216 (20,158 epithelial 3-tier cells;
split_manifest.json). For each of the 6 final-epoch checkpoints x the frozen
baseline: macro-F1 + cell-level bootstrap 95% CI (n_boot=1000, seed=42),
rho(raw), rho(L2-normed), label-shuffle null z (n=100, seed=42).

Metric code is imported UNCHANGED from the G1 rebuild pipeline
(scripts/scgpt_frozen_diag_rebuild.py): l2_norm_rows, label_shuffle_null,
fit_eval_probe (frozen split self-probe); rho itself is
corneal_fm/evaluation.marker_gradient_correlation (locked, untouched).

Naive arm decode: final-epoch cls head argmax -> train author class ->
3-tier mapping = data-driven MAJORITY cell_class_4 per author class in the
TRAIN h5ad (the van Zyl author types are absent from caft.CELL_TYPE_TO_LAYER;
mapping + full contingency reported). Ordinal arm: CORN threshold decode
(sigmoid >= 0.5, count) -> {0:LSC, 1:basal, 2:differentiated}.

Single test donor -> donor-level stats degenerate; cell-level bootstrap is
the locked §8 primary (disclosed in every output).
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("<home>/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scgpt_cls  # noqa: E402
from scgpt_frozen_diag_rebuild import (  # noqa: E402  (G1 rebuild pipeline, unchanged)
    fit_eval_probe,
    l2_norm_rows,
    label_shuffle_null,
)
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
TRAIN_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_train.h5ad"
CKPT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "checkpoints"
G1_NPZ = ROOT / "artifacts" / "g1_scgpt" / "vanzyl_cell_emb.npz"
OUT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval"
MANIFEST = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "split_manifest.json"
SCGPT_PRETRAINED = ROOT / "pretrained" / "scgpt_pretrained"

LAYER_ORDER = ["LSC", "basal", "differentiated"]
TEST_DONOR = "sanes2022_Hu0216"
N_BOOT = 1000
BOOT_SEED = 42
N_NULL = 100
F1_FRZ_LOCKED = 0.9494  # G1 probe (split self-probe, full analysis set) — G0 revised baseline
G0_REVISED = "F1_ce >= 0.55 AND F1_ce >= F1_frz - 0.10 (F1_frz=0.9494)"

RUNS = [("naive", s) for s in (42, 123, 456)] + [("ordinal", s) for s in (42, 123, 456)]


def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def macro_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOT,
                seed: int = BOOT_SEED) -> dict[str, float]:
    present = sorted(set(y_true.tolist()) & set(LAYER_ORDER))
    m = float(f1_score(y_true, y_pred, labels=present, average="macro",
                       zero_division=0))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = f1_score(y_true[b], y_pred[b], labels=present,
                            average="macro", zero_division=0)
    return {
        "macro_f1": m,
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n_boot": n_boot,
        "seed": seed,
    }


def rho_and_null(emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """rho(raw), rho(L2-normed), null z — same implementation as G1 rebuild."""
    rho_raw = float(marker_gradient_correlation(emb, labels, LAYER_ORDER))
    emb_n = l2_norm_rows(emb)
    rho_n = float(marker_gradient_correlation(emb_n, labels, LAYER_ORDER))
    null = label_shuffle_null(emb_n, labels, LAYER_ORDER, n=N_NULL, seed=42)
    z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))
    return {
        "rho_raw": rho_raw,
        "rho_normed": rho_n,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_z": z,
        "null_pctile": float((null <= rho_n).mean()),
        "n_null": N_NULL,
    }


def naive_mapping() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    import anndata as ad
    a = ad.read_h5ad(TRAIN_H5AD, backed="r")
    author = a.obs["author_cell_type"].astype(str).to_numpy()
    cc4 = a.obs["cell_class_4"].astype(str).to_numpy()
    cont: dict[str, Counter] = {}
    for au, l in zip(author, cc4):
        cont.setdefault(au, Counter())[l] += 1
    mapping = {au: max(c, key=c.get) for au, c in cont.items()}
    return mapping, {au: dict(c) for au, c in cont.items()}


def load_test() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import anndata as ad
    a = ad.read_h5ad(TEST_H5AD, backed="r")
    y = np.asarray(a.obs["cell_class_4"].astype(str))
    author = np.asarray(a.obs["author_cell_type"].astype(str))
    if not (a.obs["donor_id"].astype(str) == TEST_DONOR).all():
        raise RuntimeError("test h5ad is not exclusively the locked test donor")
    log(f"test h5ad: {a.shape}, donors all={TEST_DONOR}, "
        f"y dist={dict(Counter(y.tolist()))}")
    return y, author, np.asarray(a.obs["donor_id"].astype(str))


def checkpoint_metrics(arm: str, seed: int, vocab, pargs, device,
                       y_true, naive_map) -> dict[str, Any]:
    """Forward the test cells and compute the full metric suite."""
    import anndata as ad
    import scgpt_frozen_diag_rebuild as reb

    args = json.loads((CKPT_DIR / f"{arm}_seed{seed}" / "args.json")
                      .read_text(encoding="utf-8"))
    n_cls = args["n_cls"]
    state = torch.load(CKPT_DIR / f"{arm}_seed{seed}" / "model.pt",
                       map_location="cpu", weights_only=True)
    m2 = scgpt_cls.build_finetune_model(vocab, pargs, n_cls,
                       pretrained_dir=SCGPT_PRETRAINED)
    m2.load_state_dict(state, strict=True)
    m2.to(device).eval()

    a = ad.read_h5ad(TEST_H5AD, backed="r")
    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = scgpt_cls.tokenize_chunked(x_mat, gene_ids, vocab, pargs,
                                     max_len=args["training"]["max_len"])
    embs, clss = [], []
    with torch.no_grad():
        for s in range(0, tok["genes"].shape[0], 32):
            src = tok["genes"][s:s + 32].to(device)
            vals = tok["values"][s:s + 32].to(device)
            pad = tok["mask"][s:s + 32].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = m2(src, vals, src_key_padding_mask=pad, CLS=True)
            embs.append(out["cell_emb"].float().cpu().numpy())
            clss.append(out["cls_output"].float().cpu().numpy())
    cell_emb = np.concatenate(embs, axis=0).astype(np.float32)
    cls_out = np.concatenate(clss, axis=0).astype(np.float32)

    if arm == "naive":
        naive_classes = args["naive_classes"]
        pred_author = np.array([naive_classes[int(i)] for i in cls_out.argmax(1)])
        pred = np.array([naive_map.get(c, "NON_LAYER") for c in pred_author],
                        dtype=str)
    else:
        probs = 1.0 / (1.0 + np.exp(-cls_out))
        k = (probs >= 0.5).sum(axis=-1)
        pred = np.array([LAYER_ORDER[int(i)] for i in k], dtype=str)

    f1 = macro_f1_ci(y_true, pred)
    rh = rho_and_null(cell_emb, y_true)
    return {
        "arm": arm, "seed": seed, "n_cls": n_cls, "n_eval_cells": int(len(y_true)),
        "pred_dist": dict(Counter(pred.tolist())),
        "macro_f1": f1,
        **{k: v for k, v in rh.items()},
        "checkpoint": str(CKPT_DIR / f"{arm}_seed{seed}" / "model.pt"),
    }


def load_saved_results() -> dict[str, Any]:
    """Load frozen + per-run jsons (used by --summary-only)."""
    frozen = json.loads((OUT_DIR / "frozen_baseline.json").read_text(encoding="utf-8"))
    results = {}
    for arm, seed in RUNS:
        results[f"{arm}_{seed}"] = json.loads(
            (OUT_DIR / f"{arm}_seed{seed}.json").read_text(encoding="utf-8"))
    return frozen, results


def build_summary(frozen: dict, results: dict, naive_map: dict,
                  naive_cont: dict, y_true: np.ndarray) -> dict[str, Any]:
    """Aggregation + gates (single source of truth for both run modes)."""
    agg: dict[str, Any] = {}
    for arm in ("naive", "ordinal"):
        f1s = [results[f"{arm}_{s}"]["macro_f1"]["macro_f1"] for s in (42, 123, 456)]
        rns = [results[f"{arm}_{s}"]["rho_normed"] for s in (42, 123, 456)]
        rrs = [results[f"{arm}_{s}"]["rho_raw"] for s in (42, 123, 456)]
        zs = [results[f"{arm}_{s}"]["null_z"] for s in (42, 123, 456)]
        agg[arm] = {
            "n_seeds": 3, "seeds": [42, 123, 456],
            "macro_f1": {
                "mean": float(np.mean(f1s)), "sd": float(np.std(f1s, ddof=1)),
                "min": float(min(f1s)), "max": float(max(f1s)),
                "ci95": [float(np.mean(f1s) - 4.303 * np.std(f1s, ddof=1) / np.sqrt(3)),
                         float(np.mean(f1s) + 4.303 * np.std(f1s, ddof=1) / np.sqrt(3))],
                "ci_method": "t(df=2) mean +/- 4.303*sd/sqrt(3)",
                "per_seed": {f"s{s}": f1s[i] for i, s in enumerate((42, 123, 456))},
            },
            "rho_normed": {
                "mean": float(np.mean(rns)), "sd": float(np.std(rns, ddof=1)),
                "min": float(min(rns)), "max": float(max(rns)),
                "seed_consistency": "max-min=%.4f sd=%.4f" % (max(rns) - min(rns), np.std(rns, ddof=1)),
                "ci95": [float(np.mean(rns) - 4.303 * np.std(rns, ddof=1) / np.sqrt(3)),
                         float(np.mean(rns) + 4.303 * np.std(rns, ddof=1) / np.sqrt(3))],
                "per_seed": {f"s{s}": rns[i] for i, s in enumerate((42, 123, 456))},
            },
            "rho_raw_mean": float(np.mean(rrs)),
            "null_z": {"mean": float(np.mean(zs)), "per_seed": {f"s{s}": zs[i] for i, s in enumerate((42, 123, 456))}},
        }

    frz_rho = frozen["rho"]["rho_normed"]
    frz_probe = frozen["probe_f1"]
    naive_f1 = agg["naive"]["macro_f1"]
    ordinal_f1 = agg["ordinal"]["macro_f1"]
    naive_rho = agg["naive"]["rho_normed"]
    ordinal_rho = agg["ordinal"]["rho_normed"]

    drho_ci = [frz_rho - naive_rho["ci95"][1], frz_rho - naive_rho["ci95"][0]]
    gates = {
        "G0_revised": {
            "rule": G0_REVISED,
            "threshold_f1_ce_min": 0.55,
            "threshold_f1_ce_vs_frz": round(F1_FRZ_LOCKED - 0.10, 4),
            "per_seed": {
                f"s{s}": {
                    "f1_ce": results[f"naive_{s}"]["macro_f1"]["macro_f1"],
                    "pass_f1_ge_0.55": bool(results[f"naive_{s}"]["macro_f1"]["macro_f1"] >= 0.55),
                    "pass_f1_ge_frz_minus_0.10": bool(results[f"naive_{s}"]["macro_f1"]["macro_f1"] >= F1_FRZ_LOCKED - 0.10),
                    "G0_pass": bool(results[f"naive_{s}"]["macro_f1"]["macro_f1"] >= 0.55
                                    and results[f"naive_{s}"]["macro_f1"]["macro_f1"] >= F1_FRZ_LOCKED - 0.10),
                } for s in (42, 123, 456)
            },
            "aggregate_naive_f1_ci": naive_f1["ci95"],
            "G0_pass_aggregate": bool(naive_f1["ci95"][0] >= 0.55
                                      and naive_f1["ci95"][0] >= F1_FRZ_LOCKED - 0.10),
        },
        "G0_original_descriptive": {
            "rule": "F1_ce >= 0.55 AND F1_ce - F1_frz >= 0.10 (F1_frz=0.9494)",
            "per_seed_gain": {f"s{s}": round(results[f"naive_{s}"]["macro_f1"]["macro_f1"] - F1_FRZ_LOCKED, 4)
                              for s in (42, 123, 456)},
            "aggregate_gain": round(naive_f1["mean"] - F1_FRZ_LOCKED, 4),
            "note": "descriptive only (original gate superseded by G0 revision)",
        },
        "claim1_rho_ce_le_0.2": {
            "rule": "rho_ce <= 0.2 (catastrophic collapse => same-corpus principle)",
            "per_seed": {f"s{s}": results[f"naive_{s}"]["rho_normed"] for s in (42, 123, 456)},
            "aggregate_mean": naive_rho["mean"],
            "aggregate_ci95": naive_rho["ci95"],
            "PASS_rho_le_0.2": bool(naive_rho["mean"] <= 0.2),
        },
        "delta_rho": {
            "rule": "drho = rho_frz(test-donor) - rho_ce >= 0.3 with CI not crossing 0",
            "rho_frz_test_donor": frz_rho,
            "per_seed": {f"s{s}": round(frz_rho - results[f"naive_{s}"]["rho_normed"], 4)
                         for s in (42, 123, 456)},
            "aggregate_mean": round(frz_rho - naive_rho["mean"], 4),
            "aggregate_ci95": [round(drho_ci[0], 4), round(drho_ci[1], 4)],
            "PASS_drho_ge_0.3": bool(drho_ci[0] >= 0.3),
            "PASS_ci_not_crossing_0": bool(drho_ci[0] > 0),
        },
        "ordinal_rescue": {
            "rule": "rho_ord >= rho_ce + 0.2 AND the two-arm CIs do not overlap",
            "rho_ce_mean": naive_rho["mean"], "rho_ce_ci95": naive_rho["ci95"],
            "rho_ord_mean": ordinal_rho["mean"], "rho_ord_ci95": ordinal_rho["ci95"],
            "diff_ord_minus_ce": round(ordinal_rho["mean"] - naive_rho["mean"], 4),
            "PASS_rho_ord_ge_rho_ce_plus_0.2": bool(ordinal_rho["mean"] >= naive_rho["mean"] + 0.2),
            "PASS_ci_non_overlap": bool(ordinal_rho["ci95"][0] > naive_rho["ci95"][1]),
        },
    }
    return {
        "meta": {
            "eval_set": TEST_DONOR,
            "n_eval_cells": int(len(y_true)),
            "layer_order": LAYER_ORDER,
            "frozen_baseline_source": "G1 vanzyl_cell_emb.npz donor-masked",
            "f1_frz_locked_g0_baseline": F1_FRZ_LOCKED,
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED,
                          "level": "cell-level (donor-level degenerates: single test donor)"},
            "null": {"n": N_NULL, "seed": 42, "primary": "L2-normed"},
            "naive_mapping": {
                "method": "data-driven majority cell_class_4 per author class in train h5ad "
                          "(van Zyl author types absent from caft.CELL_TYPE_TO_LAYER)",
                "contingency": naive_cont,
                "mapping": naive_map,
            },
            "timestamp_utc": now_utc(),
        },
        "frozen": frozen,
        "per_run": results,
        "aggregate": agg,
        "gates": gates,
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild summary from saved per-run jsons (no GPU)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    log(f"manifest: test_donor={manifest['test_donor']} "
        f"n_test={manifest['n_test_cells']} sha_train={manifest['h5ad_sha256']['scgpt_vanzyl_train.h5ad'][:8]}")

    vocab, pargs = scgpt_cls.load_vocab(pretrained_dir=SCGPT_PRETRAINED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__}")

    # 0. naive mapping (data-driven majority from train h5ad)
    naive_map, naive_cont = naive_mapping()
    log(f"naive mapping: {len(naive_map)} author classes -> 3-tier (majority)")
    for au in sorted(naive_map):
        log(f"  {au:22s} -> {naive_map[au]:14s} {naive_cont[au]}")

    # 1. test labels
    y_true, author_test, _ = load_test()

    # 1b. summary-only path (rebuild from saved jsons)
    if args.summary_only:
        frozen, results = load_saved_results()
        summary = build_summary(frozen, results, naive_map, naive_cont, y_true)
        sp = OUT_DIR / "summary_vanzyl_6run.json"
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log("saved " + str(sp))
        log(json.dumps(summary["gates"], indent=1))
        log("DONE (summary-only)")
        return 0

    # 2. frozen baseline (G1 npz masked to test donor)
    g1 = np.load(G1_NPZ, allow_pickle=True)
    donors = np.array([str(d) for d in g1["donors"]])
    frz_mask = donors == TEST_DONOR
    frz_emb = g1["cell_emb"][frz_mask].astype(np.float32)
    frz_y = np.array([str(l) for l in np.asarray(g1["labels"])[frz_mask]])
    log(f"frozen baseline: {frz_emb.shape} cells from G1 npz (donor-masked)")
    frz_rh = rho_and_null(frz_emb, frz_y)
    frz_probe = fit_eval_probe(frz_emb, frz_y, LAYER_ORDER, n_repeats=5)
    frozen = {
        "arm": "frozen", "seed": None, "n_eval_cells": int(frz_emb.shape[0]),
        "source": "G1 vanzyl_cell_emb.npz donor-masked to " + TEST_DONOR,
        "rho": frz_rh, "probe_f1": frz_probe,
        "g1_full_analysis_reference": {
            "rho_normed": 0.8047, "z": 229.88,
            "probe_f1_macro": 0.9494, "n": 62640,
        },
    }
    log(f"frozen: rho_n={frz_rh['rho_normed']:.4f} z={frz_rh['null_z']:.1f} "
        f"probe_f1={frz_probe['f1_macro']:.4f}")
    (OUT_DIR / "frozen_baseline.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8")

    # 3. per-checkpoint metrics
    results: dict[str, Any] = {}
    for arm, seed in RUNS:
        log(f"=== eval {arm} seed{seed} ===")
        t0 = time.time()
        r = checkpoint_metrics(arm, seed, vocab, pargs, device, y_true, naive_map)
        log(f"  {arm} s{seed}: F1={r['macro_f1']['macro_f1']:.4f} "
            f"CI=[{r['macro_f1']['ci95_low']:.4f},{r['macro_f1']['ci95_high']:.4f}] "
            f"rho_n={r['rho_normed']:.4f} z={r['null_z']:.1f} "
            f"({time.time() - t0:.0f}s)")
        results[f"{arm}_{seed}"] = r
        (OUT_DIR / f"{arm}_seed{seed}.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8")

    # 4. aggregation + gates (single code path with --summary-only)
    summary = build_summary(frozen, results, naive_map, naive_cont, y_true)
    gates = summary["gates"]
    sp = OUT_DIR / "summary_vanzyl_6run.json"
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("saved " + str(sp))
    log("=== gates ===")
    log(json.dumps(gates, indent=1))
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
