#!/usr/bin/env python
"""Epidermis cross-tissue scGPT final evaluation (approved supplementary).

Evaluates 2 seeds x 2 checkpoints (best-val by val macro-F1, final-epoch
model_epoch15.pt) on the epidermis held-out test cells (1,836 cells; v14
Wang2020 product, 4-layer ordinal): macro-F1 + cell-level bootstrap 95% CI
(n_boot=1000, seed=42), rho(L2-normed) + rho(raw) + label-shuffle z
(n=100). Metric code imported UNCHANGED from the 6-run evaluation
(scripts/scgpt_eval_vanzyl.py: macro_f1_ci, rho_and_null — LAYER_ORDER
monkey-patched to the epidermis 4-layer axis; implementations untouched).
Frozen baseline = the pre-run frozen diagnostic on the same test cells
(cross_tissue/epidermis_test_frozen_diag.json).

Naive decode: cls head argmax -> args.json naive_classes (train minus the
locked val partition) -> majority cell_class_4 mapping (identity for the
epidermis: author_cell_type == cell_class_4 == ordinal_layer).
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
import scgpt_eval_vanzyl as sev  # noqa: E402  (6-run eval suite, unchanged)

EPI_LAYER_ORDER = ["basal", "suprabasal", "spinous", "differentiated"]
TEST_DONOR = "epidermis"
sev.LAYER_ORDER = EPI_LAYER_ORDER  # monkey-patch: same functions, 4-layer axis
macro_f1_ci = sev.macro_f1_ci
rho_and_null = sev.rho_and_null

CONV_CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue" / "checkpoints"
VAL_NPZ = ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue" / "val_indices.npz"
OUT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue"
SCGPT_PRETRAINED = ROOT / "pretrained" / "scgpt_pretrained"
TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue" / "scgpt_epidermis_test.h5ad"
TRAIN_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue" / "scgpt_epidermis_train.h5ad"
FROZEN_DIAG = ROOT / "artifacts" / "paper1_vanzyl" / "cross_tissue" / "epidermis_test_frozen_diag.json"

CKPT_TYPES = [("best_val", "model_best_val.pt"),
              ("final_epoch", "model_epoch15.pt")]
SEEDS = [42, 123]


def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def naive_mapping_90pct() -> tuple[list[str], dict[str, str], dict[str, dict[str, int]]]:
    """Majority cell_class_4 per author class on the TRAIN-90% subset (the
    exact cells the convergence models were trained on)."""
    import anndata as ad
    a = ad.read_h5ad(TRAIN_H5AD, backed="r")
    vd = np.load(VAL_NPZ)
    val_mask = np.zeros(a.n_obs, dtype=bool)
    val_mask[vd["val_idx"].astype(np.int64)] = True
    tr_author = a.obs["author_cell_type"].astype(str).to_numpy()[~val_mask]
    tr_cc4 = a.obs["cell_class_4"].astype(str).to_numpy()[~val_mask]
    classes = sorted(set(tr_author.tolist()))
    cont: dict[str, Counter] = {}
    for au, l in zip(tr_author, tr_cc4):
        cont.setdefault(au, Counter())[l] += 1
    mapping = {au: max(c, key=c.get) for au, c in cont.items()}
    return classes, mapping, {au: dict(c) for au, c in cont.items()}


def eval_checkpoint(ckpt_path: Path, args_json: Path, vocab, pargs, device,
                    y_true, naive_classes, naive_map) -> dict[str, Any]:
    import anndata as ad
    import scgpt_cls as sc

    args = json.loads(args_json.read_text(encoding="utf-8"))
    n_cls = args["n_cls"]
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = sc.build_finetune_model(vocab, pargs, n_cls,
                                    pretrained_dir=SCGPT_PRETRAINED)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    a = ad.read_h5ad(TEST_H5AD, backed="r")
    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = sc.tokenize_chunked(x_mat, gene_ids, vocab, pargs,
                              max_len=args["training"]["max_len"])
    embs, clss = [], []
    with torch.no_grad():
        for s in range(0, tok["genes"].shape[0], 32):
            src = tok["genes"][s:s + 32].to(device)
            vals = tok["values"][s:s + 32].to(device)
            pad = tok["mask"][s:s + 32].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(src, vals, src_key_padding_mask=pad, CLS=True)
            embs.append(out["cell_emb"].float().cpu().numpy())
            clss.append(out["cls_output"].float().cpu().numpy())
    cell_emb = np.concatenate(embs, axis=0).astype(np.float32)
    cls_out = np.concatenate(clss, axis=0).astype(np.float32)

    pred_author = np.array([naive_classes[int(i)] for i in cls_out.argmax(1)],
                           dtype=str)
    pred = np.array([naive_map.get(c, "NON_LAYER") for c in pred_author],
                    dtype=str)
    f1 = macro_f1_ci(y_true, pred)
    rh = rho_and_null(cell_emb, y_true)
    return {
        "seed": args["seed"], "n_cls": n_cls, "n_eval_cells": int(len(y_true)),
        "checkpoint": str(ckpt_path),
        "pred_dist": dict(Counter(pred.tolist())),
        "macro_f1": f1,
        **{k: v for k, v in rh.items()},
    }


def gpu_free_mib() -> int:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        used, total = (int(x) for x in out.stdout.strip().split(","))
        return total - used
    except Exception:
        return -1


def main() -> int:
    import subprocess, time as _t
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # GPU guard: free memory >= 4 GiB before starting (GF 1-epoch lane may
    # run in parallel; PREREG §9 verified headroom)
    waited = 0
    while True:
        free = gpu_free_mib()
        if free >= 4096:
            break
        if waited >= 30 * 60:
            log(f"ABORT: GPU free {free} MiB < 4096 after 30 min")
            return 3
        log(f"GPU free {free} MiB < 4096; wait 60s ({waited // 60} min)")
        _t.sleep(60)
        waited += 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, pargs = scgpt_cls.load_vocab(pretrained_dir=SCGPT_PRETRAINED)
    log(f"device={device} torch={torch.__version__}")

    # test labels
    import anndata as ad
    a = ad.read_h5ad(TEST_H5AD, backed="r")
    y_true = np.asarray(a.obs["cell_class_4"].astype(str))
    if not (a.obs["donor_id"].astype(str) == TEST_DONOR).all():
        raise RuntimeError("test h5ad not exclusively the locked test donor")
    log(f"test: {a.shape}, y dist={dict(Counter(y_true.tolist()))}")

    # train-90% mapping (same cells the models were trained on)
    naive_classes, naive_map, naive_cont = naive_mapping_90pct()
    log(f"train-90% naive classes: {len(naive_classes)}, mapping: "
        f"{naive_map}")

    # frozen baseline reference (pre-run frozen diagnostic, same test cells)
    frozen_diag = json.loads(FROZEN_DIAG.read_text(encoding="utf-8"))
    frozen_res = frozen_diag["results"]["epidermis_test"]
    frz_rho = float(frozen_res["rho_normed"])
    frz_z = float(frozen_res["null_z"])
    frz_f1 = float(frozen_res["probe"]["f1_macro"])
    log(f"frozen baseline (epidermis test): rho_n={frz_rho:.4f} "
        f"z={frz_z:.1f} probe_f1={frz_f1:.4f}")

    # per-seed x per-checkpoint eval
    results: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        cdir = CONV_CKPT / f"naive_seed{seed}"
        args_json = cdir / "args.json"
        for ctype, fname in CKPT_TYPES:
            ckpt = cdir / fname
            if not ckpt.is_file():
                log(f"SKIP seed{seed} {ctype}: {fname} missing")
                continue
            log(f"=== eval seed{seed} {ctype} ===")
            t0 = time.time()
            r = eval_checkpoint(ckpt, args_json, vocab, pargs, device,
                                y_true, naive_classes, naive_map)
            log(f"  F1={r['macro_f1']['macro_f1']:.4f} "
                f"CI=[{r['macro_f1']['ci95_low']:.4f},{r['macro_f1']['ci95_high']:.4f}] "
                f"rho_n={r['rho_normed']:.4f} z={r['null_z']:.1f} "
                f"({time.time() - t0:.0f}s)")
            results[f"s{seed}_{ctype}"] = r

    # aggregation per checkpoint type (mean +/- t-CI, df=2)
    agg: dict[str, Any] = {}
    for ctype, _ in CKPT_TYPES:
        f1s = [results[f"s{s}_{ctype}"]["macro_f1"]["macro_f1"] for s in SEEDS]
        rns = [results[f"s{s}_{ctype}"]["rho_normed"] for s in SEEDS]
        rrs = [results[f"s{s}_{ctype}"]["rho_raw"] for s in SEEDS]
        zs = [results[f"s{s}_{ctype}"]["null_z"] for s in SEEDS]
        f1_ci = [float(np.mean(f1s) - 12.706 * np.std(f1s, ddof=1) / np.sqrt(len(SEEDS))),
                 float(np.mean(f1s) + 12.706 * np.std(f1s, ddof=1) / np.sqrt(len(SEEDS)))]
        rho_ci = [float(np.mean(rns) - 12.706 * np.std(rns, ddof=1) / np.sqrt(len(SEEDS))),
                  float(np.mean(rns) + 12.706 * np.std(rns, ddof=1) / np.sqrt(len(SEEDS)))]
        agg[ctype] = {
            "macro_f1": {"mean": float(np.mean(f1s)), "sd": float(np.std(f1s, ddof=1)),
                         "min": float(min(f1s)), "max": float(max(f1s)),
                         "ci95": f1_ci,
                         "per_seed": {f"s{s}": f1s[i] for i, s in enumerate(SEEDS)}},
            "rho_normed": {"mean": float(np.mean(rns)), "sd": float(np.std(rns, ddof=1)),
                           "min": float(min(rns)), "max": float(max(rns)),
                           "ci95": rho_ci,
                           "per_seed": {f"s{s}": rns[i] for i, s in enumerate(SEEDS)}},
            "rho_raw_mean": float(np.mean(rrs)),
            "null_z": {"mean": float(np.mean(zs)),
                       "per_seed": {f"s{s}": zs[i] for i, s in enumerate(SEEDS)}},
            "ci_method": f"t(df={len(SEEDS) - 1}) mean +/- 12.706*sd/sqrt({len(SEEDS)})",
        }

    summary = {
        "meta": {
            "task": "Epidermis cross-tissue scGPT final evaluation "
                    "(approved supplementary)",
            "eval_set": TEST_DONOR,
            "n_eval_cells": int(len(y_true)),
            "layer_order": EPI_LAYER_ORDER,
            "checkpoints": {"best_val": "model_best_val.pt (val macro-F1, official)",
                            "final_epoch": "model_epoch15.pt (sensitivity)"},
            "bootstrap": {"n": 1000, "seed": 42, "level": "cell-level"},
            "null": {"n": 100, "seed": 42, "primary": "L2-normed"},
            "frozen_baseline_test_donor": {"rho_normed": frz_rho, "z": frz_z,
                                           "probe_f1_macro": frz_f1},
            "naive_mapping": {
                "method": "data-driven majority cell_class_4 per author class "
                          "on the TRAIN-90% subset (same cells as training)",
                "mapping": naive_map,
                "contingency": naive_cont,
            },
            "timestamp_utc": now_utc(),
        },
        "frozen_baseline": frozen_diag,
        "per_seed_per_ckpt": results,
        "aggregate": agg,
    }
    out = OUT_DIR / "final_test_eval.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("saved " + str(out))
    log(json.dumps(agg, indent=1))
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
