#!/usr/bin/env python
"""Fixed-axis differentiation-gradient evaluation (reviewer M1: metric circularity).

Builds a frozen-axis from the scGPT-human pretrained model on the locked
test donor Hu0216, then projects every checkpoint (and the frozen model
itself) onto that axis and reports Spearman rho vs the 3-tier layer rank.

Outputs:
    <home>/corneal_scgpt/artifacts/paper1_vanzyl/fixed_axis/fixed_axis_eval.json

Usage (on <compute-node>, GPU guard is built-in):
    nohup <home>/corneal_scgpt/repo/.venv/bin/python \
        <home>/corneal_scgpt/scripts/eval_fixed_axis.py > run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as st
import torch

# ---------------------------------------------------------------------------
# Paths / locked constants
# ---------------------------------------------------------------------------
ROOT = Path("<home>/corneal_scgpt")
PRETRAINED = ROOT / "pretrained" / "scgpt_pretrained"
TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
RUN6_CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "checkpoints"
CORN_CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "corn_lora" / "checkpoints"
RUN6_EVAL = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval"
OUT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "fixed_axis"
OUT_JSON = OUT_DIR / "fixed_axis_eval.json"

LAYER_ORDER = ["LSC", "basal", "differentiated"]
N_NULL = 100
NULL_SEED = 42
SANITY_TOL = 0.15
EXPECTED_FRZ_RHO = 0.7887175099644741  # run6/eval/frozen_baseline.json rho_normed


def _load_expected_frz_rho() -> float:
    """Read the locked frozen-baseline rho_normed from run6 eval."""
    try:
        data = json.loads((RUN6_EVAL / "frozen_baseline.json").read_text(encoding="utf-8"))
        return float(data["rho"]["rho_normed"])
    except Exception as exc:
        print(f"WARN: could not read frozen_baseline.json ({exc}); using hard-coded value",
              file=sys.stderr, flush=True)
        return EXPECTED_FRZ_RHO

# LoRA is needed for the corn_lora checkpoints.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scgpt_cls  # noqa: E402
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from scgpt_frozen_diag_rebuild import l2_norm_rows, label_shuffle_null  # noqa: E402

try:
    from scgpt_lora_corn_vanzyl import wrap_lora  # noqa: E402
except Exception as exc:  # pragma: no cover
    wrap_lora = None  # type: ignore[assignment]
    print(f"WARN: could not import wrap_lora: {exc}", flush=True)


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------
def gpu_free_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_for_gpu(free_mib: int = 4096) -> None:
    while True:
        free = gpu_free_mib()
        if free >= free_mib:
            print(f"[{_ts()}] GPU free {free} MiB >= {free_mib} MiB; proceeding",
                  flush=True)
            return
        print(f"[{_ts()}] GPU free {free} MiB < {free_mib} MiB; wait 60s",
              flush=True)
        time.sleep(60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def load_test() -> tuple[np.ndarray, np.ndarray]:
    import anndata as ad

    a = ad.read_h5ad(TEST_H5AD, backed="r")
    y = np.asarray(a.obs["cell_class_4"].astype(str))
    log(f"loaded test h5ad: {a.shape}, layer_order={LAYER_ORDER}, "
        f"dist={dict(zip(*np.unique(y, return_counts=True)))}")
    return y, np.asarray(a.obs_names, dtype=str)


def tokenize_test(vocab, pargs) -> dict[str, Any]:
    import anndata as ad

    a = ad.read_h5ad(TEST_H5AD, backed="r")
    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    # max_len used by the run6/corn_lora scripts for the same test set.
    tok = scgpt_cls.tokenize_chunked(x_mat, gene_ids, vocab, pargs, max_len=4253)
    log(f"tokenized test data: genes={tok['genes'].shape}")
    return tok


def extract_embeddings(model: torch.nn.Module, tok: dict, device: torch.device,
                       batch: int = 32) -> np.ndarray:
    embs = []
    n = tok["genes"].shape[0]
    model.to(device).eval()
    with torch.no_grad():
        for s in range(0, n, batch):
            src = tok["genes"][s:s + batch].to(device)
            vals = tok["values"][s:s + batch].to(device)
            pad = tok["mask"][s:s + batch].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(src, vals, src_key_padding_mask=pad, CLS=True)
            embs.append(out["cell_emb"].float().cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    return emb


def build_centroid_svd_axis(emb: np.ndarray, labels: np.ndarray,
                            layer_order: list[str]) -> np.ndarray:
    """Return the first SVD axis of the per-layer centroid matrix.

    Mirrors svd_axis_validation.rho_axis (and the internals of
    marker_gradient_correlation), oriented LSC -> differentiated.
    """
    ord_map = {c: i for i, c in enumerate(layer_order)}
    valid = np.array([lbl in ord_map for lbl in labels])
    lab = labels[valid]
    e = emb[valid]
    present = [c for c in layer_order if (lab == c).any()]
    if len(present) < 2:
        raise ValueError(f"need >=2 layers for axis; got {present}")
    centroids = np.vstack([e[lab == c].mean(axis=0) for c in present])
    centroids_c = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centroids_c, full_matrices=False)
    axis = vt[0].copy().astype(np.float64)
    # Orient so last layer centroid projects above first layer centroid.
    first_c = e[lab == present[0]].mean(axis=0)
    last_c = e[lab == present[-1]].mean(axis=0)
    if (last_c - first_c) @ axis < 0:
        axis = -axis
    return axis


def fixed_axis_rho(emb: np.ndarray, labels: np.ndarray, axis: np.ndarray,
                   layer_order: list[str]) -> float:
    ord_map = {c: i for i, c in enumerate(layer_order)}
    valid = np.array([lbl in ord_map for lbl in labels])
    e = emb[valid]
    lab = labels[valid]
    proj = e @ axis
    ranks = np.array([ord_map[l] for l in lab])
    return float(st.spearmanr(proj, ranks).statistic)


def fixed_axis_rho_and_null(emb: np.ndarray, labels: np.ndarray, axis: np.ndarray,
                           layer_order: list[str], n: int = N_NULL,
                           seed: int = NULL_SEED) -> dict[str, Any]:
    rho_raw = fixed_axis_rho(emb, labels, axis, layer_order)
    emb_n = l2_norm_rows(emb)
    rho_n = fixed_axis_rho(emb_n, labels, axis, layer_order)
    # Label-shuffle null on the L2-normed embeddings, using the fixed axis.
    rng = np.random.default_rng(seed)
    nulls = np.empty(n, dtype=float)
    for i in range(n):
        shuffled = rng.permutation(labels)
        nulls[i] = fixed_axis_rho(emb_n, shuffled, axis, layer_order)
    z = float((rho_n - nulls.mean()) / max(nulls.std(ddof=1), 1e-12))
    return {
        "rho_raw": rho_raw,
        "rho_normed": rho_n,
        "null_mean": float(nulls.mean()),
        "null_sd": float(nulls.std(ddof=1)),
        "null_z": z,
        "null_pctile": float((nulls <= rho_n).mean()),
        "n_null": n,
        "null_seed": seed,
    }


def load_run6_checkpoint(arm: str, seed: int, vocab, pargs,
                         device: torch.device) -> torch.nn.Module:
    run_dir = RUN6_CKPT / f"{arm}_seed{seed}"
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    n_cls = args["n_cls"]
    model = scgpt_cls.build_finetune_model(vocab, pargs, n_cls,
                                             pretrained_dir=PRETRAINED)
    state = torch.load(run_dir / "model.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def load_corn_lora_checkpoint(seed: int, vocab, pargs,
                              device: torch.device) -> torch.nn.Module:
    if wrap_lora is None:
        raise RuntimeError("wrap_lora not available; cannot load corn_lora")
    run_dir = CORN_CKPT / f"corn_lora_seed{seed}"
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    n_cls = args["n_cls"]  # = N_THRESHOLDS = 2
    model = scgpt_cls.build_finetune_model(vocab, pargs, n_cls,
                                             pretrained_dir=PRETRAINED)
    wrap_lora(model)
    # Use the best-val checkpoint (same as the lane's primary evaluation).
    state = torch.load(run_dir / "model_best_val.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def original_run6_rho(arm: str, seed: int) -> float | None:
    path = RUN6_EVAL / f"{arm}_seed{seed}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("rho_normed", data.get("rho")))


def original_corn_lora_rho(seed: int) -> float | None:
    path = CORN_CKPT / f"corn_lora_seed{seed}" / "args.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["test_donor_eval"]["best_val"]["rho_normed"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu-guard", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("Fixed-axis differentiation-gradient evaluation (reviewer M1)")
    log(f"Output: {OUT_JSON}")

    expected_frz_rho = _load_expected_frz_rho()
    log(f"Locked frozen-baseline rho_normed reference: {expected_frz_rho:.4f}")

    if not args.skip_gpu_guard:
        wait_for_gpu(free_mib=4096)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__}")

    # Locked vocab + model args.
    vocab, pargs = scgpt_cls.load_vocab(pretrained_dir=PRETRAINED)

    # Labels and tokenized test cells (once).
    y_true, barcodes = load_test()
    tok = tokenize_test(vocab, pargs)

    # ------------------------------------------------------------------
    # 1. Frozen model embeddings + fixed axis construction
    # ------------------------------------------------------------------
    log("--- extracting FROZEN model embeddings ---")
    # n_cls doesn't affect cell_emb; use 3 to keep cls_decoder shape valid.
    frz_model = scgpt_cls.build_finetune_model(vocab, pargs, 3,
                                                 pretrained_dir=PRETRAINED)
    frz_model.to(device)
    frz_emb = extract_embeddings(frz_model, tok, device)
    log(f"frozen embeddings: {frz_emb.shape}")

    log("--- building frozen centroid-SVD axis ---")
    axis = build_centroid_svd_axis(frz_emb, y_true, LAYER_ORDER)
    axis_path = OUT_DIR / "frozen_axis.npy"
    np.save(axis_path, axis)
    log(f"saved axis to {axis_path}")

    # Sanity: frozen self-projection rho should match the locked frozen baseline.
    frz_rho = fixed_axis_rho(frz_emb, y_true, axis, LAYER_ORDER)
    sanity_delta = abs(frz_rho - expected_frz_rho)
    sanity_pass = sanity_delta <= SANITY_TOL
    log(f"frozen self-projection rho = {frz_rho:.4f}; "
        f"expected ~{expected_frz_rho:.4f}; delta={sanity_delta:.4f}; "
        f"sanity gate={'PASS' if sanity_pass else 'FAIL'}")
    if not sanity_pass:
        log("SANITY GATE FAILED: axis construction is inconsistent; stopping")
        result = {
            "meta": {
                "timestamp_utc": _ts(),
                "test_donor": "sanes2022_Hu0216",
                "n_cells": int(len(y_true)),
                "layer_order": LAYER_ORDER,
                "sanity_gate": {
                    "expected_frz_rho": expected_frz_rho,
                    "observed_frz_rho": frz_rho,
                    "delta": sanity_delta,
                    "tolerance": SANITY_TOL,
                    "pass": False,
                },
            },
            "error": "sanity gate failed; no checkpoint projections written",
        }
        OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 1

    # ------------------------------------------------------------------
    # 2. Fixed-axis rho for frozen + every checkpoint
    # ------------------------------------------------------------------
    checkpoints: list[dict[str, Any]] = [
        {"name": "frozen", "type": "frozen", "seed": None},
    ]
    for arm in ("naive", "ordinal"):
        for seed in (42, 123, 456):
            checkpoints.append({"name": f"{arm}_seed{seed}", "type": "run6",
                                "arm": arm, "seed": seed})
    for seed in (42, 123, 456):
        checkpoints.append({"name": f"corn_lora_seed{seed}",
                            "type": "corn_lora", "seed": seed})

    per_checkpoint: dict[str, Any] = {}
    per_checkpoint["frozen"] = fixed_axis_rho_and_null(frz_emb, y_true, axis,
                                                        LAYER_ORDER)
    per_checkpoint["frozen"]["original_rho_normed"] = expected_frz_rho

    for ckpt in checkpoints[1:]:
        name = ckpt["name"]
        seed = ckpt["seed"]
        log(f"--- {name} ---")
        try:
            if ckpt["type"] == "run6":
                model = load_run6_checkpoint(ckpt["arm"], seed, vocab, pargs,
                                               device)
                orig_rho = original_run6_rho(ckpt["arm"], seed)
            elif ckpt["type"] == "corn_lora":
                model = load_corn_lora_checkpoint(seed, vocab, pargs, device)
                orig_rho = original_corn_lora_rho(seed)
            else:
                raise ValueError(f"unknown checkpoint type: {ckpt['type']}")
            emb = extract_embeddings(model, tok, device)
            res = fixed_axis_rho_and_null(emb, y_true, axis, LAYER_ORDER)
            res["original_rho_normed"] = orig_rho
            per_checkpoint[name] = res
            log(f"  {name}: fixed_axis_rho={res['rho_normed']:.4f} "
                f"(orig per-checkpoint rho={orig_rho}), z={res['null_z']:.1f}")
        except FileNotFoundError as exc:
            log(f"  {name}: MISSING checkpoint/file: {exc}")
            per_checkpoint[name] = {
                "error": f"missing checkpoint/file: {exc}",
                "original_rho_normed": None,
            }
        except Exception as exc:
            log(f"  {name}: ERROR {type(exc).__name__}: {exc}")
            per_checkpoint[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "original_rho_normed": None,
            }

    # ------------------------------------------------------------------
    # 3. Build summary / comparison table
    # ------------------------------------------------------------------
    comparison = []
    for name in ["frozen"] + [c["name"] for c in checkpoints[1:]]:
        entry = per_checkpoint[name]
        if "error" in entry:
            comparison.append({
                "checkpoint": name,
                "fixed_axis_rho": None,
                "fixed_axis_z": None,
                "original_rho_normed": entry.get("original_rho_normed"),
                "status": f"error: {entry['error']}",
            })
        else:
            comparison.append({
                "checkpoint": name,
                "fixed_axis_rho": entry["rho_normed"],
                "fixed_axis_z": entry["null_z"],
                "original_rho_normed": entry["original_rho_normed"],
                "status": "ok",
            })

    # Aggregate per-arm means for the successful checkpoints.
    def _arm_mean(name_prefix: str) -> dict[str, Any]:
        vals = [per_checkpoint[c["name"]] for c in checkpoints[1:]
                if c["name"].startswith(name_prefix) and "error" not in per_checkpoint[c["name"]]]
        if not vals:
            return {"n": 0}
        rhos = [v["rho_normed"] for v in vals]
        zs = [v["null_z"] for v in vals]
        return {
            "n": len(vals),
            "rho_mean": float(np.mean(rhos)),
            "rho_sd": float(np.std(rhos, ddof=1)),
            "z_mean": float(np.mean(zs)),
            "per_seed": {str(c["seed"]): per_checkpoint[c["name"]]["rho_normed"]
                         for c in checkpoints[1:] if c["name"].startswith(name_prefix)
                         and "error" not in per_checkpoint[c["name"]]},
        }

    result = {
        "meta": {
            "timestamp_utc": _ts(),
            "test_donor": "sanes2022_Hu0216",
            "n_cells": int(len(y_true)),
            "layer_order": LAYER_ORDER,
            "axis_construction": {
                "method": "centroid-SVD (first right singular vector of per-layer centroid matrix), oriented LSC -> differentiated",
                "source": "frozen scGPT-human pretrained model embeddings on test donor",
                "axis_path": str(axis_path),
                "axis_norm": float(np.linalg.norm(axis)),
            },
            "sanity_gate": {
                "expected_frz_rho": expected_frz_rho,
                "observed_frz_rho": frz_rho,
                "delta": sanity_delta,
                "tolerance": SANITY_TOL,
                "pass": True,
            },
            "null": {"n": N_NULL, "seed": NULL_SEED},
        },
        "per_checkpoint": per_checkpoint,
        "comparison_table": comparison,
        "aggregate": {
            "naive": _arm_mean("naive_"),
            "ordinal": _arm_mean("ordinal_"),
            "corn_lora": _arm_mean("corn_lora_"),
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"saved {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
