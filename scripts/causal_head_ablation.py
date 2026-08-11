#!/usr/bin/env python
"""Causal head-ablation for the Paper 1 Results R1.2 "head-proximal overwrite" claim.

The observation (R1.2, currently descriptive): naive fine-tuning (lambda=0)
collapses the layerwise differentiation-gradient rho of the head-proximal
layers (L5-L12 -> ~0/negative, L12 ~ -0.042) while early layers (L1-L4)
retain it; the frozen Geneformer rises monotonically to a L12 peak of ~0.806.

This experiment randomises the classification head to test whether the head
is the causal medium of the gradient overwrite. Three variants over the same
14,775 external epithelial cells (GSE153515) and the same reference
pseudotime definition (layer-centroid PCA axis, Spearman rho vs layer ordinal):

  A  naive FT checkpoint 7473 as-is:   trained body + trained head
                                        (sanity control; must bit-exactly
                                         reproduce L12 rho -0.04239536391926124)
  B  naive FT body + RANDOMLY re-initialised classification head (no training,
     no fine-tuning): tests whether the collapse is stored in the BODY weights
  C  frozen Geneformer body + randomly initialised classification head
     (no training): control proving a random head does NOT corrupt the
     gradient readout of a healthy (un-trained) body

MECHANICS NOTE (important for interpretation): hidden states are produced by
the bert body alone (collect_layerwise_cls calls model.bert(...)); the
classification head never participates in hidden-state production at
inference. Consequently variant B is EXPECTED to be bit-identical to variant A
and variant C bit-identical to the frozen reference. The ablation therefore
establishes, quantitatively, (i) whether the collapse is intrinsic to the
trained body weights rather than a readout artefact of the trained head, and
(ii) that a random head leaves a healthy body's gradient untouched. It does
NOT (and cannot, without a training arm) prove that the classification head's
gradient path specifically was the overwrite medium during training.

SANITY GATE (fail-closed, tol 1e-6, same as measure_layerwise_rho.py):
  A: |rho_A[12] - results_0.0_42.json value| <= 1e-6
  B: |rho_B[12] - rho_A[12]| <= 1e-6  AND  hidden states max-abs-diff(B, A) == 0
  C: |rho_C[12] - layerwise_rho_frozen_gf.json value| <= 1e-6

Outputs (artifacts/paper1_layerwise_rho/):
  causal_head_ablation_{variant}.json   (13-layer rho + sanity gate per variant)
  causal_head_ablation_summary.json     (all variants + key-layer table + gate)
figures/paper1/figure_causal_head_ablation.{pdf,png}

Usage:
    python3 scripts/causal_head_ablation.py \
        [--variants A,B,C] [--batch-size 32] [--head-seed 0] [--no-figure]

Inference + label construction + metric mirror scripts/measure_layerwise_rho.py
exactly (which mirrors run_tier_b_eval.py / run_tier_b_baselines.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# --- GPU / environment settings (mirror the eval/training pipelines) ---------
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1+PTX")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as functional  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Local project import (read-only reuse; the function is NOT modified).
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (identical to measure_layerwise_rho.py / Paper 1 eval contract)
# ---------------------------------------------------------------------------

SAVE_DIR = PROJECT_ROOT / "save"
EXT_DATA = PROJECT_ROOT / "data" / "gse153515_annotated.h5ad"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_layerwise_rho"
REF_JSON_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
TOKEN_DATASET = REF_JSON_DIR / "external_tokenized" / "cell.dataset"
FIG_DIR = PROJECT_ROOT / "figures" / "paper1"

# Naive lambda=0 seed=42 fine-tuned checkpoint (latest; matches the model used
# for results_0.0_42.json and summary.json lam0.0_seed42).
FT_CKPT = (
    SAVE_DIR / "caft_gf_lam0.0_seed42_20260731-211824" / "model" / "checkpoint-7473"
)
FT_REF_RHO = -0.04239536391926124  # results_0.0_42.json
FROZEN_REF_RHO = 0.8063606041550953  # layerwise_rho_frozen_gf.json

# Geneformer-V2-104M frozen weights (HF cache snapshot, as in baselines).
PRETRAINED_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--ctheodoris--Geneformer"
    / "snapshots"
    / "04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5"
    / "Geneformer-V2-104M"
)

LAYER_ORDER_4 = ["LSC", "TAC", "basal", "differentiated"]

N_HIDDEN_STATES = 13
LAYER_LABELS = ["embedding_output"] + [f"transformer_layer_{k}" for k in range(1, 13)]
KEY_LAYERS = [3, 5, 8, 12]  # L3 / L5 / L8 / L12 (transformer-layer indices)

N_FT_CLASSES = 31  # 31 author_cell_type classes (num_labels for the head)
BATCH_SIZE = 32  # MUST stay 32 for bit-exact parity with the reference runs

SANITY_TOL = 1e-6


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# External test data (identical to measure_layerwise_rho._load_external_cells)
# ---------------------------------------------------------------------------

def _load_external_cells() -> tuple[object, np.ndarray]:
    """Tokenized epithelial cells (14,775) + harmonized layer labels in order."""
    import anndata as ad
    from datasets import load_from_disk

    dataset = load_from_disk(str(TOKEN_DATASET))
    adata = ad.read_h5ad(EXT_DATA)
    labels = adata.obs["cell_type_harmonized"].astype(str)
    mask = np.asarray(labels != "NON_LAYER")
    keep_idx = np.where(mask)[0]
    dataset = dataset.select(keep_idx)
    labels_epi = np.asarray(labels[mask].to_numpy(), dtype=str)
    print(f"{_ts()}  Epithelial filter: {len(keep_idx)}/{labels.shape[0]} cells "
          f"(NON_LAYER excluded); labels {labels_epi.shape}")
    return dataset, labels_epi


def _collate_batch(batch: list[dict]):
    """Right-pad input_ids within a batch (identical to run_tier_b_eval)."""
    max_len = max(len(b["input_ids"]) for b in batch)
    padded = np.zeros((len(batch), max_len), dtype=np.int64)
    lengths = np.zeros(len(batch), dtype=np.int64)
    for i, b in enumerate(batch):
        ids = np.asarray(b["input_ids"], dtype=np.int64)
        padded[i, : len(ids)] = ids
        lengths[i] = int(b["length"])
    return torch.from_numpy(padded), torch.from_numpy(lengths)


# ---------------------------------------------------------------------------
# Layer-wise CLS collection (identical to measure_layerwise_rho, but keeps the
# full per-layer CLS matrix for the B-vs-A hidden-state equivalence check)
# ---------------------------------------------------------------------------

def collect_layerwise_cls(
    model,
    tokenized_dataset,
    *,
    device: str,
    batch_size: int = BATCH_SIZE,
    l2_normalize: bool,
    fp16_autocast: bool,
    progress_name: str = "batches",
) -> np.ndarray:
    """CLS (position 0) of every hidden state for all cells.

    Returns (13, n_cells, 768) float32 numpy. For frozen Geneformer the CLS
    is L2-normalised (arm protocol in run_tier_b_baselines); for fine-tuned
    checkpoints it is the raw CLS (arm protocol in run_tier_b_eval).
    """
    model.eval()
    loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_batch,
    )
    layer_cls: list[list[np.ndarray]] = [[] for _ in range(N_HIDDEN_STATES)]
    n_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for i, (input_ids, length) in enumerate(loader):
            input_ids = input_ids.to(device)
            length = length.to(device)
            attn_mask = (
                torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                < length.unsqueeze(1)
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=fp16_autocast and device == "cuda",
            ):
                if hasattr(model, "bert"):
                    bert_out = model.bert(
                        input_ids, attention_mask=attn_mask,
                        output_hidden_states=True,
                    )
                else:
                    bert_out = model(
                        input_ids=input_ids, attention_mask=attn_mask,
                        output_hidden_states=True,
                    )
            hidden = bert_out.hidden_states
            if len(hidden) != N_HIDDEN_STATES:
                raise RuntimeError(
                    f"expected {N_HIDDEN_STATES} hidden states, got {len(hidden)}"
                )
            for li, h in enumerate(hidden):
                cls = h[:, 0].float()  # CLS at position 0
                if l2_normalize:
                    cls = functional.normalize(cls, dim=-1)
                layer_cls[li].append(cls.cpu().numpy())
            if (i + 1) % 20 == 0 or i + 1 == n_batches:
                print(f"{_ts()}    [{progress_name}] batch {i + 1}/{n_batches} "
                      f"({time.time() - t0:.0f}s)")
    return np.stack(
        [np.concatenate(part, axis=0) for part in layer_cls], axis=0
    )


def _measure_rhos(layer_cls: np.ndarray, labels: np.ndarray) -> list[float]:
    return [
        marker_gradient_correlation(
            layer_cls[i], labels, layer_order=LAYER_ORDER_4
        )
        for i in range(N_HIDDEN_STATES)
    ]


# ---------------------------------------------------------------------------
# Model construction per variant
# ---------------------------------------------------------------------------

def build_variant_model(
    variant: str, device: str, head_seed: int, torch_dtype,
) -> tuple[nn.Module, dict]:
    """Build the model for a variant; returns (model, provenance metadata)."""
    meta: dict = {"head_seed": head_seed}

    if variant in ("A", "B"):
        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(
            str(FT_CKPT), num_labels=N_FT_CLASSES, torch_dtype=torch_dtype,
        )
        meta.update({
            "body_state": "fine-tuned (naive lambda=0, seed42, checkpoint-7473)",
            "checkpoint_path": str(FT_CKPT),
        })
        if variant == "B":
            # Replace ONLY the classification head with a fresh random init
            # (same default Linear init scheme transformers uses). The trained
            # bert body (incl. pooler) is untouched.
            torch.manual_seed(head_seed)
            head = nn.Linear(model.config.hidden_size, N_FT_CLASSES)
            model.classifier = head.to(device=device, dtype=torch_dtype)
            meta["head_state"] = "random (re-initialised, not trained)"
        else:
            meta["head_state"] = "trained (from checkpoint)"
    else:  # C
        from transformers import BertModel

        model = BertModel.from_pretrained(str(PRETRAINED_DIR)).to(device)
        # Attach a random classification head on the frozen body (the head
        # never participates in hidden-state production; present only so the
        # variant mirrors the "frozen body + random head" ablation arm).
        torch.manual_seed(head_seed)
        head = nn.Linear(model.config.hidden_size, N_FT_CLASSES)
        model.classifier = head.to(device=device)
        meta.update({
            "body_state": "frozen Geneformer-V2-104M (HF snapshot 04c2b2e8)",
            "checkpoint_path": str(PRETRAINED_DIR),
            "head_state": "random (fresh init, not trained)",
        })
    model = model.to(device)
    return model, meta


def _variant_protocol(variant: str) -> dict:
    """Collection protocol per variant (mirrors the arm protocols)."""
    if variant == "C":
        return {"l2_normalize": True, "fp16_autocast": True,
                "dtype": "fp16 autocast (fp32 weights)"}
    return {"l2_normalize": False, "fp16_autocast": False,
            "dtype": "bfloat16"}


# ---------------------------------------------------------------------------
# Sanity gates (fail-closed, tol 1e-6)
# ---------------------------------------------------------------------------

def _gate_passed(diff: float) -> bool:
    return diff <= SANITY_TOL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", default="A,B,C",
        help="Comma-separated variants to run (default A,B,C)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="BATCH SIZE MUST REMAIN 32 for bit-exact parity")
    parser.add_argument("--head-seed", type=int, default=0,
                        help="RNG seed for the random classification head(s)")
    parser.add_argument("--no-figure", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    if args.batch_size != BATCH_SIZE:
        print(f"{_ts()}  WARNING: batch-size {args.batch_size} != {BATCH_SIZE}; "
              f"bit-exact parity with reference runs is NOT guaranteed.")
    variants = [v.strip().upper() for v in args.variants.split(",")]
    bad = [v for v in variants if v not in ("A", "B", "C")]
    if bad:
        print(f"{_ts()}  ERROR: unknown variant(s) {bad}; must be A, B or C.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{_ts()}  Device: {device}; variants: {variants}; head_seed: "
          f"{args.head_seed}")
    if device != "cuda":
        print(f"{_ts()}  ERROR: CUDA required (as in run_tier_b_baselines).")
        return 1
    torch_dtype = torch.bfloat16

    dataset, labels = _load_external_cells()
    n_cells = len(dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    cls_cache: dict[str, np.ndarray] = {}
    gate_failed = False

    for variant in variants:
        print(f"\n{_ts()}  === Variant {variant} ===")
        proto = _variant_protocol(variant)
        model, meta = build_variant_model(variant, device, args.head_seed,
                                          torch_dtype)
        layer_cls = collect_layerwise_cls(
            model, dataset, device=device, batch_size=args.batch_size,
            l2_normalize=proto["l2_normalize"],
            fp16_autocast=proto["fp16_autocast"],
            progress_name=f"variant {variant}",
        )
        cls_cache[variant] = layer_cls
        del model
        torch.cuda.empty_cache()

        rhos = _measure_rhos(layer_cls, labels)
        gate: dict = {}

        if variant == "A":
            diff = abs(rhos[-1] - FT_REF_RHO)
            passed = _gate_passed(diff)
            gate = {"layer12_rho": rhos[-1], "reference_rho": FT_REF_RHO,
                    "reference_source": "results_0.0_42.json "
                                        "(marker_gradient_rank_correlation)",
                    "abs_diff": diff, "passed": bool(passed)}
        elif variant == "B":
            if "A" in cls_cache:
                max_abs = float(np.abs(layer_cls - cls_cache["A"]).max())
            else:
                # Compare against the persisted A result if A was not run here.
                a_path = OUT_DIR / "causal_head_ablation_A.json"
                if a_path.is_file():
                    a_cls = np.load(OUT_DIR / "causal_head_ablation_A_cls.npy")
                    max_abs = float(np.abs(layer_cls - a_cls).max())
                else:
                    print(f"{_ts()}  ERROR: variant B requires variant A "
                          f"(run --variants A,B or A,B,C).")
                    return 1
            rho_a = results["A"]["marker_gradient_rho"][-1] if "A" in results \
                else _read_rho_a()
            diff = abs(rhos[-1] - rho_a)
            passed = _gate_passed(diff) and max_abs == 0.0
            gate = {"layer12_rho": rhos[-1],
                    "reference_rho": rho_a,
                    "reference_source": "variant A (same body, trained head)",
                    "abs_diff_vs_A": diff,
                    "hidden_max_abs_diff_vs_A": max_abs,
                    "passed": bool(passed)}
        else:  # C
            diff = abs(rhos[-1] - FROZEN_REF_RHO)
            passed = _gate_passed(diff)
            gate = {"layer12_rho": rhos[-1], "reference_rho": FROZEN_REF_RHO,
                    "reference_source": "layerwise_rho_frozen_gf.json",
                    "abs_diff": diff, "passed": bool(passed)}

        gate_failed |= not passed
        print(f"{_ts()}  SANITY variant {variant}: rho[-1]={rhos[-1]:.8f} "
              f"ref={gate.get('reference_rho', 'n/a')} "
              f"-> {'PASS' if passed else 'FAIL'}")
        if variant == "B":
            print(f"{_ts()}    hidden max-abs-diff vs A: "
                  f"{gate['hidden_max_abs_diff_vs_A']:.3e}")

        entry = {
            "variant": variant,
            "description": _VARIANT_DESC[variant],
            "n_layers": N_HIDDEN_STATES,
            "layer_labels": LAYER_LABELS,
            "key_layers": {f"L{k}": rhos[k] for k in KEY_LAYERS},
            "n_cells": int(n_cells),
            "layer_order": LAYER_ORDER_4,
            "l2_normalized": proto["l2_normalize"],
            "fp16_autocast": proto["fp16_autocast"],
            "dtype": proto["dtype"],
            "batch_size": args.batch_size,
            **meta,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "marker_gradient_rho": rhos,
            "sanity_gate": gate,
        }
        results[variant] = entry

        out_json = OUT_DIR / f"causal_head_ablation_{variant}.json"
        with open(out_json, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"{_ts()}  Saved {out_json}")
        if variant == "A":
            np.save(OUT_DIR / "causal_head_ablation_A_cls.npy", layer_cls)

    # --- Summary table ------------------------------------------------------
    print(f"\n{_ts()}  === Summary (Spearman rho per layer) ===")
    header = ["variant"] + [f"L{k}" for k in KEY_LAYERS] + ["L12 gate"]
    print("  " + "  ".join(f"{h:>10}" for h in header))
    for v in ("A", "B", "C"):
        if v not in results:
            continue
        e = results[v]
        row = [v] + [f"{e['key_layers'][f'L{k}']:.4f}" for k in KEY_LAYERS] \
              + ["PASS" if e["sanity_gate"]["passed"] else "FAIL"]
        print("  " + "  ".join(f"{x:>10}" for x in row))
    print("  " + "  ".join(f"{x:>10}" for x in
          ["frozen_ref", "0.7656", "0.7868", "0.5687", "0.8064", "-"]))

    # --- Summary JSON -------------------------------------------------------
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": "causal_head_ablation",
        "purpose": ("Test whether the classification head is the causal medium "
                    "of the Paper 1 R1.2 head-proximal overwrite (naive FT "
                    "collapses L5-L12 gradient rho; frozen GF does not)."),
        "data": "GSE153515 external epithelial cells",
        "n_cells": int(n_cells),
        "sanity_tolerance": SANITY_TOL,
        "sanity_gate_all_passed": not gate_failed,
        "key_layers": {f"L{k}": k for k in KEY_LAYERS},
        "frozen_reference_key_layers": {f"L{k}": v for k, v in
                                        zip(KEY_LAYERS, [0.7656, 0.7868,
                                                         0.5687, 0.8064])},
        "variants": results,
    }
    with open(OUT_DIR / "causal_head_ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{_ts()}  Saved {OUT_DIR / 'causal_head_ablation_summary.json'}")

    # --- Figure -------------------------------------------------------------
    if not args.no_figure:
        _make_figure(results)

    if gate_failed:
        print(f"\n{_ts()}  SANITY GATE FAILED — refusing to report success.")
        for v, e in results.items():
            if not e["sanity_gate"]["passed"]:
                print(f"    {v}: {e['sanity_gate']}")
        return 2

    print(f"\n{_ts()}  SANITY GATE PASSED for all {len(results)} variants "
          f"(tol {SANITY_TOL:.0e}).")
    print(f"{_ts()}  Done.")
    return 0


def _read_rho_a() -> float:
    """Layer-12 rho of variant A from its persisted JSON (for standalone B)."""
    path = OUT_DIR / "causal_head_ablation_A.json"
    with open(path) as f:
        return float(json.load(f)["marker_gradient_rho"][-1])


# ---------------------------------------------------------------------------
# Figure (style mirrors generate_paper1_results_figures.py)
# ---------------------------------------------------------------------------

_VARIANT_DESC = {
    "A": ("naive FT checkpoint as-is: trained body + trained head "
          "(sanity control)"),
    "B": ("fine-tuned body + randomly re-initialised classification head, "
          "no training"),
    "C": ("frozen Geneformer body + randomly initialised classification head, "
          "no training"),
}

# Okabe-Ito colorblind-safe palette (same as generate_paper1_results_figures.py)
OI_BLUE = "#0072B2"
OI_BLUEGREEN = "#009E73"
OI_ORANGE = "#E69F00"
FROZEN_RHO_COLOR = "#333333"


def _make_figure(results: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#000000",
        "legend.frameon": False,
        "svg.fonttype": "none",
    })

    layers = list(range(1, 13))  # transformer layers 1..12 (drop embedding output)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    styles = {
        "A": (OI_BLUE, 1.6, "-", "o"),
        "B": (OI_BLUEGREEN, 1.6, "--", "s"),
        "C": (OI_ORANGE, 1.6, ":", "^"),
    }
    # Frozen reference (persisted layerwise rho, unchanged by the ablation)
    frozen_path = OUT_DIR / "layerwise_rho_frozen_gf.json"
    if frozen_path.is_file():
        with open(frozen_path) as f:
            frozen_rho = json.load(f)["marker_gradient_rho"]
        ax.plot(layers, [frozen_rho[i] for i in layers],
                color=FROZEN_RHO_COLOR, lw=2.4, ls="-",
                label="frozen GF (reference)", zorder=4)

    for v in ("A", "B", "C"):
        if v not in results:
            continue
        color, lw, ls, marker = styles[v]
        rho = results[v]["marker_gradient_rho"]
        ys = [rho[i] for i in layers]
        ax.plot(layers, ys, color=color, lw=lw, ls=ls, marker=marker,
                markevery=1, markersize=3, zorder=3,
                label=f"{v}: {_VARIANT_DESC[v].split(':')[0]}")
    ax.axhline(0.0, color=OI_BLUE, ls=":", lw=0.7, zorder=1)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(layers)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Differentiation gradient \u03c1 (Spearman)")
    ax.set_title("Causal head-ablation: is the classifier head the overwrite "
                 "medium? (GSE153515, 14,775 cells)", fontsize=9,
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "figure_causal_head_ablation.pdf"
    out_png = FIG_DIR / "figure_causal_head_ablation.png"
    fig.savefig(out_pdf, dpi=300, metadata={
        "Title": "Causal head-ablation (R1.2 head-proximal overwrite)",
        "Author": "Corneal-FM (Paper1, Ophthalmology Science)",
        "Subject": "Randomised-head ablation, GSE153515",
        "Creator": "scripts/causal_head_ablation.py",
    })
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"{_ts()}  Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    sys.exit(main())
