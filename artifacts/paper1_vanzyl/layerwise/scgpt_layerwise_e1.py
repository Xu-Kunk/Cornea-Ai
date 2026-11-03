#!/usr/bin/env python
"""E1: scGPT layerwise differentiation-gradient analysis (van Zyl test donor).

For each config {frozen, s42_bestval, s42_final, s123_bestval, s123_final},
forward the locked test donor (Hu0216, 20,158 cells) and capture the [CLS]
(position-0) hidden state at every depth (pre-encoder embedding L0 and the
12 transformer layer outputs L1..L12) via forward hooks. Per layer:
rho(raw), rho(L2-normed), label-shuffle z (n=100, seed 42) — same metric
implementation as the G1/6-run pipelines (marker_gradient_correlation +
label_shuffle_null imported unchanged from scgpt_frozen_diag_rebuild).

The frozen final-layer rho must reproduce the G1 frozen test-donor value
(0.7887) as a pipeline sanity check.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/summer/corneal_scgpt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scgpt_cls  # noqa: E402
from scgpt_frozen_diag_rebuild import l2_norm_rows, label_shuffle_null  # noqa: E402
from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

TEST_H5AD = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "scgpt_vanzyl_test.h5ad"
CONV_CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "convergence" / "checkpoints"
SCGPT_PRETRAINED = ROOT / "pretrained" / "scgpt_pretrained"
OUT_DIR = ROOT / "artifacts" / "paper1_vanzyl" / "layerwise"
LAYER_ORDER = ["LSC", "basal", "differentiated"]
N_NULL = 100

CONFIGS = [
    ("frozen", None),
    ("s42_bestval", CONV_CKPT / "naive_seed42" / "model_best_val.pt"),
    ("s42_final", CONV_CKPT / "naive_seed42" / "model_epoch15.pt"),
    ("s123_bestval", CONV_CKPT / "naive_seed123" / "model_best_val.pt"),
    ("s123_final", CONV_CKPT / "naive_seed123" / "model_epoch15.pt"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, pargs = scgpt_cls.load_vocab(pretrained_dir=SCGPT_PRETRAINED)
    args_ref = json.loads((CONV_CKPT / "naive_seed42" / "args.json")
                          .read_text(encoding="utf-8"))
    n_cls = args_ref["n_cls"]
    log(f"device={device} n_cls={n_cls} torch={torch.__version__}")

    import anndata as ad
    a = ad.read_h5ad(TEST_H5AD, backed="r")
    y_true = np.asarray(a.obs["cell_class_4"].astype(str))
    keep = np.array([g in vocab for g in a.var_names])
    gene_ids = np.array([vocab[g] for g in a.var_names[keep]], dtype=np.int64)
    x_mat = a[:, keep].X
    if hasattr(x_mat, "to_memory"):
        x_mat = x_mat.to_memory()
    tok = scgpt_cls.tokenize_chunked(x_mat, gene_ids, vocab, pargs,
                                     max_len=args_ref["training"]["max_len"])
    pad_id = vocab[pargs["pad_token"]]
    n = tok["genes"].shape[0]
    log(f"test: {n} cells, seq_len={tuple(tok['genes'].shape)}, "
        f"y dist={dict(zip(*np.unique(y_true, return_counts=True)))}")

    results: dict[str, Any] = {}
    for name, ckpt in CONFIGS:
        log(f"=== {name} ===")
        t0 = time.time()
        model = scgpt_cls.build_finetune_model(vocab, pargs, n_cls,
                                               pretrained_dir=SCGPT_PRETRAINED)
        if ckpt is not None:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        model.to(device).eval()

        # per-layer [CLS] capture via forward hooks
        layer_cls: dict[int, list[np.ndarray]] = {i: [] for i in range(13)}
        handles = []
        for i, layer in enumerate(model.transformer_encoder.layers):
            def make_hook(idx: int, is_input: bool = False):
                def hook(module, inp, out):
                    src = inp[0] if is_input else out
                    layer_cls[idx].append(
                        src[:, 0, :].detach().float().cpu().numpy())
                return hook
            if i == 0:
                handles.append(layer.register_forward_hook(make_hook(0, True)))
            handles.append(layer.register_forward_hook(make_hook(i + 1)))

        with torch.no_grad():
            for s in range(0, n, 32):
                src = tok["genes"][s:s + 32].to(device)
                vals = tok["values"][s:s + 32].to(device)
                pad = tok["mask"][s:s + 32].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    model(src, vals, src_key_padding_mask=pad, CLS=False)
        for h in handles:
            h.remove()

        layers: dict[str, Any] = {}
        for i in range(13):
            emb = np.concatenate(layer_cls[i], axis=0).astype(np.float32)
            emb_n = l2_norm_rows(emb)
            rho_raw = float(marker_gradient_correlation(emb, y_true, LAYER_ORDER))
            rho_n = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
            null = label_shuffle_null(emb_n, y_true, LAYER_ORDER, n=N_NULL, seed=42)
            z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))
            layers[f"L{i}"] = {"rho_raw": rho_raw, "rho_normed": rho_n,
                               "null_mean": float(null.mean()),
                               "null_sd": float(null.std(ddof=1)),
                               "z": z, "n_null": N_NULL}
        results[name] = layers
        log(f"  L12 rho_n={layers['L12']['rho_normed']:.4f} z={layers['L12']['z']:.1f} "
            f"({time.time() - t0:.0f}s)")

    out = {
        "meta": {
            "task": "E1 scGPT layerwise rho (van Zyl test donor Hu0216, 20,158 cells)",
            "layer_order": LAYER_ORDER,
            "checkpoints": {n: (str(c) if c else "pretrained frozen")
                            for n, c in CONFIGS},
            "sanity": "frozen L12 rho should reproduce G1 frozen test-donor 0.7887",
            "metric_source": "scgpt_frozen_diag_rebuild (unchanged)",
            "timestamp_utc": now_utc(),
        },
        "per_layer": results,
    }
    op = OUT_DIR / "layerwise_rho.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("saved " + str(op))

    # quick line plot (working artifact; des-1 restyle later)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    style = {"frozen": ("k", "--"), "s42_bestval": ("tab:blue", "-"),
             "s42_final": ("tab:red", "-"), "s123_bestval": ("tab:green", "-"),
             "s123_final": ("tab:purple", "-")}
    xs = list(range(13))
    for name, (c, ls) in style.items():
        ys = [results[name][f"L{i}"]["rho_normed"] for i in range(13)]
        ax.plot(xs, ys, color=c, ls=ls, lw=1.8, label=name)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{i}" for i in range(13)])
    ax.set_xlabel("Transformer depth (L0 = embedding, L1--L12 encoder layers)")
    ax.set_ylabel("Differentiation-gradient Spearman rho (L2-normed)")
    ax.legend(fontsize=8)
    ax.set_title("scGPT layerwise rho (van Zyl test donor Hu0216)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "layerwise_rho.png", dpi=300)
    log("saved layerwise_rho.png")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
