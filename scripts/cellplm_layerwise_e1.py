#!/usr/bin/env python
"""E1 extension: CellPLM layerwise differentiation-gradient analysis
(third architecture, cell-as-token, FROZEN 2026-08-20).

Runs on p5070 with /home/summer/corneal_scgpt/venv/bin/python. For the locked
test donor Hu0216 (20158 cells), captures per-layer embeddings of the CellPLM
OmicsFormer (cell-as-token: 1 cell = 1 token) and computes per layer the same
metrics as the E1 scGPT layerwise analysis (marker_gradient_correlation +
label_shuffle_null, unchanged):

  * L0  = embedder output (OmicsEmbeddingLayer, (n_cells, enc_hid))
  * L1..L4 = Flowformer encoder layer outputs (encoder.layers[i])
  * LAT = latent (post pre_latent_norm + VAE mu) -- the representation the
          official CellPLM pipeline uses for rho/eval

Configs (frozen): frozen (unfitted pretrained), naive (cellplm/seed42,
rho_n=-0.046 collapsed), corn (cellplm/corn/seed42, rho_n=0.920 rescued).

Sanity gates (locked seed json values, tol 2e-3): LAT rho(L2-normed) must
reproduce frozen 0.8652 / naive -0.0464 / corn 0.9203.

GPU guard: waits until torch.cuda.mem_get_info()[0] >= 4 GiB (poll 60 s,
max wait budget 5 h).
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

PROJECT_ROOT = Path("/home/summer/cellplm_vanzyl")
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS))

from CellPLM.pipeline.cell_type_annotation import (  # noqa: E402
    CellTypeAnnotationDefaultModelConfig,
    CellTypeAnnotationDefaultPipelineConfig,
    CellTypeAnnotationPipeline,
)
from run_cellplm_corn import (  # noqa: E402  (locked arm C3-CORN machinery)
    LAYER_ORDER,
    N_ORDINAL,
    N_THRESHOLDS,
    CORNAnnotationHead,
    label_shuffle_null,
)

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "vanzyl_cellplm.h5ad"
CKPT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "ckpt"
NAIVE_CKPT = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "seed42" / "model_best_valid.pt"
CORN_CKPT = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm" / "corn" / "seed42" / "model_best_valid.pt"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm_layerwise"
TEST_DONOR = "sanes2022_Hu0216"
RHO_TOL = 2e-3
NULL_N, NULL_SEED = 100, 42
MIN_FREE_GPU = 4 * 1024**3
MAX_WAIT_S = 5 * 3600
POLL_S = 60
LAYER_NAMES = ["L0", "L1", "L2", "L3", "L4", "LAT"]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def _gpu_state() -> tuple[bool, float]:
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        free, _total = torch.cuda.mem_get_info(0)
        return free >= MIN_FREE_GPU, free / 2**30
    except Exception:
        return False, 0.0


def wait_for_gpu() -> None:
    t0 = time.time()
    while True:
        ok, free_gib = _gpu_state()
        if ok:
            log(f"GPU guard passed: free {free_gib:.2f} GiB >= 4 GiB")
            return
        waited = time.time() - t0
        log(f"GPU busy/contended: free {free_gib:.2f} GiB; waited {waited / 60:.0f} min")
        if waited > MAX_WAIT_S:
            raise RuntimeError(f"GPU guard timed out after {MAX_WAIT_S / 3600:.0f} h")
        time.sleep(POLL_S)


def build_pipeline(device: str, corn: bool) -> CellTypeAnnotationPipeline:
    """Build the pipeline; for corn, swap in the CORN head (mechanical port,
    identical to corn_advantage_cellplm.build_pipeline)."""
    model_config = CellTypeAnnotationDefaultModelConfig.copy()
    model_config["out_dim"] = 3
    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix="20230926_85M",
        overwrite_config=model_config,
        pretrain_directory=str(CKPT_DIR),
    )
    pipeline.model.to(device)
    if corn:
        in_dim = pipeline.model.head.mlp[1].in_features
        corn_head = CORNAnnotationHead(
            in_dim=in_dim,
            hidden_dim=in_dim,
            num_thresholds=N_THRESHOLDS,
            num_layers=CellTypeAnnotationDefaultModelConfig["dec_layers"],
            dropout=CellTypeAnnotationDefaultModelConfig["model_dropout"],
            norm=None,
            batch_num=0,
            n_ordinal=N_ORDINAL,
            class_to_ordinal={},
        ).to(device)
        pipeline.model.head = corn_head
    return pipeline


def forward_capture(pipeline, te_panel, device: str):
    """Forward test cells with hooks on embedder (L0) + encoder layers (L1..L4);
    also return the official latent (LAT = VAE mu, the eval representation).
    Returns dict layer_name -> (n_cells, hid) np arrays, aligned to row order.
    """
    from CellPLM.utils.data import TranscriptomicDataset, XDict
    from torch.utils.data import DataLoader

    dataset = TranscriptomicDataset(te_panel, None, ["batch"], order_required=True)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    captured: dict[str, list] = {k: [] for k in LAYER_NAMES}
    handles = []

    def make_hook(key: str):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            captured[key].append(out.detach().float().cpu().numpy())
        return hook

    handles.append(pipeline.model.embedder.register_forward_hook(make_hook("L0")))
    for i, layer in enumerate(pipeline.model.encoder.layers):
        handles.append(layer.register_forward_hook(make_hook(f"L{i + 1}")))

    latents, order_list = [], []
    pipeline.model.eval()
    with torch.no_grad():
        for data_dict in dataloader:
            input_dict = {}
            for k in data_dict:
                if k == "x_seq":
                    input_dict[k] = data_dict[k].to(device)
                elif k not in ["gene_list", "split", "order_list"]:
                    input_dict[k] = data_dict[k].to(device)
            input_dict["loss_mask"] = torch.ones(data_dict["x_seq"].shape[0]).bool().to(device)
            out_dict, _ = pipeline.model(XDict(input_dict), data_dict["gene_list"])
            latents.append(out_dict["latent"].cpu())
            order_list.append(input_dict.get("order_list", torch.arange(data_dict["x_seq"].shape[0])).cpu())
    for h in handles:
        h.remove()

    emb_lat = torch.cat(latents).numpy()
    order = torch.cat(order_list).numpy()
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))

    out: dict[str, np.ndarray] = {}
    for k, parts in captured.items():
        if k == "LAT":
            continue  # LAT populated below (official eval representation)
        emb = np.concatenate(parts, axis=0).astype(np.float32)
        out[k] = emb[inv]
    out["LAT"] = emb_lat[inv]
    return out


def config_setup(cfg: str, device: str, max_cells: int | None = None):
    """Return (pipeline, te_panel, y_true, locked_rho) for one config."""
    import random
    import anndata as ad
    import hdf5plugin  # noqa: F401

    full = ad.read_h5ad(DATA_PATH)
    tr_va = full[full.obs["split"].astype(str) != "test"].copy()
    te = full[full.obs["split"].astype(str) == "test"].copy()
    assert te.n_obs == 20158 and (te.obs["donor_id"].astype(str) == TEST_DONOR).all()
    if max_cells is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(te.n_obs, size=min(max_cells, te.n_obs), replace=False)
        te = te[idx].copy()
        log(f"[SMOKE] capped test cells to {te.n_obs}")
    corn = cfg == "corn"
    # FROZEN reproducibility (recorded deviation D-F1): the AE latent layer
    # (latent.layers.1.lat_2lat) is NOT in the pretrained checkpoint and is
    # randomly initialised at pipeline build -- the locked frozen baselines in
    # the seed jsons differ across seeds (42: 0.865 / 123: 0.803 / 456: 0.860)
    # for exactly this reason. To reproduce the locked seed42 frozen baseline
    # (0.8652), the frozen pipeline must be built under the seed-42 RNG state
    # (run_cellplm_vanzyl.set_seed(42) at run start). naive/corn are
    # unaffected (trained lat_2lat comes from the checkpoint).
    if cfg == "frozen":
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    pipeline = build_pipeline(device, corn)
    pipeline.common_preprocess(
        tr_va, CellTypeAnnotationDefaultPipelineConfig["hvg"], ["batch"], False
    )
    gene_list = list(pipeline.gene_list)
    te_panel = te[:, te.var.index.isin(gene_list)].copy()
    log(f"[{cfg}] panel {len(gene_list)} genes; test panel {te_panel.shape[1]}")

    if cfg == "naive":
        state = torch.load(NAIVE_CKPT, map_location="cpu")["model_state_dict"]
        pipeline.model.load_state_dict(state, strict=True)
        log(f"[{cfg}] loaded naive checkpoint")
    elif cfg == "corn":
        state = torch.load(CORN_CKPT, map_location="cpu")["model_state_dict"]
        pipeline.model.load_state_dict(state, strict=True)
        log(f"[{cfg}] loaded corn checkpoint")

    y_true = te_panel.obs["celltype"].values.astype(str)

    # locked reference (for the LAT sanity gate)
    if cfg == "frozen":
        ref = None  # frozen LAT gate = the naive seed json frozen baseline
    elif cfg == "naive":
        ref = json.loads((PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm"
                          / "seed42" / "seed42.json").read_text())["eval"]["rho_normed"]
    else:
        ref = json.loads((PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm"
                          / "corn" / "seed42" / "seed42.json").read_text())["eval"]["rho_normed"]
    return pipeline, te_panel, y_true, ref


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", nargs="+",
                    default=["frozen", "naive", "corn"])
    ap.add_argument("--max-cells", type=int, default=None,
                    help="cap test cells (smoke aid only; real run uses all "
                         "20158 -- flagged in meta)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__} max_cells={args.max_cells}")

    frozen_gate = None
    results: dict[str, Any] = {}
    for cfg in args.configs:
        if device == "cuda":
            wait_for_gpu()
        pipeline, te_panel, y_true, ref = config_setup(cfg, device,
                                                       args.max_cells)
        if cfg == "frozen":
            frozen_gate = ref  # None -> use naive seed json frozen baseline below
        # use the naive seed json frozen baseline as the shared frozen gate
        if frozen_gate is None and cfg == "frozen":
            frozen_gate = json.loads(
                (PROJECT_ROOT / "artifacts" / "paper1_vanzyl" / "cellplm"
                 / "seed42" / "seed42.json").read_text())["frozen_baseline"]["rho_normed"]

        layers = forward_capture(pipeline, te_panel, device)
        per_layer: dict[str, Any] = {}
        for name in LAYER_NAMES:
            emb = layers[name]
            emb_n = l2_norm_rows(emb)
            rho_raw = float(marker_gradient_correlation(emb, y_true, LAYER_ORDER))
            rho_n = float(marker_gradient_correlation(emb_n, y_true, LAYER_ORDER))
            null = label_shuffle_null(emb_n, y_true, NULL_N, NULL_SEED)
            z = float((rho_n - null.mean()) / max(float(null.std(ddof=1)), 1e-12))
            per_layer[name] = {
                "rho_raw": rho_raw,
                "rho_normed": rho_n,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std(ddof=1)),
                "z": z,
                "n_null": NULL_N,
            }
            log(f"  [{cfg}] {name}: rho_n={rho_n:.4f} rho_raw={rho_raw:.4f} z={z:.1f}")

        # LAT sanity gate vs locked seed json
        lat_rho = per_layer["LAT"]["rho_normed"]
        if cfg == "frozen":
            gate = frozen_gate
        else:
            gate = ref
        gate_ok = gate is not None and abs(lat_rho - gate) <= RHO_TOL
        log(f"[{cfg}] LAT gate: {lat_rho:.4f} vs locked {gate} -> "
            f"{'PASS' if gate_ok else 'FAIL'}")
        results[cfg] = {
            "per_layer": per_layer,
            "lat_gate": {"locked": gate, "observed": lat_rho,
                         "tol": RHO_TOL, "pass": bool(gate_ok)},
        }
        del pipeline
        if device == "cuda":
            torch.cuda.empty_cache()

    out = {
        "meta": {
            "task": "E1 extension: CellPLM layerwise rho (cell-as-token, "
                    "van Zyl test donor Hu0216, 20158 cells)",
            "layer_order": LAYER_ORDER,
            "layers": LAYER_NAMES,
            "layer_meaning": {"L0": "embedder output",
                              "L1..L4": "Flowformer encoder layers",
                              "LAT": "latent (VAE mu) = official eval representation"},
            "checkpoints": {
                "frozen": "unfitted pretrained 20230926_85M",
                "naive": "cellplm/seed42/model_best_valid.pt",
                "corn": "cellplm/corn/seed42/model_best_valid.pt",
            },
            "metric_source": "corn_advantage_cellplm / run_cellplm_vanzyl "
                             "(unchanged) + corneal_fm.evaluation",
            "sanity": "LAT rho(L2-normed) vs locked seed json (tol 2e-3): "
                      "frozen 0.8652 / naive -0.0464 / corn 0.9203",
            "timestamp_utc": now_utc(),
        },
        "per_config": results,
    }
    op = OUT_DIR / "cellplm_layerwise_rho.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("saved " + str(op))

    # quick line plot (working artifact; des-1 restyle later)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    style = {"frozen": ("k", "--"), "naive": ("tab:red", "-"),
             "corn": ("tab:green", "-")}
    xs = list(range(len(LAYER_NAMES)))
    for cfg, (c, ls) in style.items():
        if cfg not in results:
            continue
        ys = [results[cfg]["per_layer"][n]["rho_normed"] for n in LAYER_NAMES]
        ax.plot(xs, ys, color=c, ls=ls, lw=1.8, label=cfg)
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(LAYER_NAMES)
    ax.set_xlabel("CellPLM depth (L0 = embedder, L1--L4 Flowformer, LAT = latent)")
    ax.set_ylabel("Differentiation-gradient Spearman rho (L2-normed)")
    ax.legend(fontsize=8)
    ax.set_title("CellPLM layerwise rho (van Zyl test donor Hu0216)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cellplm_layerwise_rho.png", dpi=300)
    log("saved cellplm_layerwise_rho.png")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
