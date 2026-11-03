#!/usr/bin/env python
"""GSE202352 (Wiedemann 2023, human skin) TP-OAC cross-tissue eval.

Mirrors the locked GSE147424 eval (scripts/epidermis_tpoac_eval.py) stages
tokenize / forward / metrics on the new cross-tissue control dataset:

  1. tokenize : Geneformer V2 tokenizer (custom attrs cell_type,
     author_cell_type, disease, tissue, donor_id, layer; nproc=4;
     collapse_gene_ids=True) -> artifacts/paper1_epidermis_eval_gse202352/
     external_tokenized/cell.dataset
  2. forward  : layer-12 CLS (bf16, GPU, batch 32) for the SAME 6
     checkpoints as the GSE147424 eval (frozen HF snapshot, ce_naive,
     corns_ordinal, caft, aa_ft_newaxis, ss_newaxis) ->
     artifacts/paper1_epidermis_eval_gse202352/embeddings.npz
  3. metrics  : marker_gradient_rank_correlation(emb, ordinal_layer) +
     continuous spearmanr(proj onto eval axis, s-score) ->
     artifacts/paper1_epidermis_eval_gse202352/gse202352_rho_results.json
  4. sanity   : ordinal label balance, embedding shapes vs n_cells, frozen
     arm rho in [0, 0.5] (expected WEAK: frozen Geneformer's skin axis is
     weak for this cross-tissue control).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "artifacts" / "paper1_epidermis_eval_gse202352" / "gse202352_processed.h5ad"
OUT_DIR = ROOT / "artifacts" / "paper1_epidermis_eval_gse202352"
TOKEN_CACHE = OUT_DIR / "external_tokenized"
EMB_NPZ = OUT_DIR / "embeddings.npz"
RHO_JSON = OUT_DIR / "gse202352_rho_results.json"

BASAL_MARKERS = ["KRT5", "KRT14", "TP63"]
DIFF_MARKERS = ["KRT1", "KRT10", "IVL", "FLG"]
LAYER_ORDER = ["basal", "suprabasal", "spinous", "differentiated"]
ORDER_MAP = {c: i for i, c in enumerate(LAYER_ORDER)}
DATASET_NAME = "Wiedemann2023_GSE202352"

# checkpoint -> (label, path) — IDENTICAL to the locked GSE147424 eval
CKPTS = [
    ("frozen", ROOT / ".hf_snapshot_placeholder"),
    ("ce_naive", ROOT / "save/caft_gf_lam0.0_seed42_20260731-211824/model/checkpoint-7473"),
    ("corns_ordinal", ROOT / "save/ordinal_gf_lam0.0_seed42_20260805-171141/model/checkpoint-7473"),
    ("caft", ROOT / "save/caft_gf_lam0.5_seed42_20260801-200953/model/checkpoint-7473"),
    ("aa_ft_newaxis", ROOT / "save/method_axis_ft_lam1.0_seed42_20260810-203641/model/checkpoint-7473"),
    ("ss_newaxis", ROOT / "save/method_soft_spearman_lam0.5_seed42_20260811-040352/model/checkpoint-7473"),
]


def _hf_snapshot() -> Path:
    import glob

    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M")))
    if not hits:
        raise FileNotFoundError("Geneformer-V2-104M snapshot not found in HF cache")
    return Path(hits[0])


def stage_tokenize() -> int:
    import anndata as ad
    from geneformer import TranscriptomeTokenizer

    from corneal_fm import gf_compat

    gf_compat.install()
    TOKEN_CACHE.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(DATA)
    obs = a.obs.copy().reset_index(drop=True)
    obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    obs["layer"] = obs["ordinal_idx"].astype("int64")
    for col in ("cell_type", "author_cell_type", "disease", "tissue", "donor_id"):
        if col not in obs.columns:
            obs[col] = "skin" if col == "tissue" else ("healthy" if col == "disease" else "unknown")
    obs["cell_type"] = obs["ordinal_layer"].astype(str)
    obs["author_cell_type"] = obs["ordinal_layer"].astype(str)
    a.obs = obs
    tmp = Path(tempfile.mkdtemp(prefix="gse202352_tokenize_"))
    tmp_h5ad = tmp / "gse202352.h5ad"
    a.write_h5ad(tmp_h5ad)
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "cell_type": "cell_type",
            "author_cell_type": "author_cell_type",
            "disease": "disease",
            "tissue": "tissue",
            "donor_id": "donor_id",
            "layer": "layer",
        },
        nproc=4,
        chunk_size=512,
        model_version="V2",
        use_h5ad_index=True,
        collapse_gene_ids=True,
    )
    tk.tokenize_data(
        data_directory=tmp,
        output_directory=str(TOKEN_CACHE),
        output_prefix="cell",
        file_format="h5ad",
    )
    shutil.rmtree(tmp, ignore_errors=True)
    from datasets import load_from_disk

    ds = load_from_disk(str(TOKEN_CACHE / "cell.dataset"))
    print(f"tokenized {len(ds)} cells -> {TOKEN_CACHE / 'cell.dataset'}")
    return len(ds)


def _collate(batch):
    import torch

    max_len = max(len(b["input_ids"]) for b in batch)
    padded = np.zeros((len(batch), max_len), dtype=np.int64)
    lengths = np.zeros(len(batch), dtype=np.int64)
    for i, b in enumerate(batch):
        ids = np.asarray(b["input_ids"], dtype=np.int64)
        padded[i, : len(ids)] = ids
        lengths[i] = int(b["length"])
    return torch.from_numpy(padded), torch.from_numpy(lengths)


def stage_forward(batch_size: int = 32) -> None:
    import torch
    from datasets import load_from_disk
    from torch.utils.data import DataLoader

    ds = load_from_disk(str(TOKEN_CACHE / "cell.dataset"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    import anndata as ad

    a = ad.read_h5ad(DATA)
    ordinal = np.asarray(a.obs["ordinal_idx"].astype(int))
    marker_s = np.asarray(a.obs["marker_score"].astype(float))
    assert len(ds) == len(ordinal), (len(ds), len(ordinal))

    out: dict[str, np.ndarray] = {
        "ordinal": ordinal, "marker_score": marker_s,
    }
    for name, ckpt in CKPTS:
        path = _hf_snapshot() if name == "frozen" else ckpt
        if not Path(path).exists():
            print(f"SKIP {name}: {path} missing")
            continue
        print(f"forward {name} from {path} ...", flush=True)
        from transformers import BertForSequenceClassification

        model = BertForSequenceClassification.from_pretrained(
            str(path), torch_dtype=torch.bfloat16
        ).to("cuda")
        model.eval()
        cls_list = []
        with torch.no_grad():
            for i, (input_ids, length) in enumerate(loader):
                input_ids = input_ids.to("cuda")
                length = length.to("cuda")
                attn = torch.arange(input_ids.shape[1], device="cuda")[None, :] < length[:, None]
                out_b = model.bert(input_ids, attention_mask=attn, output_hidden_states=True)
                cls_list.append(out_b.hidden_states[-1][:, 0].float().cpu().numpy())
                if (i + 1) % 25 == 0:
                    print(f"   batch {i + 1}/{len(loader)}", flush=True)
        out[name] = np.concatenate(cls_list)
        del model
        torch.cuda.empty_cache()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB_NPZ, **out)
    print(f"saved {EMB_NPZ}: " + ", ".join(f"{k}={v.shape}" for k, v in out.items()))


def stage_metrics() -> None:
    from scipy.stats import spearmanr

    from corneal_fm.evaluation import marker_gradient_correlation

    z = np.load(EMB_NPZ)
    ordinal = z["ordinal"]
    labels = np.array([LAYER_ORDER[k] for k in ordinal])
    s = z["marker_score"]
    rows = []
    for name in [n for n, _ in CKPTS]:
        if name not in z:
            continue
        emb = z[name]
        rho_ord = marker_gradient_correlation(emb, labels, layer_order=LAYER_ORDER)
        cents = np.vstack([emb[ordinal == k].mean(axis=0) for k in range(4)])
        cc = cents - cents.mean(axis=0)
        _, _, vt = np.linalg.svd(cc, full_matrices=False)
        axis = vt[0]
        first_c = emb[ordinal == 0].mean(axis=0)
        last_c = emb[ordinal == 3].mean(axis=0)
        if (last_c - first_c) @ axis < 0:
            axis = -axis
        rho_cont = float(spearmanr(emb @ axis, s).statistic)
        rows.append({"checkpoint": name, "rho_L12_ordinal": float(rho_ord), "rho_cont_marker": rho_cont})
        print(f"{name:16s} rho_ordinal={rho_ord:.4f}  rho_cont(marker)={rho_cont:.4f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RHO_JSON, "w") as f:
        json.dump({"dataset": DATASET_NAME, "layer_order": LAYER_ORDER,
                   "markers": {"basal": BASAL_MARKERS, "differentiated": DIFF_MARKERS},
                   "results": rows}, f, indent=2)
    print(f"saved {RHO_JSON}")


def stage_sanity() -> dict:
    """Verify: (a) ordinal balance, (b) emb shape == n_cells, (c) frozen rho in [0,0.5]."""
    import anndata as ad

    checks: dict[str, bool] = {}
    a = ad.read_h5ad(DATA)
    counts = a.obs["ordinal_layer"].value_counts().to_dict()
    total = sum(counts.values())
    fracs = {c: counts.get(c, 0) / total for c in LAYER_ORDER}
    checks["a_ordinal_balanced"] = all(0.05 < fracs[c] < 0.60 for c in LAYER_ORDER)
    print("ordinal counts:", counts, "| fracs:", {k: round(v, 3) for k, v in fracs.items()})

    z = np.load(EMB_NPZ)
    n_cells = a.n_obs
    emb_ok = {k: z[k].shape[0] == n_cells for k in z if k not in ("ordinal", "marker_score")}
    checks["b_emb_shapes_match"] = all(emb_ok.values())
    print("emb shapes vs n_cells:", emb_ok, f"(n_cells={n_cells})")

    with open(RHO_JSON) as f:
        rho = json.load(f)
    frozen = next(r for r in rho["results"] if r["checkpoint"] == "frozen")
    rho_ord = frozen["rho_L12_ordinal"]
    checks["c_frozen_rho_in_range"] = 0.0 <= rho_ord <= 0.5
    print(f"frozen rho_L12_ordinal = {rho_ord:.4f} (expected weak [0, 0.5])")

    checks["all_pass"] = all(checks.values())
    print("SANITY:", "PASS" if checks["all_pass"] else "FAIL", json.dumps(checks, indent=1))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tokenize", "forward", "metrics", "sanity"], required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.stage == "tokenize":
        stage_tokenize()
    elif args.stage == "forward":
        stage_forward(args.batch_size)
    elif args.stage == "metrics":
        stage_metrics()
    else:
        stage_sanity()


if __name__ == "__main__":
    main()
