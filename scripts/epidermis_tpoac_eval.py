#!/usr/bin/env python
"""Epidermis cross-tissue TP-OAC validation — Wang 2020 (GSE147424).

Stages (run with --stage tokenize | forward | metrics):
  1. marker_axis : impute NaN->0, z-scored differentiation score
     s = mean(z(log1p KRT1,KRT10,IVL,FLG)) - mean(z(log1p KRT5,KRT14,TP63)),
     4 ordinal quartile bins (basal < suprabasal < spinous < differentiated),
     resolvability statistics.
  2. tokenize    : Geneformer V2 tokenizer (symbols via default gene mapping).
  3. forward     : layer-12 CLS (raw, matching run_method_arms_eval) per
     checkpoint on GPU: frozen, CE-naive (caft lam0.0), CORN-ordinal lam0.0,
     CAFT lam0.5, new-axis AA-FT lam1.0, new-axis SS lam0.5.
  4. metrics     : marker_gradient_rank_correlation(emb, ordinal) + continuous
     spearmanr(proj onto eval axis, s). TP-OAC: frozen/ordinal preserve the
     axis, CE-naive collapses.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "epidermis" / "wang2020_epidermis_wang2020.h5ad"
OUT_DIR = ROOT / "artifacts" / "paper1_epidermis_eval"
TOKEN_CACHE = OUT_DIR / "external_tokenized"
EMB_NPZ = OUT_DIR / "embeddings.npz"

BASAL_MARKERS = ["KRT5", "KRT14", "TP63"]
DIFF_MARKERS = ["KRT1", "KRT10", "IVL", "FLG"]
LAYER_ORDER = ["basal", "suprabasal", "spinous", "differentiated"]
ORDER_MAP = {c: i for i, c in enumerate(LAYER_ORDER)}

# checkpoint -> (label, path)
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


def _load_marker_adata():
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(DATA)
    x = a.X
    if sp.issparse(x):
        x = x.toarray()
    x = np.nan_to_num(x, nan=0.0)  # 12.8% NaN (dropout); impute -> 0
    return a, x


def stage_marker_axis() -> dict:
    import anndata as ad
    from scipy.stats import spearmanr

    a, x = _load_marker_adata()
    var = a.var_names
    idx = {g: i for i, g in enumerate(var)}
    log1p = np.log1p(x)

    def group_mean(genes):
        return np.mean([log1p[:, idx[g]] for g in genes if g in idx], axis=0)

    basal_m = group_mean(BASAL_MARKERS)
    diff_m = group_mean(DIFF_MARKERS)
    # Retain only marker-positive cells (cells with zero expression of ALL 7
    # epidermal markers carry no differentiation signal — analogous to the
    # cornea NON_LAYER exclusion; ~55% of the raw Wang download).
    pos = (basal_m > 0) | (diff_m > 0)

    def group_zscore(genes, mask):
        m = group_mean(genes)
        mv = m[mask]
        return (m - mv.mean()) / (mv.std() + 1e-9)

    basal_z = group_zscore(BASAL_MARKERS, pos)
    diff_z = group_zscore(DIFF_MARKERS, pos)
    s = np.full(len(pos), np.nan)
    s[pos] = diff_z[pos] - basal_z[pos]  # high = differentiated
    s_ret = s[pos]
    q = np.quantile(s_ret, [0.25, 0.5, 0.75])
    ordinal = np.full(len(pos), -1, dtype=int)
    ordinal[pos] = np.digitize(s_ret, q).clip(0, 3)  # 0..3

    stats = {
        "n_cells_raw": int(len(pos)),
        "n_cells_marker_positive": int(pos.sum()),
        "frac_marker_positive": float(pos.mean()),
        "n_genes": int(a.n_vars),
        "s_q25_q50_q75_retained": [float(x) for x in q],
        "frac_strong_basal_retained": float((s_ret < -1.0).mean()),
        "frac_strong_diff_retained": float((s_ret > 1.0).mean()),
        "ordinal_counts_retained": np.bincount(ordinal[pos], minlength=4).tolist(),
    }
    # resolvability: does the protocol-14 crude annotation track s?
    ct = np.asarray(a.obs["cell_type"].astype(str))
    ct_order = np.full(len(ct), np.nan)
    ct_order[ct == "Basal"] = 0.0
    ct_order[ct == "Spinous"] = 1.0
    ct_order[ct == "Granular"] = 2.0
    ok = pos & np.isfinite(ct_order)
    if ok.sum() >= 10:
        stats["spearman_s_vs_ct_order"] = float(spearmanr(s[ok], ct_order[ok]).statistic)
    for label, m in (("Basal", 0.0), ("Spinous", 1.0), ("Granular", 2.0)):
        sel = pos & (ct_order == m)
        if sel.sum():
            stats[f"mean_s_{label}"] = float(s[sel].mean())
    # save the processed cell-level data for the tokenizer (marker-positive only)
    obs = a.obs[pos].copy()
    obs["marker_score"] = s[pos]
    obs["ordinal_layer"] = [LAYER_ORDER[k] for k in ordinal[pos]]
    obs["ordinal_idx"] = ordinal[pos].astype(int)
    out_h5 = OUT_DIR / "wang2020_processed.h5ad"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import scipy.sparse as sp

    a2 = ad.AnnData(X=sp.csr_matrix(x[pos]), obs=obs, var=a.var)
    a2.write_h5ad(out_h5)
    stats["processed_h5ad"] = str(out_h5)
    stats["processed_cells"] = int(a2.n_obs)
    print(json.dumps(stats, indent=1))
    return stats


def stage_tokenize():
    import anndata as ad
    from geneformer import TranscriptomeTokenizer

    from corneal_fm import gf_compat

    gf_compat.install()
    TOKEN_CACHE.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(OUT_DIR / "wang2020_processed.h5ad")
    obs = a.obs.copy().reset_index(drop=True)
    obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    obs["layer"] = obs["ordinal_idx"].astype("int64")
    for col in ("cell_type", "author_cell_type", "disease", "tissue", "donor_id"):
        if col not in obs.columns:
            obs[col] = "skin" if col == "tissue" else ("healthy" if col == "disease" else "unknown")
    obs["cell_type"] = obs["ordinal_layer"].astype(str)
    obs["author_cell_type"] = obs["ordinal_layer"].astype(str)
    a.obs = obs
    tmp = Path(tempfile.mkdtemp(prefix="epidermis_tokenize_"))
    tmp_h5ad = tmp / "wang2020.h5ad"
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


def stage_forward(batch_size: int = 256):
    import torch
    from datasets import load_from_disk
    from torch.utils.data import DataLoader


    ds = load_from_disk(str(TOKEN_CACHE / "cell.dataset"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    # reorder epi labels to match ds order (ds preserves adata order; ordinal_idx aligns)
    import anndata as ad

    a = ad.read_h5ad(OUT_DIR / "wang2020_processed.h5ad")
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
        # Use the checkpoint's OWN config num_labels (caft/naive heads are
        # 31-class; the method-arm heads 4-class) — the head is unused here.
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


def stage_metrics():
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
        # continuous variant: projection onto the eval (centroid-PCA) axis vs marker score
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
    with open(OUT_DIR / "epidermis_rho_results.json", "w") as f:
        json.dump({"dataset": "Wang2020_GSE147424", "layer_order": LAYER_ORDER,
                   "markers": {"basal": BASAL_MARKERS, "differentiated": DIFF_MARKERS},
                   "results": rows}, f, indent=2)
    print(f"saved {OUT_DIR / 'epidermis_rho_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["marker_axis", "tokenize", "forward", "metrics"], required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    if args.stage == "marker_axis":
        stage_marker_axis()
    elif args.stage == "tokenize":
        stage_tokenize()
    elif args.stage == "forward":
        stage_forward(args.batch_size)
    else:
        stage_metrics()


if __name__ == "__main__":
    main()
