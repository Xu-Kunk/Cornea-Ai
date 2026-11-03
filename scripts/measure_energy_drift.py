#!/usr/bin/env python
"""A5 — Energy-drift diagnostic across Geneformer-V2-104M fine-tuning arms
(Paper 1 energy-model extension; precedent: ELI arXiv:2203.14952).

For every already-trained checkpoint the script measures, per hidden state,
how far the fine-tuned [CLS] representation has DRIFTED from the frozen
pretrained geometry — a third, mechanism-independent diagnostic alongside the
marker-gradient rho (order preservation) and LAA (adjacency preservation):

  1. MMD drift      : RBF MMD^2(pre_cls[l], post_cls[l]) with median-heuristic
                      bandwidth, computed on L2-normalised embeddings (both
                      sides), so the drift is scale-invariant and directly
                      comparable across layers and arms.
  2. Discriminator AUC: per-layer logistic discriminator (small EBM proxy)
                      distinguishing pre vs post [CLS] embeddings
                      (5-fold stratified CV AUC, z-scored features).

Reference ("pre") is always frozen Geneformer; each fine-tuned checkpoint is
a "post" arm. Outputs JSON into artifacts/paper1_energy_drift/ and embeds the
existing per-layer rho curves from artifacts/paper1_layerwise_rho/ for direct
rho-vs-drift cross-reference (rho is NOT recomputed here).

Two embedding sources:
  --source percell   (default) : reuse the existing per-cell FINAL-layer CLS
                      embeddings in artifacts/paper1_tier_b_eval/percell_*.npz
                      (full 14,775 external cells; zero model inference).
  --source layerwise : live inference over the external epithelial cells,
                      all 13 hidden states, mirroring
                      scripts/measure_layerwise_rho.py (frozen: L2-normalised
                      fp16 autocast; fine-tuned: raw bf16/fp32, same arm
                      protocols). --max-cells subsamples for CPU feasibility.

Existing fine-tuned arms (locked checkpoints, read-only):
  naive_s42  : save/caft_gf_lam0.0_seed42_*/model/checkpoint-7473
  caft_s42   : save/caft_gf_lam0.5_seed42_*/model/checkpoint-7473
  (+ any other trained arm discovered under save/ with model/checkpoint-*:
   ordinal_gf_*/hp_aft_gf_*/energy_anchor_gf_* — included automatically when
   present; percell source skips them with a warning since no npz exists).

Usage:
  # final-layer drift, full data, zero inference (seconds):
  /home/dq/venv-ds/bin/python3 scripts/measure_energy_drift.py --source percell
  # per-layer drift smoke on a subsample (CPU, ~20-30 min @300 cells):
  /home/dq/venv-ds/bin/python3 scripts/measure_energy_drift.py \
      --source layerwise --max-cells 300 --device cpu
  # full per-layer run (queue on GPU when free, or overnight CPU @3000 cells):
  /home/dq/venv-ds/bin/python3 scripts/measure_energy_drift.py \
      --source layerwise --max-cells 3000 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from measure_layerwise_rho import (  # noqa: E402  (read-only reuse; scripts/ on path)
    LAYER_LABELS,
    N_HIDDEN_STATES,
    PRETRAINED_DIR,
    SAVE_DIR,
    _collate_batch,
    _load_external_cells,
    collect_layerwise_cls,
)

SAVE_DIR = PROJECT_ROOT / "save"
PERCELL_DIR = PROJECT_ROOT / "artifacts" / "paper1_tier_b_eval"
RHO_DIR = PROJECT_ROOT / "artifacts" / "paper1_layerwise_rho"
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper1_energy_drift"
N_FT_CLASSES = 31
DEFAULT_SEED = 42
N_FOLDS = 5

# arm_name -> (npz stem, layerwise rho json stem, human label)
ARMS: dict[str, dict] = {
    "naive_s42": {"npz": "percell_naive_s42", "rho": "layerwise_rho_naive_s42", "label": "naive (CAFT lam 0.0, s42)"},
    "caft_s42": {"npz": "percell_caft_s42", "rho": "layerwise_rho_caft_s42", "label": "CAFT (lam 0.5, s42)"},
    "naive_s123": {"npz": "percell_naive_s123", "rho": "layerwise_rho_naive_s123", "label": "naive (CAFT lam 0.0, s123)"},
    "caft_s123": {"npz": "percell_caft_s123", "rho": "layerwise_rho_caft_s123", "label": "CAFT (lam 0.5, s123)"},
    "naive_s456": {"npz": "percell_naive_s456", "rho": "layerwise_rho_naive_s456", "label": "naive (CAFT lam 0.0, s456)"},
    "caft_s456": {"npz": "percell_caft_s456", "rho": "layerwise_rho_caft_s456", "label": "CAFT (lam 0.5, s456)"},
}
DEFAULT_ARMS = ["naive_s42", "caft_s42"]  # task-specified pair


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _median_bandwidth(A: np.ndarray, B: np.ndarray, seed: int, max_pairs: int = 2000) -> float:
    """Median pairwise distance over a deterministic subsample of A ∪ B."""
    rng = np.random.RandomState(seed)
    na, nb = A.shape[0], B.shape[0]
    ia = rng.choice(na, size=min(na, max_pairs), replace=False)
    ib = rng.choice(nb, size=min(nb, max_pairs), replace=False)
    S = np.vstack([A[ia], B[ib]])
    m = S.shape[0]
    sq = (S * S).sum(1)
    # chunked pairwise squared distances (matmul identity, bounded memory)
    meds: list[np.ndarray] = []
    chunk = 256
    for i in range(0, m, chunk):
        for j in range(i, m, chunk):
            d2 = sq[i : i + chunk, None] + sq[j : j + chunk][None, :] - 2.0 * (
                S[i : i + chunk] @ S[j : j + chunk].T
            )
            meds.append(np.sqrt(np.maximum(d2, 0.0))[np.triu_indices_from(d2)])
    all_d = np.concatenate(meds)
    sigma = float(np.median(all_d))
    return sigma if sigma > 0 else 1.0


def _kernel_sum(A: np.ndarray, B: np.ndarray, sigma: float, chunk: int = 512) -> float:
    """sum_{i,j} exp(-||a_i - b_j||^2 / (2 sigma^2)), chunked (matmul identity)."""
    sq_a = (A * A).sum(1)
    sq_b = (B * B).sum(1)
    total = 0.0
    for i in range(0, A.shape[0], chunk):
        ai = A[i : i + chunk]
        sa = sq_a[i : i + chunk]
        for j in range(0, B.shape[0], chunk):
            d2 = sa[:, None] + sq_b[j : j + chunk][None, :] - 2.0 * (ai @ B[j : j + chunk].T)
            total += float(np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma * sigma)).sum())
    return total


def rbf_mmd2(A: np.ndarray, B: np.ndarray, sigma: float) -> float:
    """Biased RBF MMD^2 between A (n,H) and B (m,H), L2-normalised inputs."""
    n, m = A.shape[0], B.shape[0]
    kxx = _kernel_sum(A, A, sigma) / (n * n)
    kyy = _kernel_sum(B, B, sigma) / (m * m)
    kxy = _kernel_sum(A, B, sigma) / (n * m)
    return float(kxx + kyy - 2.0 * kxy)


def discriminator_auc(
    pre: np.ndarray, post: np.ndarray, seed: int, n_folds: int = N_FOLDS
) -> float:
    """5-fold stratified CV AUC of a logistic discriminator on z-scored features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([pre, post])
    y = np.concatenate([np.zeros(pre.shape[0]), np.ones(post.shape[0])])
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])))
    return float(np.mean(aucs))


def drift_metrics(pre: np.ndarray, post: np.ndarray, seed: int) -> dict:
    """mmd_drift + auc for one layer; inputs (n,H)/(m,H) float32."""
    pre_n = _l2_normalize(pre)
    post_n = _l2_normalize(post)
    sigma = _median_bandwidth(pre_n, post_n, seed=seed)
    mmd = rbf_mmd2(pre_n, post_n, sigma)
    auc = discriminator_auc(pre_n, post_n, seed=seed)
    return {
        "mmd_drift": round(mmd, 6),
        "discriminator_auc": round(auc, 6),
        "bandwidth_sigma": round(sigma, 6),
        "n_pre": int(pre.shape[0]),
        "n_post": int(post.shape[0]),
    }


# ---------------------------------------------------------------------------
# Embedding sources
# ---------------------------------------------------------------------------

def load_percell(arm: str) -> np.ndarray:
    """FINAL-layer [CLS] embeddings (n, 768) from the existing percell npz."""
    npz = PERCELL_DIR / f"{ARMS[arm]['npz']}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"percell embeddings not found: {npz}")
    return np.load(str(npz))["emb_ext"].astype(np.float32)


def load_layerwise_percell(arm: str) -> np.ndarray:
    """(13, n, 768) per-layer CLS via live inference (mirror measure_layerwise_rho)."""
    if arm == "frozen_gf":
        raise ValueError("frozen_gf handled separately (no checkpoint dir)")
    lam, seed = _arm_to_lam_seed(arm)
    pattern = f"caft_gf_lam{lam}_seed{seed}_*"
    save_dirs = sorted(SAVE_DIR.glob(pattern))
    if not save_dirs:
        raise FileNotFoundError(f"no save dir for {arm} (pattern {pattern})")
    ckpt_dirs = sorted((save_dirs[-1] / "model").glob("checkpoint-*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"no checkpoints under {save_dirs[-1] / 'model'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained(
        str(ckpt_dirs[-1]), num_labels=N_FT_CLASSES, torch_dtype=torch_dtype
    ).to(device)
    layer_cls = collect_layerwise_cls(
        model, _CACHED_EXTERNAL_DS, device=device, batch_size=_CACHED_BS,
        l2_normalize=False, fp16_autocast=False,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return layer_cls


def _arm_to_lam_seed(arm: str) -> tuple[float, int]:
    """'naive_s42' -> (0.0, 42); 'caft_s42' -> (0.5, 42)."""
    kind, _, seed = arm.rpartition("_")
    if seed.startswith("s"):
        seed = seed[1:]
    lam = 0.0 if kind == "naive" else 0.5
    return lam, int(seed)


_CACHED_EXTERNAL_DS = None
_CACHED_BS = 32
_CACHED_LABELS = None


def load_external_subsampled(max_cells: int, seed: int):
    """(n, n) tokenized external epithelial cells + layer labels, deterministic subsample.

    The subsampled dataset REPLACES the module cache so every collector
    (frozen, FT checkpoints, extra arms) runs over the SAME cells — pre/post
    drift must compare matched cell sets.
    """
    global _CACHED_EXTERNAL_DS
    if _CACHED_EXTERNAL_DS is None:
        ds, labels = _load_external_cells()
        _CACHED_EXTERNAL_DS = ds
        _CACHED_LABELS = labels
    n = len(_CACHED_EXTERNAL_DS)
    if max_cells is not None and max_cells < n:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n, size=max_cells, replace=False))
        _CACHED_EXTERNAL_DS = _CACHED_EXTERNAL_DS.select(idx.tolist())
        _CACHED_LABELS = _CACHED_LABELS[idx]
    return _CACHED_EXTERNAL_DS, _CACHED_LABELS


def collect_frozen_layerwise(ds, device: str, batch_size: int) -> np.ndarray:
    """Frozen Geneformer per-layer CLS (L2-normalised, fp16 autocast on cuda)."""
    from transformers import BertModel

    model = BertModel.from_pretrained(str(PRETRAINED_DIR)).to(device)
    layer_cls = collect_layerwise_cls(
        model, ds, device=device, batch_size=batch_size,
        l2_normalize=True, fp16_autocast=True,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return layer_cls


# ---------------------------------------------------------------------------
# Rho cross-reference (read-only reuse of the locked layerwise rho JSONs)
# ---------------------------------------------------------------------------

def load_rho_curve(arm: str) -> list[float] | None:
    """Per-layer marker-gradient rho from artifacts/paper1_layerwise_rho/."""
    if arm == "frozen_gf":
        p = RHO_DIR / "layerwise_rho_frozen_gf.json"
    else:
        stem = ARMS.get(arm, {}).get("rho")
        if stem is None:
            return None  # non-CAFT arm: no locked rho curve exists yet
        p = RHO_DIR / f"{stem}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)["marker_gradient_rho"]


def discover_extra_arms() -> list[str]:
    """Any other trained arm under save/ (ordinal/hp_aft/energy_anchor) with checkpoints."""
    out: list[str] = []
    for pat in ("ordinal_gf_*", "hp_aft_gf_*", "energy_anchor_gf_*"):
        for d in sorted(SAVE_DIR.glob(pat)):
            if any((d / "model").glob("checkpoint-*")):
                name = d.name
                # collapse timestamp: name -> ordinal/hp_aft/energy_anchor + lam + seed
                key = name.replace("_", " ").split()
                out.append(name)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["percell", "layerwise"], default="percell")
    ap.add_argument("--arms", default=None,
                    help="Comma-separated post arms (default: naive_s42,caft_s42). "
                         "All 6 CAFT arms available; other trained arms are auto-discovered.")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Subsample external cells for layerwise inference (default: all).")
    ap.add_argument("--auc-max-cells", type=int, default=4000,
                    help="Per-side cell cap for the discriminator AUC (default 4000).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=["cpu", "auto"], default="cpu",
                    help="cpu (default; GPU is owned by fix-7) or auto (cuda if free).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    global _CACHED_BS
    _CACHED_BS = args.batch_size
    print(f"{_ts()}  device={device} source={args.source} seed={args.seed}")

    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(DEFAULT_ARMS)
    known = set(ARMS) | {"frozen_gf"}
    for a in arms:
        if a not in known:
            print(f"{_ts()}  ERROR: unknown arm {a!r}; known: {sorted(known)}")
            return 1
    print(f"{_ts()}  post arms: {arms}")

    # ---- collect pre (frozen) + post embeddings ---------------------------
    if args.source == "percell":
        # final layer only: frozen npz + per-arm npz (full 14,775 cells)
        pre_full = np.load(str(PERCELL_DIR / "percell_frozen_gf.npz"))["emb_ext"].astype(np.float32)
        # frozen npz emb is L2-normalised; post npz emb is raw — drift normalises
        # both sides internally, so the protocol asymmetry is handled.
        n_layers = 1
        layer_labels = ["final_hidden_state_12"]
        pre_by_layer: list[np.ndarray] = [pre_full]
        post_by_layer: dict[str, list[np.ndarray]] = {}
        for arm in arms:
            if arm == "frozen_gf":
                post_by_layer[arm] = [pre_full]
                continue
            if arm not in ARMS:
                print(f"{_ts()}  WARN {arm}: no percell npz (only caft arms have one); skipping.")
                continue
            post_by_layer[arm] = [load_percell(arm)]
    else:  # layerwise
        n_layers = N_HIDDEN_STATES
        layer_labels = list(LAYER_LABELS)
        ds, _labels = load_external_subsampled(args.max_cells, args.seed)
        n_cells = len(ds)
        print(f"{_ts()}  external cells used: {n_cells}")
        t0 = time.time()
        frozen_cls = collect_frozen_layerwise(ds, device, args.batch_size)
        pre_by_layer = [frozen_cls[i] for i in range(n_layers)]
        print(f"{_ts()}  frozen layerwise CLS collected ({time.time() - t0:.0f}s)")
        post_by_layer: dict[str, list[np.ndarray]] = {}
        for arm in arms:
            if arm == "frozen_gf":
                post_by_layer[arm] = pre_by_layer
                continue
            t0 = time.time()
            cls = load_layerwise_percell(arm) if arm in ARMS else _collect_extra_arm(arm, ds, device, args.batch_size)
            if cls is None:
                continue
            post_by_layer[arm] = [cls[i] for i in range(n_layers)]
            print(f"{_ts()}  {arm} layerwise CLS collected ({time.time() - t0:.0f}s)")

        # auto-discover other trained arms (ordinal / hp_aft / energy_anchor)
        extra = [a for a in discover_extra_arms() if a not in post_by_layer]
        for arm in extra:
            t0 = time.time()
            cls = _collect_extra_arm(arm, ds, device, args.batch_size)
            if cls is None:
                continue
            post_by_layer[arm] = [cls[i] for i in range(n_layers)]
            print(f"{_ts()}  {arm} (auto-discovered) layerwise CLS collected ({time.time() - t0:.0f}s)")

    # ---- per-layer drift metrics -----------------------------------------
    results: dict[str, dict] = {}
    for arm, posts in post_by_layer.items():
        if arm == "frozen_gf":
            continue  # reference vs itself: trivial, skip
        per_layer = []
        for l in range(n_layers):
            pre = pre_by_layer[l]
            post = posts[l]
            if post.shape[0] > args.auc_max_cells:
                rng = np.random.RandomState(args.seed)
                post = post[rng.choice(post.shape[0], args.auc_max_cells, replace=False)]
            if pre.shape[0] > args.auc_max_cells:
                rng = np.random.RandomState(args.seed)
                pre = pre[rng.choice(pre.shape[0], args.auc_max_cells, replace=False)]
            m = drift_metrics(pre, post, seed=args.seed)
            m["layer"] = layer_labels[l]
            per_layer.append(m)
        rho = load_rho_curve(arm)
        rec = {
            "arm": arm,
            "label": ARMS.get(arm, {}).get("label", arm),
            "source": args.source,
            "n_layers": n_layers,
            "layer_labels": layer_labels,
            "drift": per_layer,
            "rho_layerwise_reference": rho,
            "protocol": {
                "normalization": "L2-normalised both sides for drift metrics",
                "mmd": "RBF biased MMD^2, median-heuristic sigma",
                "auc": f"logistic discriminator, {N_FOLDS}-fold stratified CV, z-scored",
                "seed": args.seed,
            },
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        results[arm] = rec
        out_json = OUT / f"energy_drift_{arm}.json"
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"{_ts()}  saved {out_json}")

    # ---- summary ----------------------------------------------------------
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": args.source,
        "device": device,
        "n_layers": n_layers,
        "layer_labels": layer_labels,
        "arms": {k: v for k, v in results.items()},
        "note": ("drift = RBF MMD^2 + logistic-discriminator AUC of pre (frozen GF) vs "
                 "post (fine-tuned) [CLS] per hidden state; rho_layerwise_reference from "
                 "artifacts/paper1_layerwise_rho (not recomputed)."),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{_ts()}  summary -> {OUT / 'summary.json'}")
    return 0


def _collect_extra_arm(name: str, ds, device: str, batch_size: int):
    """Inference for a discovered (non-CAFT) trained arm, e.g. ordinal/hp_aft/energy_anchor."""
    save_dir = SAVE_DIR / name
    ckpt_dirs = sorted((save_dir / "model").glob("checkpoint-*"))
    if not ckpt_dirs:
        print(f"{_ts()}  WARN {name}: no checkpoints; skipping.")
        return None
    from transformers import BertForSequenceClassification

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = BertForSequenceClassification.from_pretrained(
        str(ckpt_dirs[-1]), num_labels=N_FT_CLASSES, torch_dtype=torch_dtype
    ).to(device)
    cls = collect_layerwise_cls(
        model, ds, device=device, batch_size=batch_size,
        l2_normalize=False, fp16_autocast=False,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return cls  # (13, n, 768)


if __name__ == "__main__":
    sys.exit(main())
