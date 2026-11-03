"""Paper 1 — per-cell probabilities, calibration and OOD analysis (Items 9-10).

Implements the per-cell probability collection for the 7 Paper-1 models
(6 fine-tuned CAFT checkpoints + frozen Geneformer-V2-104M) on the external
GSE153515 test set and the internal holdout, plus the downstream calibration /
OOD analysis and the epoch-2 tier-B re-evaluation needed for Supplementary
Table S1.

Modes:
  (default)  collect per-cell predictions/probabilities/embeddings/donor IDs
             and save percell_{arm}_s{seed}.npz; fail-closed sanity gate vs
             the existing results_*.json artifacts (tol 1e-6).
  --epoch2   tier-B external evaluation on the epoch-2 checkpoints (step
             4982) for Supplementary Table S1; sanity gate reruns the
             epoch-3 checkpoints (step 7473) and requires bit-exact agreement
             with the existing results JSONs.
  --postprocess-only   CPU-only: per-class reliability diagrams (Figure S1)
             and OOD analysis (Figure S2) from the saved npz files.
  --s1-table assemble docs/manuscript/paper1_supplementary_table_s1.md from
             the epoch-2/epoch-3 results JSONs.

No existing script or evaluation module is modified; all metric functions are
reused read-only from corneal_fm/evaluation.py and the two tier-B scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
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

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import load_from_disk  # noqa: E402

# Read-only reuse of the tier-B pipeline helpers (existing files untouched).
from run_tier_b_baselines import (  # noqa: E402
    _class_bucket_map,
    _frozen_embeddings,
    _geneformer_pad_id,
    harmonize_probe_outputs,
)
from run_tier_b_eval import (  # noqa: E402
    BOOT_SEED,
    CONFIG_PATH,
    EVAL_CLASSES,
    EXT_DATA,
    HARMONIZED_MAP,
    LAYER_ORDER_4,
    MANIFEST_PATH,
    N_BOOT,
    OUT_DIR,
    SAVE_DIR,
    TOKEN_DATASET,
    TRAIN_DATA,
    _load_id2label,
    map_predictions,
    run_inference,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import BertForSequenceClassification, BertModel  # noqa: E402

from corneal_fm.evaluation import (  # noqa: E402
    donor_level_macro_f1,
    evaluate_on_external_test,
    expected_calibration_error,
    layer_adjacency_accuracy,
    marker_gradient_correlation,
)

# ---------------------------------------------------------------------------
# Constants (Paper 1 locked evaluation contract)
# ---------------------------------------------------------------------------

REF_JSON_DIR = OUT_DIR  # artifacts/paper1_tier_b_eval
PERCELL_DIR = OUT_DIR
FIG_DIR = PROJECT_ROOT / "figures" / "paper1"
S1_PATH = PROJECT_ROOT / "docs" / "manuscript" / "paper1_supplementary_table_s1.md"

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
TOKENIZED_TRAIN = (
    SAVE_DIR
    / "caft_gf_lam0.5_seed42_20260801-200953"
    / "corneal_layer_subset_tk"
    / "cell.dataset"
)

LAMBDAS = [0.0, 0.5]
SEEDS = [42, 123, 456]
ARM_NAMES = {0.0: "naive", 0.5: "caft"}
BATCH_SIZE = 32
SANITY_TOL = 1e-6
EPOCH2_STEP = 4982
EPOCH3_STEP = 7473
PROBE_MAX_ITER = 5000
PROBE_SEED = 42

FROZEN_REF_LABEL = "frozen_gf"
FROZEN_SAVE_DIR_NAME = "caft_gf_lam0.5_seed42_20260801-200953"  # frozen ref run


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def _panel_tag(ax, label: str, size: float = 8.0,
             x: float = -0.08) -> None:
    """Axis-outside top-left panel label (ReAct v8fix / Cell Press)."""
    ax.text(x, 1.05, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def _parse_runs(runs_arg: str) -> list[tuple[float, int]]:
    runs: list[tuple[float, int]] = []
    for token in runs_arg.split(","):
        lam_s, seed_s = token.split(":")
        runs.append((float(lam_s), int(seed_s)))
    return runs


def _discover_save_dir(lam: float, seed: int) -> Path:
    matches = sorted(SAVE_DIR.glob(f"caft_gf_lam{lam}_seed{seed}_*"))
    if not matches:
        raise FileNotFoundError(f"no save dir for lam={lam} seed={seed}")
    return matches[-1]


def _checkpoint_path(save_dir: Path, step: int | None = None) -> Path:
    if step is None:
        ckpts = sorted((save_dir / "model").glob("checkpoint-*"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoints in {save_dir / 'model'}")
        return ckpts[-1]
    ckpt = save_dir / "model" / f"checkpoint-{step}"
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint {ckpt} not found")
    return ckpt


def _load_external_cells() -> tuple[object, ad.AnnData]:
    """Tokenized external cells (epithelial subset) + matching h5ad subset."""
    dataset = load_from_disk(str(TOKEN_DATASET))
    adata = ad.read_h5ad(str(EXT_DATA))
    labels = adata.obs["cell_type_harmonized"].astype(str)
    mask = np.asarray(labels != "NON_LAYER")
    dataset = dataset.select(np.where(mask)[0])
    adata = adata[mask].copy()
    return dataset, adata


def _load_internal_cells(save_dir: Path) -> tuple[object, np.ndarray, np.ndarray]:
    """Per-run prepared test holdout, filtered to the 4 epithelial classes.

    Returns (dataset_epi, y_true, donor_ids). Layer -1 (NON_LAYER) is dropped;
    layers 0..4 map through HARMONIZED_MAP to LSC/TAC/basal/differentiated.
    """
    ds = load_from_disk(str(save_dir / "prepared" / "caft_labeled_test.dataset"))
    layers = np.asarray(ds["layer"])
    keep = np.where(np.isin(layers, [0, 1, 2, 3, 4]))[0]
    ds = ds.select(keep)
    y_true = np.array([HARMONIZED_MAP[int(lab)] for lab in np.asarray(ds["layer"])], dtype=str)
    if "donor_id" in ds.column_names:
        donors = np.asarray(ds["donor_id"], dtype=str)
    else:
        donors = np.full(len(ds), "internal_split", dtype=str)
    return ds, y_true, donors


def _load_ft_model(ckpt_path: Path, id2label: dict, device: torch.device) -> BertForSequenceClassification:
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return (
        BertForSequenceClassification.from_pretrained(
            str(ckpt_path), num_labels=len(id2label), torch_dtype=torch_dtype
        )
        .to(device)
        .eval()
    )


def _recompute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    emb: np.ndarray,
    donors: np.ndarray,
) -> dict:
    """Recompute macro-F1 / LAA / rho / ECE exactly as evaluate_on_external_test."""
    mf1 = donor_level_macro_f1(
        y_true, y_pred, donors, EVAL_CLASSES, n_boot=N_BOOT, seed=BOOT_SEED
    )
    laa = layer_adjacency_accuracy(y_true, emb, LAYER_ORDER_4)
    rho = marker_gradient_correlation(emb, y_true, LAYER_ORDER_4)
    idx_map = {c: i for i, c in enumerate(EVAL_CLASSES)}
    y_idx = np.array([idx_map[lab] for lab in y_true], dtype=int)
    ece = expected_calibration_error(y_idx, probs)
    return {
        "macro_f1": mf1["macro_f1"],
        "ci_lower": mf1["ci_lower"],
        "ci_upper": mf1["ci_upper"],
        "laa": float(laa),
        "rho": float(rho),
        "ece": float(ece),
    }


def _ref_metrics_from_json(results: dict) -> dict:
    return {
        "macro_f1": results["primary_endpoint"]["macro_f1"],
        "ci_lower": results["primary_endpoint"]["ci_lower"],
        "ci_upper": results["primary_endpoint"]["ci_upper"],
        "laa": results["secondary_endpoints"]["layer_adjacency_accuracy"],
        "rho": results["secondary_endpoints"]["marker_gradient_rank_correlation"],
        "ece": results["secondary_endpoints"]["expected_calibration_error"],
    }


def _sanity_gate(computed: dict, ref: dict, label: str) -> bool:
    fields = ["macro_f1", "ci_lower", "ci_upper", "laa", "rho", "ece"]
    ok = True
    for f in fields:
        diff = abs(float(computed[f]) - float(ref[f]))
        passed = diff <= SANITY_TOL
        ok = ok and passed
        print(
            f"    {f:>9}: computed={computed[f]!r} ref={ref[f]!r} "
            f"diff={diff:.3e} {'PASS' if passed else 'FAIL'}"
        )
    print(f"    SANITY {label}: {'PASS' if ok else 'FAIL'} (tol {SANITY_TOL})")
    return ok


def _save_percell_npz(
    out_path: Path,
    *,
    y_true_ext: np.ndarray,
    y_pred_ext: np.ndarray,
    probs_ext: np.ndarray,
    emb_ext: np.ndarray,
    donor_ext: np.ndarray,
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    probs_int: np.ndarray,
    emb_int: np.ndarray,
    donor_int: np.ndarray,
) -> None:
    np.savez_compressed(
        out_path,
        y_true_ext=_str(y_true_ext),
        y_pred_ext=_str(y_pred_ext),
        probs_ext=np.asarray(probs_ext, dtype=np.float64),
        emb_ext=np.asarray(emb_ext, dtype=np.float32),
        donor_ext=_str(donor_ext),
        y_true_int=_str(y_true_int),
        y_pred_int=_str(y_pred_int),
        probs_int=np.asarray(probs_int, dtype=np.float64),
        emb_int=np.asarray(emb_int, dtype=np.float32),
        donor_int=_str(donor_int),
    )


def _str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype="<U64")


def _save_meta(
    out_path: Path,
    *,
    model: str,
    arm: str,
    lam: float | None,
    seed: int | None,
    checkpoint: str,
    n_ext: int,
    n_int: int,
    gate_ext: dict | None,
    gate_int: dict | None,
) -> None:
    meta = {
        "model": model,
        "arm": arm,
        "lambda": lam,
        "seed": seed,
        "checkpoint_path": checkpoint,
        "n_cells_external": n_ext,
        "n_cells_internal": n_int,
        "sanity_gate_external": gate_ext,
        "sanity_gate_internal": gate_int,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.with_suffix(".npz.meta.json").write_text(json.dumps(meta, indent=2))


def _percell_name(lam: float, seed: int) -> str:
    return f"percell_{ARM_NAMES[lam]}_s{seed}.npz"


# ---------------------------------------------------------------------------
# Mode 1: per-cell collection (external + internal) with sanity gate
# ---------------------------------------------------------------------------


def _collect_fine_tuned(lam: float, seed: int, device: torch.device, batch_size: int) -> bool:
    save_dir = _discover_save_dir(lam, seed)
    ckpt_path = _checkpoint_path(save_dir)
    id2label = _load_id2label(save_dir, ckpt_path)
    model = _load_ft_model(ckpt_path, id2label, device)

    ext_ds, ext_adata = _load_external_cells()
    emb_ext, logits_ext = run_inference(model, ext_ds, batch_size=batch_size, device=device)
    pred_ext, probs_ext = map_predictions(logits_ext, id2label)
    y_true_ext = ext_adata.obs["cell_type_harmonized"].astype(str).to_numpy()
    donor_ext = ext_adata.obs["donor_id"].astype(str).to_numpy()

    int_ds, y_true_int, donor_int = _load_internal_cells(save_dir)
    emb_int, logits_int = run_inference(model, int_ds, batch_size=batch_size, device=device)
    pred_int, probs_int = map_predictions(logits_int, id2label)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Sanity gate on the external split vs the existing results JSON.
    ref_path = REF_JSON_DIR / f"results_{lam}_{seed}.json"
    ref = _ref_metrics_from_json(json.loads(ref_path.read_text()))
    computed = _recompute_metrics(y_true_ext, pred_ext, probs_ext, emb_ext, donor_ext)
    gate_ok = _sanity_gate(computed, ref, f"lam={lam} seed={seed} (external)")
    if not gate_ok:
        return False

    out_path = PERCELL_DIR / _percell_name(lam, seed)
    _save_percell_npz(
        out_path,
        y_true_ext=y_true_ext,
        y_pred_ext=pred_ext,
        probs_ext=probs_ext,
        emb_ext=emb_ext,
        donor_ext=donor_ext,
        y_true_int=y_true_int,
        y_pred_int=pred_int,
        probs_int=probs_int,
        emb_int=emb_int,
        donor_int=donor_int,
    )
    _save_meta(
        out_path,
        model="Geneformer-V2-104M (fine-tuned)",
        arm=ARM_NAMES[lam],
        lam=lam,
        seed=seed,
        checkpoint=str(ckpt_path),
        n_ext=int(len(y_true_ext)),
        n_int=int(len(y_true_int)),
        gate_ext=computed,
        gate_int=None,
    )
    print(
        f"{_ts()} saved {out_path.name} "
        f"(ext={len(y_true_ext)} int={len(y_true_int)} gate={'PASS' if gate_ok else 'FAIL'})"
    )
    return True


def _frozen_predictions(
    model: BertModel,
    train_ds: object,
    test_ds: object,
    device: torch.device,
    batch_size: int,
    *,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2-normalized embeddings + logistic probe -> (emb, predictions, probs)."""
    pad = _geneformer_pad_id()
    train_emb = _frozen_embeddings(model, train_ds, batch_size=batch_size, pad_token_id=pad)
    test_emb = _frozen_embeddings(model, test_ds, batch_size=batch_size, pad_token_id=pad)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train_emb)
    x_te = scaler.transform(test_emb)
    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED, solver="lbfgs"
    ).fit(x_tr, y_train)
    proba_31 = probe.predict_proba(x_te)
    pred_31 = probe.predict(x_te)
    class_bucket = _class_bucket_map(list(probe.classes_))
    predictions, probabilities = harmonize_probe_outputs(
        proba_31, pred_31, list(probe.classes_), class_bucket
    )
    return test_emb, predictions, probabilities


def _collect_frozen(device: torch.device, batch_size: int) -> bool:
    """Frozen arm: external (probe on full corpus, bit-exact w/ baselines) +
    internal (leak-free probe on lam0.5_s42 prepared train split)."""
    model = BertModel.from_pretrained(str(PRETRAINED_DIR)).to(device).eval()

    # --- external: reproduce results_frozen_gf.json probe exactly -----------
    train_ds = load_from_disk(str(TOKENIZED_TRAIN))
    ext_ds, ext_adata = _load_external_cells()
    y_train_ext = np.asarray(train_ds["author_cell_type"], dtype=str)
    emb_ext, pred_ext, probs_ext = _frozen_predictions(
        model, train_ds, ext_ds, device, batch_size, y_train=y_train_ext
    )
    y_true_ext = ext_adata.obs["cell_type_harmonized"].astype(str).to_numpy()
    donor_ext = ext_adata.obs["donor_id"].astype(str).to_numpy()

    ref_path = REF_JSON_DIR / "results_frozen_gf.json"
    ref = _ref_metrics_from_json(json.loads(ref_path.read_text()))
    computed = _recompute_metrics(y_true_ext, pred_ext, probs_ext, emb_ext, donor_ext)
    gate_ext_ok = _sanity_gate(computed, ref, "frozen_gf (external)")
    if not gate_ext_ok:
        return False

    # --- internal: leak-free probe on the prepared train split --------------
    save_dir = SAVE_DIR / FROZEN_SAVE_DIR_NAME
    prep = save_dir / "prepared"
    with open(prep / "caft_id_class_dict.pkl", "rb") as fh:
        id2label = pickle.load(fh)
    int_train_ds = load_from_disk(str(prep / "caft_labeled_train.dataset"))
    y_train_int = np.array([id2label[int(lab)] for lab in np.asarray(int_train_ds["label"])], dtype=str)
    int_test_ds, y_true_int, donor_int = _load_internal_cells(save_dir)
    emb_int, pred_int, probs_int = _frozen_predictions(
        model, int_train_ds, int_test_ds, device, batch_size, y_train=y_train_int
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_path = PERCELL_DIR / "percell_frozen_gf.npz"
    _save_percell_npz(
        out_path,
        y_true_ext=y_true_ext,
        y_pred_ext=pred_ext,
        probs_ext=probs_ext,
        emb_ext=emb_ext,
        donor_ext=donor_ext,
        y_true_int=y_true_int,
        y_pred_int=pred_int,
        probs_int=probs_int,
        emb_int=emb_int,
        donor_int=donor_int,
    )
    _save_meta(
        out_path,
        model="Geneformer-V2-104M (frozen)",
        arm="frozen",
        lam=None,
        seed=None,
        checkpoint=str(PRETRAINED_DIR),
        n_ext=int(len(y_true_ext)),
        n_int=int(len(y_true_int)),
        gate_ext=computed,
        gate_int=None,
    )
    print(
        f"{_ts()} saved {out_path.name} "
        f"(ext={len(y_true_ext)} int={len(y_true_int)} gate={'PASS' if gate_ext_ok else 'FAIL'})"
    )
    return True


# ---------------------------------------------------------------------------
# Mode 2: epoch-2 tier-B external evaluation (Supplementary Table S1)
# ---------------------------------------------------------------------------


def _tierb_external_eval(
    ckpt_path: Path,
    save_dir: Path,
    device: torch.device,
    batch_size: int,
    *,
    dataset: object,
    external_adata: ad.AnnData,
    train_adata: object,
    config: dict,
    split_manifest: dict,
    torch_dtype: torch.dtype,
) -> dict:
    id2label = _load_id2label(save_dir, ckpt_path)
    model = (
        BertForSequenceClassification.from_pretrained(
            str(ckpt_path), num_labels=len(id2label), torch_dtype=torch_dtype
        )
        .to(device)
        .eval()
    )
    embeddings, logits_31 = run_inference(model, dataset, batch_size=batch_size, device=device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    predictions, probabilities = map_predictions(logits_31, id2label)
    return evaluate_on_external_test(
        train_adata=train_adata,
        external_adata=external_adata,
        embeddings=embeddings,
        predictions=predictions,
        probabilities=probabilities,
        config=config,
        split_manifest=split_manifest,
        classes=EVAL_CLASSES,
        layer_order=LAYER_ORDER_4,
        n_boot=N_BOOT,
        seed=BOOT_SEED,
    )


def run_epoch2(device: torch.device, batch_size: int, runs: list[tuple[float, int]]) -> int:
    """Tier-B external eval on checkpoint-4982; sanity gate reruns 7473 first."""
    dataset, external_adata = _load_external_cells()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    split_manifest = json.loads(MANIFEST_PATH.read_text())
    train_adata = ad.read_h5ad(str(TRAIN_DATA), backed="r")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    all_ok = True

    for lam, seed in runs:
        save_dir = _discover_save_dir(lam, seed)

        # Sanity gate: rerun epoch-3 checkpoint, must match existing JSON.
        ckpt3 = _checkpoint_path(save_dir, EPOCH3_STEP)
        print(f"{_ts()} sanity rerun lam={lam} seed={seed} ckpt={ckpt3.name}")
        results3 = _tierb_external_eval(
            ckpt3,
            save_dir,
            device,
            batch_size,
            dataset=dataset,
            external_adata=external_adata,
            train_adata=train_adata,
            config=config,
            split_manifest=split_manifest,
            torch_dtype=torch_dtype,
        )
        ref = _ref_metrics_from_json(json.loads((REF_JSON_DIR / f"results_{lam}_{seed}.json").read_text()))
        computed = _ref_metrics_from_json(results3)
        gate_ok = _sanity_gate(computed, ref, f"lam={lam} seed={seed} epoch-3 rerun")
        all_ok = all_ok and gate_ok
        if not gate_ok:
            continue

        # Epoch-2 checkpoint (step 4982) -> new results JSON.
        ckpt2 = _checkpoint_path(save_dir, EPOCH2_STEP)
        print(f"{_ts()} eval lam={lam} seed={seed} ckpt={ckpt2.name}")
        results2 = _tierb_external_eval(
            ckpt2,
            save_dir,
            device,
            batch_size,
            dataset=dataset,
            external_adata=external_adata,
            train_adata=train_adata,
            config=config,
            split_manifest=split_manifest,
            torch_dtype=torch_dtype,
        )
        results2["provenance"]["epoch"] = 2
        results2["provenance"]["checkpoint"] = ckpt2.name
        out = REF_JSON_DIR / f"results_epoch2_{lam}_{seed}.json"
        out.write_text(json.dumps(results2, indent=2))
        print(f"{_ts()} saved {out.name}")

    if not all_ok:
        print("SANITY GATE FAILED for one or more epoch-3 reruns — epoch-2 results incomplete.")
        return 2
    print("SANITY GATE PASSED for all epoch-3 reruns; epoch-2 results saved.")
    return 0


# ---------------------------------------------------------------------------
# Mode 3: postprocessing (CPU) — Figures S1 + S2 + CSVs
# ---------------------------------------------------------------------------


def _load_percell_all() -> dict[str, dict]:
    percell: dict[str, dict] = {}
    percell["frozen_gf"] = _load_percell_file(PERCELL_DIR / "percell_frozen_gf.npz")
    for lam in LAMBDAS:
        for seed in SEEDS:
            name = _percell_name(lam, seed)
            percell[f"{ARM_NAMES[lam]}_s{seed}"] = _load_percell_file(PERCELL_DIR / name)
    return percell


def _load_percell_file(path: Path) -> dict:
    data = np.load(path)
    return {
        "y_true_ext": data["y_true_ext"],
        "y_pred_ext": data["y_pred_ext"],
        "probs_ext": data["probs_ext"],
        "emb_ext": data["emb_ext"],
        "donor_ext": data["donor_ext"],
        "y_true_int": data["y_true_int"],
        "y_pred_int": data["y_pred_int"],
        "probs_int": data["probs_int"],
        "emb_int": data["emb_int"],
        "donor_int": data["donor_int"],
    }


def _reliability_bins(y_bin: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per-bin reliability rows over half-open ``[lo, hi)`` bins (last closed).

    Matches the binning convention fixed in ``corneal_fm/evaluation.py`` on
    2026-08-05: ``np.digitize(p, edges, right=False)`` assigns every
    probability to exactly one bin, with ``p == 1.0`` falling in the final
    (right-closed) bin. The previous implementation used ``p > lo`` for bin 0
    without an upper bound, which silently captured every cell with
    probability > 0 in bin 0 and distorted the Figure S1 reliability diagrams
    (the same defect as the old ECE binning).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "mean_pred": float(p[mask].mean()) if n else float("nan"),
                "frac_positive": float(y_bin[mask].mean()) if n else float("nan"),
            }
        )
    return rows


def run_postprocess() -> int:
    """CPU: per-class reliability (Fig S1) + OOD analysis (Fig S2)."""
    import matplotlib  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415, E402

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from generate_paper1_results_figures import (  # noqa: PLC0415, E402
        figure_metadata,
        panel_label,
        setup_style,
    )

    setup_style()
    # Explicit TrueType embedding (Type 42) — Cell Press requirement; the
    # imported setup_style already sets it, kept here for self-containment.
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    oi_blue = "#0072B2"
    oi_orange = "#E69F00"
    oi_gray = "#999999"
    # v8fix arm-color lock: naive=blue, CAFT=orange (was vermillion)
    arm_colors = {0.0: oi_blue, 0.5: oi_orange, "frozen": oi_gray}
    arm_labels = {0.0: "Fine-tune λ=0", 0.5: "CAFT λ=0.5", "frozen": "Frozen GF"}

    percell = _load_percell_all()

    # ------------------------------------------------------------------ S1 --
    classes = EVAL_CLASSES
    n_bins = 10
    s1_rows: list[dict] = []

    # Drop classes with no positive cells in the per-cell data (e.g. TAC can be
    # absent from the internal holdout).  If only three classes remain, compress
    # from a 2x2 grid to a single row so the empty panel does not waste space.
    present_classes = []
    for cls in classes:
        cls_idx = classes.index(cls)
        for pc in percell.values():
            if (pc["y_true_ext"] == cls).any():
                present_classes.append(cls)
                break

    if len(present_classes) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.57))
        axes = axes.ravel()
        layout_note = None
    else:
        n_cols = max(len(present_classes), 1)
        fig, axes = plt.subplots(1, n_cols, figsize=(6.5, 2.45))
        axes = np.atleast_1d(axes)
        missing = [c for c in classes if c not in present_classes]
        layout_note = f"Classes with no true cells omitted: {', '.join(missing)}." if missing else None

    for ax, cls in zip(axes, present_classes):
        cls_idx = classes.index(cls)
        # per-family per-seed curves, keyed by bin index for envelope/mean
        fam_curves: dict[str, list[tuple[list[float], list[float], list[int]]]] = {
            "naive": [], "caft": []}
        frozen_line: tuple[list[float], list[float]] | None = None
        for name, pc in percell.items():
            y_bin = (pc["y_true_ext"] == cls).astype(int)
            p = np.asarray(pc["probs_ext"][:, cls_idx], dtype=float)
            rows = _reliability_bins(y_bin, p, n_bins)
            for r in rows:
                s1_rows.append({"model": name, "class": cls, **r})
            xs = [r["mean_pred"] for r in rows if r["n"] > 0]
            ys = [r["frac_positive"] for r in rows if r["n"] > 0]
            bins = [r["bin"] for r in rows if r["n"] > 0]
            if name == "frozen_gf":
                frozen_line = (xs, ys)
            else:
                fam = "naive" if name.startswith("naive") else "caft"
                lam = 0.0 if fam == "naive" else 0.5
                fam_curves[fam].append((xs, ys, bins))
                # light per-seed lines behind the family mean
                ax.plot(xs, ys, color=arm_colors[lam], lw=0.8, alpha=0.35,
                        zorder=2)
        if frozen_line is not None:
            ax.plot(*frozen_line, color=oi_gray, lw=1.4, zorder=3)
        for fam, color in (("naive", arm_colors[0.0]), ("caft", arm_colors[0.5])):
            if not fam_curves[fam]:
                continue
            agg: dict[int, list[tuple[float, float]]] = {}
            for xs, ys, bins in fam_curves[fam]:
                for x, y, b in zip(xs, ys, bins):
                    agg.setdefault(int(b), []).append((x, y))
            xs_m, ys_m, lo_l, hi_l = [], [], [], []
            for b in sorted(agg):
                pts = agg[b]
                xs_m.append(float(np.mean([p[0] for p in pts])))
                ys_v = [p[1] for p in pts]
                ys_m.append(float(np.mean(ys_v)))
                if len(pts) >= 2:
                    lo_l.append(float(min(ys_v)))
                    hi_l.append(float(max(ys_v)))
                else:
                    lo_l.append(float("nan"))
                    hi_l.append(float("nan"))
            xs_m = np.asarray(xs_m)
            ys_m = np.asarray(ys_m)
            lo_ok = ~np.isnan(lo_l)
            if lo_ok.any():
                ax.fill_between(xs_m[lo_ok], np.asarray(lo_l)[lo_ok],
                                np.asarray(hi_l)[lo_ok], color=color,
                                alpha=0.12, lw=0, zorder=1)
            # family mean as the solid line (seeds 42/123/456 merged)
            ax.plot(xs_m, ys_m, color=color, lw=1.6, zorder=3)
        ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed fraction")
        ax.set_title(cls)
    # Single shared legend below the panels.
    leg_handles = [
        plt.Line2D([0], [0], color=oi_gray, lw=1.6, label="Frozen GF"),
        plt.Line2D([0], [0], color=arm_colors[0.0], lw=1.6,
                   label="naive \u03bb=0 (seeds 42/123/456)"),
        plt.Line2D([0], [0], color=arm_colors[0.5], lw=1.6,
                   label="CAFT \u03bb=0.5 (seeds 42/123/456)"),
    ]
    for ax, letter in zip(axes, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        _panel_tag(ax, letter)
    fig.legend(handles=leg_handles, loc="lower center", ncol=3, fontsize=6.0,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    n_note = "n = 14,775 epithelial cells (GSE153515 external test)"
    if layout_note:
        n_note = layout_note + "  " + n_note
    fig.text(0.5, 0.005, n_note, ha="center", va="bottom",
             fontsize=6, color="#555555")
    # v8fix: extra bottom margin + tight bounding box so the shared legend is
    # not clipped by the figure edge.
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    out_pdf = FIG_DIR / "figureS1_calibration.pdf"
    out_png = FIG_DIR / "figureS1_calibration.png"
    fig.savefig(out_pdf, dpi=300, metadata=figure_metadata(
        "Figure S1. Per-class reliability diagrams (external test, 10 bins)."),
        bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(s1_rows).to_csv(FIG_DIR / "figureS1_calibration.csv", index=False)
    print(f"{_ts()} Figure S1 + figureS1_calibration.csv written")

    # ------------------------------------------------------------------ S2 --
    # (a) max-probability density: external vs internal, faceted by arm.
    arms = [("frozen", ["frozen_gf"]), (0.0, ["naive_s42", "naive_s123", "naive_s456"]), (0.5, ["caft_s42", "caft_s123", "caft_s456"])]
    # Use GridSpec so the lower row can span the full width with only two
    # panels, avoiding the empty 1/3 dead space of a 2x3 grid.
    fig2 = plt.figure(figsize=(6.5, 4.3))
    gs = fig2.add_gridspec(
        2, 3, left=0.08, right=0.94, top=0.93, bottom=0.10,
        wspace=0.35, hspace=0.45,
    )
    ax_a = fig2.add_subplot(gs[0, 0])
    ax_b = fig2.add_subplot(gs[0, 1])
    ax_c = fig2.add_subplot(gs[0, 2])
    ax_d = fig2.add_subplot(gs[1, 0:2])
    ax_e = fig2.add_subplot(gs[1, 2])

    s2_rows: list[dict] = []
    for col, (arm, names) in enumerate(arms):
        ax = (ax_a, ax_b, ax_c)[col]
        for split_key, split_label in (("ext", "External"), ("int", "Internal")):
            maxp = np.concatenate([np.max(percell[n][f"probs_{split_key}"], axis=1) for n in names])
            for n in names:
                per_n = np.max(percell[n][f"probs_{split_key}"], axis=1)
                s2_rows.append(
                    {"model": n, "split": split_label, "maxprob_mean": float(per_n.mean()), "maxprob_q25": float(np.quantile(per_n, 0.25)), "maxprob_median": float(np.median(per_n)), "maxprob_q75": float(np.quantile(per_n, 0.75))}
                )
            sns.kdeplot(maxp, ax=ax, label=split_label, clip=(0, 1), linewidth=1.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Max predicted probability")
        ax.set_ylabel("Density")
        ax.set_title(arm_labels[arm])
        # v8fix: per-panel legend for internal vs external density curves.
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    panel_label(fig2, ax_a, "A")
    panel_label(fig2, ax_b, "B")
    panel_label(fig2, ax_c, "C")

    # (b) per-donor ECE + per-donor mean confidence (external, 2 donors).
    donors = np.unique(percell["frozen_gf"]["donor_ext"])
    model_names = ["frozen_gf"] + [f"{ARM_NAMES[lam]}_s{seed}" for lam in LAMBDAS for seed in SEEDS]
    # v8fix: locked donor palette (donor_1 = Okabe-Ito blue, donor_2 = orange)
    # and underscore-free display names on the shared axis.
    donor_colors = {"donor_1": oi_blue, "donor_2": oi_orange}


    def _display_name(name: str) -> str:
        if name == "frozen_gf":
            return "Frozen GF"
        arm, seed = name.rsplit("_s", 1)
        return f"{'naive' if arm == 'naive' else 'CAFT'} s{seed}"


    ece_by_donor: dict[str, list[float]] = {d: [] for d in donors}
    conf_by_donor: dict[str, list[float]] = {d: [] for d in donors}
    idx_map = {c: i for i, c in enumerate(classes)}
    for name in model_names:
        pc = percell[name]
        for d in donors:
            mask = pc["donor_ext"] == d
            y_idx = np.array([idx_map[lab] for lab in pc["y_true_ext"][mask]], dtype=int)
            ece = expected_calibration_error(y_idx, np.asarray(pc["probs_ext"][mask]))
            ece_by_donor[d].append(float(ece))
            conf_by_donor[d].append(float(np.max(pc["probs_ext"][mask], axis=1).mean()))
    x = np.arange(len(model_names))
    width = 0.38
    for d, off in zip(donors, (-width / 2, width / 2)):
        ax_d.bar(x + off, ece_by_donor[d], width, label=str(d),
                 color=donor_colors.get(str(d), oi_gray))
        ax_e.bar(x + off, conf_by_donor[d], width, label=str(d),
                 color=donor_colors.get(str(d), oi_gray))
    for ax, title, ylabel in ((ax_d, "Per-donor ECE (external)", "ECE"),
                                (ax_e, "Per-donor mean confidence (external)", "Mean confidence")):
        ax.set_xticks(x)
        ax.set_xticklabels([_display_name(n) for n in model_names],
                           rotation=45, ha="right", fontsize=6.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    # Shared donor legend, placed outside the right of panel D so it never
    # overlaps the bars in E.
    handles = [
        plt.Line2D([0], [0], color=donor_colors[d], marker="s", ls="none",
                     markersize=7, label=f"Donor {d[-1]}")
        for d in donors
    ]
    ax_d.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=False,
                bbox_to_anchor=(0.0, 1.04))
    _panel_tag(ax_d, "D")
    _panel_tag(ax_e, "E", x=-0.24)
    # v8fix: extra bottom margin for the -line model labels (45° rotation).
    fig2.text(0.5, 0.005,
              "n = 14,775 epithelial cells (GSE153515 external test)",
              ha="center", va="bottom", fontsize=6, color="#555555")
    fig2.tight_layout(rect=(0, 0.14, 1, 0.93))
    out_pdf2 = FIG_DIR / "figureS2_ood.pdf"
    out_png2 = FIG_DIR / "figureS2_ood.png"
    meta2 = figure_metadata(
        "Figure S2. OOD analysis: max-probability density and per-donor calibration."
    )
    fig2.savefig(out_pdf2, dpi=300, metadata=meta2, bbox_inches="tight",
                 facecolor="white")
    fig2.savefig(out_png2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    per_donor_rows = []
    for name, (d) in [(n, d) for n in model_names for d in donors]:
        per_donor_rows.append(
            {
                "model": name,
                "donor": str(d),
                "ece": ece_by_donor[d][model_names.index(name)],
                "mean_confidence": conf_by_donor[d][model_names.index(name)],
            }
        )
    pd.DataFrame(per_donor_rows).to_csv(FIG_DIR / "figureS2_ood_perdonor.csv", index=False)
    pd.DataFrame(s2_rows).to_csv(FIG_DIR / "figureS2_ood_maxprob.csv", index=False)
    print(f"{_ts()} Figure S2 + figureS2_ood_*.csv written")
    return 0


# ---------------------------------------------------------------------------
# Mode 4: Supplementary Table S1 (CPU)
# ---------------------------------------------------------------------------


def _per_epoch_train_loss(save_dir: Path) -> dict[int, float]:
    """Mean train loss per epoch from trainer_state.json log_history (ckpt-7473)."""
    ts = json.loads((save_dir / "model" / f"checkpoint-{EPOCH3_STEP}" / "trainer_state.json").read_text())
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for entry in ts.get("log_history", []):
        if "loss" not in entry or "epoch" not in entry:
            continue
        epoch = int(entry["epoch"])  # floor: 0->epoch1, 1->epoch2, 2->epoch3
        sums[epoch] = sums.get(epoch, 0.0) + float(entry["loss"])
        counts[epoch] = counts.get(epoch, 0) + 1
    return {epoch + 1: sums[epoch] / counts[epoch] for epoch in sorted(sums)}


def run_s1_table() -> int:
    """Assemble docs/manuscript/paper1_supplementary_table_s1.md."""
    rows: list[dict] = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            save_dir = _discover_save_dir(lam, seed)
            losses = _per_epoch_train_loss(save_dir)
            for epoch, ckpt_name in ((2, f"checkpoint-{EPOCH2_STEP}"), (3, f"checkpoint-{EPOCH3_STEP}")):
                if epoch == 3:
                    res = json.loads((REF_JSON_DIR / f"results_{lam}_{seed}.json").read_text())
                else:
                    res = json.loads((REF_JSON_DIR / f"results_epoch2_{lam}_{seed}.json").read_text())
                rows.append(
                    {
                        "arm": ARM_NAMES[lam],
                        "lambda": lam,
                        "seed": seed,
                        "epoch": epoch,
                        "checkpoint": ckpt_name,
                        "macro_f1": res["primary_endpoint"]["macro_f1"],
                        "ci_lower": res["primary_endpoint"]["ci_lower"],
                        "ci_upper": res["primary_endpoint"]["ci_upper"],
                        "LAA": res["secondary_endpoints"]["layer_adjacency_accuracy"],
                        "rho": res["secondary_endpoints"]["marker_gradient_rank_correlation"],
                        "ECE": res["secondary_endpoints"]["expected_calibration_error"],
                        "mean_train_loss": round(losses.get(epoch, float("nan")), 6),
                    }
                )

    for label, fname in (("hvg_pca", "results_hvg_pca.json"), ("frozen_gf", "results_frozen_gf.json")):
        res = json.loads((REF_JSON_DIR / fname).read_text())
        rows.append(
            {
                "arm": label,
                "lambda": None,
                "seed": None,
                "epoch": None,
                "checkpoint": "n/a",
                "macro_f1": res["primary_endpoint"]["macro_f1"],
                "ci_lower": res["primary_endpoint"]["ci_lower"],
                "ci_upper": res["primary_endpoint"]["ci_upper"],
                "LAA": res["secondary_endpoints"]["layer_adjacency_accuracy"],
                "rho": res["secondary_endpoints"]["marker_gradient_rank_correlation"],
                "ECE": res["secondary_endpoints"]["expected_calibration_error"],
                "mean_train_loss": float("nan"),
            }
        )

    lines = [
        "# Supplementary Table S1. Per-seed external-test metrics by training epoch",
        "",
        "**Models.** Six fine-tuned Geneformer-V2-104M checkpoints (naive λ=0.0 and CAFT λ=0.5; seeds 42/123/456) plus two seed-independent baselines (HVG/PCA logistic probe and frozen Geneformer). Metrics are computed on the external test set (GSE153515, 14,775 epithelial cells): macro-F1 = donor-level macro F1 with 95% bootstrap CI (n_boot=1000, seed 42); LAA = layer adjacency accuracy; ρ = marker-gradient rank correlation; ECE = expected calibration error (10 bins).",
        "",
        "**Honest limitation (epoch coverage).** Training used `save_total_limit=2` (checkpoint every epoch), so the epoch-1 checkpoint was not retained. This table therefore covers epochs 2–3 only; epoch-3 rows are the final models and reproduce `artifacts/paper1_tier_b_eval/results_*.json` exactly.",
        "",
        "| Arm | λ | Seed | Epoch | Checkpoint | macro-F1 | CI lower | CI upper | LAA | ρ | ECE | Mean train loss |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def _f(v: float) -> str:
            return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

        lines.append(
            f"| {r['arm']} | {r['lambda']} | {r['seed'] or '—'} | {r['epoch'] or '—'} | {r['checkpoint']} "
            f"| {_f(r['macro_f1'])} | {_f(r['ci_lower'])} | {_f(r['ci_upper'])} "
            f"| {_f(r['LAA'])} | {_f(r['rho'])} | {_f(r['ECE'])} | {_f(r['mean_train_loss'])} |"
        )
    lines += [
        "",
        "**Notes.**",
        "- Mean train loss is aggregated from `trainer_state.json` log_history (per-epoch means over logged steps).",
        "- The internal holdout used for the calibration/OOD figures is each run's own `prepared/caft_labeled_test.dataset` (19,922 cells; 6,924 epithelial after removing NON_LAYER), not the manuscript's donor-disjoint test (20,394 cells / 25 donors) — see changelog 2026-08-03-calibration-ood-s1 for the discrepancy note.",
        f"- Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
    ]
    S1_PATH.parent.mkdir(parents=True, exist_ok=True)
    S1_PATH.write_text("\n".join(lines))
    print(f"{_ts()} wrote {S1_PATH}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        default=",".join(f"{lam}:{seed}" for lam in LAMBDAS for seed in SEEDS),
        help="comma-separated LAM:SEED pairs (default: all 6)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip-frozen", action="store_true", help="skip the frozen arm")
    parser.add_argument("--frozen-only", action="store_true", help="only the frozen arm")
    parser.add_argument("--epoch2", action="store_true", help="epoch-2 tier-B eval (S1)")
    parser.add_argument("--postprocess-only", action="store_true", help="CPU figures S1/S2")
    parser.add_argument("--s1-table", action="store_true", help="assemble S1 markdown")
    args = parser.parse_args()

    if args.postprocess_only:
        return run_postprocess()
    if args.s1_table:
        return run_s1_table()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{_ts()} device={device}")

    if args.epoch2:
        runs = _parse_runs(args.runs)
        return run_epoch2(device, args.batch_size, runs)

    runs = _parse_runs(args.runs)
    all_ok = True
    if not args.frozen_only:
        for lam, seed in runs:
            print(f"{_ts()} FT lam={lam} seed={seed}")
            ok = _collect_fine_tuned(lam, seed, device, args.batch_size)
            all_ok = all_ok and ok
            if not ok:
                print(f"SANITY FAIL for lam={lam} seed={seed} — aborting")
                return 2
    if not args.skip_frozen:
        print(f"{_ts()} frozen arm")
        ok = _collect_frozen(device, args.batch_size)
        all_ok = all_ok and ok
        if not ok:
            print("SANITY FAIL for frozen arm — aborting")
            return 2
    if all_ok:
        print("SANITY GATE PASSED for all models.")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
