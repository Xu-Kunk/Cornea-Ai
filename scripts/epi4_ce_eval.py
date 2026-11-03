#!/usr/bin/env python
"""Evaluate the epi4-CE arm (4-class CE over epithelial cells) on GSE153515.

Companion of ``scripts/method_arm_gf.py --arm ce --label-space epi4``: the
checkpoints carry a 4-way softmax head over the evaluation class set
(LSC, TAC, basal, differentiated), so no 31 -> 4 class aggregation is needed —
the head logits ARE the class probabilities (softmax).

Everything else is the Tier-B external-evaluation path of
``scripts/run_tier_b_eval.py``, imported (not reimplemented): same locked
config, same split manifest, same tokenized external cache, same static +
leakage gates, same ``corneal_fm.evaluation.evaluate_on_external_test`` call
(donor-level macro-F1 + bootstrap CI, per-class F1, LAA, differentiation
gradient rho, ECE), same inference batching. The only differences are
(a) the checkpoint discovery arm tag, (b) the 4-way softmax decode, and
(c) the output directory, which is the paper's results tree rather than
``artifacts/paper1_tier_b_eval`` (no existing artifact is overwritten).

Per seed it writes, under ``--out-dir``:
    seed{seed}/eval_gse153515.json   full evaluation record
    seed{seed}/percell_epi4_{seed}.npz
        y_true_ext / y_pred_ext / probs_ext / emb_ext / donor_ext
        (same layout as artifacts/paper1_tier_b_eval/percell_*.npz, so the
        per-cell predictions and CLS embeddings stay auditable)

Usage:
    python scripts/epi4_ce_eval.py --seeds 42,123,456 \
        --out-dir results/paper1_vanzyl/epi4_ce
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import run_tier_b_eval as tierb  # noqa: E402  (the locked Tier-B eval path)
import torch  # noqa: E402
import yaml  # noqa: E402

from corneal_fm.evaluation import evaluate_on_external_test, run_static_checks  # noqa: E402

ARM_TAG = "method_ce_epi4"
CLASSES = ["LSC", "TAC", "basal", "differentiated"]


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def _num_labels(checkpoint_path: Path) -> int:
    """Head width of the checkpoint (4 for the epi4-CE arm).

    Geneformer checkpoints store ``id2label``/``label2id`` but not always a
    ``num_labels`` key, so fall back to the head's weight shape.
    """
    cfg = json.loads((checkpoint_path / "config.json").read_text())
    if "num_labels" in cfg:
        return int(cfg["num_labels"])
    id2label = cfg.get("id2label") or {}
    if id2label:
        return len(id2label)
    from safetensors.torch import load_file

    weights = load_file(str(checkpoint_path / "model.safetensors"))
    return int(weights["classifier.weight"].shape[0])


def evaluate_seed(runner_args, seed: int, token_dataset, external_adata, config,
                  split_manifest, train_adata, device, torch_dtype) -> dict:
    arm_tag = runner_args.arm_tag
    variant = getattr(runner_args, "variant", None) or Path(runner_args.out_dir).name
    ck = tierb.discover_checkpoints([(arm_tag, 0.0, seed)])
    if not ck:
        raise FileNotFoundError(
            f"no checkpoint found for arm={arm_tag} lam=0.0 seed={seed} "
            f"(looked in {tierb.SAVE_DIR})"
        )
    ck = ck[0]
    ckpt_path = ck["checkpoint_path"]
    n_labels = _num_labels(ckpt_path)
    if n_labels != len(CLASSES):
        raise ValueError(
            f"{ckpt_path} has {n_labels} head labels, expected {len(CLASSES)} "
            "for the epi4 label space"
        )
    print(f"\n{_ts()}  === seed={seed} step={ck['step']} ({ckpt_path}) ===")

    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained(
        str(ckpt_path), num_labels=n_labels, torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    embeddings, logits = tierb.run_inference(
        model, token_dataset, batch_size=runner_args.batch_size, device=device,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # 4-way softmax decode: the epi4 head outputs the evaluation classes
    # directly (argmax class index == LAYER_ORDER_4 index).
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predictions = np.asarray(
        [tierb.LAYER_ORDER_4[c] for c in np.argmax(logits, axis=1)], dtype=object
    )

    results = evaluate_on_external_test(
        train_adata=train_adata,
        external_adata=external_adata,
        embeddings=embeddings,
        predictions=predictions,
        probabilities=probabilities,
        config=config,
        split_manifest=split_manifest,
        classes=CLASSES,
        layer_order=tierb.LAYER_ORDER_4,
        n_boot=tierb.N_BOOT,
        seed=tierb.BOOT_SEED,
    )
    results["model"] = {
        "arm": arm_tag,
        "variant": variant,
        "label_space": "epi4",
        "lambda": 0.0,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "save_dir": str(ck["save_dir"]),
        "step": ck["step"],
        "num_labels": n_labels,
        "decode": "softmax over the 4-class head (argmax)",
    }
    results["notes"] = (
        "marker_gradient_rank_correlation is the Spearman rho between the CLS "
        "embedding projection on the per-layer-centroid principal axis "
        "(oriented LSC -> differentiated) and the 4-stage ordinal, computed by "
        "the same corneal_fm.evaluation code as the naive / ordinal Tier-B arms. "
        "Decode: softmax over the 4-way head (argmax); no 31->4 aggregation."
    )

    out_seed = Path(runner_args.out_dir) / f"seed{seed}"
    out_seed.mkdir(parents=True, exist_ok=True)
    (out_seed / "eval_gse153515.json").write_text(json.dumps(results, indent=2))
    np.savez(
        out_seed / f"percell_epi4_{seed}.npz",
        y_true_ext=np.asarray(external_adata.obs["cell_type_harmonized"].astype(str)),
        y_pred_ext=np.asarray(predictions, dtype=object).astype(str),
        probs_ext=probabilities.astype(np.float64),
        emb_ext=embeddings.astype(np.float32),
        donor_ext=np.asarray(external_adata.obs["donor_id"].astype(str)),
    )
    pe, se = results["primary_endpoint"], results["secondary_endpoints"]
    print(
        f"{_ts()}  seed={seed} macro_f1={pe['macro_f1']:.4f} "
        f"[{pe['ci_lower']:.4f},{pe['ci_upper']:.4f}] LAA={se['layer_adjacency_accuracy']:.4f} "
        f"rho={se['marker_gradient_rank_correlation']:.4f} ECE={se['expected_calibration_error']:.4f}"
    )
    return results


def load_external(token_cache: Path, token_dataset: Path, ext_data: Path,
                  study_accession: str, skip_tokenize: bool):
    """External test set for the locked Tier-B path, with a working cache hit.

    ``run_tier_b_eval.tokenize_external`` tests ``token_dataset.is_file()``,
    which is always False for an HF ``save_to_disk`` dataset *directory*, so its
    ``--skip-tokenize`` silently re-tokenizes and rewrites the cache. Here the
    cache is honoured when the directory exists, using exactly the same
    alignment steps (NON_LAYER filter + ``obs['study']`` declaration) as the
    locked function; otherwise the locked function is called unchanged.
    """
    if skip_tokenize and token_dataset.exists():
        import anndata as ad
        from datasets import load_from_disk

        dataset = load_from_disk(str(token_dataset))
        adata = ad.read_h5ad(str(ext_data))
        labels = adata.obs["cell_type_harmonized"].astype(str)
        mask = np.asarray(labels != "NON_LAYER")
        dataset = dataset.select(np.where(mask)[0])
        adata = adata[mask].copy()
        adata.obs["study"] = study_accession
        print(f"{_ts()}  Cached tokenized dataset reused: {token_dataset} "
              f"({len(dataset)} epithelial cells)")
        return dataset, adata
    return tierb.tokenize_external(
        ext_data, token_cache, token_dataset, study_accession,
        skip_tokenize=skip_tokenize,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated seeds in the save-dir name.")
    parser.add_argument("--out-dir", default="results/paper1_vanzyl/epi4_ce")
    parser.add_argument("--arm-tag", default=ARM_TAG,
                        help="Save-dir prefix of the arm to evaluate "
                             "(default: method_ce_epi4 = the drop_nonepi arm; "
                             "pass method_axis_ft for the masked-parity arm).")
    parser.add_argument("--variant", default=None,
                        help="Label recorded in the eval JSON (default: the "
                             "out-dir basename, e.g. drop_nonepi / mask_nonepi).")
    parser.add_argument("--batch-size", type=int, default=tierb.BATCH_SIZE)
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Reuse the cached tokenized external test dataset.")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    paths = tierb.resolve_run_paths(None)          # locked GSE153515 paths
    config_path, ext_data = paths["config_path"], paths["ext_data"]
    token_cache, token_dataset = paths["token_cache"], paths["token_dataset"]
    print(f"{_ts()}  config={config_path}\n{_ts()}  external={ext_data}")

    static = run_static_checks(
        config_path=str(config_path), split_manifest_path=str(tierb.MANIFEST_PATH)
    )
    print(f"{_ts()}  Static checks: {static}")

    device = tierb._detect_device()
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"{_ts()}  device={device} dtype={torch_dtype}")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(tierb.MANIFEST_PATH) as f:
        split_manifest = json.load(f)

    dataset, external_adata = load_external(
        token_cache, token_dataset, ext_data,
        (config.get("external_test") or {}).get("accession", "GSE153515"),
        skip_tokenize=args.skip_tokenize,
    )
    print(f"{_ts()}  External eval cells: {len(dataset)}")

    import anndata as ad
    train_adata = ad.read_h5ad(str(tierb.TRAIN_DATA), backed="r")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        evaluate_seed(args, seed, dataset, external_adata, config,
                      split_manifest, train_adata, device, torch_dtype)
    print(f"{_ts()}  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
