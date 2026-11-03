#!/usr/bin/env python
"""CPU feasibility validation for HPA-FT (this study main method) — GPU is owned
by fix-7 (D5 ordinal), so this validates everything short of training:

  (a) load an existing fine-tuned checkpoint + construct the LWC loss
      (per-layer continuity over hidden states 3-10 [+ final 12]) and run one
      forward/backward on a small epithelial batch — must be finite and must
      flow gradient into the cont layers;
  (b) headfreeze freeze policy: requires_grad == False on encoder.layer[4:]
      (L5-L12), trainable head/embeddings/L1-L4, frozen params excluded from
      the HF optimizer;
  (c) damp policy (--upper-lr-scale 0.1): 4 optimizer groups with the upper
      layers at 10% LR;
  (d) save-dir convention save/hp_aft_gf_<mode>_lam<λ>_seed<s>_<ts>/ ready.

Runs entirely on CPU with fp32. No training is started.

Usage:
    <venv>/bin/python3 scripts/feasibility_hp_aft.py
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402

from corneal_fm.caft import load_pretrained_classifier  # noqa: E402
from corneal_fm.config import get_save_dir  # noqa: E402
from corneal_fm.hp_aft import (  # noqa: E402
    build_hpa_trainer,
    count_parameters,
    parse_cont_layers,
)

CKPT = (
    PROJECT_ROOT / "save/caft_gf_lam0.0_seed42_20260731-211824/model/checkpoint-7473"
)
PREP_TRAIN = (
    PROJECT_ROOT / "save/caft_gf_lam0.0_seed42_20260731-211824/prepared/caft_labeled_train.dataset"
)
IDC_PKL = (
    PROJECT_ROOT / "save/caft_gf_lam0.0_seed42_20260731-211824/prepared/caft_id_class_dict.pkl"
)
N_FT_CLASSES = 31
MAX_TOKENS = 128  # CPU truncation for the feasibility batch only
N_SAMPLE = 3

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def small_epithelial_batch():
    """3 epithelial cells spanning layers LSC(0)/basal(2)/squamous(4), truncated."""
    import pickle

    ds = load_from_disk(str(PREP_TRAIN))
    idc = pickle.load(open(IDC_PKL, "rb"))
    layer_col = ds["layer"]
    idx_by_layer: dict[int, int] = {}
    for i, lyr in enumerate(layer_col):
        if int(lyr) >= 0 and int(lyr) not in idx_by_layer:
            idx_by_layer[int(lyr)] = i
        if len(idx_by_layer) >= N_SAMPLE:
            break
    pick = [idx_by_layer.get(lyr) for lyr in sorted(idx_by_layer)[:N_SAMPLE]]
    pick = [i for i in pick if i is not None][:N_SAMPLE]
    if len(pick) < 2:
        raise RuntimeError(f"need >=2 epithelial cells, got {len(pick)}")
    rows = [ds[i] for i in pick]
    small = Dataset.from_dict({
        "input_ids": [r["input_ids"][:MAX_TOKENS] for r in rows],
        "length": [min(int(r["length"]), MAX_TOKENS) for r in rows],
        "label": [int(r["label"]) for r in rows],
        "layer": [int(r["layer"]) for r in rows],
    })
    batch = {
        "input_ids": torch.tensor(small["input_ids"], dtype=torch.long),
        "length": torch.tensor(small["length"], dtype=torch.long),
        "label": torch.tensor(small["label"], dtype=torch.long),
    }
    print(f"  sampled cells: labels={small['label']} layers={small['layer']} "
          f"len={len(pick)} batch_seq_len={batch['input_ids'].shape}")
    return small, batch, idc


def minimal_targs(tmp: Path) -> dict:
    return {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "weight_decay": 0.001,
        "logging_steps": 10,
        "save_strategy": "no",
        "eval_strategy": "no",
        "fp16": False,
        "bf16": False,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": [],
    }


def restore_trainable(model) -> None:
    for p in model.parameters():
        p.requires_grad = True


def main() -> int:
    t0 = time.time()
    print("=" * 70)
    print("HPA-FT CPU feasibility (no training)")
    print("=" * 70)

    # ---- (a) LWC loss construction + forward/backward -------------------
    print("\n[A] LWC: load checkpoint + per-layer continuity loss")
    model = load_pretrained_classifier(str(CKPT), num_classes=N_FT_CLASSES)  # fp32 CPU
    model.eval()
    n_train, n_total = count_parameters(model)
    print(f"  checkpoint loaded: {n_total/1e6:.1f}M params (trainable {n_train/1e6:.1f}M)")
    check("checkpoint loads on CPU", True, f"total={n_total}")

    small, batch, idc = small_epithelial_batch()
    tmp1 = PROJECT_ROOT / "tmp_trainer" / "hp_aft_feas_lwc"
    tmp1.mkdir(parents=True, exist_ok=True)
    trainer = build_hpa_trainer(
        model, small, idc,
        mode="lwc", hpa_lambda=0.5, cont_layers="3-10", final_cont=True,
        training_args_kwargs=minimal_targs(tmp1), output_directory=str(tmp1),
    )
    check("mode=lwc effective cont layers", trainer.cont_layers == list(range(3, 11)) + [12],
          f"cont_layers={trainer.cont_layers} (3-10 + final-cont=12)")

    trainer._precompute_centroids()  # real path over the small dataset
    cc = trainer.cont_layer_centroids
    check("per-layer centroids computed", cc is not None and cc.shape[0] == len(trainer.cont_layers),
          f"centroids shape={None if cc is None else tuple(cc.shape)}")

    loss = trainer.compute_loss(model, {k: v.clone() for k, v in batch.items()})
    loss_val = float(loss.detach())
    l_cont = None
    # L_cont is logged only every 50 steps; re-derive for the report
    check("LWC loss forward finite", torch.isfinite(loss).item(), f"loss={loss_val:.4f}")
    if trainer.cont_layer_centroids is not None:
        l_cont = _continuity_value(trainer, model, batch)
    check("L_cont < 0 (cosine adjacency active)", l_cont is not None and l_cont < 0,
          f"L_cont={l_cont:.4f}" if l_cont is not None else "n/a")

    loss.backward()
    g3 = model.bert.encoder.layer[2].attention.self.query.weight.grad  # hidden 3
    g12 = model.bert.encoder.layer[11].attention.self.query.weight.grad  # hidden 12
    check("gradient reaches cont layers (L3, L12)", g3 is not None and g12 is not None,
          f"|g|L3={None if g3 is None else float(g3.abs().max()):.2e} "
          f"|g|L12={None if g12 is None else float(g12.abs().max()):.2e}")

    # ---- (b) headfreeze freeze policy -----------------------------------
    print("\n[B] headfreeze: freeze L5-L12")
    restore_trainable(model)
    tmp2 = PROJECT_ROOT / "tmp_trainer" / "hp_aft_feas_hf"
    tmp2.mkdir(parents=True, exist_ok=True)
    trainer2 = build_hpa_trainer(
        model, small, idc,
        mode="headfreeze", hpa_lambda=0.5, cont_layers="2-4",
        training_args_kwargs=minimal_targs(tmp2), output_directory=str(tmp2),
    )
    check("headfreeze defaults to freeze", trainer2.freeze_upper and not trainer2.damp_upper,
          f"freeze_upper={trainer2.freeze_upper} damp={trainer2.damp_upper}")
    check("headfreeze effective cont layers (trainable only)",
          trainer2.cont_layers == [2, 3, 4], f"cont_layers={trainer2.cont_layers}")
    f1 = model.bert.encoder.layer[4].attention.self.query.weight.requires_grad   # L5 frozen
    f12 = model.bert.encoder.layer[11].attention.self.query.weight.requires_grad  # L12 frozen
    e1 = model.bert.encoder.layer[3].attention.self.query.weight.requires_grad   # L4 trainable
    head = model.classifier.weight.requires_grad
    emb = model.bert.embeddings.word_embeddings.weight.requires_grad
    check("L5/L12 frozen, L4/head/embeddings trainable",
          (not f1) and (not f12) and e1 and head and emb,
          f"L5={f1} L12={f12} L4={e1} head={head} emb={emb}")
    n_tr, _ = count_parameters(model)
    check("freeze count consistent", n_tr < n_total,
          f"trainable now {n_tr/1e6:.1f}M of {n_total/1e6:.1f}M")

    opt = trainer2.create_optimizer()
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    n_trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    check("frozen params excluded from optimizer",
          n_opt == n_trainable_params, f"opt params={n_opt} trainable={n_trainable_params}")
    check("no damp: HF default LR groups", len(opt.param_groups) == 2,
          f"groups={len(opt.param_groups)} lr0={opt.param_groups[0]['lr']:.2e}")

    # frozen layers produce constant hidden states -> dropped cont layers
    restore_trainable(model)
    tmp2b = PROJECT_ROOT / "tmp_trainer" / "hp_aft_feas_hf_drop"
    tmp2b.mkdir(parents=True, exist_ok=True)
    trainer2b = build_hpa_trainer(
        model, small, idc,
        mode="headfreeze", hpa_lambda=0.5, cont_layers="3-10",
        training_args_kwargs=minimal_targs(tmp2b), output_directory=str(tmp2b),
    )
    check("frozen cont layers dropped (3-10 -> 3,4)",
          trainer2b.cont_layers == [3, 4], f"effective={trainer2b.cont_layers}")

    # ---- (c) damp policy --------------------------------------------------
    print("\n[C] damp: --upper-lr-scale 0.1")
    restore_trainable(model)
    tmp3 = PROJECT_ROOT / "tmp_trainer" / "hp_aft_feas_damp"
    tmp3.mkdir(parents=True, exist_ok=True)
    trainer3 = build_hpa_trainer(
        model, small, idc,
        mode="headfreeze", hpa_lambda=0.5, cont_layers="2-4",
        freeze_upper=False, upper_lr_scale=0.1,
        training_args_kwargs=minimal_targs(tmp3), output_directory=str(tmp3),
    )
    check("damp: freeze off, damp on", (not trainer3.freeze_upper) and trainer3.damp_upper,
          f"freeze_upper={trainer3.freeze_upper} damp={trainer3.damp_upper}")
    opt3 = trainer3.create_optimizer()
    lrs = sorted({g["lr"] for g in opt3.param_groups})
    check("damp: 4 groups, upper lr = 10%", len(opt3.param_groups) == 4 and lrs == [5e-6, 5e-5],
          f"lrs={lrs} groups={len(opt3.param_groups)}")
    n_upper = len(opt3.param_groups[2]["params"]) + len(opt3.param_groups[3]["params"])
    check("damp: upper group non-empty", n_upper > 0, f"upper params={n_upper}")

    # ---- (d) parse validation + save-dir convention -----------------------
    print("\n[D] CLI parsing + save-dir convention")
    check("parse '3-10'", parse_cont_layers("3-10") == list(range(3, 11)),
          f"{parse_cont_layers('3-10')}")
    check("parse 'final'", parse_cont_layers("final") == [12])
    check("parse '3,4,10'", parse_cont_layers("3,4,10") == [3, 4, 10])
    try:
        parse_cont_layers("13")
        check("parse invalid '13' rejected", False)
    except ValueError:
        check("parse invalid '13' rejected", True)

    sd = get_save_dir("hp_aft_gf", "lwc_lam0.5_seed42")
    pat = re.compile(r"^hp_aft_gf_lwc_lam0\.5_seed42_\d{8}-\d{6}$")
    check("save-dir convention hp_aft_gf_<mode>_lam<λ>_seed<s>_<ts>",
          bool(pat.match(sd.name)), sd.name)
    shutil.rmtree(sd, ignore_errors=True)  # keep save/ clean (validation only)
    print(f"  (validated dir removed: {sd})")

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"HPA-FT CPU feasibility: {n_pass}/{len(RESULTS)} checks passed "
          f"({time.time() - t0:.0f}s, CPU only, no training started)")
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    print("=" * 70)
    return 0 if n_pass == len(RESULTS) else 1


def _continuity_value(trainer, model, batch):
    """Re-derive L_cont for the report (mirrors compute_loss internals)."""
    import torch.nn.functional as F  # noqa: N812  (mirrors corneal_fm/caft.py style)

    from corneal_fm.caft import NON_LAYER

    labels = batch["label"]
    input_ids = batch["input_ids"]
    device = input_ids.device
    with torch.no_grad():
        attn = torch.arange(input_ids.shape[1], device=device)[None, :] < batch["length"][:, None]
        out = model.bert(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        hidden = out.hidden_states
        layers = torch.tensor(
            [trainer.class_to_layer.get(int(c), NON_LAYER) for c in labels.tolist()],
            device=device,
        )
        valid = layers != NON_LAYER
        if not bool(valid.any()):
            return None
        lv = layers[valid]
        terms = []
        for ci, lyr_idx in enumerate(trainer.cont_layers):
            z = F.normalize(hidden[lyr_idx][:, 0], dim=-1)[valid]
            sims = []
            for zi, lyr in zip(z, lv.tolist()):
                cands = [i for i in (lyr - 1, lyr + 1) if 0 <= i < trainer.n_layers]
                if cands:
                    c = trainer.cont_layer_centroids[ci][cands]
                    sims.append((zi @ c.t()).mean())
            if sims:
                terms.append(-torch.stack(sims).mean())
    return float(torch.stack(terms).mean()) if terms else None


if __name__ == "__main__":
    sys.exit(main())
