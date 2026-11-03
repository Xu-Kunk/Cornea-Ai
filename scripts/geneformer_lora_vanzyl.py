#!/usr/bin/env python
"""GF LoRA: Geneformer V2-104M LoRA fine-tuning on van Zyl (third architecture, LoRA arm).

Extends the LoRA demonstration to Geneformer (the primary this study model).
scGPT LoRA (E4) achieved final-epoch rho 0.834 (2-seed mean); GF LoRA tests whether
the same QKV LoRA recipe (r=8, alpha=16, trunk-frozen) preserves the differentiation
gradient in the Geneformer backbone.

Architecture: Geneformer = BertForSequenceClassification (HuggingFace).
BertSelfAttention has separate query/key/value nn.Linear modules — simpler than
scGPT's packed in_proj_weight. The LoRA wrapper adds low-rank adapters to each.

Protocol mirrors scGPT E4-LoRA: custom training loop, 15 epochs, locked 90/10 val
split, locked test donor Hu0216, same metric suite (rho, z, macro-F1, bootstrap CI).

Key lessons from scGPT E4-v1 fixer failure:
  - requires_grad_(False) FREEZES the base trunk AND the freshly created adapters
  - Must explicitly re-enable lora_A_/lora_B_ gradients after freezing
  - B=0 init verified via identity check (max|diff| < 1e-4)
"""

from __future__ import annotations

import argparse
import glob as pyglob
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path("<home>/corneal_scgpt")
sys.path.insert(0, str(PROJECT))

from corneal_fm.evaluation import marker_gradient_correlation  # noqa: E402
from datasets import load_from_disk
from sklearn.metrics import f1_score

# ── Paths ─────────────────────────────────────────────────────────────────
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
HF_SNAP_GLOB = os.path.join(
    HF_CACHE, "models--ctheodoris--Geneformer", "snapshots", "*", "Geneformer-V2-104M"
)
DATA_DIR = PROJECT / "data"
ARTIFACTS = PROJECT / "artifacts" / "paper1_vanzyl"
GF_TOKENIZED_DIR = DATA_DIR / "gf_tokenized_vanzyl.dataset"

# ── Protocol constants (LOCKED, mirrors scGPT E4-LoRA) ────────────────────
LAYER_ORDER = ["LSC", "basal", "differentiated"]
N_NULL = 100
LORA_R = 8
LORA_ALPHA = 16
EPOCHS = 15
LR = 5e-5
PER_DEVICE_BATCH = 1  # 16GB GPU limit for 104M model w/ long sequences
GRAD_ACCUM = 32  # keep effective batch = 32
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-3
SEEDS = [42, 123, 456]


def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _locate_pretrained() -> str:
    hits = pyglob.glob(HF_SNAP_GLOB)
    if not hits:
        log("ERROR: Geneformer-V2-104M weights not found in HF cache.")
        sys.exit(1)
    return hits[0]


def _locate_token_dictionary() -> str:
    hits = pyglob.glob(
        os.path.join(HF_CACHE, "models--ctheodoris--Geneformer", "snapshots", "*", "token_dictionary.pkl")
    )
    if not hits:
        log("ERROR: token_dictionary.pkl not found.")
        sys.exit(1)
    return hits[0]


# ── LoRA machinery for BertSelfAttention ──────────────────────────────────


class LoRABertSelfAttention(nn.Module):
    """Wraps BertSelfAttention with LoRA on query/key/value projections.

    BertSelfAttention stores query, key, value as separate nn.Linear modules
    (no packed in_proj_weight), making the LoRA grafting simpler than scGPT.
    B=0 init ensures the wrapper output is identical to the base attention.
    """

    def __init__(self, attn: nn.Module, r: int = LORA_R, alpha: int = LORA_ALPHA):
        super().__init__()
        self.num_attention_heads = attn.num_attention_heads
        self.attention_head_size = attn.attention_head_size
        self.all_head_size = attn.all_head_size
        self.dropout = attn.dropout
        self.scaling = alpha / r

        d = attn.query.weight.shape[0]
        # Copy base weights, frozen
        for proj_name in ("query", "key", "value"):
            src = getattr(attn, proj_name)
            linear = nn.Linear(d, d, bias=src.bias is not None)
            linear.weight.data.copy_(src.weight.data)
            if src.bias is not None:
                linear.bias.data.copy_(src.bias.data)
            linear.requires_grad_(False)
            setattr(self, proj_name, linear)

        # LoRA adapters (B=0 init)
        for proj_name in ("query", "key", "value"):
            setattr(self, f"lora_A_{proj_name}", nn.Parameter(torch.zeros(r, d)))
            setattr(self, f"lora_B_{proj_name}", nn.Parameter(torch.zeros(d, r)))
            nn.init.normal_(getattr(self, f"lora_A_{proj_name}"), 0.0, 0.02)
            nn.init.zeros_(getattr(self, f"lora_B_{proj_name}"))

    def _delta(self, name: str) -> torch.Tensor:
        return self.scaling * (
            getattr(self, f"lora_B_{name}") @ getattr(self, f"lora_A_{name}")
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        return x.view(B, S, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False,
                past_key_values=None, **kwargs):
        q = F.linear(hidden_states, self.query.weight + self._delta("query"), self.query.bias)
        k = F.linear(hidden_states, self.key.weight + self._delta("key"), self.key.bias)
        v = F.linear(hidden_states, self.value.weight + self._delta("value"), self.value.bias)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            hidden_states.shape[0], -1, self.all_head_size)
        # No out_proj here: BertSdpaSelfAttention in transformers>=4.57 does NOT
        # have out_proj — the output projection is in BertSelfOutput (parent
        # BertAttention module). The wrapped attention returns the raw concat'd
        # head outputs; BertSelfOutput applies the dense projection + dropout.
        return (attn_out, attn_weights) if output_attentions else (attn_out,)


def wrap_gf_lora(model, r: int = LORA_R, alpha: int = LORA_ALPHA) -> list[str]:
    """Replace every BertSelfAttention with LoRA wrapper; freeze trunk."""
    wrapped = []
    for i, layer in enumerate(model.bert.encoder.layer):
        layer.attention.self = LoRABertSelfAttention(layer.attention.self, r=r, alpha=alpha)
        wrapped.append(i)
    # Freeze everything
    model.requires_grad_(False)
    # Re-enable classifier head
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    # CRITICAL: re-enable LoRA adapters (fix from scGPT E4-v1 failure)
    n_adapt = 0
    for name, p in model.named_parameters():
        if "lora_A_" in name or "lora_B_" in name:
            p.requires_grad_(True)
            n_adapt += p.numel()
    n_head = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    log(f"LoRA wrapped {len(wrapped)} layers (r={r}, alpha={alpha}); "
        f"trainable = head {n_head/1e6:.3f}M + adapters {n_adapt/1e6:.3f}M")
    return [n for n, p in model.named_parameters() if p.requires_grad]


# ── Metrics ────────────────────────────────────────────────────────────────


def macro_f1_ci(y_true, y_pred, n_boot=1000, seed=42) -> dict:
    present = sorted(set(y_true.tolist()) & set(LAYER_ORDER))
    m = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = f1_score(y_true[b], y_pred[b], labels=present,
                            average="macro", zero_division=0)
    return {"macro_f1": m, "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5))}


def rho_and_null(emb, labels) -> dict:
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    rho_n = float(marker_gradient_correlation(emb_n, labels, LAYER_ORDER))
    rng = np.random.default_rng(42)
    nulls = np.empty(N_NULL)
    for i in range(N_NULL):
        shuffled = labels[rng.permutation(len(labels))]
        nulls[i] = marker_gradient_correlation(emb_n, shuffled, LAYER_ORDER)
    return {"rho_normed": rho_n,
            "null_mean": float(nulls.mean()), "null_sd": float(nulls.std(ddof=1)),
            "null_z": float((rho_n - nulls.mean()) / max(nulls.std(ddof=1), 1e-12)),
            "null_pctile": float((nulls <= rho_n).mean())}


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "gf_lora")
    ap.add_argument("--identity-check", action="store_true")
    args = ap.parse_args()
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} seed={seed}")

    # ── Load tokenized dataset ──────────────────────────────────────────
    from transformers import BertForSequenceClassification
    ds = load_from_disk(str(GF_TOKENIZED_DIR))
    n_total = len(ds)
    log(f"Loaded {n_total} tokenized cells from {GF_TOKENIZED_DIR}")

    # ── Load author->3-tier mapping from source h5ad ────────────────────
    import anndata as ad
    src_h5ad = DATA_DIR / "vanzyl_annotated.h5ad"
    a = ad.read_h5ad(str(src_h5ad))
    author_all = a.obs["author_cell_type"].astype(str).to_numpy()
    cc4_all = a.obs["cell_class_4"].astype(str).to_numpy()
    donor_all = a.obs["donor_id"].astype(str).to_numpy() if "donor_id" in a.obs else None

    # Majority mapping: author_cell_type -> cell_class_4
    cont: dict[str, Counter] = {}
    for au, l in zip(author_all, cc4_all):
        cont.setdefault(au, Counter())[l] += 1
    naive_map = {au: max(c, key=c.get) for au, c in cont.items()}
    author_classes = sorted(set(author_all.tolist()))
    class_to_id = {c: i for i, c in enumerate(author_classes)}
    log(f"n_author_classes={len(author_classes)} n_total_cells={n_total}")

    # ── Split: train/val (90/10 stratified by label) ────────────────────
    # Use fixed val_indices.npz from scGPT convergence split if available
    val_npz = ARTIFACTS / "run6" / "convergence" / "val_indices.npz"
    if val_npz.is_file():
        vd = np.load(val_npz)
        val_mask = np.zeros(n_total, dtype=bool)
        val_mask[vd["val_idx"].astype(np.int64)] = True
        train_idx = np.where(~val_mask)[0].tolist()
        val_idx_a = np.where(val_mask)[0].tolist()
        log(f"Locked val split: {len(train_idx)} train, {len(val_idx_a)} val")
    else:
        # Fallback: stratified 90/10 by author class
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        labels_enc = np.array([class_to_id.get(c, 0) for c in author_all])
        tr_idx, va_idx = next(sss.split(np.arange(n_total), labels_enc))
        train_idx, val_idx_a = tr_idx.tolist(), va_idx.tolist()
        log(f"Stratified 90/10 split: {len(train_idx)} train, {len(val_idx_a)} val")

    # Build dataset indices
    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx_a)
    y_true_val = cc4_all[val_idx_a]

    # ── Load model ──────────────────────────────────────────────────────
    from corneal_fm.caft import load_pretrained_classifier
    pretrained_dir = _locate_pretrained()

    if args.identity_check:
        # fp32 for identity check (LoRA adapters are fp32, need matching dtype)
        base = load_pretrained_classifier(pretrained_dir, len(author_classes))
        base.to(device).eval()
        lora_model = load_pretrained_classifier(pretrained_dir, len(author_classes))
        lora_model.to(device)
        batch = [train_ds[i] for i in range(4)]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch],
            batch_first=True, padding_value=0).to(device)
        attn_mask = (input_ids != 0).to(device)
        with torch.no_grad():
            out_base = base(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            emb_base = out_base.hidden_states[-1][:, 0]

        wrap_gf_lora(lora_model)
        lora_model.to(device).eval()
        with torch.no_grad():
            out_lora = lora_model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            emb_lora = out_lora.hidden_states[-1][:, 0]
        diff = (emb_base - emb_lora).abs().max().item()
        log(f"IDENTITY CHECK (fp32): B=0 LoRA vs base max|diff| = {diff:.2e}")
        if diff > 1e-4:
            log("IDENTITY CHECK FAILED")
            return 1
        log("IDENTITY CHECK PASS")
        return 0

    model = load_pretrained_classifier(pretrained_dir, len(author_classes),
                                       torch_dtype=torch.bfloat16)
    model.to(device)
    model.config.pad_token_id = 0  # Geneformer pad token
    wrap_gf_lora(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Data loaders ────────────────────────────────────────────────────
    class GFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, indices, label_fn):
            self.data = hf_ds
            self.indices = indices
            self.label_fn = label_fn

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            item = self.data[self.indices[idx]]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "label": torch.tensor(self.label_fn(item), dtype=torch.long),
            }

    def label_fn(item):
        # Use the author_cell_type stored in the dataset
        # The label is the author class index
        return class_to_id.get(item.get("author_cell_type", ""), 0)

    tr_dataset = GFDataset(train_ds, list(range(len(train_ds))), label_fn)
    val_dataset = GFDataset(val_ds, list(range(len(val_ds))), label_fn)

    def collate_fn(batch):
        # Pad variable-length sequences to max length in batch
        # GF dataset stores input_ids as Python lists, need to convert to tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch],
            batch_first=True, padding_value=0)
        labels = torch.stack([b["label"] for b in batch])
        attn_mask = (input_ids != 0)
        return input_ids, attn_mask, labels

    loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=PER_DEVICE_BATCH, shuffle=True, collate_fn=collate_fn,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ── Training loop ───────────────────────────────────────────────────
    run_dir = args.out / "checkpoints" / f"gf_lora_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    opt_step = 0
    best_val_f1, best_val_f1_ep = -1.0, None
    best_val_rho, best_val_rho_ep = None, None
    traj_path = run_dir / "trajectory.jsonl"
    t_all = time.time()
    epochs = 1 if args.dry_run else EPOCHS

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        opt.zero_grad(set_to_none=True)
        for step, (input_ids, attn_mask, y) in enumerate(loader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attn_mask, labels=y)
                loss = outputs.loss
            (loss.float() / GRAD_ACCUM).backward()
            running.append(float(loss.detach().float()))
            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
                opt.step()
                opt_step += 1
                if opt_step <= WARMUP_STEPS:
                    for g in opt.param_groups:
                        g["lr"] = LR * opt_step / max(WARMUP_STEPS, 1)
                else:
                    for g in opt.param_groups:
                        g["lr"] = LR
                opt.zero_grad(set_to_none=True)
        train_loss = float(np.mean(running[-min(200, len(running)):]))

        # Validation
        model.eval()
        vemb, vlogits = [], []
        with torch.no_grad():
            for input_ids, attn_mask, _y in val_loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
                vemb.append(out.hidden_states[-1][:, 0].float().cpu().numpy())
                vlogits.append(out.logits.float().cpu().numpy())
        vemb = np.concatenate(vemb, axis=0).astype(np.float32)
        vlogits = np.concatenate(vlogits, axis=0)
        vpred = np.array([naive_map.get(author_classes[int(i)], "NON_LAYER")
                          for i in vlogits.argmax(1)], dtype=str)
        present = sorted(set(y_true_val.tolist()) & set(LAYER_ORDER))
        val_f1 = float(f1_score(y_true_val, vpred, labels=present,
                                average="macro", zero_division=0))
        rh = rho_and_null(vemb, y_true_val)
        if val_f1 > best_val_f1:
            best_val_f1, best_val_f1_ep = val_f1, epoch
            torch.save(model.state_dict(), run_dir / "model_best_val.pt")
        if best_val_rho is None or rh["rho_normed"] > best_val_rho:
            best_val_rho, best_val_rho_ep = rh["rho_normed"], epoch
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:02d}.pt")
        # Check adapter activity
        ad_norm = 0.0
        for name, p in model.named_parameters():
            if "lora_B_" in name:
                ad_norm = max(ad_norm, float(p.abs().max()))
        rec = {"epoch": epoch, "train_loss_mean": train_loss, "val_f1": val_f1,
               "val_rho_normed": rh["rho_normed"], "val_z": rh["null_z"],
               "best_val_f1": best_val_f1, "best_val_f1_epoch": best_val_f1_ep,
               "max_lora_B_abs": ad_norm,
               "wall_s": round(time.time() - t_all, 1)}
        with traj_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_f1={val_f1:.4f} val_rho={rh['rho_normed']:.4f} "
            f"val_z={rh['null_z']:.1f} ({time.time()-t_all:.0f}s)")

    torch.save(model.state_dict(), run_dir / "model.pt")

    # ── Test donor evaluation ───────────────────────────────────────────
    log("Evaluating on test donor...")
    test_mask = np.zeros(n_total, dtype=bool)
    if donor_all is not None:
        test_mask |= (donor_all == "sanes2022_Hu0216")
    else:
        # Fallback: assume last 20% of cells are test
        test_mask[-n_total // 5:] = True
    test_indices = np.where(test_mask)[0].tolist()
    test_ds = ds.select(test_indices)
    y_test = cc4_all[test_indices]
    log(f"Test donor: {len(test_indices)} cells, distribution: {dict(Counter(y_test.tolist()))}")

    test_dataset = GFDataset(test_ds, list(range(len(test_ds))), label_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Evaluate best-val and final-epoch checkpoints
    ev = {}
    for ctype, fname in [("best_val", "model_best_val.pt"), ("final_epoch", "model_epoch15.pt")]:
        ck = run_dir / fname
        if not ck.is_file():
            log(f"SKIP {ctype}: {fname} missing")
            continue
        st = torch.load(ck, map_location="cpu", weights_only=True)
        m2 = load_pretrained_classifier(_locate_pretrained(), len(author_classes))
        wrap_gf_lora(m2)
        m2.load_state_dict(st, strict=True)
        m2.to(device).eval()
        temb, tlogits = [], []
        with torch.no_grad():
            for input_ids, attn_mask, _y in test_loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = m2(input_ids, attention_mask=attn_mask, output_hidden_states=True)
                temb.append(out.hidden_states[-1][:, 0].float().cpu().numpy())
                tlogits.append(out.logits.float().cpu().numpy())
        temb = np.concatenate(temb, axis=0).astype(np.float32)
        tlogits = np.concatenate(tlogits, axis=0)
        tpred = np.array([naive_map.get(author_classes[int(i)], "NON_LAYER")
                          for i in tlogits.argmax(1)], dtype=str)
        f1_res = macro_f1_ci(y_test, tpred)
        rh_res = rho_and_null(temb, y_test)
        ev[ctype] = {"macro_f1": f1_res, **rh_res,
                     "pred_dist": dict(Counter(tpred.tolist()))}
        log(f"  test {ctype}: F1={f1_res['macro_f1']:.4f} "
            f"rho={rh_res['rho_normed']:.4f} z={rh_res['null_z']:.1f}")

    summary = {
        "arm": "GF LoRA (QKV r=8 alpha=16; head + adapters trained, base trunk frozen)",
        "seed": seed, "epochs": epochs, "n_cls": len(author_classes),
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
        "val_split": {"n_train": len(train_idx), "n_val": len(val_idx_a)},
        "best_val_f1": best_val_f1, "best_val_f1_epoch": best_val_f1_ep,
        "best_val_rho": best_val_rho, "best_val_rho_epoch": best_val_rho_ep,
        "test_donor_eval": ev,
        "wall_s": time.time() - t_all,
        "timestamp_utc": now_utc(),
    }
    (run_dir / "args.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "runs_status").write_text(f"DONE\n{now_utc()}\n", encoding="utf-8")
    log("saved " + str(run_dir))
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())