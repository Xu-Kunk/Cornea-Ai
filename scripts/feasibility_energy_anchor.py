#!/usr/bin/env python
"""CPU feasibility for the A1 EnergyAnchor arm (GPU owned by fix-7).

Verifies, without starting any training:
  (a) EnergyAnchorTrainer init computes frozen per-stage penultimate-layer
      reference centroids (shape/finiteness) + median-heuristic bandwidth;
  (b) compute_loss with anchor_lambda>0: loss = L_cls + lambda_e * L_anchor,
      finite, and gradient flows into the anchor layer (encoder.layer[10],
      the penultimate transformer layer) but NOT into the frozen centroids;
  (c) masked path: an all-non-epithelial batch yields loss == L_cls
      (no anchor term) with valid grads;
  (d) anchor_lambda=0 => loss == L_cls (plain CE control);
  (e) checkpoint-free, dataset-only reference: reference is computed from the
      freshly loaded pretrained model at init (no fine-tuned weights).

Writes artifacts/paper1_energy_drift/feasibility_a1.json.
"""
import json
import sys
import time
from pathlib import Path

PROJECT = Path("<local>/")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import torch
from datasets import load_from_disk

from corneal_fm.caft import load_pretrained_classifier
from corneal_fm.energy_anchor import build_energy_anchor_trainer
from corneal_fm.ordinal import class_to_ordinal_map

PRETRAINED = (
    Path.home() / ".cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots/"
    "04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5/Geneformer-V2-104M"
)
PREP = PROJECT / "save/caft_gf_lam0.0_seed42_20260731-211824/prepared/caft_labeled_train.dataset"
IDC_PKL = PROJECT / "save/caft_gf_lam0.0_seed42_20260731-211824/prepared/caft_id_class_dict.pkl"
OUT = PROJECT / "artifacts/paper1_energy_drift"
OUT.mkdir(parents=True, exist_ok=True)

import pickle
idc = pickle.load(open(IDC_PKL, "rb"))
stage_map = class_to_ordinal_map(idc)
print("idc classes:", len(idc), "stage_map sample:", {k: v for k, v in list(stage_map.items())[:8]})

ds = load_from_disk(str(PREP))
labels = ds["label"]
epi_idx = [i for i, c in enumerate(labels) if stage_map.get(int(c), -1) != -1]
non_idx = [i for i, c in enumerate(labels) if stage_map.get(int(c), -1) == -1]
print("epi:", len(epi_idx), "non-epi:", len(non_idx))
# 32 epithelial + 32 non-epithelial, in that order -> batch1 exercises anchor,
# batch2 exercises the masked (all-non-epi) path.
sub = ds.select(epi_idx[:8] + non_idx[:8])
keep = {"input_ids", "length", "label"}
drop_cols = [c for c in sub.column_names if c not in keep]
if drop_cols:
    sub = sub.remove_columns(drop_cols)
print("subset:", len(sub), "cols:", sub.column_names)

torch.set_num_threads(16)
model = load_pretrained_classifier(PRETRAINED, num_classes=len(idc), torch_dtype=None)
targs = {
    "per_device_train_batch_size": 8,
    "num_train_epochs": 1,
    "use_cpu": True,
    "report_to": [],
    "logging_steps": 1,
    "save_strategy": "no",
    "eval_strategy": "no",
    "remove_unused_columns": False,
}

t0 = time.time()
tr = build_energy_anchor_trainer(
    model, sub, idc, anchor_lambda=0.5, training_args_kwargs=targs,
    output_directory=str(PROJECT / "tmp_trainer" / "feas_energy_anchor_model"),
)
init_s = time.time() - t0
print(f"\ninit+reference precompute: {init_s:.1f}s")

res = {
    "ok": True,
    "centroids_shape": None,
    "centroid_norms": None,
    "centroid_counts": None,
    "bandwidth_sigma": None,
    "loss_anchor": None,
    "loss_cls": None,
    "loss_total": None,
    "l_anchor": None,
    "grad_anchor_layer_norm": None,
    "grad_centroid_requires_grad": None,
    "loss_all_nonepi": None,
    "loss_lambda0": None,
    "init_precompute_s": round(init_s, 1),
}

c = tr.stage_centroids
res["centroids_shape"] = list(c.shape)
res["centroid_norms"] = [round(float(torch.norm(c[i]).item()), 4) for i in range(c.shape[0])]
res["bandwidth_sigma"] = round(float(tr.kernel_bandwidth), 6)
print("centroids:", c.shape, "norms:", res["centroid_norms"], "sigma:", res["bandwidth_sigma"])
assert torch.isfinite(c).all(), "non-finite centroids"

# (b) one forward/backward on the epithelial batch
loader = torch.utils.data.DataLoader(
    sub.select(range(8)), batch_size=8,
    collate_fn=tr.data_collator, shuffle=False,
)
batch = next(iter(loader))
print("collated keys:", list(batch.keys()))
loss = tr.compute_loss(tr.model, batch)
print("loss_total:", float(loss))
loss.backward()

l_cls_attr = loss.detach()
grad_norms = []
for n, p in tr.model.named_parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        grad_norms.append((n, float(p.grad.norm().item())))
anchor_grads = [(n, g) for n, g in grad_norms if n.startswith("bert.encoder.layer.10")]
print("layers with nonzero grad (first 14):", [n.split(".")[3] for n, _ in grad_norms[:14]])
print("encoder.layer.10 grads:", anchor_grads[:2])

res["loss_total"] = round(float(loss.item()), 6)
res["grad_anchor_layer_norm"] = round(anchor_grads[0][1], 6) if anchor_grads else 0.0
res["grad_centroid_requires_grad"] = tr.stage_centroids.requires_grad
assert len(anchor_grads) > 0, "no gradient into anchor layer (penultimate)"
assert tr.stage_centroids.requires_grad is False, "reference centroids must be frozen"

# (c) all-non-epi batch -> loss == L_cls (no anchor term); no grads needed
non_batch = next(iter(torch.utils.data.DataLoader(
    sub.select(range(8, 16)), batch_size=8, collate_fn=tr.data_collator, shuffle=False)))
with torch.no_grad():
    loss_ne = tr.compute_loss(tr.model, non_batch)
res["loss_all_nonepi"] = round(float(loss_ne.item()), 6)
print("all-non-epi loss:", float(loss_ne))

# (d) lambda=0 -> loss == L_cls (no grads needed)
tr2 = build_energy_anchor_trainer(
    load_pretrained_classifier(PRETRAINED, num_classes=len(idc), torch_dtype=None),
    sub, idc, anchor_lambda=0.0, training_args_kwargs=targs,
    output_directory=str(PROJECT / "tmp_trainer" / "feas_energy_anchor_model0"),
)
epi_batch = next(iter(torch.utils.data.DataLoader(
    sub.select(range(8)), batch_size=8, collate_fn=tr2.data_collator, shuffle=False)))
with torch.no_grad():
    loss0 = tr2.compute_loss(tr2.model, epi_batch)
res["loss_lambda0"] = round(float(loss0.item()), 6)
print("lambda=0 loss:", float(loss0))
assert tr2.stage_centroids is None

with open(OUT / "feasibility_a1.json", "w") as f:
    json.dump(res, f, indent=2)
print("\nfeasibility OK ->", OUT / "feasibility_a1.json")
