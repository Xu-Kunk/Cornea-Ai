"""Fixed subsample v3: backed view write (proven to work from v1)."""
import os
from pathlib import Path

import os
from pathlib import Path

import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data")))
SRC = DATA / "30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad"
TRAIN_OUT = DATA / "corneal_100k_subsample.h5ad"
TEST_OUT = DATA / "corneal_20k_test.h5ad"
SEED = 42

print("Loading source (backed)...", flush=True)
adata = sc.read_h5ad(SRC, backed='r')
print(f"Source: {adata.n_obs} cells, {adata.n_vars} genes", flush=True)

# Step 1: Subsample 120k stratified
sub_frac = 120000 / adata.n_obs
_, sub_idx = train_test_split(
    np.arange(adata.n_obs), test_size=sub_frac,
    stratify=adata.obs['cell_type'], random_state=SEED
)
print(f"Subsampled {len(sub_idx)} cells", flush=True)

# Step 2: Split 120k → 100k + 20k
n_test = 20000
train_idx, test_idx = train_test_split(
    sub_idx, test_size=n_test,
    stratify=adata.obs['cell_type'].iloc[sub_idx], random_state=SEED
)
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}", flush=True)

# Verify no overlap + cell type coverage
assert len(set(train_idx) & set(test_idx)) == 0
ct = adata.obs['cell_type']
for name in sorted(ct.cat.categories):
    tc = (ct.iloc[train_idx] == name).sum()
    tec = (ct.iloc[test_idx] == name).sum()
    print(f"  {name:<40} train={tc:>5}  test={tec:>4}", flush=True)

# Write test
print(f"\nWriting test ({len(test_idx)} cells)...", flush=True)
adata[test_idx].write(TEST_OUT)
print(f"  Done: {TEST_OUT}", flush=True)

# Write train — USE DIRECT backed view write (proven to work)
print(f"Writing train ({len(train_idx)} cells)...", flush=True)
adata[train_idx].write(TRAIN_OUT)
print(f"  Done: {TRAIN_OUT}", flush=True)

# Verify
import os
for f in [TRAIN_OUT, TEST_OUT]:
    sz = os.path.getsize(f) / 1e9
    print(f"  {os.path.basename(f)}: {sz:.2f} GB", flush=True)

print(f"\nDone: {len(train_idx)} train + {len(test_idx)} test", flush=True)
