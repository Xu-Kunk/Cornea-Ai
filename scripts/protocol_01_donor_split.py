"""Protocol 01 v4.1: Donor-disjoint split (balanced).

Critical fix: Donors have highly unequal cell counts (some donors contribute
thousands of cells, others only a few). Simple GroupShuffleSplit assigns
large donors to one side, creating extreme imbalance (609 vs 19k cells).

Fix: Greedy donor assignment — sort donors by cell count descending, then
assign each donor to whichever split has fewer cells. This ensures both
train (~83%) and test (~17%) have sufficient cell counts with zero donor
overlap.

Since the original 10GB source was cleaned up, we merge the existing
100k + 20k files and re-split.
"""
import os
import scanpy as sc
import numpy as np
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_SRC = DATA_DIR / "corneal_100k_subsample.h5ad"
TEST_SRC = DATA_DIR / "corneal_20k_test.h5ad"

TRAIN_OUT = DATA_DIR / "corneal_100k_donor_split.h5ad"
TEST_OUT = DATA_DIR / "corneal_20k_donor_split.h5ad"

TEST_FRACTION = 0.17  # ~17% of cells for test

# Step 1: Merge existing splits
print("=" * 60)
print("  Protocol 01 v4.1: Donor-Disjoint Split (Balanced)")
print("=" * 60)
print(f"\nLoading {TRAIN_SRC.name} ...")
adata_train = sc.read_h5ad(str(TRAIN_SRC))
print(f"  {adata_train.n_obs} cells")
print(f"Loading {TEST_SRC.name} ...")
adata_test = sc.read_h5ad(str(TEST_SRC))
print(f"  {adata_test.n_obs} cells")

assert len(set(adata_train.obs_names) & set(adata_test.obs_names)) == 0, \
    "Train/test cell overlap!"

adata = sc.concat([adata_train, adata_test], join="inner")
del adata_train, adata_test
print(f"Merged: {adata.n_obs} cells × {adata.n_vars} genes")

# Step 2: Greedy balanced donor assignment
donors = adata.obs["donor_id"].values
donor_sizes = adata.obs.groupby("donor_id").size().sort_values(ascending=False)

print(f"\nTotal donors: {len(donor_sizes)}")
print(f"Donor size range: {donor_sizes.min()} – {donor_sizes.max()} cells")
print(f"Top 5 donors by size:")
for d, n in donor_sizes.head().items():
    print(f"  {d}: {n} cells")

rng = np.random.default_rng(SEED)
total_cells = len(adata)
target_test_cells = int(total_cells * TEST_FRACTION)

train_donor_names = []
test_donor_names = []
train_cell_count = 0
test_cell_count = 0

for donor, size in donor_sizes.items():
    # Add donor to whichever split is further below its target
    test_gap = target_test_cells - test_cell_count
    train_gap = (total_cells - target_test_cells) - train_cell_count
    
    if test_gap > train_gap:
        test_donor_names.append(donor)
        test_cell_count += size
    else:
        train_donor_names.append(donor)
        train_cell_count += size

train_donor_set = set(train_donor_names)
test_donor_set = set(test_donor_names)
assert len(train_donor_set & test_donor_set) == 0, "Donor overlap!"

print(f"\nDonor assignment:")
print(f"  Train: {len(train_donor_set)} donors, {train_cell_count} cells ({100*train_cell_count/total_cells:.1f}%)")
print(f"  Test:  {len(test_donor_set)} donors, {test_cell_count} cells ({100*test_cell_count/total_cells:.1f}%)")
print(f"  Donor overlap: {len(train_donor_set & test_donor_set)} ✅")

# Step 3: Build index arrays
train_mask = adata.obs["donor_id"].isin(train_donor_set).values
test_mask = ~train_mask
train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]

# Step 4: Verify
print(f"\nFinal split:")
print(f"  Train: {len(train_idx)} cells from {len(train_donor_set)} donors")
print(f"  Test:  {len(test_idx)} cells from {len(test_donor_set)} donors")
print(f"  Donor-disjoint: {len(train_donor_set & test_donor_set) == 0} ✅")
print(f"  Cell-disjoint: {len(set(train_idx) & set(test_idx)) == 0} ✅")

ct = adata.obs["cell_type"]
print(f"\nCell type coverage:")
all_types = sorted(ct.unique())
missing_in_test = []
for name in all_types:
    tc = (ct.iloc[train_idx] == name).sum()
    tec = (ct.iloc[test_idx] == name).sum()
    flag = "" if tec > 0 else " ⚠️ MISSING"
    if tec == 0:
        missing_in_test.append(name)
    print(f"  {name:<45} train={tc:>6}  test={tec:>5}{flag}")

if missing_in_test:
    print(f"\n⚠️ Cell types missing from test: {missing_in_test}")

# Step 5: Write
print(f"\nWriting train ({len(train_idx)} cells) → {TRAIN_OUT.name}")
adata[train_idx].write(str(TRAIN_OUT))
print(f"Writing test ({len(test_idx)} cells) → {TEST_OUT.name}")
adata[test_idx].write(str(TEST_OUT))

for f in [TRAIN_OUT, TEST_OUT]:
    sz = os.path.getsize(str(f)) / 1e9
    print(f"  {f.name}: {sz:.2f} GB")

# Step 6: Summary
print(f"\n{'=' * 60}")
print(f"  DONE")
print(f"  Train: {len(train_idx)} cells from {len(train_donor_set)} donors")
print(f"  Test:  {len(test_idx)} cells from {len(test_donor_set)} donors")
print(f"  Target test fraction: {TEST_FRACTION} → actual: {100*len(test_idx)/total_cells:.1f}%")
print(f"  Donor overlap: 0 ✅")
print(f"  Old files preserved: {TRAIN_SRC.name}, {TEST_SRC.name}")
print(f"  New files: {TRAIN_OUT.name}, {TEST_OUT.name}")
print(f"{'=' * 60}")
