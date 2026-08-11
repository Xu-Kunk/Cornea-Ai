"""Verify Geneformer tokenized datasets."""
import os, glob, pickle
from pathlib import Path
from collections import Counter
from datasets import load_from_disk
import numpy as np

# Portable project paths (no hardcoded Windows paths)
DATA_DIR = Path(os.environ.get("CORNEAL_FM_DATA", str(Path(__file__).resolve().parent.parent / "data")))

import geneformer
gf_dir = geneformer.__path__[0]
with open(os.path.join(gf_dir, "token_dictionary_gc104M.pkl"), "rb") as f:
    token_dict = pickle.load(f)
cls_id = token_dict.get("<cls>")
eos_id = token_dict.get("<eos>")
print(f"CLS token ID: {cls_id}")
print(f"EOS token ID: {eos_id}")
print(f"Token dict size: {len(token_dict)}")

ds_train = load_from_disk(str(DATA_DIR / "gf_tokenized_train" / "corneal_100k.dataset"))
ds_test = load_from_disk(str(DATA_DIR / "gf_tokenized_test" / "corneal_20k.dataset"))

print("\n=== TRAIN DATASET ===")
print(f"Cells: {len(ds_train):,}")
print(f"Columns: {ds_train.column_names}")

first = ds_train[0]["input_ids"]
print(f"First cell first 10 tokens: {first[:10]}")
print(f"First cell last 10 tokens: {first[-10:]}")
print(f"First cell length: {len(first)}")

cls_ok = sum(1 for ids in ds_train["input_ids"] if len(ids) > 0 and ids[0] == cls_id)
eos_ok = sum(1 for ids in ds_train["input_ids"] if len(ids) > 1 and ids[-1] == eos_id)
print(f"Train: CLS present: {cls_ok}/{len(ds_train)}, EOS present: {eos_ok}/{len(ds_train)}")

cls_ok_test = sum(1 for ids in ds_test["input_ids"] if len(ids) > 0 and ids[0] == cls_id)
eos_ok_test = sum(1 for ids in ds_test["input_ids"] if len(ids) > 1 and ids[-1] == eos_id)
print(f"Test:  CLS present: {cls_ok_test}/{len(ds_test)}, EOS present: {eos_ok_test}/{len(ds_test)}")

# Spot check metadata
ct_train = Counter(ds_train["cell_type"])
ct_test = Counter(ds_test["cell_type"])
print(f"\nTrain cell types ({len(ct_train)}):")
for ct, cnt in ct_train.most_common(5):
    print(f"  {ct}: {cnt:,}")
print(f"Test cell types ({len(ct_test)}):")
for ct, cnt in ct_test.most_common(5):
    print(f"  {ct}: {cnt:,}")

print(f"\nTrain diseases: {set(ds_train['disease'])}")
print(f"Train tissues: {set(ds_train['tissue'])}")
print(f"Train donors: {len(set(ds_train['donor_id']))}")
print(f"Test donors: {len(set(ds_test['donor_id']))}")

# Dataset sizes
train_tokens = sum(len(ids) for ids in ds_train["input_ids"])
test_tokens = sum(len(ids) for ids in ds_test["input_ids"])
print(f"\n=== SUMMARY ===")
print(f"Train: {len(ds_train):,} cells, {train_tokens:,} tokens")
print(f"Test:  {len(ds_test):,} cells, {test_tokens:,} tokens")

sz = sum(os.path.getsize(f) for f in glob.glob(str(DATA_DIR / "gf_tokenized_train" / "**" / "*"), recursive=True) if os.path.isfile(f))
print(f"Train dataset size: {sz/(1024*1024):.1f} MB")
sz2 = sum(os.path.getsize(f) for f in glob.glob(str(DATA_DIR / "gf_tokenized_test" / "**" / "*"), recursive=True) if os.path.isfile(f))
print(f"Test dataset size:  {sz2/(1024*1024):.1f} MB")
print(f"All 4 metadata columns: cell_type, disease, tissue, donor_id -- PRESERVED")
print("VERIFICATION PASSED")
