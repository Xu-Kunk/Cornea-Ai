"""
Convert GSE147424 (Wang 2020 epidermis) to h5ad.
Nature Protocols s41596-025-01220-1 compliant preprocessing.
"""
import gzip, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data"))) / "epidermis" / "wang2020_epidermis" / "extracted"
OUTPUT = Path(os.environ.get("CORNEAL_FM_DATA", str(PROJECT_ROOT / "data"))) / "epidermis" / "wang2020_epidermis_wang2020.h5ad"

def load_sample(path: Path) -> pd.DataFrame:
    """Load a single gzipped count matrix."""
    with gzip.open(path, 'rt') as f:
        df = pd.read_csv(f, index_col=0)
    # Drop the empty first column name
    df.index.name = 'gene'
    return df

def main():
    print("Loading 17 samples...")
    samples = sorted(DATA_DIR.glob("GSM*.txt.gz"))
    dfs = []
    for p in samples:
        print(f"  {p.name}...")
        df = load_sample(p)
        dfs.append(df)

    # Combine: genes as rows, cells as columns
    # All samples share same genes, just different cells
    combined = pd.concat(dfs, axis=1)
    print(f"Combined: {combined.shape[0]} genes × {combined.shape[1]} cells")

    # Create AnnData (genes × cells → cells × genes)
    adata = sc.AnnData(
        X=sparse.csr_matrix(combined.T.values),
        var=pd.DataFrame(index=combined.index),
        obs=pd.DataFrame(index=combined.columns),
    )
    adata.var_names = combined.index.tolist()

    # Annotate epidermal layers based on marker genes
    print("\nAnnotating cell types...")
    basal_markers = ['KRT14', 'KRT5']
    spinous_markers = ['KRT1', 'KRT10']
    granular_markers = ['IVL', 'FLG']

    cell_types = []
    for i in range(adata.n_obs):
        row = adata.X[i].toarray().flatten()
        basal = np.mean([row[adata.var_names.get_loc(g)] for g in basal_markers if g in adata.var_names])
        spinous = np.mean([row[adata.var_names.get_loc(g)] for g in spinous_markers if g in adata.var_names])
        granular = np.mean([row[adata.var_names.get_loc(g)] for g in granular_markers if g in adata.var_names])
        mx = max(basal, spinous, granular)
        if mx == basal and basal > 0:
            cell_types.append('Basal')
        elif mx == spinous and spinous > 0:
            cell_types.append('Spinous')
        elif mx == granular and granular > 0:
            cell_types.append('Granular')
        else:
            cell_types.append('Undefined')

    adata.obs['cell_type'] = cell_types
    adata.obs['layer'] = adata.obs['cell_type'].map({
        'Basal': 'basal',
        'Spinous': 'spinous',
        'Granular': 'granular',
    }).fillna('undefined')

    # Basic QC
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"\nAfter QC: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

    # Save
    adata.write(OUTPUT)
    print(f"\nSaved: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size/(1024**2):.1f}MB")

if __name__ == "__main__":
    main()
