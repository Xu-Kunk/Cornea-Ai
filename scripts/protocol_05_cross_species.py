"""
Cross-species data assembly for Corneal-FM v4.1 E3 validation.
Handles: Rat DED (OMIX008264, local count.txt), Macaque (GSE276184, GEO MTX).

China network: Uses multiple GEO mirrors with retry + fallback.
"""
import os, sys, gzip, tarfile, io, time, logging
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CORNEAL_FM_DATA", str(ROOT / "data")))
RAT_SRC = Path(os.environ.get("DED_SEQ_DATA", str(ROOT.parent / "DED-seq"))) / "data" / "rat_scRNA" / "data"

# --- Configuration ---
RAT_FILES = {
    "C1": "control", "C2": "control", "C3": "control",
    "DE1": "DED", "DE2": "DED", "DE3": "DED",
}
N_RAT_GENES = 19581  # from C1.count.txt line count

# GEO mirrors for China network (try in order)
GEO_MIRRORS = [
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE276nnn/GSE276184/suppl/GSE276184_RAW.tar",
]
GEO_FALLBACK = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE276184&format=file"


# ============================================================
# RAT DED: Assemble count.txt → h5ad
# ============================================================
def assemble_rat_ded():
    """Assemble 6 rat count.txt files into a single AnnData with condition labels."""
    logger.info("=" * 50)
    logger.info("RAT DED: Assembling 6 count.txt → h5ad")
    logger.info("=" * 50)

    all_data = []
    all_barcodes = []
    all_conditions = []
    gene_names = None

    for fname, condition in RAT_FILES.items():
        fpath = RAT_SRC / f"{fname}.count.txt"
        if not fpath.exists():
            logger.error(f"  MISSING: {fpath}")
            continue

        logger.info(f"  Loading {fname} ({condition})...")
        # Count files: tab-separated, genes as rows (column 0), cells as columns
        df = pd.read_csv(fpath, sep='\t', index_col=0)
        
        if gene_names is None:
            gene_names = df.index.tolist()
            logger.info(f"  {len(gene_names)} genes detected")
        
        # Each column is a cell barcode
        n_cells = df.shape[1]
        logger.info(f"  {fname}: {n_cells} cells, {df.shape[0]} genes")
        
        # Convert to sparse matrix (genes × cells)
        mat = csr_matrix(df.values.T)  # cells × genes
        all_data.append(mat)
        
        for bc in df.columns:
            all_barcodes.append(f"{fname}_{bc}")
            all_conditions.append(condition)

    # Concatenate all samples
    X = csr_matrix(np.vstack([m.toarray() for m in all_data]))
    logger.info(f"  Total: {X.shape[0]} cells × {X.shape[1]} genes")

    # Build AnnData
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"condition": all_conditions}, index=all_barcodes),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var_names = gene_names

    # Add raw counts
    adata.raw = adata.copy()

    # Save
    outpath = DATA / "rat_ded.h5ad"
    adata.write(outpath)
    logger.info(f"  Saved: {outpath} ({outpath.stat().st_size/1e6:.1f} MB)")
    
    # Stats
    for cond in ["control", "DED"]:
        n = (adata.obs["condition"] == cond).sum()
        logger.info(f"  {cond}: {n} cells")
    
    return adata


# ============================================================
# MACAQUE: Download GSE276184 MTX → h5ad
# ============================================================
def download_with_retry(url: str, outpath: Path, max_retries: int = 3) -> bool:
    """Download with retry and China-friendly timeouts."""
    import urllib.request
    import ssl

    ctx = ssl._create_unverified_context()
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  Download attempt {attempt}/{max_retries}: {url[:80]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                data = resp.read()
            
            outpath.write_bytes(data)
            logger.info(f"  Downloaded: {len(data)/1e6:.1f} MB")
            return True
        
        except Exception as e:
            logger.warning(f"  Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                logger.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
    
    return False


def assemble_macaque():
    """Download GSE276184 MTX and convert to h5ad."""
    logger.info("=" * 50)
    logger.info("MACAQUE: Downloading GSE276184 MTX → h5ad")
    logger.info("=" * 50)

    tar_path = DATA / "GSE276184_RAW.tar"
    extract_dir = DATA / "GSE276184_extracted"
    extract_dir.mkdir(exist_ok=True)

    # Download
    downloaded = False
    for url in GEO_MIRRORS:
        if download_with_retry(url, tar_path):
            downloaded = True
            break
    
    if not downloaded:
        logger.warning("  GEO FTP failed, trying alternative download...")
        if download_with_retry(GEO_FALLBACK, tar_path, max_retries=3):
            downloaded = True
    
    if not downloaded:
        # Try per-sample download
        logger.info("  Trying per-sample download...")
        samples = [
            "GSM8492741_0d_s1", "GSM8492742_0d_s2",
            "GSM8492743_1d_s1", "GSM8492744_1d_s2",
            "GSM8492745_3d_s1", "GSM8492746_3d_s2",
        ]
        
        all_adatas = []
        for sample in samples:
            sample_dir = extract_dir / sample
            sample_dir.mkdir(exist_ok=True)
            
            base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8492nnn/{sample}/suppl"
            files_downloaded = 0
            
            for ftype in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]:
                fname = f"{sample}_{ftype}"
                url = f"{base_url}/{fname}"
                fpath = sample_dir / fname
                
                if download_with_retry(url, fpath, max_retries=2):
                    files_downloaded += 1
            
            if files_downloaded == 3:
                logger.info(f"  Loading {sample}...")
                try:
                    adata = sc.read_10x_mtx(sample_dir, var_names='gene_symbols', cache=False)
                    adata.obs["sample"] = sample
                    adata.obs["timepoint"] = sample.split("_")[1]  # 0d, 1d, 3d
                    all_adatas.append(adata)
                except Exception as e:
                    logger.warning(f"  Failed to load {sample}: {e}")
        
        if all_adatas:
            adata = ad.concat(all_adatas, join='outer')
        else:
            logger.error("  ALL download methods failed. Manual download needed.")
            logger.error(f"  URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE276184")
            return None
    else:
        # Extract and load from tar
        logger.info("  Extracting tar...")
        with tarfile.open(tar_path, 'r') as tf:
            tf.extractall(extract_dir)
        
        # Load individual samples from extracted MTX dirs
        all_adatas = []
        for sample_dir in sorted(extract_dir.glob("GSM*")):
            if sample_dir.is_dir():
                try:
                    adata = sc.read_10x_mtx(sample_dir, var_names='gene_symbols', cache=False)
                    sample_name = sample_dir.name.split("_")[0] + "_" + sample_dir.name.split("_")[1] + "_" + sample_dir.name.split("_")[2]
                    # Parse timepoint from filename
                    for part in sample_dir.name.split("_"):
                        if part in ["0d", "1d", "3d"]:
                            adata.obs["timepoint"] = part
                            break
                    adata.obs["sample"] = sample_dir.name
                    all_adatas.append(adata)
                    logger.info(f"  Loaded {sample_dir.name}: {adata.n_obs} cells")
                except Exception as e:
                    logger.warning(f"  Failed {sample_dir.name}: {e}")
        
        if not all_adatas:
            logger.error("  No samples loaded from tar")
            return None
        
        adata = ad.concat(all_adatas, join='outer')

    # Keep only uninjured samples for cross-species (0d = normal)
    if "timepoint" in adata.obs.columns:
        adata_normal = adata[adata.obs["timepoint"] == "0d"].copy()
    else:
        adata_normal = adata.copy()
    
    logger.info(f"  Total macaque cells: {adata.n_obs}")
    logger.info(f"  Normal (0d) cells: {adata_normal.n_obs}")

    # Save raw counts
    adata_normal.raw = adata_normal.copy()

    outpath = DATA / "macaque_cornea.h5ad"
    adata_normal.write(outpath)
    logger.info(f"  Saved: {outpath} ({outpath.stat().st_size/1e6:.1f} MB)")
    
    return adata_normal


# ============================================================
# VERIFY
# ============================================================
def verify_all():
    """Verify all cross-species data files."""
    logger.info("=" * 50)
    logger.info("VERIFICATION")
    logger.info("=" * 50)

    files = {
        "Human train": DATA / "corneal_100k_subsample.h5ad",
        "Human test": DATA / "corneal_20k_test.h5ad",
        "Mouse DED": DATA / "mouse_ded.h5ad",
        "Rat DED": DATA / "rat_ded.h5ad",
        "Macaque": DATA / "macaque_cornea.h5ad",
        "Orthologs": DATA / "mouse_human_orthologs.csv",
    }

    all_ok = True
    for name, fpath in files.items():
        if fpath.exists():
            sz = fpath.stat().st_size
            if fpath.suffix == '.h5ad':
                a = sc.read_h5ad(fpath, backed='r')
                logger.info(f"  ✅ {name}: {a.n_obs:,} cells × {a.n_vars:,} genes ({sz/1e6:.1f} MB)")
            else:
                logger.info(f"  ✅ {name}: ({sz/1e6:.1f} MB)")
        else:
            logger.warning(f"  ❌ {name}: MISSING")
            all_ok = False

    return all_ok


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger.info("Cross-Species Data Assembly for Corneal-FM v4.1")
    logger.info(f"Root: {ROOT}")
    logger.info("")

    # Rat (local)
    assemble_rat_ded()

    # Macaque (download)
    assemble_macaque()

    # Verify
    ok = verify_all()
    logger.info(f"\n{'✅ All data ready' if ok else '❌ Some data missing — check logs above'}")
