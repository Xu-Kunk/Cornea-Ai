"""
CELLxGENE epidermis download — optimized for China network.
Uses Census REST API + HTTP fallback with chunked transfer + resume.

Reference: Nature Protocols s41596-025-01220-1 (scGPT fine-tune protocol)

Usage:
    python scripts/protocol_12_download_epidermis.py --source tabula_sapiens
    python scripts/protocol_12_download_epidermis.py --all
"""
import argparse, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "epidermis"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_API = "https://census.cellxgene.cziscience.com"

def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "CAFT/1.0"})
    return s

def download(url: str, dest: Path, desc: str = "") -> bool:
    tmp = Path(str(dest) + ".part")
    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
    try:
        s = get_session()
        r = s.get(url, headers=headers, stream=True, timeout=(30, 7200))
        if r.status_code not in (200, 206):
            print(f"  HTTP {r.status_code}")
            return False
        total = int(r.headers.get("content-length", 0))
        mode = "ab" if r.status_code == 206 else "wb"
        dl = tmp.stat().st_size if tmp.exists() else 0
        with open(tmp, mode) as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
                    dl += len(chunk)
        tmp.rename(dest)
        mb = dest.stat().st_size / (1024*1024)
        print(f"  {desc}: {mb:.0f}MB saved")
        return True
    except Exception as e:
        print(f"  {desc} error: {e}")
        return False

def try_census():
    """Try downloading via CELLxGENE Census API."""
    print("\n=== Tabula Sapiens via Census API ===")
    import subprocess
    # Try pip install first
    mirrors = [
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple/",
    ]
    for mirror in mirrors:
        print(f"  Trying pip install from {mirror}...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "cellxgene-census", "scanpy",
             "-i", mirror, "--trusted-host", mirror.split("//")[1].split("/")[0]],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode == 0:
            print("  Install OK")
            break
    else:
        print("  pip install failed. Use manual download instead.")
        return False

    try:
        import cellxgene_census
        import scanpy as sc
        census = cellxgene_census.open_soma(census_version="latest")
        adata = cellxgene_census.get_anndata(
            census=census, organism="Homo sapiens",
            obs_value_filter="tissue_general == 'skin'",
        )
        out = DATA_DIR / "tabula_sapiens_skin.h5ad"
        adata.write(out)
        print(f"  {adata.n_obs} skin cells -> {out}")
        return True
    except Exception as e:
        print(f"  Census failed: {e}")
        return False

def manual_download():
    """Manual download instructions for China network."""
    print("\n=== Manual Download Guide ===")
    print("CELLxGENE Explorer (browser, may need VPN):")
    print("  Tabula Sapiens:")
    print("    https://cellxgene.cziscience.com/collections/e5f58829")
    print("    -> Click Download -> h5ad format")
    print()
    print("  Ganier 2024 Skin (PNAS):")
    print("    https://cellxgene.cziscience.com/collections/c353707f")
    print("    -> Click Download -> h5ad format")
    print()
    print("Alternative — GEO (no VPN needed):")
    print("  Wang 2020: GSE147424")
    print("    wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147424/suppl/")
    print()
    print("SkinMap (open access):")
    print("  https://www.skinmap.io/studies")
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tabula_sapiens")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    success = try_census()
    if not success:
        manual_download()

    print(f"\nFiles in {DATA_DIR}:")
    for f in sorted(DATA_DIR.glob("*")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
