"""
Download epidermis scRNA-seq from GEO (accessible in China).
Uses NCBI GEO FTP + E-utils — no VPN needed.

Datasets:
  GSE147424 — Wang 2020 epidermis (246 cit., KRT14/KRT10 gradient)
"""
import os, sys, gzip, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "epidermis"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

DATASETS = {
    "wang2020_epidermis": {
        "geo_id": "GSE147424",
        "title": "Wang 2020 — Human epidermis scRNA-seq (246 cit.)",
        "ftp_path": "GSE147nnn/GSE147424/suppl/",
        "files": ["GSE147424_RAW.tar"],
    },
}

def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "CAFT/1.0"})
    return s

def download_geo(geo_id: str, ftp_path: str, fname: str, dest_dir: Path) -> bool:
    """Download a file from GEO FTP."""
    url = f"{GEO_FTP}/{ftp_path}{fname}"
    dest = dest_dir / fname
    tmp = Path(str(dest) + ".part")

    s = get_session()
    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"

    try:
        r = s.get(url, headers=headers, stream=True, timeout=(30, 7200))
        if r.status_code not in (200, 206):
            print(f"  HTTP {r.status_code} for {fname}")
            return False

        total = int(r.headers.get("content-length", 0))
        mode = "ab" if r.status_code == 206 else "wb"
        dl = tmp.stat().st_size if tmp.exists() else 0

        with open(tmp, mode) as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
                    dl += len(chunk)
                    if total:
                        mb = dl / (1024**2)
                        total_mb = total / (1024**2)
                        pct = dl / total * 100
                        if int(pct) % 20 == 0:
                            print(f"  {fname}: {mb:.0f}/{total_mb:.0f}MB ({pct:.0f}%)")

        tmp.rename(dest)
        print(f"  {fname}: {dest.stat().st_size/(1024**2):.0f}MB ✓")
        return True
    except Exception as e:
        print(f"  {fname} error: {e}")
        return False

def download_all():
    """Parallel download of all GEO datasets."""
    tasks = []
    for key, info in DATASETS.items():
        dest_dir = DATA_DIR / key
        dest_dir.mkdir(exist_ok=True)
        for fname in info["files"]:
            tasks.append((info["geo_id"], info["ftp_path"], fname, dest_dir))

    print(f"Downloading {len(tasks)} files from GEO...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(download_geo, gid, path, fname, ddir): fname
            for gid, path, fname, ddir in tasks
        }
        for f in futures:
            f.result()

if __name__ == "__main__":
    download_all()
    print(f"\nFiles in {DATA_DIR}:")
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(DATA_DIR)} ({f.stat().st_size/1024**2:.0f}MB)")
