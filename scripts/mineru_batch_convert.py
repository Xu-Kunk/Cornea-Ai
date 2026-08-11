#!/usr/bin/env python3
"""MinerU API batch PDF→Markdown converter.

Usage:
    python scripts/mineru_batch_convert.py [--batch-size 5] [--poll-interval 10]

Reads all PDFs from docs/SCI-refeference/, uploads via MinerU cloud API,
polls for completion, downloads results, and extracts Markdown files
into docs/SCI-refeference-markdown/.
"""

import argparse
import json
import os
import sys
import time
import zipfile
import io
import shutil
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "https://mineru.net/api/v4"
TOKEN = os.environ.get(
    "MINERU_TOKEN",
    "",  # REDACTED: was a hardcoded MinerU API key; set MINERU_TOKEN instead
)
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

PDF_DIR = Path("docs/SCI-refeference")
OUT_DIR = Path("docs/SCI-refeference-markdown")
PROGRESS_FILE = OUT_DIR / "_progress.json"


def get_pdf_files() -> list[Path]:
    """Return sorted list of PDF files in the reference directory."""
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No PDFs found in {PDF_DIR}")
        sys.exit(1)
    return pdfs


def load_progress() -> dict:
    """Load progress tracker."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": {}, "failed": {}, "pending": []}


def save_progress(progress: dict):
    """Save progress tracker."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


def get_upload_urls(file_names: list[str]) -> dict:
    """Get presigned upload URLs for a batch of files."""
    resp = requests.post(
        f"{API_BASE}/file-urls/batch",
        headers=HEADERS,
        json={
            "files": [{"name": n} for n in file_names],
            "model_version": "vlm",
            "is_ocr": True,
            "enable_formula": True,
            "enable_table": True,
            "language": "en",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"API error: {data}")
    return data["data"]  # {batch_id, file_urls: [...]}


def upload_file(presigned_url: str, file_path: Path) -> bool:
    """Upload a file to a presigned URL via PUT.
    
    NOTE: Do NOT set Content-Type header — the presigned OSS URL signature
    does not include it, and adding one causes 403 SignatureDoesNotMatch.
    """
    try:
        with open(file_path, "rb") as f:
            resp = requests.put(presigned_url, data=f, timeout=300)
        if resp.status_code not in (200, 201):
            print(f"\n  Upload HTTP {resp.status_code} for {file_path.name}")
            return False
        return True
    except Exception as e:
        print(f"\n  Upload error for {file_path.name}: {e}")
        return False


def poll_batch(batch_id: str, n_files: int, poll_interval: int = 10, timeout: int = 1800) -> list[dict]:
    """Poll batch status until all files are done or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(
            f"{API_BASE}/extract-results/batch/{batch_id}",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            print(f"  Poll error: {data.get('msg')}")
            time.sleep(poll_interval)
            continue

        results = data["data"].get("extract_result", [])
        states = [r.get("state", "unknown") for r in results]
        done_count = sum(1 for s in states if s in ("done", "failed"))

        print(f"  [{batch_id[:8]}] {done_count}/{n_files} done", end="\r")

        if done_count >= n_files:
            print()
            return results

        time.sleep(poll_interval)

    print(f"\n  Timeout waiting for batch {batch_id[:8]}")
    # Return whatever we have
    resp = requests.get(f"{API_BASE}/extract-results/batch/{batch_id}", headers=HEADERS, timeout=30)
    return resp.json().get("data", {}).get("extract_result", [])


def download_and_extract_md(zip_url: str, out_path: Path) -> bool:
    """Download result zip and extract the main markdown file."""
    try:
        resp = requests.get(zip_url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find the main markdown file (usually named like the PDF with .md)
            md_files = [n for n in zf.namelist() if n.endswith(".md")]
            if not md_files:
                print(f"  No .md file in zip for {out_path.stem}")
                return False
            # Pick the largest .md file (usually the full content)
            md_files.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
            content = zf.read(md_files[0]).decode("utf-8", errors="replace")
            out_path.write_text(content, encoding="utf-8")
            return True
    except Exception as e:
        print(f"  Download/extract failed for {out_path.stem}: {e}")
        return False


def process_batch(batch_pdfs: list[Path], batch_idx: int, poll_interval: int) -> dict:
    """Process a batch of PDFs: upload → poll → download markdown."""
    names = [p.name for p in batch_pdfs]
    print(f"\n=== Batch {batch_idx + 1}: {len(names)} files ===")

    # Step 1: Get presigned URLs
    try:
        batch_data = get_upload_urls(names)
    except Exception as e:
        print(f"  Failed to get upload URLs: {e}")
        return {"completed": {}, "failed": {n: str(e) for n in names}}

    batch_id = batch_data["batch_id"]
    file_urls = batch_data["file_urls"]
    print(f"  Batch ID: {batch_id[:12]}...")

    # Step 2: Upload files
    completed = {}
    failed = {}
    for pdf_path, url in zip(batch_pdfs, file_urls):
        print(f"  Uploading: {pdf_path.name[:60]}...", end=" ")
        if upload_file(url, pdf_path):
            print("OK")
        else:
            print("FAILED")
            failed[pdf_path.name] = "upload_failed"

    if not any(True for p in batch_pdfs if p.name not in failed):
        return {"completed": completed, "failed": failed}

    # Step 3: Poll for completion
    uploaded_count = len(batch_pdfs) - len(failed)
    print(f"  Waiting for {uploaded_count} files to process...")
    results = poll_batch(batch_id, len(batch_pdfs), poll_interval=poll_interval)

    # Step 4: Download markdown
    for result in results:
        fname = result.get("file_name", "")
        state = result.get("state", "unknown")
        zip_url = result.get("full_zip_url", "")

        if state == "done" and zip_url:
            out_path = OUT_DIR / (Path(fname).stem + ".md")
            print(f"  Downloading: {fname[:60]}...", end=" ")
            if download_and_extract_md(zip_url, out_path):
                print("OK")
                completed[fname] = str(out_path)
            else:
                print("FAILED")
                failed[fname] = "download_failed"
        elif state == "failed":
            err = result.get("err_msg", "unknown error")
            print(f"  FAILED: {fname[:50]} — {err}")
            failed[fname] = err
        elif fname not in failed:
            failed[fname] = f"state={state}"

    return {"completed": completed, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="MinerU batch PDF→Markdown")
    parser.add_argument("--batch-size", type=int, default=5, help="Files per batch")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between polls")
    parser.add_argument("--resume", action="store_true", help="Skip already completed files")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pdfs = get_pdf_files()
    progress = load_progress()

    if args.resume:
        done_names = set(progress["completed"].keys())
        all_pdfs = [p for p in all_pdfs if p.name not in done_names]
        print(f"Resuming: {len(done_names)} already done, {len(all_pdfs)} remaining")

    print(f"Total PDFs to process: {len(all_pdfs)}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Batch size: {args.batch_size}")

    # Split into batches
    batches = [all_pdfs[i : i + args.batch_size] for i in range(0, len(all_pdfs), args.batch_size)]

    total_completed = dict(progress["completed"])
    total_failed = dict(progress["failed"])

    for i, batch in enumerate(batches):
        result = process_batch(batch, i, args.poll_interval)
        total_completed.update(result["completed"])
        total_failed.update(result["failed"])

        # Save progress after each batch
        progress["completed"] = total_completed
        progress["failed"] = total_failed
        save_progress(progress)

        print(f"\n  Progress: {len(total_completed)} completed, {len(total_failed)} failed")

        # Pause between batches to respect rate limits (429 observed at <3s)
        if i < len(batches) - 1:
            time.sleep(5)

    # Final summary
    print("\n" + "=" * 60)
    print(f"DONE: {len(total_completed)} converted, {len(total_failed)} failed")
    if total_failed:
        print("\nFailed files:")
        for name, err in total_failed.items():
            print(f"  - {name}: {err}")
    print(f"\nMarkdown output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
