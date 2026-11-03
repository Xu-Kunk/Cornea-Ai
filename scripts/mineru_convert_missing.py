#!/usr/bin/env python3
"""Convert reference PDFs that lack a paired Markdown file.

Reads every ``*.pdf`` under ``docs/SCI-reference/``, detects those without a
sibling ``*.md`` of the same stem, and converts them via the MinerU cloud API
(``model_version="vlm"`` with OCR, formula, and table extraction), writing each
Markdown next to its source PDF. Tracks results in ``_progress.json``.

The converter runs each missing PDF as its own batch so a single slow or
oversized file does not block the rest.

Usage:
    MINERU_TOKEN=<token> python scripts/mineru_convert_missing.py [--dry-run]

Environment:
    MINERU_TOKEN   required; MinerU API bearer token

Note (2026-10-19): the converted markdown was merged into the topic folders
(01-scFM/ ... 08-Unclassified/) in 2026-09; OUT_DIR is a legacy target and this
script is retained for provenance only.
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

API_BASE = "https://mineru.net/api/v4"
PDF_DIR = Path("docs/SCI-reference")
PROGRESS_FILE = PDF_DIR / "_progress.json"
TIMEOUT_S = 3600
POLL_INTERVAL_S = 10
UPLOAD_TIMEOUT_S = 600

TOKEN = os.environ.get("MINERU_TOKEN")
if not TOKEN:
    raise RuntimeError("请先设置环境变量 MINERU_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}


def find_missing_pdfs() -> list[Path]:
    """Return sorted PDFs that have no same-stem .md sibling."""
    missing = []
    for pdf in sorted(PDF_DIR.rglob("*.pdf")):
        md = pdf.with_suffix(".md")
        if not md.exists():
            missing.append(pdf)
    return missing


def load_progress() -> dict:
    """Load the progress tracker, defaulting to an empty record."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"completed": {}, "failed": {}, "pending": []}


def save_progress(progress: dict) -> None:
    """Persist the progress tracker."""
    PROGRESS_FILE.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_upload_urls(file_name: str) -> dict:
    """Request a presigned upload URL for one file."""
    resp = requests.post(
        f"{API_BASE}/file-urls/batch",
        headers=HEADERS,
        json={
            "files": [{"name": file_name}],
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
    return data["data"]


def upload_file(presigned_url: str, file_path: Path) -> None:
    """Upload a PDF to the presigned URL.

    Content-Type must not be set: the OSS presigned signature does not include
    it, and adding one triggers 403 SignatureDoesNotMatch.
    """
    with open(file_path, "rb") as f:
        resp = requests.put(presigned_url, data=f, timeout=UPLOAD_TIMEOUT_S)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"upload HTTP {resp.status_code}")


def poll_batch(batch_id: str, n_files: int) -> list[dict]:
    """Poll a batch until every file reaches a terminal state."""
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        resp = requests.get(
            f"{API_BASE}/extract-results/batch/{batch_id}",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            time.sleep(POLL_INTERVAL_S)
            continue
        results = data["data"].get("extract_result", [])
        states = [r.get("state", "unknown") for r in results]
        done_count = sum(1 for s in states if s in ("done", "failed"))
        if done_count >= n_files:
            return results
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"polling timeout for batch {batch_id}")


def download_and_extract_md(zip_url: str, out_path: Path) -> None:
    """Download a result zip and write its largest Markdown file."""
    resp = requests.get(zip_url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        md_files = [n for n in zf.namelist() if n.endswith(".md")]
        if not md_files:
            raise RuntimeError("no .md file in result zip")
        md_files.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        content = zf.read(md_files[0]).decode("utf-8", errors="replace")
        out_path.write_text(content, encoding="utf-8")


def convert_one(pdf: Path) -> str:
    """Convert one PDF to Markdown, returning the output path as text."""
    batch_data = get_upload_urls(pdf.name)
    batch_id = batch_data["batch_id"]
    file_urls = batch_data["file_urls"]
    upload_file(file_urls[0], pdf)
    results = poll_batch(batch_id, 1)
    result = results[0]
    if result.get("state") != "done" or not result.get("full_zip_url"):
        raise RuntimeError(result.get("err_msg", f"state={result.get('state')}"))
    out_path = pdf.with_suffix(".md")
    download_and_extract_md(result["full_zip_url"], out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MinerU convert missing PDFs")
    parser.add_argument("--dry-run", action="store_true", help="only list missing PDFs")
    args = parser.parse_args()

    missing = find_missing_pdfs()
    if not missing:
        print("No PDFs missing a Markdown sibling.")
        return
    print(f"{len(missing)} PDF(s) missing Markdown:")
    for pdf in missing:
        print(f"  {pdf}")

    if args.dry_run:
        return

    progress = load_progress()
    for pdf in missing:
        print(f"\n=== Converting: {pdf.name[:70]} ===")
        try:
            out = convert_one(pdf)
        except Exception as exc:  # record and continue with remaining files
            print(f"  FAILED: {exc}")
            progress["failed"][pdf.name] = str(exc)
        else:
            print(f"  OK -> {out}")
            progress["completed"][pdf.name] = out
        save_progress(progress)
        time.sleep(2)

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
