#!/usr/bin/env python
"""Make each epi4-CE eval record self-contained and citable (additive only).

The evaluation record already carries ``per_class`` (precision/recall/F1/support
per class) and ``confusion`` — both are emitted by
``corneal_fm.evaluation.evaluate_on_external_test`` inside the same call that
produces the primary endpoint. This script adds, without touching any computed
value:

  * ``notes``: an explicit sentence naming the source of ``per_class`` /
    ``confusion`` and the archived per-cell arrays that back them;
  * ``archived_inputs``: paths + sha256 of the per-cell npz and of the
    per-seed checkpoint, so a cited number (e.g. TAC F1 = 0.0) is traceable;
  * ``record_sha256_at_enrichment``: the hash of the file *before* enrichment.

Every file is verified key-by-key (all pre-existing values identical) and a
ledger (``provenance_enrichment.json``) records before/after hashes per file.

Usage:
    python scripts/epi4_ce_enrich_provenance.py --variant-dir results/paper1_vanzyl/epi4_ce/drop_nonepi
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 123, 456)
PER_CLASS_NOTE = (
    "per_class and confusion in this record are produced by "
    "corneal_fm.evaluation.evaluate_on_external_test from the same predictions "
    "as the primary endpoint; the per-cell predictions, labels and CLS "
    "embeddings behind them are archived in the per-cell npz listed under "
    "archived_inputs (y_true_ext / y_pred_ext / probs_ext / emb_ext / donor_ext)."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative if possible, else the absolute path (external test dirs)."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _diff_keys(old: dict, new: dict) -> list[str]:
    """Keys whose values changed (must be empty for a pure enrichment)."""
    changed = []
    for key in old:
        if old[key] != new.get(key):
            changed.append(key)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant-dir", required=True)
    args = ap.parse_args()
    vdir = Path(args.variant_dir)
    if not vdir.is_absolute():
        vdir = PROJECT_ROOT / vdir

    ledger = {"generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
              "variant_dir": _rel(vdir),
              "additive_only": True, "files": []}

    for seed in SEEDS:
        rec = vdir / f"seed{seed}" / "eval_gse153515.json"
        npz = vdir / f"seed{seed}" / f"percell_epi4_{seed}.npz"
        if not rec.is_file():
            print(f"skip seed {seed}: no record")
            continue
        before = json.loads(rec.read_text())
        sha_before = _sha256(rec)
        ckpt = Path(before.get("model", {}).get("checkpoint", ""))
        ckpt_files = {}
        if ckpt.is_dir():
            for name in ("model.safetensors", "config.json"):
                f = ckpt / name
                if f.is_file():
                    ckpt_files[name] = {"path": str(f), "sha256": _sha256(f)}

        enriched = json.loads(json.dumps(before))  # deep copy
        notes = enriched.get("notes", "")
        if PER_CLASS_NOTE not in notes:
            enriched["notes"] = (notes + " " if notes else "") + PER_CLASS_NOTE
        enriched["archived_inputs"] = {
            "percell": {
                "path": _rel(npz) if npz.is_file() else None,
                "sha256": _sha256(npz) if npz.is_file() else None,
            },
            "checkpoint": ckpt_files,
            "n_eval_cells": before.get("provenance", {}).get("n_eval_cells"),
            "n_eval_donors": before.get("provenance", {}).get("n_eval_donors"),
        }
        enriched["record_sha256_at_enrichment"] = sha_before

        # ``notes`` is prose and is extended on purpose; every computed value
        # (endpoints, per_class, confusion, provenance, model) must be identical.
        changed = [k for k in _diff_keys(before, enriched) if k != "notes"]
        if changed:
            raise SystemExit(f"seed {seed}: enrichment altered existing keys {changed}")
        notes_extended = before.get("notes") != enriched.get("notes")
        rec.write_text(json.dumps(enriched, indent=2))
        ledger["files"].append({
            "seed": seed,
            "record": _rel(rec),
            "sha256_before": sha_before,
            "sha256_after": _sha256(rec),
            "added_keys": ["archived_inputs", "record_sha256_at_enrichment"],
            "notes_extended": notes_extended,
            "per_class_f1": {k: v["f1"] for k, v in before.get("per_class", {}).items()},
            "confusion_rows": len(before.get("confusion", [])),
        })
        print(f"seed {seed}: enriched (per_class={len(before.get('per_class', {}))} classes, "
              f"confusion present={bool(before.get('confusion'))})")

    (vdir / "provenance_enrichment.json").write_text(json.dumps(ledger, indent=2))
    print(f"ledger -> {vdir / 'provenance_enrichment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
