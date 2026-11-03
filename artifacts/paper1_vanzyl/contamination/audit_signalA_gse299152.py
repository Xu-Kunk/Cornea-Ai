#!/usr/bin/env python3
"""scContam-style Signal A cell-level contamination check: GSE299152 vs Genecorpus-104M.

Reuses the exact constants/implementation of the 2026-08-05 audit
(artifacts/paper1_data_audit/audit_contam.py): vectorized MinHash (m=128,
bands=16, rows=8, K=2048, prime 2^61-1) + banded LSH + exact-Jaccard candidate
verification, with positive (inject 200 query cells) and negative (200 random
fingerprints) controls. Corpus reference = the same bounded sample of 7
Genecorpus-104M shards (0.54% of the ~104M-cell corpus) cached locally.

Purpose: GSE299152 is the second external test set; the study-level NOT-IN
verdict rests on (a) GEO public date 2025-11-18, (b) absence of its paper DOI
(10.1093/pnasnexus/pgaf348, PNAS Nexus 2025) from the Genecorpus-104M
provenance table (NatCS 2026 MOESM3), and (c) absence from CELLxGENE census
2023-05-15 (scGPT). This run adds an independent cell-level negative signal.

Caveat (as in the 2026-08-05 run): a 0.54% corpus sample means absence from
the sample is NOT proof of absence from the full corpus; the decisive evidence
remains the provenance table + temporal cutoff. The value is corroboration and
symmetric methodology with the GSE153515 audit.
"""
from __future__ import annotations

import gc
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.ipc


def wait_for_memory(min_gb: float = 2.5, max_wait_s: int = 600) -> None:
    """Pre-flight: wait until the shared box has >= min_gb available memory."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        avail = float(out.splitlines()[1].split()[6])  # free -g: Mem row, 'available'
        if avail >= min_gb:
            print(f"  memory OK: {avail:.0f} GB available (waited {time.time()-t0:.0f}s)", flush=True)
            return
        print(f"  memory low ({avail:.0f} GB avail), waiting 30s ...", flush=True)
        time.sleep(30)
    print("  WARNING: max_wait reached with low memory; proceeding anyway", flush=True)

OUT = Path("/home/dq/Codex/Ai/artifacts/paper1_vanzyl/contamination")
SHARD_DIR = Path("/home/dq/Codex/Ai/artifacts/paper1_data_audit/corpus_shards")
DATA_GSE = Path("/home/dq/Codex/Ai/data/gse299152/gse299152_annotated.h5ad")
GENEFORMER_PKG = Path("/home/dq/venv-ds/lib/python3.12/site-packages/geneformer")

REPO = "theodoris-lab/Genecorpus-104M"
N_SHARDS_TOTAL = 1202
SHARDS = [0, 600, 1199]  # reduced sample (3 shards, ~0.23% of corpus) to fit shared-box memory
TOP_K = 2048
M_PERM = 128
BANDS = 16
ROWS = 8
PRIME = (1 << 61) - 1
SEED = 42
SPECIAL_TOKENS = {0, 1, 2, 3}


def corpus_fingerprints() -> tuple[list[np.ndarray], int, dict]:
    """Extract per-cell fingerprints (top-K rank-encoded token IDs, specials stripped)."""
    fingerprints: list[np.ndarray] = []
    stats = {"n_cells": 0, "n_tokens_total": 0, "n_short_cells": 0, "n_skipped": 0}
    for shard in SHARDS:
        path = SHARD_DIR / f"data-{shard:05d}-of-{N_SHARDS_TOTAL:05d}.arrow"
        if not path.exists() or path.stat().st_size < 1e8:
            print(f"  shard {shard}: NOT AVAILABLE, skipping", flush=True)
            continue
        t0 = time.time()
        try:
            with pyarrow.ipc.open_stream(path) as reader:
                table = reader.read_all()
        except pyarrow.lib.ArrowInvalid:
            with pyarrow.ipc.open_file(path) as reader:
                table = reader.read_all()
        col = table.column("input_ids")
        n = col.length()
        for i in range(n):
            arr = [t for t in col[i].as_py() if t not in SPECIAL_TOKENS][:TOP_K]
            stats["n_tokens_total"] += len(arr)
            if len(arr) < 100:
                stats["n_skipped"] += 1
                continue
            if len(arr) < TOP_K:
                stats["n_short_cells"] += 1
            fingerprints.append(np.asarray(arr, dtype=np.uint16))
        stats["n_cells"] += n
        print(f"  shard {shard}: {n} cells; running total fingerprints {len(fingerprints)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    stats["n_fingerprints"] = len(fingerprints)
    return fingerprints, stats["n_cells"], stats


def minhash_matrix(fps: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    a = rng.integers(1, PRIME, size=M_PERM, dtype=np.int64)
    b = rng.integers(0, PRIME, size=M_PERM, dtype=np.int64)
    H = np.empty((len(fps), M_PERM), dtype=np.int64)
    for i, fp in enumerate(fps):
        t = fp.astype(np.int64)
        hv = (a[:, None] * t[None, :] + b[:, None]) % PRIME
        H[i] = hv.min(axis=1)
    return H


def banded_lsh(H: np.ndarray) -> list[dict[tuple, list[int]]]:
    """Band buckets: return bands[j][bucket] = list of row ids. No band_ids (saves ~2-4 GB)."""
    bands = [dict() for _ in range(BANDS)]
    for j in range(BANDS):
        sl = H[:, j * ROWS:(j + 1) * ROWS]
        keys = [bytes(v.tobytes()) for v in sl]
        for row, k in enumerate(keys):
            bands[j].setdefault(k, []).append(row)
    return bands


def exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=False).size
    union = np.union1d(a, b).size
    return inter / union if union else 0.0


def gse_fingerprints_v2(counts: np.ndarray, var_ensembl: np.ndarray) -> list[np.ndarray]:
    """Replicate Geneformer V2 rank encoding for GSE cells -> token fingerprints."""
    med = pickle.load(open(GENEFORMER_PKG / "gene_median_dictionary_gc104M.pkl", "rb"))
    td = pickle.load(open(GENEFORMER_PKG / "token_dictionary_gc104M.pkl", "rb"))
    meds = np.array([med.get(e, 0.0) for e in var_ensembl], dtype=np.float64)
    tok = np.array([td.get(e, -1) for e in var_ensembl], dtype=np.int64)
    valid = (meds > 0) & (tok >= 4)
    meds_v, tok_v = meds[valid], tok[valid]
    fps = []
    n_nomatch = int((~valid).sum())
    for i in range(counts.shape[0]):
        c = counts[i].astype(np.float64)
        total = c.sum()
        if total <= 0:
            fps.append(np.array([], dtype=np.uint16))
            continue
        frac = c / total
        metric = frac[valid] / meds_v
        order = np.argsort(-metric)[:TOP_K]
        fps.append(tok_v[order].astype(np.uint16))
    return fps, n_nomatch


def gse_fingerprints_raw(counts: np.ndarray, var_ensembl: np.ndarray) -> list[np.ndarray]:
    """scContam cell_fingerprint (top-K by raw expression) -> token fingerprints."""
    td = pickle.load(open(GENEFORMER_PKG / "token_dictionary_gc104M.pkl", "rb"))
    tok = np.array([td.get(e, -1) for e in var_ensembl], dtype=np.int64)
    valid = tok >= 4
    tok_v = tok[valid]
    fps = []
    for i in range(counts.shape[0]):
        c = counts[i].astype(np.float64)[valid]
        order = np.argsort(-c)[:TOP_K]
        fps.append(tok_v[order].astype(np.uint16))
    return fps


def query_all(Hq: np.ndarray, qfps: list[np.ndarray], bands: list[dict],
              ref_fps: list[np.ndarray], ref_ids: list[int]) -> list[dict]:
    out = []
    for q in range(len(qfps)):
        cands: set[int] = set()
        for j in range(BANDS):
            sl = Hq[q, j * ROWS:(j + 1) * ROWS]
            key = bytes(sl.tobytes())
            cands.update(bands[j].get(key, []))
        best = 0.0
        best_id = -1
        for rid in cands:
            jac = exact_jaccard(qfps[q], ref_fps[rid])
            if jac > best:
                best, best_id = jac, rid
        out.append({"query": int(q), "ref_id": int(best_id) if best_id >= 0 else None,
                    "top_jaccard": round(best, 4), "n_candidates": len(cands)})
    return out


def main() -> None:
    t_all = time.time()
    print("STEP 0: memory pre-flight", flush=True)
    wait_for_memory()

    print("STEP 1: corpus fingerprints (cached shards)", flush=True)
    ref_fps, n_corpus_cells, stats = corpus_fingerprints()
    print(f"  corpus cells sampled: {n_corpus_cells}; fingerprints kept: {len(ref_fps)}", flush=True)

    print("STEP 2: GSE299152 fingerprints (V2-replicated + raw)", flush=True)
    import anndata as ad
    gse = ad.read_h5ad(str(DATA_GSE), backed="r")
    cnt = gse.layers["counts"][:]
    counts = cnt.toarray() if hasattr(cnt, "toarray") else np.asarray(cnt)
    var_ensembl = np.asarray(gse.var["gene_ids"], dtype=object)
    gse_fps_v2, n_nomatch = gse_fingerprints_v2(counts, var_ensembl)
    gse_fps_raw = gse_fingerprints_raw(counts, var_ensembl)
    print(f"  GSE299152 cells: {counts.shape[0]}; genes unmapped to V2 vocab: {n_nomatch}", flush=True)

    print("STEP 3: MinHash + banded LSH", flush=True)
    rng = np.random.default_rng(SEED)
    H_ref = minhash_matrix(ref_fps, rng)
    H_gse_v2 = minhash_matrix(gse_fps_v2, rng)
    H_gse_raw = minhash_matrix(gse_fps_raw, rng)
    print(f"  minhash done: ref {H_ref.shape}, gse_v2 {H_gse_v2.shape}", flush=True)
    bands = banded_lsh(H_ref)
    print(f"  LSH built: {len(ref_fps)} refs, {sum(len(b) for b in bands)} buckets", flush=True)
    del H_ref
    gc.collect()

    print("STEP 4: query GSE299152 (V2-replicated fingerprints)", flush=True)
    res_v2 = query_all(H_gse_v2, gse_fps_v2, bands, ref_fps, list(range(len(ref_fps))))
    del H_gse_v2
    gc.collect()
    print("STEP 4b: query GSE299152 (raw-expression fingerprints)", flush=True)
    res_raw = query_all(H_gse_raw, gse_fps_raw, bands, ref_fps, list(range(len(ref_fps))))
    del H_gse_raw
    gc.collect()

    print("STEP 5: positive control (inject 200 query cells into index)", flush=True)
    ctrl_rng = np.random.default_rng(7)
    inject = ctrl_rng.choice(len(gse_fps_v2), size=200, replace=False)
    ctrl_fps = ref_fps + [gse_fps_v2[i] for i in inject]
    H_ctrl = minhash_matrix(ctrl_fps, rng)
    bands_c = banded_lsh(H_ctrl)
    pos = []
    for k, qi in enumerate(inject):
        Hq = H_ctrl[len(ref_fps) + k]
        cands: set[int] = set()
        for j in range(BANDS):
            key = bytes(Hq[j * ROWS:(j + 1) * ROWS].tobytes())
            cands.update(bands_c[j].get(key, []))
        best = max((exact_jaccard(gse_fps_v2[qi], ctrl_fps[rid]) for rid in cands), default=0.0)
        pos.append({"injected": int(qi), "n_candidates": len(cands), "best_jaccard": round(best, 4)})
    del H_ctrl, bands_c, ctrl_fps
    gc.collect()
    pos_jac = [p["best_jaccard"] for p in pos]
    pos_recall_09 = float(np.mean([p["best_jaccard"] >= 0.9 for p in pos]))

    print("STEP 6: negative control (200 random fingerprints)", flush=True)
    neg_fps = [np.sort(rng.choice(np.arange(4, 20275), size=TOP_K, replace=False)).astype(np.uint16)
               for _ in range(200)]
    H_neg = minhash_matrix(neg_fps, rng)
    neg_hits = []
    for q in range(200):
        cands: set[int] = set()
        for j in range(BANDS):
            key = bytes(H_neg[q, j * ROWS:(j + 1) * ROWS].tobytes())
            cands.update(bands[j].get(key, []))
        best = max((exact_jaccard(neg_fps[q], ref_fps[rid]) for rid in cands), default=0.0)
        neg_hits.append({"query": q, "n_candidates": len(cands), "top_jaccard": round(best, 4)})
    neg_over05 = float(np.mean([h["top_jaccard"] >= 0.5 for h in neg_hits]))

    top_jac_v2 = np.array([r["top_jaccard"] for r in res_v2])
    top_jac_raw = np.array([r["top_jaccard"] for r in res_raw])
    summary = {
        "method": "scContam Signal A (vectorized MinHash m=128, bands=16, rows=8, K=2048; "
                  "same constants as scContam defaults; exact-Jaccard verified candidates)",
        "reference": f"{REPO}, shards sampled: {SHARDS} "
                     f"(~{n_corpus_cells/1e6:.2f}M of ~104M cells, ~{100*n_corpus_cells/103877737:.2f}% of corpus)",
        "benchmark": "GSE299152 (23,970 cells, annotated; donors 2046/2206/3748)",
        "vocab": "token_dictionary_gc104M.pkl (20,275 tokens; gene tokens 4..20274)",
        "corpus_stats": {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in stats.items()},
        "gse_stats": {"n_cells": int(counts.shape[0]), "genes_unmapped": int(n_nomatch)},
        "query_v2_replicated": {
            "n_queries": len(res_v2),
            "n_cells_top_jaccard_ge_0.5": int((top_jac_v2 >= 0.5).sum()),
            "n_cells_top_jaccard_ge_0.7": int((top_jac_v2 >= 0.7).sum()),
            "max_top_jaccard": float(top_jac_v2.max()),
            "top_jaccard_quantiles": [float(np.quantile(top_jac_v2, q)) for q in [0.5, 0.9, 0.99, 1.0]],
        },
        "query_raw_expression": {
            "n_cells_top_jaccard_ge_0.5": int((top_jac_raw >= 0.5).sum()),
            "n_cells_top_jaccard_ge_0.7": int((top_jac_raw >= 0.7).sum()),
            "max_top_jaccard": float(top_jac_raw.max()),
        },
        "positive_control_inject_200": {
            "median_best_jaccard": round(float(np.median(pos_jac)), 4),
            "fraction_recalled_at_0.9": round(float(pos_recall_09), 4),
            "n_candidates_median": int(np.median([p["n_candidates"] for p in pos])),
        },
        "negative_control_200_random": {
            "fraction_candidates_ge_0.5": round(float(neg_over05), 4),
            "max_top_jaccard": round(float(max(h["top_jaccard"] for h in neg_hits)), 4),
        },
        "runtime_s": round(time.time() - t_all, 1),
        "download_dir": str(SHARD_DIR),
    }
    out = OUT / "signalA_gse299152.json"
    out.write_text(json.dumps(summary, indent=2))
    print("DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
