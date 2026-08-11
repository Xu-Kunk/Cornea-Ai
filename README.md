# Corneal-FM

**Corneal-FM fine-tuning reproducibility for Paper 1** — a study of tissue-specific
fine-tuning of a single-cell foundation model on human corneal scRNA-seq, including
same-corpus / out-of-corpus / cross-tissue evaluation, FDR-controlled null analysis,
and locked-seed reproduction of every reported figure and table.

## Repository structure

```
.
├── corneal_fm/               # core library (CAFT fine-tuner, ordinal head, GPR, evaluation)
├── scripts/                  # pipeline scripts (data prep → training → eval → figures)
├── config/                   # locked run configurations (*.yml)
├── tests/                    # pytest suite (CPU-only; GPU tests self-skip)
├── results/                  # provenance artifacts referenced by Paper 1
│   ├── paper1_fair_split/    #   donor-disjoint split manifest
│   ├── paper1_fdr_nulls/     #   BH-FDR table + null controls + contamination audit
│   ├── paper1_epidermis_eval/#   cross-tissue (epidermis) gradient-correlation results
│   └── scgpt_repro/frozen_diag/  # frozen-model diagnostic runs (json + logs)
├── requirements-core.txt     # pinned runtime dependencies (Python 3.12)
├── requirements-ci.txt       # CPU-only test dependencies (GitHub Actions)
├── requirements-gx10.txt     # ASUS GX10 / DGX Spark (aarch64, CUDA 13) extras
├── pyproject.toml            # packaging + ruff + pytest config
├── Dockerfile                # NGC pytorch:25.11-py3 image for GX10
└── .github/workflows/ci.yml  # lint + CPU tests + pyright
```

## Installation

Requires **Python 3.12**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-core.txt
```

Notes:

- `requirements-core.txt` pins a **CPU** `torch==2.12.1` wheel. For GPU training,
  install a CUDA build of torch first, then the rest of the file.
- `config/env.yml` is a legacy CPU/x86 conda reference; the production GPU target is
  the NVIDIA DGX Spark / ASUS GX10 (see `requirements-gx10.txt` and `Dockerfile`).
- Set `CORNEAL_FM_DATA` to this repository root before running shell pipelines
  (several `scripts/*.sh` read it). Python scripts resolve the repo root relative to
  their own location and need no environment variables.

## Data

Raw data is **not** bundled (see `.gitignore`). Download it as follows:

| Dataset | Source | Purpose | Script |
|---|---|---|---|
| Corneal scRNA-seq | CELLxGENE collection `0f7d022a` | training / same-corpus test | `scripts/download_wiedemann.py` |
| Corneal external test | GEO **GSE153515** | out-of-domain same-tissue eval | `scripts/protocol_15_annotate_gse153515.py` |
| Corneal external test | GEO **GSE299152** | true out-of-corpus eval | `scripts/protocol_15b_annotate_gse299152.py` |
| Epidermis (cross-tissue) | GEO **GSE147424** (Wang 2020) | cross-tissue eval | `scripts/protocol_12_download_epidermis.py`, `scripts/protocol_13_download_geo.py` |

Downloads use NCBI GEO FTP / CELLxGENE Census and do not require a VPN.

## Pipeline overview

```
fair split ──► training arms ──► Tier-B eval ──► FDR / nulls ──► figures
```

1. **Fair split** — donor-disjoint train/test split:
   `scripts/prepare_paper1_fair_split.py` (manifest in `results/paper1_fair_split/`).
2. **Training arms** — CE-naive (`scripts/caft_gf.py`), ordinal head
   (`scripts/ordinal_gf.py`), and the arm grid (`scripts/run_method_arms.py`),
   with `config/default_train_args.yml`.
3. **Tier-B evaluation** — `scripts/run_tier_b_eval.py` on the external tests
   (configs `config/external_test_gse153515.yml`, `config/external_test_gse299152.yml`).
4. **FDR / null controls** — `scripts/fdr_nulls_analysis.py` (label/feature-shuffle
   nulls + Benjamini–Hochberg at q=0.05; tables in `results/paper1_fdr_nulls/`).
5. **Figures** — `scripts/generate_paper1_data_figures.py`,
   `scripts/generate_paper1_results_figures.py`,
   `scripts/generate_paper1_supplementary_figures.py`,
   `scripts/generate_figure_domains.py`.

## Reproducibility locks

- Seeds **42 / 123 / 456** (3-seed arms; deterministic subsetting in
  `scripts/deterministic_subset.py`).
- Locked run configuration in `config/*.yml` (no ad-hoc flag changes in the paper runs).
- scGPT pinned at commit **`cebd6fae`** (see `scripts/protocol_07_finetune_scgpt.py`).
- Provenance for every reported artifact lives in `results/` and is machine-independent.

## Citation

[Paper 1 preprint: to be added]

Fine-tuned checkpoints: [fine-tuned checkpoints: to be added]

## License

MIT — see `pyproject.toml` (`license = {text = "MIT"}`).
