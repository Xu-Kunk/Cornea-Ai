# Corneal-FM

Evaluation code and aggregated results for a study of how classification fine-tuning
affects the differentiation geometry of single-cell foundation model representations,
using human corneal epithelium as a four-stage differentiation continuum.

## Layout

| Path | Contents |
|---|---|
| `corneal_fm/` | Package: tokenization, training loops, ordinal and parameter-efficient heads, evaluation metrics |
| `scripts/` | Experiment drivers, per-figure producers, metric and audit utilities |
| `config/` | Locked evaluation configurations |
| `results/` | Aggregated results (JSON/CSV) for every reported comparison |
| `figures/paper1/` | Produced figures (vector PDF and 600 dpi PNG) with `figure_manifest.json` |
| `tables/paper1/` | Aggregated CSV tables |
| `tests/` | Unit and smoke tests |
| `artifacts/env_locks/` | Frozen dependency lists for the two environments used |

## Data

All single-cell data are public: the CELLxGENE Discover human ocular surface atlas
(collection `0f7d022a-46c7-4e64-be4c-e34adbb78089`), and GEO accessions GSE153515,
GSE299152, GSE147424, GSE202352, GSE186433, GSE155683 and GSE147979. Model weights
(Geneformer-V2-104M, scGPT-human, CellPLM) are available from their own repositories.

Not distributed here: raw count matrices, model checkpoints, per-cell embedding and
prediction arrays, and tokenized caches.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-core.txt      # analysis stack
pip install -r requirements-ci.txt        # test and lint tooling
```

## Running

```bash
python -m pytest tests/                    # metric and pipeline tests
python scripts/run_tier_b_eval.py --help   # main external evaluation
python scripts/figure2_layerwise_rho.py    # one producer per figure; each writes its
                                           # PDF/PNG and updates the figure manifest
```

Metric implementations live in `corneal_fm/evaluation.py` (donor-level macro-F1 with
bootstrap intervals, layer adjacency accuracy, differentiation-gradient Spearman
correlation, expected calibration error).

## Licence

MIT — see `LICENSE`.
