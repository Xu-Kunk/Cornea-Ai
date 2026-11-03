# Corneal-FM — Single-Cell Foundation Model Fine-Tuning Evaluation

**Paper 1:** *Annotation accuracy is not a proxy for differentiation order in fine-tuned single-cell foundation models* (active manuscript title, v31)

Target: **Cell Press Patterns** (Research Article; direct submission).

## Current Status

**Compiled snapshot: Paper 1 v31 — 2026-10-23.** Active source `docs/manuscript/patterns/latex/paper1.tex` builds to `latex/build/paper1.pdf`: 29 pages, 0 errors, 0 undefined, 0 overfull. Supplementary source `supplementary/supplementary_information.tex` builds to `supplementary/build/supplementary_information.pdf`: 30 pages (Figures S1–S13, Tables S1–S12, Supplementary Methods SM1–SM12 + Tables SM1/SM2, Data S1–S3). Bibliography `latex/paper1.bib` has 88 entries. Package metadata: `PATTERNS_ACTIVE_PACKAGE.md`; checksums: `CHECKSUMS.sha256`. Cell Press LaTeX template v1.10.

**v26 provenance:** the 2026-10-17 review revision was promoted to the active manuscript on 2026-10-19. The former review package `revision_2026-10-16/` is merged and frozen at `docs/manuscript/patterns/archive/revision_2026-10-16/`; the v25 source, PDF and bibliography are archived at `latex/archive/v25-2026-10-16/`. Changelog: `docs/changes/2026-10-19-paper1-v26-promotion-and-package-reorganization.md`. Remaining figure/resource holds are not cleared.

**v27 provenance:** a CellPLM training-budget dose response was added on 2026-10-19 (18 runs, three seeds per arm, remote GPU). It shows that CellPLM retains the differentiation gradient at 3 and 15 epochs and loses it only after macro-F1 saturates, whereas Geneformer loses it within one epoch. Results subsection *The gradient loss follows task saturation* and Figure 6 panel C were added; the cross-architecture claim is now stated in terms of the timing of the loss. Changelog: `docs/changes/2026-10-19-cellplm-dose-response.md`.

**Manuscript organization (2026-10-19):** active Patterns package is `docs/manuscript/patterns/`; `latex/paper1.tex` is the single source of truth; the merged 2026-10-17 review package is frozen at `archive/revision_2026-10-16/`; bioRxiv v18 is frozen in `docs/manuscript/archive/biorxiv-v18/`; old `docs/manuscript/latex/` is archived in `docs/manuscript/archive/old-latex/`.

**Figure system (图件系统, 2026-10-19 rebuild):** all 19 figures (6 main, 13 supplementary) are one-figure-one-producer, built from the hand-maintained deck `figures/paper1/design/figure1_final.pptx` (Figure 1; the old `scripts/figure1_layout_pptx.py` generator is retired and must not be re-run) and `scripts/figure2_layerwise_rho.py` … `scripts/figureS13_continuity_metrics.py` (Figures 2–6, S1–S13) through the shared contract `scripts/paper1_figure_style.py` (locked four-tier type scale tick 7.0 / axis 7.5 / title 8.0 / letter 10.0; arm palette; structural ink `#B0B0B0`; spine 0.5 pt; ticks 8×2 px outward; `save_figure()` rejects Type 3 fonts) and the fail-fast loaders in `scripts/paper1_figure_data.py`. `figures/paper1/figure_manifest.json` records all 19 entries with producing script, every input path + SHA-256, output PDF/PNG hashes and physical size; the contract and measured Patterns house style are in `figures/paper1/FIGURE_SYSTEM.md`, and caption debt is tracked in `figures/paper1/CAPTION_DEBT.md`. `scripts/qa_figure_type_floor.py` passes 19/19; every figure is 178.0 mm wide, Type 3-free and exported as 600 dpi PNG. Superseded artwork (30 files), CSVs no longer read (12) and the Figure 3 provenance set are archived under `figures/paper1/archive/superseded-2026-10-19/`, `figures/paper1/archive/stale-csv-2026-10-19/` and `figures/paper1/archive/figure3-provenance-2026-10-19/`; design assets moved to `figures/paper1/design/`. The 13 legacy producers are archived in `scripts/archive/legacy-figures-2026-10-19/`; `scripts/generate_paper1_results_figures.py` is retained as a **library only** (imported by `analyze_trajectory_continuity.py` and `sens_umap_fig.py`, with a LIBRARY ONLY banner) — its figure entry point is superseded and must not be run. Changelog: `docs/changes/2026-10-19-paper1-figure-system-rebuild.md`.

| Component | Status |
|-----------|--------|
| Active manuscript | v31 (2026-10-23): `latex/paper1.tex` → `latex/build/paper1.pdf`, 29 pages, 0/0/0; bib 88 entries; SI 30 pages. Single source of truth |
| Review package | `archive/revision_2026-10-16/` — the 2026-10-17 review revision, **merged into v26 and frozen**; no longer an independent candidate |
| Figure 3 | Full gene-program figure (58 verified records) wired into the active manuscript; production PDF at `figures/paper1/figure3_biological_readouts.pdf` |
| Figure 8A | Separate earlier standalone Panel A (three-seed) candidate; not the full Figure 3 replacement and still unadopted |
| Figure system | 6 main + 14 supplementary figures rebuilt 2026-10-19, one producer each; see `figures/paper1/figure_manifest.json` and `figures/paper1/FIGURE_SYSTEM.md`. Figure 8 B–D renderer binding remains open |
| Author list | ✅ Kunhui Xu, Chenchen Wang, Li Fenfen, Qi Dai; Li Fenfen and Qi Dai (co-corresponding) |
| Pre-submission review | Historical simulated review record; no longer represents current acceptance |
| Benchmark code | ✅ GitHub `Xu-Kunk/Cornea-Ai` (to be made public before submission — hard gate) |
| Graphical abstract | ⏳ Deferred (Patterns does not require it) |

## Core Scientific Claim

Supervised fine-tuning of single-cell foundation models collapses the continuous differentiation gradient in a **corpus- and architecture-dependent manner**:

- **Same-corpus** (Geneformer on GSE153515): catastrophic — frozen ρ 0.806 → naive ρ −0.008; official one-epoch ρ = 0.058 (six seeds)
- **Out-of-corpus** (GSE299152, post-cutoff): mild — ρ 0.707 → 0.641
- **Cross-tissue** (epidermis): reversed — ρ 0.240 → 0.455 (naive, seed-42; ordinal head 0.500)

**Rescue recipe**: an ordinal head (unconditional cumulative-threshold BCE; **not** standard conditional CORN) + LoRA reached ρ = 0.924 ± 0.001 and macro-F1 = 0.966 ± 0.002 in the scGPT same-corpus 15-epoch best-validation comparison. The earlier ordinal-only result used a different 3-epoch budget and checkpoint convention; the smaller SD is therefore **not** evidence of a LoRA causal stabilization effect.

**Cross-architecture spectrum**: Geneformer shows a large same-corpus CE collapse; CellPLM collapses at the entry layer (L1), with a three-seed CE mean of −0.007 ± 0.035 and an ordinal arm of 0.918 ± 0.002; scGPT-human shows pronounced seed heterogeneity (five seeds spanning ρ = −0.09 to 0.833). Two nominal CE arms differed (−0.008 ± 0.044 vs +0.210 ± 0.016), so architecture alone does not fix the outcome; the cause is not fully explained.

## Evidence Chain

- Three-architecture evidence (Geneformer, scGPT-human, CellPLM Flowformer)
- Layerwise collapse localization (head-proximal for Geneformer, entry-layer for CellPLM)
- BH-FDR: 21 frozen association tests plus 6 entries pending at freeze time; 20 positive associations were significant. This is not a test of an adaptation difference.
- SVD-axis validation: ρ metric is not a PC artifact
- Fixed-frozen-axis control: rescue recipe projection control (ρ 0.921 vs 0.924)
- Sensitivity analysis: several SVD-axis variants were PARTIAL, not all robust; marker/rank conclusions are reported variant by variant
- Literature-anchored biological context (Cotsarelis 1989 stem cells; Fuchs & Green 1980 keratin switch); this is background, not independent biological validation
- Donor-level contamination audit: GSE153515 exposure presumed; inferential weight on GSE299152

## Alternative objectives and implementation caveat

The alternative-objective records cover nominal CAFT (its continuity term did not execute, so it is **not** a valid negative), AA-FT (partial), soft-Spearman (unstable), flow (unstable), and axis-anchored adaptation (partial). The ordinal head (unconditional cumulative-threshold BCE) preserved the gradient in the reported same-corpus test; it is not standard conditional CORN. NFIB and CDH3 are part of the five-gene stemness program, not independent validation.

## Project Structure

```
corneal_fm/           # Core Python package (config, trainer, benchmark, utils)
scripts/              # Analysis pipelines, eval scripts, launchers
artifacts/            # All results, figures, preprints, audit docs
  paper1_preprint/    # v3–v18 preprint PDFs, submission package, layout audit
  paper1_vanzyl/      # van Zyl arm results (corn_lora, cellplm, cross_tissue, etc.)
  paper1_fdr_nulls/   # BH-FDR tables, contamination audit, donor-level audit
  paper1_tier_b_eval/ # Main protocol evaluation results
docs/
  manuscript/patterns/        # ACTIVE: v31 Patterns package
    latex/paper1.tex          # Single source of truth; build/paper1.pdf (29 pp)
    latex/archive/            # v25 source/PDF/bib (v25-2026-10-16/), older builds and reports
    supplementary/            # SI source + build/ (30 pp) + archive/
    archive/revision_2026-10-16/  # Merged/frozen 2026-10-17 review revision
    PATTERNS_ACTIVE_PACKAGE.md    # Package metadata; CHECKSUMS.sha256 alongside
  manuscript/archive/         # Historical drafts, old latex, biorxiv v18
  changes/            # Dated changelogs (archived by month)
  research/           # Paper design docs, journal fit assessments
  SCI-reference/    # Literature library (PDF + MD, 8 topic categories)
tasks/                # Agent context: todo.md, knowledge.md, lessons.md
save/                 # Training run outputs (gitignored; not in public snapshot)
data/                 # PCS-compliant: raw/ (read-only) + processed/ (derived)
  data/raw/           # Original GEO downloads, 10x matrices, orthologs
  data/processed/     # Annotated/split/tokenized data from scripts
```

## Code Release Boundary

This repository is the **minimal reproducible package** for the paper. The public snapshot includes analysis code, scripts, configuration, environment locks, README, and aggregated JSON/CSV outputs. The following items are intentionally excluded and must be obtained from the indicated sources:

| Item | Reason | How to obtain |
|------|--------|---------------|
| Raw/processed human scRNA-seq data | Large, subject to data-use agreements and original study consents | GEO / ArrayExpress accession numbers cited in the manuscript; `scripts/download_*.py` will be provided where permitted |
| Pre-trained foundation model weights | License and size constraints | Official Geneformer / scGPT / CellPLM model hubs or release packages |
| Fine-tuned checkpoints under `save/` | Run artifacts, large, reproducible from code + data | Re-run the training scripts with the downloaded data |
| Generated figures, preprints, tables under `artifacts/` | Presentation outputs, reproducible from code + data | Re-run the analysis pipelines |
| Sensitive logs or private correspondence | Privacy / embargo | Not applicable |

No source code, manuscript text, author list, or scientific conclusions were changed for this boundary definition.

## Key Commands

```bash
# Compile the active v31 manuscript (outputs to latex/build/)
cd docs/manuscript/patterns/latex
pdflatex -output-directory=build paper1.tex
(cd build && BSTINPUTS=.. bibtex paper1)
pdflatex -output-directory=build paper1.tex
pdflatex -output-directory=build paper1.tex

# Layout audit
python3 artifacts/paper1_preprint/pdf_layout_audit.py

# Run tests
pytest tests/
```

> Note: `numbered.bst` lives in `latex/`, so bibtex is run from `build/` with `BSTINPUTS=..`. The standard `pdflatex+bibtex+pdflatex×2` sequence above produces `latex/build/paper1.pdf` (35 pages, 0 errors / 0 undefined / 0 overfull).

## References

- Geneformer: \cite{theodoris2023} — Nature 2023
- scGPT: \cite{cui2024scgpt} — Nature Methods 2024
- CellPLM: \cite{wen2024cellplm} — bioRxiv 2024, OpenReview BKXvPDekud
- CORN: \cite{shi2023corn} — Pattern Analysis and Applications 2023 (DOI 10.1007/s10044-023-01181-9)
- Patterns: Cell Press data-science journal (JIF 10.8, 2025 JCR)
