# Caption debt — figures rebuilt 2026-10-19 (discharged 2026-10-22)

Every rebuilt figure moved statistics, interval definitions and provenance off
the canvas and into the caption (Patterns house style, rule 9). This file
collects the replacement caption material so the manuscript update is
mechanical.

**Status 2026-10-22.** No entry remains PENDING. The last open row (Figure 1)
closed when the figure was redrawn and re-verified on 2026-10-22; the other
rows, the "known caption changes", the claim-boundary items and the
post-rebuild follow-ups were all carried into the v26-v30 text (cover letter
and `patterns_submission_extras.md` re-checked: no stale figure names; the
extras file has since been archived). The file is retained as the record of
what the rebuild had to fix, not as an open work list.

Status legend: **READY** = figure final and caption text supplied;
**PENDING** = figure still being rebuilt.

| Figure | Stem | Status |
|---|---|---|
| 1 | `figure1_pipeline_overview` | READY (redrawn 2026-10-22: rings/ellipses dropped, panel E removed, canvas labels A–D; `design/figure1_final.pptx` hand-maintained, caption revised to match) |
| 2 | `figure2_layerwise_rho` | READY |
| 3 | `figure3_biological_readouts` | READY |
| 4 | `figure4_external_eval` | READY |
| 5 | `figure5_domain_boundary` | READY |
| 6 | `figure6_cross_architecture_spectrum` | READY (panel B) |
| S1 | `figureS1_calibration` | READY |
| S2 | `figureS2_ood` | READY |
| S3 | `figureS3_per_seed_reproducibility` | READY |
| S4 | `figureS4_gse153515_qc` | READY |
| S5 | `figureS5_training_dynamics` | READY |
| S6 | `figureS6_per_donor_breakdown` | READY |
| S7 | `figureS7_layerwise_rho_per_seed` | READY |
| S8 | `figureS8_internal_external_inversion` | READY |
| S9 | `figureS9_annotation_detail` | READY |
| S10 | `figureS10_tac_sensitivity` | READY |
| S11 | `figureS11_ordinal_boundaries` | READY |
| S12 | `figureS12_trajectory_continuity` | READY (renamed) |
| S13 | `figureS13_annotation_overview` | READY (renamed) |
| S14 | `figureS14_continuity_metrics` | READY (renamed) |

## Known caption changes already established

**Figure 5** — the manuscript currently describes panels (A) and (B) and a
hatched ordinal key. The rebuilt figure is a **single panel**: the two legacy
panels were a bar rendering and a line rendering of the same three values.
Caption must drop the (A)/(B) wording and the hatch key.

**Figure 6 panel B** — the manuscript describes a per-epoch validation
trajectory with dashed intermediate epochs. That data is not archived anywhere
(verified locally and on the remote host); the panel is now a **per-seed peak
scatter**. Caption must state: x = epoch of the validation-ρ peak, y = that peak,
one point per seed, n = 5 seeds, dotted connector = the epoch at which the
checkpoint was selected on validation macro-F1 (ring), peaks early (epochs 1–6)
while selection followed macro-F1.

**Figure 3** — caption still lists `Cross-entropy, Ordinal, CAFT-intended*`.
Replace with the canonical labels: `frozen, CE-naive, ordinal head,
CAFT-intended, Axis-anchored, Soft-Spearman`.

**Figure S11** — canvas no longer carries the box-whisker definition or the
provenance line. Caption must add: boxes = median with 1.5×IQR whiskers,
outliers not shown; arm = ordinal head, λ = 0, seed 42, checkpoint-7473, epoch 3;
GSE153515, n = 14,775; three-seed overall consistency 0.7172 / 0.8719 / 0.7877
(mean 0.7923, range 0.7172–0.8719); permutation null 0.1751, 1,000 draws, seed 42.

**Figure S1** — caption must carry the cell count and donor count that were
removed from the canvas (14,775 epithelial cells, 2 donors, GSE153515) and the
fact that the TAC panel has 171 true cells (1.2%).

## Replacement caption texts (verbatim, ready to paste)

### Figure S3
**Figure S3. Per-seed reproducibility of the endpoint and continuity metrics.** Points are individual seeds 42, 123 and 456 (marker shape) for CE-naive and CAFT-intended (colour); the short horizontal tick is the arithmetic mean of the three seeds. (A) Donor-level macro-F1. (B) Differentiation-gradient Spearman ρ between representation-derived cell ordering and prespecified stage ranks; the dotted line marks ρ = 0. (C) Layer adjacency accuracy. (D) Expected calibration error. All values are from the GSE153515 same-corpus external holdout (14,775 epithelial cells, two donors); donor-level macro-F1 is the mean of the two per-donor scores. No interval, test or correction is shown; three seeds per arm.

### Figure S6
**Figure S6. Per-donor breakdown of the primary endpoint.** Lines join the two held-out donors for each seed (marker shape: 42, 123, 456) and epoch (solid, epoch 3; dashed, epoch 2); panels show CE-naive and CAFT-intended. Values are per-donor donor-level macro-F1 (donor_1 = GSM4646295, donor_2 = GSM4646296; 14,775 epithelial cells across both donors). All three seeds and both epochs are plotted for both arms; no interval, test or correction is shown.

### Figure S7
**Figure S7. Layerwise differentiation-gradient Spearman ρ per seed.** Hidden states L1–L12 (L0 excluded: constant [CLS]); thin lines are individual seeds (linestyle and marker: 42, 123, 456) and the thick line is the arithmetic mean of the three seeds of that arm. Frozen Geneformer is a single reference run. The external set is GSE153515 (14,775 epithelial cells, two donors). No band, interval, test or correction is shown; three seeds per adapted arm.

### Figure S8
**Figure S8. Internal-versus-external ranking inversion.** Internal (training-split) and external (GSE153515) donor-level macro-F1 for HVG/PCA and frozen Geneformer, with all six adapted runs at their external values. Internal values are single training-split estimates without an interval. External values carry each result's donor-level bootstrap 95% CI (asymmetric bounds; 1,000 donor resamples, bootstrap seed 42, two donors). Adapted seeds are individual points (marker shape: 42, 123, 456; colour: arm) and the open diamond is the three-seed mean. External set: 14,775 epithelial cells, two donors. No test or correction is shown.

### Figure S10
**Figure S10. Sensitivity of the primary endpoint to the rare TAC class.** (A) Donor-level macro-F1 under the locked four-class schema (filled circles) and the three-class schema with TAC excluded (open squares); horizontal bars are the donor-level bootstrap 95% CI (1,000 donor resamples, bootstrap seed 42), drawn per run and never pooled. (B) Difference between the two endpoints, computed within each seed (points) and then averaged over the three seeds (tick); no interval is drawn for the difference because the two endpoints were bootstrapped separately. (C) Pooled cell-level per-class F1 under the four-class schema. GSE153515 external test: 14,775 epithelial cells, two donors, 171 TAC cells (1.2%). CE-naive and CAFT-intended have three seeds each (42, 123, 456); frozen is one reference run.

## Claim-boundary problems found while rebuilding (need author attention)

1. **The old S12 artwork was the wrong figure.** `pdftotext` on the retired
   `figure3_trajectory_continuity.pdf` returns "frozen / naive FT / CORN /
   CORN+LoRA" and "0.924 / 0.833 / 0.806" — it was the three-domain boundary bar
   chart, not the DPT-vs-reference continuity figure its caption describes. The
   caption's numbers match the analysis output, so **the caption was right and
   the artwork was wrong**. The rebuilt `figureS12_trajectory_continuity` now
   matches its caption.

2. **S12 — resolved: the manuscript does not overstate, the analysis script
   did.** Checked all three places: the main-text Results sentence cites only
   frozen 0.850 [0.842-0.858] vs naive 0.323-0.348 and never mentions CAFT; the
   SI Figure S12 caption says "CAFT is intermediate (rho = 0.372-0.428)", which
   the data supports; the Discussion makes no continuity or trajectory claim.
   The overstatement lived only in `scripts/analyze_trajectory_continuity.py`,
   whose docstring attributed "CAFT preserves the differentiation trajectory" to
   the Discussion — a claim the v26 Discussion no longer contains. Docstring
   corrected on 2026-10-19 to state the measured outcome and forbid restating
   the old framing. No manuscript text change needed.

3. **S9's caption says "marker expression"; the figure plots marker-set scores.**
   Panel A draws cluster-level **marker-set scores** from
   `results/gse153515_annotation/marker_scores.csv` (colour) with cluster size
   (dot area). Per-gene expression is not reachable: both
   `data/processed/gse153515_annotated.h5ad` and
   `data/processed/gse153515_limbal_stem.h5ad` carry Ensembl identifiers in a
   column named `gene_symbol`, so no marker can be looked up by symbol in
   either file (checked directly). The lane's claim that "6 of 15 curated
   markers survive the gene filter" is **not reproducible from the var index**
   and should be dropped unless a symbol mapping is produced.

4. **S5's caption uses retired arm names and cites a CSV that is no longer
   emitted.** Caption says "Naive fine-tuning (λ = 0) / CAFT (λ = 0.5)" and
   "plotted values in `figureS5_training_dynamics.csv`"; the figure now reads
   `trainer_state.json` directly and uses the canonical labels.

5. **Every S4/S5/S9/S12/S13/S14 caption still carries the old producer name and,
   for S12/S13/S14, the old filename in `\includegraphics` and in the "Files:"
   line.** Must be repointed to the new stems.


## Post-rebuild follow-ups (2026-10-19)

- **Figure S2 panel (E) restored.** The rebuild had silently dropped it; it now plots the per-donor
  mean maximum predicted probability from `figureS2_ood_perdonor.csv`, cross-checked against the
  per-cell archives. Caption carries "(E) Per-donor mean maximum predicted probability (confidence)
  on the external set."
- **Figure S11 Files line** no longer cites `figureS11_corn_boundary.csv` (stale artefact of the
  retired producer); the CSV remains on disk but is not referenced.
- **Captions** in all 20 figures were rewritten to match the rebuilt artwork; the eight SI captions
  whose "Generated by" clause named a retired producer now name the new `scripts/figure*` producer.
- **Not yet checked:** `cover_letter_patterns_2026-10-09.md` and `patterns_submission_extras.md` were
  not scanned for stale figure names.
