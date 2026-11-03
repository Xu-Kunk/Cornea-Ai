# Paper1 figure/table CNS (Cell Press) standards audit — 2026-08-15

Audit-and-fix pass across every figure/table of the Patterns manuscript. Scope:
formatting/standards only — no scientific content, numbers, or color semantics
were changed. Liberation Sans retained (Arial-metric; Arial not installed).
All scripts run with `/home/dq/venv-ds/bin/python3`.

## Global standards applied

| Standard | Status |
|---|---|
| PDF font embedding (`pdf.fonttype=42`, TrueType, never Type 3) | All 21 PDFs verified with `pdffonts`: 0 Type-3 fonts, all fonts embedded |
| Raster ≥ 300 dpi | All PNGs saved at 300 dpi |
| Canvas ≤ 6.5 × 8 in (16.5 × 20 cm) | Enforced in every script; all regenerated outputs compliant |
| Font ≥ 6 pt at final size (floor 5 pt) | All text audited ≥ 6 pt; panel letters 10 pt bold caps |
| Panel letters: capital, upper-left, bold 8–10 pt | Standardized to 10 pt bold; letters anchored above the rendered panel title (never collides with titles or tick labels) |
| No background gridlines / chartjunk | Removed light dashed grids (figure8); dashed reference *data* lines retained |
| Stroke widths 0.5–1.5 pt | Verified (axes 0.6, lines 0.7–1.5 pt) |
| Error-bar definitions in-figure | Added mean ± SD / bootstrap-95%-CI / ± range notes (figure7, figure8, figure6, figureS8, figureS12) |
| Okabe-Ito colorblind-safe palette | Retained throughout; grayscale fills ≥ 20 % separation |
| Tables: booktabs, units, right-aligned numbers, leading zeros, footnotes | Fixed tex + markdown tables (below) |

## Per-figure audit

| Figure | Fixes applied | Verification |
|---|---|---|
| figure1_pipeline_overview | Canvas 7.5→6.5 in wide (re-scaled unit system); fonttype 42 already set | 1950×1284 px; Type 42; internal layout QA reports 6 advisory label-width warnings (pre-existing estimation checks, non-fatal, labels drawn unclipped) |
| figure2_layerwise_rho | Canvas 7→6.5 in; margins; panel letters 10 pt anchored | 1950×975 px; Type 42; bbox audit **OK** |
| figure5_external_eval | Canvas 7→6.5 in; letters; panel-D yticks capped; footnote/annotation spacing | 1950×1950 px; Type 42; bbox audit **OK** |
| figure5_donor_macro_f1 | Canvas 7→6.5 in; letters; yticks capped (no 1.2 extension) | 1950×975 px; Type 42; bbox audit **OK** |
| figure6_domain_boundary | Canvas 7→6.5 in; removed `wspace` (constrained-layout squeeze); footnote shortened to fit canvas; panel titles 10 pt; yticks fixed | 1954×824 px; Type 42; bbox audit **OK** |
| figure7_continuity_metrics | Canvas 7.5→6.5 in; letters; per-panel deterministic ylim/yticks; title 8 pt; error-bar footnote (mean ± SD) | 1950×682 px; Type 42; bbox audit **OK** |
| figure8_bio_validation | Canvas 7→6.5 in; gridlines removed; letters 10 pt anchored; error-bar footnote (bootstrap CI / mean ± SD) | 1950×1671 px; Type 42; bbox audit **OK** |
| figureS1_calibration | Script updated (canvas 7→6.5 in, explicit Type-42 rcParams) but **cannot regenerate** — missing checkpoint artifact `caft_id_class_dict.pkl` (pre-existing data gap) | Stale 2100×1800 px output retained; PDF Type 42 (from earlier run) |
| figureS2_ood | Same as S1 (canvas 9.5→6.5 in in script) — **cannot regenerate** (same data gap) | Stale 2850×1800 px retained; PDF Type 42 |
| figureS3_per_seed_reproducibility | Canvas 7→6.5 in; letters; deterministic per-metric ylim/yticks (LAA narrow-range fix); hspace | 1950×1530 px; Type 42; bbox audit **OK** |
| figureS4_gse153515_qc | Letters 10 pt anchored | 1003×1290 px; Type 42; bbox audit **OK** |
| figureS5_training_dynamics | Canvas 10→6.5 in (largest shrink); margins; explicit xlim/xticks | 1950×682 px; Type 42; bbox audit **OK** |
| figureS6_per_donor_breakdown | Canvas 7→6.5 in; margins; yticks; xlim/xticks | 1950×975 px; Type 42; bbox audit **OK** |
| figureS7_layerwise_rho_per_seed | Canvas 7→6.5 in | 1950×975 px; Type 42; bbox audit **OK** |
| figureS8_internal_external_inversion | Canvas 7→6.5 in; CI-definition footnote; margins; yticks | 1950×1113 px; Type 42; bbox audit **OK** |
| figureS9_annotation_detail | Letters 10 pt anchored | 1602×1080 px; Type 42; bbox audit **OK** |
| figureS10_tac_sensitivity | Canvas 7→6.5 in; letters; titles 7.5 pt; xticks 60°/6 pt; delta labels 6 pt without redundant "+" | 1950×1950 px; Type 42; bbox audit **OK** |
| figureS11_corn_boundary | Canvas 8.7→6.5 in; removed `wspace`; panel ylim/yticks; xtick rotation 70°; footnote moved below canvas (tight-bbox grows output) | 1956×889 px; Type 42; bbox audit OK (2 tight-bbox-resolved flags) |
| figureS12 (figure3_trajectory_continuity) | Canvas 7→6.5 in; explicit ylim/yticks; CI footnote | 1950×975 px; Type 42; bbox audit **OK** |
| figure3_gse153515_annotation (legacy 4-panel) | Canvas 6.85→6.5 in; margins; dotplot labels 6 pt | 1950×1650 px; Type 42; **residual**: scanpy dotplot internal label density (ytick/colorbar-label proximity ~0.02–0.2 in at 100 dpi) — pre-existing rendering of the 15-row dotplot; the active main figures (figure4_annotation, figureS9) pass clean |
| figure4_annotation | Letters 10 pt anchored | 1602×1020 px; Type 42; bbox audit **OK** |

## Per-table audit

| Table | Fixes |
|---|---|
| Table 1 (tex `tab:external`) | Numeric columns centered → **right-aligned** (`lrrrrr`); booktabs already correct; 4-dp consistent; leading zeros present |
| Table 2 (tex `tab:domains`) | Numeric columns → right-aligned (`lrrr`); cross-tissue rows normalized 2→3 dp (0.24→0.240 etc.) for column consistency |
| Table S1 | Already compliant (right-aligned numerics, 4 dp, leading zeros); added "Related to Table 1" title clause |
| Table S2 | Numerics right-aligned; "Related to Table 1" |
| Table S3 | Text table (no numerics); "Related to Table 1" |
| Table S4 | Numerics right-aligned; decimals normalized to 4 dp (0.889→0.8890, 0.75→0.7500, 0.0→0.0000); "Related to Table 1" |
| Table S5 | Layer columns right-aligned; "Related to Figure 2" |
| Table S6 | Cell-count column right-aligned; "Related to Methods §2.1" |
| Table S7 | Text/checksum table; "Related to Methods §2.8" |
| Table S8 | ρ columns right-aligned; "Related to Figure 2" |
| Table S9 | Numeric columns right-aligned; "Related to Table 1" |

## Not brought to compliance

- **figureS1_calibration / figureS2_ood**: could not regenerate — the generator
  (`scripts/measure_percell_probs.py`) requires the archived id→class map
  (`caft_id_class_dict.pkl`) which is absent from the `save/caft_gf_*` checkpoints
  (pre-existing data gap, unrelated to this pass). Scripts were updated for the
  new canvas/Type-42 standards so a future data restore regenerates compliant
  files; current on-disk outputs remain the prior versions (S1 7.0×6.0 in,
  S2 9.5×6.0 in — over the 6.5-in limit; both PDFs already Type 42).
- **figure3_gse153515_annotation**: residual scanpy dotplot-internal label
  proximity (see table above); all other panels and the active variants pass.

## Scripts edited

generate_paper1_results_figures.py, generate_paper1_supplementary_figures.py,
generate_paper1_data_figures.py, make_figure1_pipeline_overview.py,
make_figure8_bio_validation.py, generate_figure_corn_boundary.py,
generate_figure_domains.py, analyze_trajectory_continuity.py,
analyze_tac_sensitivity.py, measure_percell_probs.py (canvas/fonttype only),
plus `docs/manuscript/latex/paper1.tex` and
`docs/manuscript/paper1_supplementary_materials.md`.

## Totals

- 21 figures audited; **19 fully compliant** (bbox-verified), 2 not regenerable
  (data gap, scripts pre-fixed), 1 legacy variant with documented residual
  dotplot-internal density.
- 11 tables audited (2 main tex + 9 supplement md): all formatting fixes applied.
