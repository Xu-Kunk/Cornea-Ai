# this study figure system — build contract

Target venue: the target journal. Two-column width **178 mm (7.01 in)**;
single column **85 mm**. This document is the contract for the full rebuild of
the 20 figure set figures. If a figure violates it, the figure is wrong.

## 1. Non-negotiables

1. **One figure, one producer.** Each output stem has exactly one script that
   writes it. Never edit a rendered PDF/PNG, never post-process a figure in a
   second script, and never let two scripts write the same stem.
2. **No numeric literals in plotting code.** Every plotted value comes from a
   data file, `paper1_figure_data.MANUSCRIPT_VALUES`, or a computation over
   those. No hand-typed coordinates, bar heights, labels of counts, axis limits
   chosen to flatter a result, or hard-coded p-values.
3. **Every figure declares its intent** in the module docstring:
   - `core_claim`: one sentence the figure proves;
   - `hero_panel`: the single panel that carries the claim;
   - `stats_shown`: tests/intervals/CIs displayed, or `none`;
   - `source_data`: exact input paths (and the manifest records their hashes).
4. **All output goes through `paper1_figure_style.save_figure`.** No direct
   `fig.savefig`, no alternate PDF writer. That function enforces 600 dpi,
   white background, preserved canvas, TrueType fonts, and provenance.
5. **The manifest is the provenance record.** `figures/paper1/figure_manifest.json`
   is append-only per figure name: script, UTC timestamp, output hashes, input
   hashes, physical size. A figure is not "done" until its entry exists.

## 2. Shared modules

- `scripts/paper1_figure_style.py` — rcParams, `figsize()`, `panel_label()`,
  `save_figure()`, `check_value()`, arm/cell-type palettes.
- `scripts/paper1_figure_data.py` — fail-fast loaders, path resolvers, named
  accessors, and `MANUSCRIPT_VALUES`.
- `scripts/paper1_figure_data.resolve_data_path("gse153515_annotated.h5ad")`
  returns `data/processed/...`, not the legacy `data/...` path.

Call `set_style()` once at the start of every producer. It is idempotent.

## 3. Arm vocabulary (canonical)

| key | display label |
|---|---|
| `frozen` | frozen |
| `ce_naive` | CE-naive |
| `ordinal` | ordinal head |
| `ordinal_cont` | ordinal + continuity |
| `lora` | LoRA |
| `ordinal_lora` | ordinal + LoRA |
| `caft_intended` | CAFT-intended |
| `hvg_pca` | HVG/PCA |

Retired aliases — **never** use in a new figure: `CORN`, `ordinal-naive`,
`naive FT`, `naive`, `Fine-tune λ=0`. Import `ARMS`/`ARM_COLORS` from the
style module instead of retyping a legend.

## 4. Typography and geometry

- Type floor: **7 pt at final printed width**. Tick and legend labels are 7 pt;
  axis labels 7.5 pt; panel letters 8 pt bold via `panel_label()`.
- Use `figsize(width_mm, height_mm)` and check the return in the manifest.
  Do not scale a figure in LaTeX to hide oversized type.
- Arial only, embedded as TrueType (`pdf.fonttype=42`). A PDF containing a
  Type 3 font fails the build; `save_figure` raises before returning.
- Top and right spines off by default. Panel letters use the shared offset.

## 5. Colour rules

- Arm colour is assigned by `ARM_COLORS` and is never reused for a cell type,
  donor, dataset, or heatmap category.
- Cell types use `CELLTYPE_COLORS` only (`LSC`, `TAC`, `basal`,
  `differentiated`). These are disjoint from every arm hex.
- One meaning per colour across all 20 figures. If a new category is needed,
  add it deliberately to the shared module, not locally.
- Heatmaps use `DIVERGING` (signed) or `SEQUENTIAL` (magnitude), never a
  rainbow map, and always include a labelled colour bar with units.
- Accessibility: adjacent arms in `ARMS` order remain distinguishable under
  deuteranopia; do not reorder a legend to break that pairing.

## 6. Data and statistics

- Load with `paper1_figure_data` helpers; a missing file raises with the
  absolute path. Do not catch and continue.
- `load_csv` returns a DataFrame; `load_npz` returns the archive; `load_json`
  returns the parsed document.
- Every plotted figure set number is asserted with
  `check_value(label, value, MANUSCRIPT_VALUES[...])` before it is drawn.
- Report the statistic actually shown: n, interval type, seed count, and
  correction. `stats_shown: none` is honest; an unlabelled bar is not.
- Never invent a value. If a number is absent from data and
  `MANUSCRIPT_VALUES`, stop and escalate.

## 7. QA loop

1. **Render** from a clean interpreter with the repo `.venv`.
2. **Visual check** at final width: type ≥7 pt, no clipping, no overlap, panel
   letters aligned, colour meanings consistent, hero panel obvious.
3. **At most two revision rounds** per figure. Each round fixes only the listed
   defects and re-renders through `save_figure` (new manifest entry).
4. If the figure still fails after two rounds, **stop and escalate** to the
   orchestrator with the defect list and the current manifest hashes. Do not
   start a third speculative redesign.

## 8. Definition of done

- One producer script, docstring fields present, no numeric literals.
- PDF + PNG written by `save_figure`; `pdffonts` shows no Type 3.
- PNG is 600 dpi; canvas size equals the declared mm size.
- Manifest entry matches the hashes on disk and lists every input.
- Figure passes the two-round visual check, or is escalated with evidence.

## the journal house style (measured 2026-10-19)

Derived from three published *the journal* papers already in the local library
(scELMo 2026, BioLLM 2025, GSFM 2026; `docs/SCI-reference/01-scFM/`).
Page width 603.4 pt, single-column full-width text block 550.6 pt, body text
7.62 pt bbox, in-figure text 6.28-6.72 pt bbox (approximately 7 pt type).

| # | Rule | Detail |
|---|---|---|
| 1 | Footprint | Full-width artwork occupies ~483-490 pt of the 550.6 pt measure, centred, with ~25-30 pt white each side. Never edge-to-edge. |
| 2 | Caption | Below the artwork, left-aligned; bold "Figure N." + bold short title, then prose. **Panel descriptors "(A) ... (B) ..." live in the caption, not beside the panels.** |
| 3 | Panel letters | Uppercase bold, plain, no box or fill; outside top-left; ~1.3-1.5x axis-label height. Single-panel figures carry no letter. |
| 4 | In-figure titles | Short, **regular weight**, centred, about axis-label size, and only where needed. Long explanation stays in the caption. |
| 5 | Axes | Thin (~1 pt) frame; ticks point **outward**, short (~2 pt), left/bottom only; axis text never bold. |
| 6 | Axis wording | Units or direction in parentheses, e.g. "Running times (s)", "HumanEval Score (up)". |
| 7 | Legends | **Unframed**; outside right or in an empty inside corner; marker/swatch-only keys; one shared key for stacked sub-panels. |
| 8 | Colour | 2-8 hues, mid-to-high saturation; **grey carries the reference/baseline role**; hue first, marker shape second. |
| 9 | Statistics | Test, threshold, correction and n live **in the caption**. In-artwork marks are minimal: asterisks appended to labels, or small brackets. No floating "n = ..." above groups and no p-value text inside the plot area. |
| 10 | Uncertainty | Distributions (box / violin / jittered points) are preferred over capped error bars; the caption states what the interval is. |
| 11 | Density | 2-5 lettered panels, each possibly containing 2-4 sub-axes; ~20-30 pt gutters; near-square sub-panels. |
| 12 | Recurring | One type family; black text except where colour is the datum; **all explanatory prose and all statistics outside the artwork**. |

### Consequences for this figure set

- No explanatory footnote, statistics paragraph or claim sentence inside any figure.
- No framed legend, no hatching, no filled header pills, no rounded container boxes around analysis panels.
- Panel letters at 9.5 pt (set by `panel_label`); ticks outward at 2 pt (set by `set_style`).
- Figures are still built at 178 mm because that is the figure set's text width; the
  journal's own 25-30 pt inset is a layout decision made after acceptance.
