> 2026-10-09：MJS（Patterns #1 → CRM #2）取消 —— Cell Press 官方不允许单刊 MJS；改为**直投 Patterns**。
> 2026-10-19：v26 接接入。审阅包文字升为 active 正文；v25 归档。

# Patterns active package — Paper 1

## Current manuscript

| Item | Value |
|---|---|
| Title | *Annotation accuracy is not a proxy for differentiation order in fine-tuned single-cell foundation models* |
| Version | **v28** (post-check corrections + SI layout/guidance build, 2026-10-21) |
| Main source | `latex/paper1.tex` |
| Bibliography | `latex/paper1.bib` (87 entries) |
| Compiled PDF | `latex/build/paper1.pdf` — 35 pages, 0 errors / 0 undefined / 0 overfull (`7d55fbab…`) |
| Supplementary source | `supplementary/supplementary_information.tex` |
| Supplementary PDF | `supplementary/build/supplementary_information.pdf` — 27 pages, 0 errors / 0 undefined / 0 overfull (`99bb6412…`); 178 mm text block so the 178 mm figure sources print 1:1 |
| Submission-copy SI | `paper1_supplementary_information.pdf` (byte-identical to the build) |
| Authors | Kunhui Xu, Chenchen Wang, Fenfen Li, Xiaoyu Chen, Xinyu Wang, Yi Zhang, Tiancheng Pu, Jiahao Tang, Hongguang Cui, Peng Guo, Qi Dai; Hongguang Cui, Peng Guo & Qi Dai (co-corresponding); Qi Dai (lead contact) |
| Target | Cell Press *Patterns*, Research Article, direct submission |

## Provenance hashes (2026-10-19)

Authoritative full hashes live in `CHECKSUMS.sha256` (`sha256sum -c CHECKSUMS.sha256`, 7/7 pass). The abridged values below are for quick eyeballing only and were re-derived on 2026-10-19 after the figure-system rebuild.

| File | SHA-256 (abridged) |
|---|---|
| `latex/paper1.tex` | `efd23d11…` |
| `latex/paper1.bib` | `912b9c27…4e9ffa` |
| `latex/build/paper1.pdf` | `c3d1b0fd…` |
| `supplementary/supplementary_information.tex` | `45f4f2fd…` |
| `supplementary/build/supplementary_information.pdf` | `ce869b72…` |
| `figures/paper1/figure3_biological_readouts.pdf` | `0a796383…a1516bc` |
| `figures/paper1/figure_manifest.json` | `833dff2e…`; 20 entries, each with script + input/output SHA-256 |

> Earlier revisions of this table carried pre-rebuild values (`61556542…`, `f4ee5359…`,
> `dfc89942…`, `40936214…`, `d6722cde…`). They no longer match the files and must not be quoted.

## Version history

| Version | Date | State |
|---|---|---|
| v25 | 2026-10-16 | Archived: `latex/archive/v25-2026-10-16/` (source, PDF, bib, build) |
| v26 | 2026-10-19 | **Active** |

## What changed at v26

Text promoted from the 2026-10-17 review revision (Intro 717 / Results 2,779 / Discussion 934 + Limitations 202 / Methods 14 sections / Summary 133), plus:

- Results §4: classical-baseline comparison paragraph (SI Table S10).
- Discussion ¶1: one sentence on the classical baselines.
- Restored the Discussion readout sentence that the direct-style pass had dropped
  ("…identified by fixed expression differences, not by axis-dependent gradient retention").
- Figure 6 caption: legacy arm alias `CORN` removed.
- Bibliography: 87 entries (supersedes the v25 86-entry file; adds `emmaemb2026` and 15 audited metadata corrections).

### v26.1 — submission-hygiene corrections (2026-10-19)

Two line-level corrections made while assembling `submission/`; no scientific content changed:

- Removed the `Review copy --- 2026-10-17.` line after `\maketitle` (was printed on page 1 of the PDF).
- Results §1: the rare-TAC per-class pointer corrected from `Figure S8` to `Figure S10`
  (S10 is the TAC-sensitivity figure; the reference at §3 was already correct).

Recompiled: 35 pages, 0 errors / 0 undefined / 0 overfull (11 underfull, unchanged).
`latex/paper1.tex` → `8cd6d7e3…`, `latex/build/paper1.pdf` → `7453ed2a…`.

### v26.2 — title change (2026-10-19)

Title changed from the descriptive *“Classification gains coexist with differentiation-gradient loss in single-cell foundation models”* to the declarative **“Annotation accuracy is not a proxy for differentiation order in fine-tuned single-cell foundation models.”**

Rationale: Patterns evaluation papers title on a present-tense proposition (cf. *Generalization of biological foundation models requires prospective validation across contexts*, Patterns 2026). `is not a proxy for` states the non-implication the data support — accuracy rose (0.4421 → 0.6492) while the gradient fell (0.8064 → −0.0082) with layer adjacency flat (0.9665 → 0.9768) — without claiming universal collapse. `differentiation order` also removes the “optimization gradient” ambiguity that `differentiation-gradient` carried in a title about fine-tuning, and matches the Summary's own wording.

Changed in four files: `latex/paper1.tex` (`\title`), `README.md`, `PATTERNS_ACTIVE_PACKAGE.md`, `submission/01_cover_letter.md`. Body terminology (`differentiation gradient`) is unchanged.

Recompiled: 35 pages, 0 errors / 0 undefined / 0 overfull.
`latex/paper1.tex` → `ff093924…`, `latex/build/paper1.pdf` → `56ba6e02…`,
`submission/MANIFEST.sha256` → `2ff026e1…`.

### v26.3 — Results restructure and an implications paragraph (2026-10-19)

Structural edit only; no numbers, data, or claims changed.

- **Results split into six sections, each carrying a proposition.** The former
  “Fine-tuning outcomes vary across datasets, model configurations, and checkpoints”
  was not a claim and carried six datasets in one section. It is now:
  - *The gradient loss shrinks out of corpus and reverses in epidermis* (GSE299152,
    GSE147424, GSE202352, official-budget control, GSE186433/GSE155683).
  - *Loss depth and seed sensitivity differ across architectures* (van Zyl arm,
    scGPT checkpoints, head initialization, CellPLM).
- **Heading de-duplication.** “Fine-tuning” appeared in four of five headings. Now:
  *Adaptation flattens…* and *Controls reproduce the loss and separate the readout
  from the axis choice*.
- **Discussion gains a closing implications paragraph** (“Three checks follow for
  practice…”), stating what a reader should do: evaluate the ordering the downstream
  analysis needs, read intermediate layers, treat seed/checkpoint selection as part of
  the result, and note that stage-ordered supervision repairs a known corpus but does
  not safeguard an unseen one. This also introduces the manuscript’s only `However`.

Recompiled: 35 pages, 0 errors / 0 undefined / 0 overfull.
`latex/paper1.tex` → `28f9bbb1…`, `latex/build/paper1.pdf` → `fe243e7b…`,
`submission/MANIFEST.sha256` → `6cdedce2…`.

### v26.4 — Introduction compression and sentence splits (2026-10-19)

Readability edit only; no numbers, data, or claims changed.

- **Introduction: 717 → 472 words.** The three per-paper literature paragraphs
  (146 + 149 + 138 words, 13 citations, each “X reports Y. Its limitation is Z.”)
  became two grouped paragraphs: one on annotation/adaptation benchmarks, one on
  measurements of order in frozen or adjacent tasks. Each ends with the gap it leaves.
  All 19 citation keys in the Introduction are preserved; none was dropped or moved.
- **Seven balanced `while/whereas` sentences split** where the subordinate clause carried
  the finding (Results §1 ×2, §3 budget control, §4 ×2, §6 FDR; Discussion ¶1).
  Connective profile over Intro+Results+Discussion: `while` 20 → 14,
  `rather than` 13 → 11, `however` 0 → 1, `because` 0 → 2.

Recompiled: 35 pages, 0 errors / 0 undefined / 0 overfull.
`latex/paper1.tex` → `5ce4937e…`, `latex/build/paper1.pdf` → `0a29930f…`,
`submission/MANIFEST.sha256` → `5878f6dc…`.

### v26.5 — read-through corrections (2026-10-19)

Ten defects found by a full read-through plus an independent cross-reference and
numeric audit. No scientific claim changed.

| # | Line | Correction |
|---|---|---|
| 1 | Results §4 | Ordinal+LoRA final-checkpoint macro-F1 `0.9634` → **`0.9635`** (SI Table S6 per-seed mean 0.96347) |
| 2 | Results §6 | “Six classical baselines — harmony, **three** HVG/PCA variants, and scVI” → **four** (SI Table S10 lists four) |
| 3 | Figure 6 caption | z-scores attributed to “Tables S7 and S8” → **Table S7** only (S8 has no z column) |
| 4 | Figure 5 caption | Removed “See also Table S3” (S3 is same-corpus layerwise ρ; the three-domain values are main Table 2) |
| 5 | Results §4 | Removed the duplicated sentence “Representative single-run axis profiles are shown in Figure 3B”, which also appears in Results §2 |
| 6 | Methods §3 | GSE153515 QC retention `89.9%` → **`90.0%`** (15,827/17,587 = 89.99%) |
| 7 | Results §6 | Added a closing synthesis sentence after the FDR family description |
| 8 | Results §6 / Methods §7 | Added the missing Figure S12 pointer (DPT) and Figure S10 pointer (TAC sensitivity) |
| 9 | Table 1 note | Added: CAFT-intended rows did not execute the continuity objective (§5) |

Deliberately **not** changed: CellPLM’s 38,233 fitting / 4,249 validation cells. The source
manifest (`results/paper1_vanzyl/cellplm/data_manifest.json`) records these values, so they
differ from scGPT’s 38,235 / 4,247 by two cells for a real pipeline reason, not by typo.

Recompiled: 35 pages, 0 errors / 0 undefined / 0 overfull.
`latex/paper1.tex` → `7283b92f…`, `latex/build/paper1.pdf` → `5324a36a…`,
`submission/MANIFEST.sha256` → `26e4711d…`.

### v27 — CellPLM dose-response (2026-10-19)

Three new CellPLM arms were run on the remote GPU (RTX PRO 5000 72GB, host `summer-NUC13RNGi7`, reachable as SSH `p5070`) to test whether the CellPLM collapse depends on the training budget. Scripts `/home/summer/cellplm_vanzyl/scripts/run_cellplm_vanzyl_e3.py` (sha256 `3c98fc39fac7…`) and `run_cellplm_corn_lora.py` (`2c1b31185dc8…`); 18 runs, seeds 42/123/456, test donor Hu0216 unchanged.

| arm | n | ρ | macro-F1 | epochs |
|---|---|---|---|---|
| frozen | — | 0.8652 | — | — |
| CE | 3 | +0.8792 ± 0.0122 | 0.8347 | 3 |
| CE | 3 | +0.8622 ± 0.0440 | 0.9592 | 15 |
| CE | 3 | +0.7791 ± 0.0946 | 0.9712 | 30 |
| CE | 3 | +0.1223 ± 0.2905 | 0.9721 | 60 |
| CE | 3 | −0.0071 ± 0.0346 | 0.9728 | 83–118 |
| ordinal + LoRA | 3 | +0.9226 ± 0.0031 | 0.9144 | 15 |
| ordinal + LoRA | 3 | +0.9217 ± 0.0003 | 0.9499 | 143 / 154 / 454 |
| ordinal only (existing) | 3 | +0.9180 ± 0.0021 | 0.9728 | 78–129 |

**Finding.** The CellPLM collapse is not immediate. At the Geneformer protocol budget (3 epochs) CellPLM retains the gradient at 0.879; macro-F1 saturates by 30 epochs (0.971 of a 0.973 final value) while ρ continues to fall to 0.122 (60 epochs) and −0.007 (convergence). The loss therefore follows continued optimization after task saturation, not the approach to accuracy. Ordinal + LoRA holds ρ at 0.922 at the convergence budget even in a seed that ran 454 epochs.

**Why this matters.** The manuscript previously juxtaposed CellPLM's convergence-budget −0.007 with Geneformer's three-epoch −0.008 as a cross-architecture spectrum. That comparison was budget-confounded; the control quantifies the confound at Δρ ≈ 0.886. The claim is now stated as a difference in the *timing* of the loss.

**Changed files.** New Results subsection *The gradient loss follows task saturation* (7 subsections now); Figure 6 gains panel C (layout 2×2), produced by `scripts/figure6_cross_architecture_spectrum.py`; §4 CellPLM paragraph now forward-references the budget dependence; Discussion gains the onset sentence; Summary gains "timing" and "training budget". Results archived under `results/paper1_vanzyl/cellplm_e{,3,15,30,60}` and `cellplm_corn_lora{,_conv}`; embeddings under `artifacts/paper1_vanzyl/`; the six original convergence JSONs were also pulled from the remote for the first time.

Changelog: `docs/changes/2026-10-19-cellplm-dose-response.md`.

Recompiled: 35 pages, 0 errors / 0 undefined / 0 overfull (Summary 136 words).
`latex/paper1.tex` → `d7ad97eb…`, `latex/build/paper1.pdf` → `2a53263b…`,
`figures/paper1/figure_manifest.json` → `833dff2e…`, `submission/MANIFEST.sha256` → `bb50f0c5…`.

### v28 — SI Table S11 (2026-10-19)

Added `Table S11. Related to Figure 6: CellPLM training-budget dose response` to the Supplementary Information: one row per CellPLM arm with per-seed epochs, ρ and macro-F1 plus mean ± SD, including the frozen reference and both convergence rows. The SI is now 26 pages (Tables S1–S11). The main-text Supplemental Information Index and the new Results subsection both reference S11; the SI copy `paper1_supplementary_information.pdf` was refreshed and is byte-identical to the build.

Recompiled: main 35 pages, SI 26 pages, both 0 errors / 0 undefined / 0 overfull.
`latex/paper1.tex` → `efd23d11…`, `latex/build/paper1.pdf` → `c3d1b0fd…`,
`supplementary/supplementary_information.tex` → `45f4f2fd…`,
`supplementary/build/supplementary_information.pdf` → `ce869b72…`,
`submission/MANIFEST.sha256` → `e09c7a12…`.

### v28 post-check corrections (2026-10-20)

Submission-time fact corrections and the Figure 1 alignment pass, followed later the same
day by a two-sentence Discussion edit (see the conclusion-pass sub-block below); no new
experiments and no further text or caption changes beyond the items listed. The SI was
never touched.

- `latex/paper1.tex:79` — Introduction summary recast so the separate Geneformer
  ordinal-head recipe is stated to lose its advantage on the post-cutoff test
  (the prior combined same-corpus/GF-ordinal phrasing was split).
- `latex/paper1.tex:163` — Figure 1 caption: (C) coverage varies by model and domain;
  (D) the middle readout is named layer adjacency accuracy (LAA); (E) describes a
  positive-association test under BH–FDR, explicitly *not* a test of differences
  between adaptation arms.
- `submission/02_highlights_etoc_bigpicture.md` — Bigger Picture closing sentence now
  refers to the new corneal dataset (Bigger Picture 278 words).
- Figure 1 artifacts refreshed end-to-end from the producer
  `scripts/figure1_layout_pptx.py` (layout spec, FODP → PPTX → PDF → PNG, manifest
  `figure1` entry); the panel-D readout card was changed to LAA.

Recompiled with the existing flow (`pdflatex → bibtex → pdflatex ×2`): main **35 pages,
0 errors / 0 undefined / 0 overfull** (11 underfull, unchanged). The SI is byte-identical
and was not recompiled.

Current package hashes after this first post-check pass (2026-10-20) — superseded by the
conclusion-pass table below:

| File | SHA-256 (abridged) |
|---|---|
| `latex/paper1.tex` | `594013ad…` |
| `latex/build/paper1.pdf` | `198ea832…` |
| `figures/paper1/figure1_pipeline_overview.pdf` | `62aa3bba…` |
| `figures/paper1/figure1_pipeline_overview.png` | `54cfab78…` |
| `figures/paper1/figure_manifest.json` | `f4ebe5e1…` |
| `submission/MANIFEST.sha256` | `f7cde83f…` |
| `latex/paper1.bib` (`912b9c27…`), `supplementary/supplementary_information.tex` (`45f4f2fd…`), `supplementary/build/supplementary_information.pdf` and its copy (`ce869b72…`) | unchanged |

Reassembly: `python3 scripts/assemble_patterns_submission.py` then `--check` both exit 0;
`sha256sum -c MANIFEST.sha256` 42/42 pass; `03_manuscript.pdf` is byte-identical to
`latex/build/paper1.pdf` and `04_supplementary.pdf` to the SI build. Hand-authored
`01`/`02`/`07`/`08` are unchanged. G1–G4 and the holds listed above remain open.

#### Conclusion pass — Discussion closing paragraph (2026-10-20, later)

Two sentences in the Discussion closing paragraph (`latex/paper1.tex:267`) were replaced:

- “Stage-ordered supervision was the cheapest intervention that worked in our same-corpus
  tests.” → “Stage-ordered supervision retained the gradient in our same-corpus tests.”
- “However, it did not transfer out of corpus, so it repairs a known corpus rather than
  safeguarding an unseen one.” → “However, the separate Geneformer ordinal-head recipe lost
  its advantage on the post-cutoff test; the combined ordinal+LoRA recipe has not been
  evaluated out of corpus.”

Only these two sentences changed: inverse substitution reproduces the previous
`594013ad…` tex byte-for-byte, so the Summary and every other source are unchanged. The
same `pdflatex → bibtex → pdflatex ×2` flow gives main **35 pages, 0 errors / 0 undefined /
0 overfull** (11 underfull, unchanged); the content change is confined to **PDF page 15**
with no float repagination.

Current package hashes (2026-10-20, conclusion pass):

| File | SHA-256 (abridged) |
|---|---|
| `latex/paper1.tex` | `744dd81f…` |
| `latex/build/paper1.pdf` | `adc1b703…` |
| `submission/MANIFEST.sha256` | `6f739ed7…` |
| `latex/paper1.bib` (`912b9c27…`), `supplementary/supplementary_information.tex` (`45f4f2fd…`), `supplementary/build/supplementary_information.pdf` and its copy (`ce869b72…`), `figures/paper1/figure1_pipeline_overview.pdf`/`.png` (`62aa3bba…`/`54cfab78…`), `figures/paper1/figure_manifest.json` (`f4ebe5e1…`) | unchanged |

The first-pass table above (`594013ad…` / `198ea832…` / `f7cde83f…`) is now a dated,
superseded intermediate snapshot, not the current package. Reassembly re-verified:
`assemble_patterns_submission.py` and `--check` exit 0; `sha256sum -c MANIFEST.sha256`
42/42; `03` byte-identical to the build; `04`/SI copy byte-identical to the SI build;
all 20 figure sources unchanged. G1–G4 and holds remain open.

Changelog: `docs/changes/2026-10-20-paper1-figure1-factual-alignment.md`.

### 2026-10-21 — SI layout + guidance-figure correction build (current snapshot)

Built from two authorised source lanes: the SI layout pass (`supplementary_information.tex`,
designer lane) and the guidance/caption corrections in `latex/paper1.tex` (guidance lane).
No original data, training runs or figure artwork were regenerated; no scientific numbers changed.

**What this build contains**

- SI: automatic S-numbering with no hand-repeated “Figure Sx.” label; 178 mm text block so the
  178 mm figure sources print **1:1**; Table S2 and S4 column re-layout (~10 pt); Table S5 heading
  protected with `needspace`; one DOI gain “j.”; SI Related-to anchors synchronised with the main
  SI index; Table S11 states the logit convention explicitly. The last mechanical edit was the
  Figure S12 caption (left panel: DPT vs the frozen stage-centroid reference; right panel: three-seed
  mean line with a per-decile min–max band), recompiled here.
- Main: Figure 6 panel A caption now matches the artwork — Geneformer official-budget row draws the
  **three** plotted seeds 42/123/456 (mean 0.096; the six-seed 0.058 stays in the text and Table S8),
  five-colour key (frozen dark gray; CE-naive blue, one blue for every CE row; ordinal head green;
  LoRA pink-purple; ordinal + LoRA orange-red) plus the point / black mean-tick / dotted ρ = 0
  semantics, with per-row table pointers (S8; S6–S7; S9; S11). Figure 3C drops the unverified
  “Wang 2020” attribution (keeps GSE147424 and 18,355 cells). Supplementary Information Index:
  Table S4 → Table 1, Table S9 → Figure 6, Table S10 → Table 3.

**Build**

| Item | Value |
|---|---|
| Main | `pdflatex → bibtex → pdflatex ×2`; **35 pages**, 0 error / 0 undefined / 0 overfull, 11 underfull; `7d55fbab…` |
| SI | pure `pdflatex`, three passes (aux stable from pass 2); **27 pages**, 0 error / 0 undefined / 0 overfull, 69 underfull (count read from the designer trial log before recompiling; no claim made against the old narrower layout); `7eb9469f…` (pre-wording pass) → `99bb6412…` after the S2/S4 wording pass below |
| Main content diff vs the 2026-10-20 build | pages 10, 13, 28 only (the three edited locations); page count unchanged at 35, no float re-pagination |
| SI copy | `paper1_supplementary_information.pdf` synced, byte-identical to the SI build |

Current hashes (2026-10-21, after the S2/S4 wording pass): `latex/paper1.tex` `99c6b236…`; SI
`supplementary_information.tex` `15208dab…`; `latex/paper1.bib` `912b9c27…` (unchanged);
`figures/paper1/figure_manifest.json` `f4ebe5e1…` (unchanged); `submission/MANIFEST.sha256`
`dc74942d…`. CHECKSUMS header stays **v28** with the “SI layout/guidance + S2/S4 wording
post-check 2026-10-21” note (not promoted to v29).

Reassembly: `assemble_patterns_submission.py` and `--check` exit 0; `sha256sum -c MANIFEST.sha256`
42/42; `03` byte-identical to the main build and `04`/SI copy to the SI build; all 40 figure copies
byte-identical; hand-authored `01`/`02`/`07`/`08` unchanged; the `09` COI draft is outside the
42-file machine set and keeps its final attestation Off.

**2026-10-21 (later) — S2/S4 wording pass (author-confirmed).** The only source change is four
authorised text replacements in `supplementary_information.tex`; no number, table value or inference
was edited. S2 row 2 now reads “Classification safeguard / Historical CI-overlap screen
(descriptive) / PASS (screen)”; the S2 note states that observed macro-F1 values were close to the
naive arm and that overlapping bootstrap intervals are descriptive, not equivalence or
non-inferiority; S4 marks scFoundation “Unknown (pretraining corpus not publicly enumerated)” and UCE
“Inferred out-of-corpus (census 2023-04-25)”. These follow the existing Methods statements
(lines 346/401/403/457); **no formal non-inferiority test and no new corpus audit were performed.**
Rebuilt SI: **27 pages**, 0 error / 0 undefined / 0 overfull, 69 underfull, 26 `\headheight`
warnings (non-blocking; measured 13.8 pt header/body gap, no overlap); `99bb6412…`. The aux file is
unchanged, so all 26 Contents page references (14 figures + 11 tables + one Data range) still resolve.
Content diff versus `7eb9469f…` is confined to SI pages 17–18; every measured-value token (774) is
identical, and the only two raw-token differences are the label fragments “F1” (in “Macro-F1
non-inferiority”) and “95” (in “95% CIs”), both inside the rewritten S2 wording.

**Verification status.** `built; authorized layout/guidance visual verification PASS` (the earlier
2026-10-21 "visual verification pending" stage is closed). The observer reviewed the final main
`7d55fbab…` pages 10/13/28 and all **27/27** SI pages at 150 dpi; the parent session independently
re-checked the 7 sources and the 42 packaged hashes, the SI labels and the PDF copies. Scope of that
PASS is **this round's SI layout and guidance captions only** — it does not certify the statistical
conclusions and does not guarantee acceptance at *Patterns*.

That whole-issue PASS covers revision `7eb9469f…` and remains reusable for the pages the later wording
pass did not touch. The S2/S4 wording pass itself is **`built; S2/S4 wording-pass visual verification
PASS`**: the observer reviewed SI pages 16–19 at 150 dpi plus magnified S2/S4 crops, and pages 16/19
are pixel-identical to the previous revision (no new page break or orphaned table head). No research
data changed and no non-inferiority test or corpus audit was added.

Recorded, not adjusted: SI p19/p24 note pages carry large trailing whitespace; the p23 "Notes."
heading and its text straddle a page break; some long paths leave wider word gaps. The fancyhdr
`\headheight` warning is non-blocking — measured header/body minimum spacing is 13.8 pt with no
overlap — and is not claimed to have disappeared.

**Not claimed by this round.** No new statistical analysis or corpus audit was performed: the S2
CI-overlap screen is now labelled descriptive (“PASS (screen)”), and the S4 corpus states are marked
Unknown / inferred per the author-confirmed wording (see the wording-pass block above). The two
earlier open wording items are therefore **closed as reworded** — not resolved by new evidence, and
no formal non-inferiority test was run. Prior 2026-10-20 hash tables above are explicitly historical.

Changelog: `docs/changes/2026-10-21-patterns-si-layout-and-guidance.md`.

## Figure system

The system holds 19 figures (6 main, 13 supplementary). All 20 were rebuilt on
2026-10-19; supplementary Figure S8 was retired and the survivors renumbered S9→S8 … S14→S13 on 2026-10-22:

- **One producer per figure**: `scripts/figure1_pipeline_overview.py` … `scripts/figureS13_continuity_metrics.py`.
  The 13 superseded producers are archived in `scripts/archive/legacy-figures-2026-10-19/`.
  `scripts/generate_paper1_results_figures.py` is retained as a **library only** (its palette and
  save helpers are imported by `analyze_trajectory_continuity.py` and `sens_umap_fig.py`); its
  figure entry point is superseded and must not be run.
- **Shared contract**: `scripts/paper1_figure_style.py` (locked four-tier type scale, arm and
  cell-type palettes, structural ink, `save_figure` with PDF font-type self-check, 600 dpi PNG and
  a hash-recorded manifest) and `scripts/paper1_figure_data.py` (fail-fast loaders; no literals).
- **Provenance**: `figures/paper1/figure_manifest.json` records, per figure, the producing script,
  every input path and its SHA-256, and the output hashes. `figures/paper1/FIGURE_SYSTEM.md` is the
  build contract plus the measured *Patterns* house style.
- **Superseded artwork**: `figures/paper1/archive/superseded-2026-10-19/` (30 files).
- **Type floor check**: `scripts/qa_figure_type_floor.py` — 19/19 figures pass.
- **Captions**: rewritten on 2026-10-19 to carry the statistics and interval definitions that were
  removed from the artwork; `figures/paper1/CAPTION_DEBT.md` records the mapping.
- Changelog: `docs/changes/2026-10-19-paper1-figure-system-rebuild.md`.

## Open holds (not cleared by v26)

1. **Figure 1 / 6 / 8 B–D / 9** label and render-source synchrony. Figures still carry baked-in `CORN` legend text.
2. **Figure 8**: the historical full-figure PDF-to-renderer binding is still unidentified; the three-seed Panel A candidate is unadopted.
3. **Probability decoding**: the disagreement rate between threshold-count and repaired-probability argmax has not been measured; ECE is not claimed to calibrate the macro-F1 decision rule.
4. **CellPLM frozen baseline** varies across runs (ρ 0.8652 / 0.8026 / 0.8600). Cause not established; do not present 0.865 as a single stable reference.
5. **Bibliography**: two keys share one DOI (`ding2025scgptretina`, `ding2026protocol`); unresolved.
6. **Code availability**: `Xu-Kunk/Cornea-Ai` must be public before submission — hard gate.
7. **Grant number** not yet inserted; **Zenodo DOI** not yet minted.
8. **FDR families**: 21-test frozen family + 6 freeze-time pending; later-completed runs are outside that family by design.

## Submission package (`submission/`)

Built on 2026-10-19. `submission/` is a **generated** folder: never edit it by hand.

| File | Role |
|---|---|
| `submission/00_CHECKLIST.md` | Deliverable status + hard gates (hand-authored) |
| `submission/01_cover_letter.md` | Cover letter, v26 wording (hand-authored) |
| `submission/02_highlights_etoc_bigpicture.md` | Highlights / eTOC / Bigger Picture (hand-authored) |
| `submission/03_manuscript.pdf` | Copy of `latex/build/paper1.pdf` |
| `submission/04_supplementary.pdf` | Copy of `supplementary/build/supplementary_information.pdf` |
| `submission/05_figures_main/` | Figure 1–6, PDF + 600 dpi PNG (12 files) |
| `submission/06_figures_supp/` | Figure S1–S13, PDF + 600 dpi PNG (26 files) |
| `submission/07_declarations.md` | CRediT / interests / AI / inclusion and diversity (hand-authored) |
| `submission/08_data_code_availability.md` | Resource availability, verbatim from the manuscript (hand-authored) |
| `submission/MANIFEST.sha256` | SHA-256 of all 40 generated files |

Rebuild and verify:

```bash
python3 scripts/assemble_patterns_submission.py          # build + table
python3 scripts/assemble_patterns_submission.py --check  # verify only, no writes
```

The assembler verifies `CHECKSUMS.sha256` before copying, checks copies are byte-identical,
requires all 19 figures against `figure_manifest.json`, and refuses to touch `00`–`02`, `07`, `08`.

**Superseded material** moved to `archive/docs/` on 2026-10-19:
`cover_letter_patterns_2026-10-09.md` (old title, v18 wording),
`patterns_submission_extras.md` (three statements conflicting with v26), and the older
duplicate `submission_requirements.md`.

## Package policy

`latex/paper1.tex` is the single source of truth. The per-section Markdown drafts and the
Markdown→LaTeX assembler are frozen inside `archive/revision_2026-10-16/` and are no longer
authoritative. Never introduce a second editable copy of the manuscript text.

Historical v18 bioRxiv PDFs remain in `../archive/biorxiv-v18/` and `artifacts/paper1_preprint/`.

### v29 — pre-submission evidence and compliance pass (2026-10-21)

Author-scoped: necessary items only; no systemic style pass; no phenomenon naming.

**Scientific evidence added.** The headline is a difference between frozen and adapted ρ, but the
archived statistics tested only positive association. A paired donor-stratified cell bootstrap
(2,000 replicates, axis refit on the resampled cells in every replicate) is now reported in
Results §1 and described in Methods §14: Δρ = 0.815 (95% CI 0.737–0.898; 0 of 2,000 replicates
≤ 0), per-seed 0.752–0.849, same direction in both donors. LAA is now quantified against a
label-permutation null: +0.010 (95% CI 0.007–0.013) against a null mean of 0.642 and a 0.316
separation, i.e. it moves detectably but negligibly. Source: `results/paper1_vanzyl/claim_audit/
delta_rho_laa.json` (gate reproduces 0.8064 / −0.0082 / 0.9665 / 0.9768 to 3.9e-05).

**Wording changed.** Summary rewritten (145 words; gap-first, paired numbers, one caveat,
benchmark ending). "layer adjacency did not flag the change" replaced in Results §1 and the
Discussion by the quantified version, because the metric does move (CI excludes 0). Three
"not a test of differences" disclaimers reduced to one Methods statement plus pointers.
Discussion ¶1 "weakened substantially" → "fell to near zero".

**Compliance.** AI declaration retitled to the Elsevier standard "in the writing process"
(12/12 published Patterns research articles use it) and rewritten in 48 words: DeepSeek-V4 series
models, served locally, for language polishing and code development; no AI-generated data,
results or images. A matching Methods §14 sentence records that archived files are the record of
record. Resource availability gains the mandated closing sentence; the Code bullet no longer
asserts public availability (repo is still private); Qi Dai carries a Lead contact footnote;
contributions use initials; funding updated to the NSFC and Zhejiang provincial grants.

**Build.** 36 pages, 0 error / 0 undefined / 0 overfull (11 underfull); SI untouched.
`latex/paper1.tex` `92281f76…`;
`latex/build/paper1.pdf` = `submission/03_manuscript.pdf` = `c59f4af61065…`;
`assemble_patterns_submission.py --check` OK; `MANIFEST.sha256` / `CHECKSUMS.sha256` verified.

**Still open.** Repository public + DOI before acceptance; COI PDF Funding field and the
"On behalf of all authors" checkbox; Bigger Picture optional at first submission; the systemic
cadence pass (mean sentence length 18.1 vs 22.7–34.2 in the published corpus; semicolons
0.87/100 words; `rather than` 11 occurrences); the five repeated post-cutoff caveats; the
Methods-resident pending-configurations sentence still in Results §6.

### v30d — storyline, evidence and figure consistency pass (2026-10-22)

Author-scoped response to the external review, executed over four lanes.

**Evidence.** Two new Geneformer arms close the label-set and step-budget confounds: Arm 6 (epithelial-only
four-class CE, 27,878 cells, 2,616 steps, ρ 0.158 ± 0.049) and Arm 7 (same labels and supervised cells with the
51,806 non-epithelial rows retained but excluded from the loss, 7,473 steps, ρ 0.254 ± 0.022). Against the
ordinal arm the step-matched contrast is Δρ = −0.579 (pooled 95% CI −0.631 to −0.541); against frozen,
+0.552 (0.514 to 0.604). The frozen-versus-naive contrast now quotes the pooled interval 0.815 (0.692 to 0.908),
so one interval convention runs through the paper. CellPLM LoRA-only now reports three seeds
(ρ 0.551 ± 0.365; epochs 184/366/80), replacing the single high seed 0.873 that the earlier draft quoted.

**Retired from the paper.** All failed or non-executing methods (CAFT-intended, axis-anchored, soft-Spearman,
flow, the Geneformer LoRA OOM record, the archived basin criterion) are gone from text, tables and figures —
including SI Table S2, which is now *Label-set and step-budget controls* with per-seed rows. The underlying
records remain public under `results/` and are pointed to from Data and Code Availability.

**Consistency.** Summary rewritten (130 words, quotes 0.254); Methods arms renumbered 1–7 with the count
corrected to five adapted configurations; Limitations restructured into four named items; first-use definitions
added (CE, LAA, ECE, PCS, DPT, domain names); LAA labelled a low-sensitivity companion metric; identifier
line-breaking fixed (no mid-token breaks); author name order corrected to Fenfen Li; funding updated to
the NSFC grant; AI declaration set to the Elsevier "writing process" heading in 48 words.

**Figures.** Figure 3 and nine supplementary figures regenerated with the retired series removed and the
panel-C legend corrected (Wang 2020 → GSE147424); S5 replaced by CE-naive versus ordinal-head training
dynamics; S12 panel letters moved inside the visible area; seven SI captions rewritten. Zero retired-arm
strings across all 20 submission figures and the SI. The sole main-text hit is the genuine module path
`corneal_fm/caft.py` in the code-availability list, kept deliberately.

**Build.** Main 34 pp / 0 error / 0 undefined / 0 overfull (14 underfull); SI 26 pp / 0 / 0 / 0 / 63 underfull.
`latex/paper1.tex` `788f1ded…`; `latex/build/paper1.pdf` = `submission/03_manuscript.pdf` `db69e72d…`;
SI `supplementary_information.tex` `aa5b1975…`, SI PDF `639ac88b…` (build, synced copy and `04` all identical);
`figure_manifest.json` `280759b9…`; `CHECKSUMS.sha256` v30d, 7/7; `assemble --check` OK.

**Open before submission.** Repository public + DOI deposit; Cell Press Declaration of Interests PDF (Funding
field and the "On behalf of all authors" checkbox) — author only; whether to attach the Bigger Picture at first
submission (optional); main-figure reordering (reviewer suggestion) deliberately deferred to the revision stage.

**Shortlisted but not executed.** Systemic cadence pass (sentence length 18.1 → 22–25, `The`-initial share,
semicolon density, `rather than` count) and compression of the repeated post-cutoff caveats — both remain
available if a referee asks for tighter prose.

### v30e — author list, contributions and funding (2026-10-22)

*Author count and funding superseded by v30k below.*

Author-scoped metadata pass; no scientific text, number, table or figure changed.

**Authors and affiliations.** The title block now lists ten authors across two affiliations — The First
Affiliated Hospital, Zhejiang University School of Medicine (1) and the Eye Research Center, Hangzhou
Institute of Medicine, Chinese Academy of Sciences, Eye Hospital, Wenzhou Medical University (2) — with
ten ORCIDs. Hongguang Cui, Peng Guo and Qi Dai are corresponding; Qi Dai is lead contact. Only the three
corresponding e-mail addresses appear.

**Contributions and funding.** The CRediT statement covers all ten authors; funding is the single NSFC
grant (82571251). The Zhejiang provincial programme was withdrawn at the author's instruction. Mirrored
into `submission/07_declarations.md`, `01_cover_letter.md` and the `00_CHECKLIST.md` R4 row.

**Build.** Main 34 pp / 0 error / 0 undefined / 0 overfull (14 underfull); SI unchanged.
`latex/paper1.tex` `ef07a568…`; `latex/build/paper1.pdf` `1d559137…`; `CHECKSUMS.sha256` v30e, 7/7;
`assemble_patterns_submission.py --check` OK.

### v30f — author list and funding (2026-10-22)

*Ten-author list and single-grant funding superseded by v30k below; kept as history.*

Ten authors with two affiliations, three corresponding authors (Hongguang Cui, Peng Guo, Qi Dai) and Qi Dai as lead contact; the previous single Wenzhou affiliation and the four-author block are gone from the main title page and from the SI cover, which now match. Funding reduced to the National Natural Science Foundation of China (82571251); the Zhejiang programme (WKJ-ZJ-26067) was withdrawn at the author's request and no longer appears in the manuscript, the declarations, the checklist or the cover letter. Contributions cover all ten authors.

Main `latex/paper1.tex` `ef07a568…`; `latex/build/paper1.pdf` = `submission/03_manuscript.pdf` `1d559137…` (34 pp); SI tex `840e5d95…`; SI PDF (build = synced = `04`) `2cbc81d8…` (26 pp); `CHECKSUMS` v30f 7/7; `MANIFEST` `7d1b9b57…`, `--check` OK.

Outstanding and author-only: the Cell Press Declaration of Interests PDF was completed against the old four-author list and must be redone (affiliations, funding, final attestation).

### v30h — legend budget and audit-language pass (2026-10-22)

Main figure legends compressed to an internal budget (\u2264120 words; tables \u2264100), measured against a corpus of 12 published *Patterns* papers (median legend 136 words, ~33 words per panel): Fig 1 155\u2192119, Fig 2 101\u219296, Fig 3 187\u2192119, Fig 4 123\u2192110, Fig 5 125\u2192111, Fig 6 297\u2192121, Table 2 82\u219261. Colour keys, panel letters, axis meanings and every statistical definition stated in a retained clause were preserved; the cut order was defensive asides \u2192 duplicated interval definitions \u2192 long table-pointer strings \u2192 panel minutiae already in Methods. The budget itself is not stated in the manuscript.

Audit-language pass: six Methods sentences reworded or deleted (`archived`/`recorded` adjectives, two "retained for traceability" clauses, an environment-lock clause, `were retained with` \u2192 `are reported with`, and a "record of record" tail), plus one orphaned meta-sentence. A full-text sweep found only four defensive constructions, all retained because they limit claims rather than defend them.

Current: `latex/paper1.tex` `d9653514\u2026`; `latex/build/paper1.pdf` = `03_manuscript.pdf` `13de5fde\u2026` (34 pp, 0/0/0, 14 underfull); SI unchanged (26 pp); `CHECKSUMS` v30h 7/7; `assemble --check` OK.

### v30i — Figure 1 redrawn from the editable deck (2026-10-22)

Figure 1 is now maintained in `design/figure1_final.pptx` and exported to PDF/PNG; the layout-spec generator is retired (marked DO NOT RUN) and the spec table plus `.fodp` are historical records. The manifest records the PPTX as the authoritative input.

Panel E (statistical control) is deleted — its observed dot sat inside the null envelope and read as a non-significant result, contradicting the legend, and the legend/artwork disagreed about whether arm contrasts were tested. Panels A–D were rebalanced into the freed space: one 7.0 mm gutter throughout (three different values and a 14.2 pt vertical break before), panel C promoted to a real card, panel D given the header band the other panels already had, D's column dividers darkened from 0.51 pt/1.41:1 to 0.8 pt/3.36:1, D's third column corrected from ten outlined shapes collapsing into nine to ten and ten with 2.83 pt gaps, C's 0.1 pt hatch textures replaced by a solid tint with a 0.6 pt 7:1 border, the mis-coloured 3.4 pt bar in B replaced by a text label, three placeholder phrases removed, the smallest type raised from 7.0 to 7.5 pt, and all connectors re-anchored with a new B/C→D arrow. The legend loses its (E) sentence and is now 93 words.

Fig 1 PDF `2efd5e74…`, PNG `e4d67f0e…`, PPTX `e862ad53…`; main PDF 34 pp; `CHECKSUMS` v30i 7/7; `assemble --check` OK.

### v30j — Figure 1 second-pass remediation (2026-10-22)

*Hashes and page-state below are v30j; v30k at the end of this file supersedes them.*

An independent audit rejected the v30i artwork. The deck was edited in place and re-exported
through the unchanged pipeline (LibreOffice 24.2 → PDF, `pdftoppm -r 600` → PNG). Seven
blocking defects were fixed; every value below was measured on the export, not read off the
edit script.

**Fixed.** (1) `B26` read “HVG/PCA” above the only reference glyph row (`B13_1..3`, y 29.8 mm),
whose stroke is `#4D4D4D` — the *frozen* colour — leaving “HVG/PCA” with no row anywhere; it now
reads **“frozen · all 3 models”** (8 pt bold, centred on the glyph group at x 98.7 mm), which also
removes the duplicated “HVG/PCA” text that survives only in the reference key. (2) The orphan rule
`S01` (50 mm at x 100.5–150.5, y 68.6, bridging the B|C gutter) is deleted, and note `S02`
(“Coverage varies by model and domain”) moved inside panel C under its third row. At the 7.5 pt
floor the sentence is 45.3 mm wide against 44.4 mm of usable card interior, so it is set on two
lines; C’s content block was lifted 5.9 mm (sub-heading 5.5 mm) to give it a 2.25 mm gap above and
a 3.03 mm margin below, and it is italic — the only italic in the figure — because panel C renders
sub-heading, row sub-lines and note in one style. (3) `F2_shaft` ran x 116.5→128.0, starting 1.2 mm
right of the CellPLM glyph and stopping 1.0 mm short of card C, so it read as “CellPLM alone feeds
the domains”; re-anchored to the card edges, 122.0→129.0 at y 31.8, heads moved to x 127.3.
(4) The missing C→D connector is added: `F4` (shaft x 151.5, y 67.2→72.9, apex 74.6) is
geometrically identical to `F3` and centred on C’s 129–174 mm span; the two vertical connectors
now match in length, weight, head and vertical span. (5) Panel A reconciled to the manifest:
`10 studies · 49 donors` → **`10 studies · 49 train donors`**, and `20,394 held-out cells` →
**`20,394 held-out cells` + `(25 donors)`** on a second 7.5 pt line (the joined string is 49.2 mm at
9 pt, 41.0 mm at 7.5 pt, against a 36 mm box); A’s block below the counts shifted 2.8 mm to keep
the split bar’s clearances. (6) “held-out” no longer carries two meanings: panel C’s first row is
**`external holdout`** (`GSE153515 · same tissue` unaltered), panel A keeps the internal split.
(7) The held-out segment’s diagonal pattern drew as ~0.2 pt hairlines (0.035 mm, ≈12 % ink) and
would not reproduce: replaced by a flat `#D9D9D9` fill whose outline becomes a 0.6 pt white
delimiter at the train/held-out junction (measured seam 0.21 mm); C’s two row separators darkened
`#D9D9D9` → `#BFBFBF` (1.41:1 → 1.95:1), and A’s block rule was matched to the same grey for the
same reason.

**Measured on the export.** Canvas 178.02 × 124.00 mm; PDF page 504.624 × 351.496 pt; PNG
4206 × 2930 px at 600 dpi; fonts ArialMT / Arial-BoldMT / Arial-ItalicMT, all TrueType, no Type 3;
run sizes 7.5 / 8 / 9 / 10 pt only, none below the 7.5 pt floor; 152 shapes; every word box inside
its card with ≥1 mm inset; no overlapping word boxes; no text outside a card.

**Recorded, not fixed.** The split bar’s two greys (`#B0B0B0` 69 %, `#D9D9D9` 85 %) sit 16 points
apart, under the “adjacent grey fills ≥20 %” house rule; the junction now carries an explicit
0.6 pt seam and each segment is labelled underneath. Panel A’s 6.9 mm top inset and panel B’s
1.4 mm caption inset, the two horizontal arrows’ different heights (A→B y 37.8, B→C y 31.8) and
panel D’s 2.6 mm column-body offset are pre-existing and were not touched. The v30i figure hashes
quoted above predate a 14:00 deck edit; the manifest already recorded
`d9257244…`/`8e6c43ce…`/`13d3b098…` at 06:00:34Z for that state.

Fig 1 PDF `b67ea94b…`, PNG `0ca186df…`, PPTX `cafe678f…`; main PDF 34 pp unchanged
(`4ab23453…`, `03_manuscript.pdf` byte-identical); `CHECKSUMS` **v30j** 7/7; `MANIFEST.sha256`
`2a4d0116…`, 42/42; `assemble_patterns_submission.py` and `--check` exit 0.

### v30k — eleventh author and second funding source (2026-10-22)

Author/funding metadata pass; no scientific text, number, table or figure changed. This block
supersedes the ten-author and single-grant statements in the v30f and v30j blocks above; those
records are left intact as history.

**Authors.** **Xiaoyu Chen** (ORCID 0000-0002-9980-5524, affiliation 2) is inserted immediately
after Fenfen Li, giving **eleven** authors across the two affiliations. Order: Kunhui Xu, Chenchen
Wang, Fenfen Li, Xiaoyu Chen, Xinyu Wang, Yi Zhang, Tiancheng Pu, Jiahao Tang, Hongguang Cui,
Peng Guo, Qi Dai. Hongguang Cui, Peng Guo and Qi Dai remain co-corresponding (only their three
e-mail addresses appear); Qi Dai remains lead contact. The CRediT statement gains
`X.C.: investigation, data curation.` after F.L.; no other role changed. The SI cover page lists
the same eleven names in the same order.

**Funding.** Two sources again, verbatim: *This work was supported by the National Natural Science
Foundation of China (82571251) and the Medical and Health Science Program of Zhejiang Province
(WKJ-ZJ-26067).* Mirrored into `submission/07_declarations.md`, `01_cover_letter.md`,
`00_CHECKLIST.md` (R4 and §12) and the COI fill guide.

**Build.** Main 34 pp / 0 error / 0 undefined / 0 overfull (14 underfull); SI 26 pp / 0 / 0 / 0
(63 underfull). `latex/paper1.tex` `ce0dcb36…`; `latex/build/paper1.pdf` = `submission/03_manuscript.pdf`
`e2e3fa07…`; SI tex `c3c8f0d7…`, SI PDF (build = synced = `04`) `60f6ebc8…`; `figure_manifest.json`
unchanged; `CHECKSUMS` **v30k** 7/7; `MANIFEST.sha256` `b06604d4…`, 42/42; `assemble --check` OK.

### v31 — 投稿冻结（2026-10-23）

v26→v31 的累积变更（详见 `docs/changes/2026-10-2*.md` 与 `submission/00_CHECKLIST.md` §12–§21）：

- **外部评审响应**：删除全部失败/未生效方法材料（CAFT/axis-anchored/soft-Spearman/flow）与变更日志式叙述；Limitations 改四条；术语首现定义；图注压缩至 92–119 词并修复 Fig 1/3/4/6 的图文不符；SI 删 S8 并重编号为 S1–S13。
- **新增证据**：两个 4 类上皮 CE 对照臂（2,616 / 7,473 步）分离标签集与步数预算；配对 bootstrap 全文统一为 pooled 口径；CellPLM 剂量-反应；引 Kendiukhov 电路映射预印本。
- **结构与文风**：主稿 35→29 页（删模板占位板块 + Methods 细节外移 + 表 3 移入 SI）；句子均长 19.2→23.2；第一人称 3.01→4.47/1000；臂名拼写统一 29 处；数字密度 10.3/100 词经实测不可降（属设计代价，已裁定接受）。
- **合规**：11 位作者 / 三通讯 / lead contact；资助两条；AI 声明改为 Elsevier 标准标题 + 48 词。

**当前哈希**：`latex/paper1.tex` `f1eca6d3…`；`latex/build/paper1.pdf` = `submission/03_manuscript.pdf` `39599233…`（29 页）；SI tex `5bb338e3…`；SI PDF（build/同步/`04`）`86d37073…`（30 页）；`CHECKSUMS` v31 7/7；`assemble --check` OK。

**目录整理（2026-10-23）**：删除 73 个 LaTeX 中间件与 2 个 `__pycache__`；五份参考文档（图件规范、文风指南、投稿要求、期刊定位、GB 对标）移入 `reference/`；`README.md` 与根 `README.md` 更新至 v31。
