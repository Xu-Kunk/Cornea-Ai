# Paper1 Pretraining-Contamination Audit (scContam-class)

**Date**: 2026-08-19
**Auditor**: Fixer (implementation lane)
**Scope**: 5 datasets of Paper1 vs scGPT (CELLxGENE census 2023-05-15) and Geneformer
(Genecorpus-104M = the corpus of the paper's GF V2-104M checkpoint; Genecorpus-30M
audited secondarily for completeness).
**Artifacts**: `contamination_audit.json` (machine-readable), this report,
`evidence/` (census 2023-05-15 datasets table, MOESM3 xlsx pointer), `signalA_gse299152.json`
(cell-level MinHash, see §4).
**Supersedes**: P1-O1 report (2026-07-31, "not executable") and
`artifacts/paper1_data_audit/contamination_provenance.json` (2026-08-05) — the V2 corpus
provenance table became available (NatCS 2026 MOESM3), enabling DOI-level resolution;
this audit adds (i) direct census datasets-table verification via `cellxgene_census` 1.18.0,
(ii) van Zyl GF-V2 membership (NEW — file-level), (iii) GSE299152 paper-DOI check
(10.1093/pnasnexus/pgaf348, NEW), (iv) a GSE299152 cell-level Signal A run.

---

## 1. Corpus / model manifest (locked)

| Model used in paper | Checkpoint | Corpus | Verification performed |
|---|---|---|---|
| scGPT-human (51.86M) | perturblab/scgpt-human | CELLxGENE census **2023-05-15** release cycle | `cellxgene_census` 1.18.0 `open_soma("2023-05-15")` datasets table read directly → **562 datasets** (evidence/census_2023-05-15_datasets.csv). scGPT repo `data/cellxgene/data_config.py` records VERSION `"2023-05-08"` ("we used the version 2023-05-08 for our experiments") — internal build of the same initial census; label differs by 7 d, no content effect for datasets published before May 2023 |
| Geneformer-V2-104M (41.4M) | ctheodoris/Geneformer snapshot 04c2b2e8…/Geneformer-V2-104M (safetensors sha256 fff5cba…, 398 MB) | **Genecorpus-104M** (theodoris-lab/Genecorpus-104M; 103,877,737 cells; 2,903 datasets; 563 unique DOIs + 44 non-DOI rows) | NatCS 2026 SI MOESM3 sheet `genecorpus_104m` (2,904 rows: source_DOI \| n_cells_passed_filtering \| organ) parsed in full. **HF dataset created 2026-02-17, lastModified 2026-03-27** |
| Geneformer V1 (secondary) | n/a (not the paper's checkpoint) | Genecorpus-30M (Nature 2023 SI workbook, 561 rows) | Workbook re-downloaded + text-searched 2026-08-19: **44 eye rows, all retina/RPE/optic choroid; zero cornea/limbus/sclera/conjunctiva/ocular-surface; 0 hits for all paper accessions** |

---

## 2. Per-dataset verdicts

| Dataset (role in paper) | GEO public | scGPT (census 2023-05-15) | GF V1 (30M) | GF V2 (104M) | Verdict for disclosure |
|---|---|---|---|---|---|
| **van Zyl 2022 PNAS** (10.1073/pnas.2200914119; scGPT train+test donor Hu0216; GF + CellPLM eval) | n/a (CELLxGENE 2022) | **IN** — census table has collection 63d03351, 6 files, 238,149 cells; our files' cell counts (37,485 central cornea; 52,309 TM wedge) match census files **exactly**; is_primary_data=True | NOT IN (zero cornea; pub. 2022-07-19 > V1 assembly) | **IN** — corpus table row DOI 2200914119, 6 rows, counts 52,309/42,901/37,485/57,422/13,900/34,132 = **exact match to census dataset files** (file-level) | **Contaminated both models. MUST disclose.** scGPT results on van Zyl are same-corpus by design; NEW: GF-V2 evaluation on van Zyl is same-corpus too |
| **GSE153515** (Li 2021; GF locked external test, scGPT probe) | 2021-01-14 | NOT IN (GEO-only; absent from census table; zero GSE accessions in census) | NOT IN (0 hits) | **IN** (study-level; DOI jtos.2020.12.004 → 53,290 cells, eye). Cell-level Signal A (2026-08-05, 0.54% sample): 0 hits ≥0.5 Jaccard; controls perfect; ~1% sample-coverage → not exclusionary | **Contaminated (GF).** Already disclosed in figure set as same-corpus (least conservative) — consistent |
| **GSE155683** (Collin/Lako 2021; GF **training** atlas: lako_adult 2,987 + lako_fetal 6,985 cells via CXG collection 0f7d022a) | 2021-04-19 | NOT IN (0f7d022a **not** in census; verified by collection_id + dataset_id) | NOT IN (0 hits; zero cornea) | **IN** (study-level; DOI jtos.2021.03.010 → 8 rows, 111,186 cells, eye) | **GF finetune set overlaps GF pretraining corpus. MUST disclose** (figure set already notes 10.4% train-atlas overlap incl. lako; confirm counts) |
| **GSE186433** (Catala 2021; GF **training** atlas: dickman 1,511 cells) | 2021-10-25 | NOT IN | NOT IN (0 hits) | **NOT IN** (study-level; DOI s41598-021-01015-w absent from all 2,904 rows; corroborates prior audit) | **Clean** — no disclosure required |
| **GSE299152** (2025; second external test, GF + scGPT) | 2025-11-18 | NOT IN (postdates census snapshot by 2.5 y) | NOT IN | **NOT IN** (study-level; paper DOI 10.1093/pnasnexus/pgaf348 absent; **zero pnasnexus DOIs** in corpus table) — *temporal caveat: corpus HF release 2026-02-17 is AFTER GSE299152's release, so the "post-cutoff" argument alone is NOT valid for V2; decisive evidence = table absence + no positive signal* | **Clean** — strongest external evidence; **fix the figure set's "postdates corpus assembly" wording** (§5) |

---

## 3. Contamination-risk conclusions (Limitations disclosure)

1. **van Zyl 2022 PNAS — contaminated for BOTH scGPT and Geneformer-V2-104M.**
   - scGPT: cell-level (census files = our files; 6 dataset files, 238,149 cells; collection
     63d03351-06be-478e-a0db-f7a653b6b19b; DOI 10.1073/pnas.2200914119).
   - GF V2-104M: file-level (corpus provenance table rows match the census dataset files
     cell-for-cell). **This is NEW relative to the current figure set**, which treats van Zyl
     only as the scGPT same-corpus arm; any GF frozen/fine-tuned numbers reported on van Zyl
     (e.g., the G1 frozen ρ ≈ 0.80 control arm) are same-corpus for GF and must be labelled as
     such in the figure set.
2. **GSE153515 — in Genecorpus-104M (study-level).** Already disclosed in figure set as the
   same-corpus (least conservative) GF external test — verdict confirmed, no action beyond
   keeping the existing contamination-aware framing.
3. **GSE155683 — in Genecorpus-104M (study-level).** It is *training* data (9,972 cells in the
   GF finetune atlas), so it does not weaken the external-test claims; the existing disclosure
   of the 10.4% train-atlas overlap covers it. Confirm the overlap table includes lako
   (8 corpus rows / 111,186 cells).
4. **GSE186433 — clean** for both corpora (study-level). No disclosure needed; prior audit
   consistent.
5. **GSE299152 — clean** (study-level) for both corpora, and now backed by a cell-level
   Signal A negative (§4). It remains the paper's most conservative external test. **One
   wording fix required**: the figure set's "deposited after corpus assembly (post-cutoff)"
   is not strictly correct for Genecorpus-104M (corpus HF release 2026-02-17 postdates
   GSE299152's 2025-11-18 release); the defensible basis is absence from the DOI-keyed
   provenance table, not temporal exclusion. For scGPT (census 2023-05-15) temporal exclusion
   holds.

---

## 4. Cell-level check (scContam-style Signal A)

- GSE153515 (prior run, 2026-08-05): 0/15,827 cells with top-Jaccard ≥ 0.5 against a
  0.54%-of-corpus sample (491,227 fingerprints); positive control recall@0.9 = 1.00,
  negative control = 0.0. Coverage caveat: a ~15.8k-cell dataset has only ~1% chance of
  falling inside the 0.56M-cell sample → pipeline validation, not exclusion evidence.
- GSE299152 (this run, 2026-08-19): 23,970 cells queried against the same 491,227-fingerprint
  reference (identical constants m=128, b=16, r=8, K=2048, seed 42). See
  `signalA_gse299152.json` for results. Interpretation mirrors GSE153515: absence in sample
  is corroborative, not conclusive.
- van Zyl cell-level query was **not run**: positive membership is already established at
  file level for both corpora (exact cell-count correspondence), so a sampled-MinHash adds no
  decision power; documented as a deliberate scope decision.

---

## 5. Methods-citable text (proposed update to "Provenance and contamination audit")

> Pretraining-corpus membership was audited for every dataset against the two corpora of the
> evaluated models. For scGPT-human (pretrained on the CELLxGENE census 2023-05-15 release;
> 33.8M unique normal human cells), membership was verified by direct read of the census
> datasets table (cellxgene-census 1.18.0, `census_info.datasets`); the only corneal/anterior-
> segment collection in the census is the van Zyl et al. ocular anterior-segment atlas
> (collection 63d03351-06be-478e-a0db-f7a653b6b19b; 6 dataset files, 238,149 cells; DOI
> 10.1073/pnas.2200914119), whose dataset files match our local AnnData files cell-for-cell
> (Central Cornea 37,485; Trabecular Meshwork and Corneal Scleral wedge 52,309 cells), so the
> van Zyl arm is same-corpus for scGPT by construction and is reported as such. For
> Geneformer-V2-104M (pretrained on Genecorpus-104M, 103.9M cells from 2,903 datasets), we
> performed a DOI-level audit against the official corpus source table (NatCS 2026
> supplementary table MOESM3, sheet genecorpus_104m; 563 unique DOIs). The van Zyl atlas DOI
> resolves to six rows whose cell counts exactly match the census dataset files (238,149
> cells, organ = eye), placing the van Zyl arm inside the Geneformer pretraining corpus at
> file level. GSE153515's primary-publication DOI (10.1016/j.jtos.2020.12.004; public
> 2021-01-14) resolves to one row (53,290 cells, organ = eye) and is reported as same-corpus
> (least conservative). Two training-atlas studies — lako/GSE155683 (DOI 10.1016/j.jtos.2021.03.010;
> 8 rows, 111,186 cells) and li/GSE147979 (DOI 10.1038/s41467-021-25968-8; 30 rows) — resolve
> to corpus rows (10,395 of 99,606 training cells, 10.4%); dickman/GSE186433
> (10.1038/s41598-021-01015-w) does not. GSE299152 (GEO public 2025-11-18; PNAS Nexus 2025,
> DOI 10.1093/pnasnexus/pgaf348) is absent from the corpus table (no PNAS Nexus DOI appears
> among all 2,904 rows); because the public corpus release (2026-02-17) postdates the
> dataset's GEO release, GSE299152's out-of-corpus status rests on provenance-table absence
> rather than on a temporal cutoff, and it is reported as the most conservative external test.
> GSE299152 also postdates the scGPT census snapshot (2023-05-15) by 2.5 years and is absent
> from the census manifest, placing it out-of-corpus for scGPT. As a corroborating cell-level
> check in the spirit of scContam, bounded Signal A fingerprint sampling (0.54% of
> Genecorpus-104M, 491,227 fingerprints; vectorized MinHash m = 128, b = 16, r = 8, banded
> LSH, exact-Jaccard confirmation) found zero hits for the GSE299152 cells (max Jaccard 0.0)
> and, in a prior run, for GSE153515 (max Jaccard 0.0), with 100% positive-control recall and
> zero negative-control false positives; because full-corpus coverage (~600 GB) was not
> attained, these runs validate the pipeline but do not constitute exclusion evidence. The
> DOI-level audit is the primary evidence.

*Required figure set edits flagged by this audit*: (i) add the van Zyl ↔ Genecorpus-104M
file-level finding to the Provenance and contamination audit subsection and to any GF van Zyl
claims; (ii) replace the "postdates corpus assembly (post-cutoff)" characterization of
GSE299152 with the provenance-table-absence basis (keep "post-cutoff" for the scGPT/census
sentence); (iii) optionally record the scGPT census label detail (repo data_config VERSION
2023-05-08 = internal build of the 2023-05-15 release).

---

## 6. Biases and limitations

1. **Study-level vs cell-level**: DOI-keyed provenance matching is study-level; a dataset
   could in principle be in a corpus via a different accession/DOI route (e.g., GEO-direct
   ingestion without a resolvable row). Direction of bias: under-reports inclusion.
2. **Sample-based cell-level check**: 0.54% corpus coverage; a negative in-sample result is
   NOT absence evidence. Direction of bias: under-reports duplication.
3. **Genecorpus-104M assembly date unknown internally**: the public HF release (2026-02-17)
   is the only hard date; actual assembly may predate GSE299152's release or not — this is
   why GSE299152's verdict rests on table absence, not timing. Direction of bias: n/a
   (handled explicitly).
4. **CELLxGENE curation-API timestamps are migration artifacts**: `created_at` for all
   collections reads 2026-06-10/11 (system migration), so API dates were not used as
   publish-time evidence; census table membership + paper publication date were used instead.
5. **scGPT census label**: repo `data_config.py` says 2023-05-08 vs paper's 2023-05-15; both
   refer to the same initial census release cycle; a 7-day content delta is possible but
   cannot affect datasets published in 2022 (van Zyl).
6. **scGPT filter caveat**: membership assumes scGPT's documented filter (is_primary_data,
   normal cells) — van Zyl cells pass (is_primary_data = True verified in local h5ad);
   however scGPT's exact snapshot of the census at training time (2023) could in principle
   differ from the 2026-read table only by datasets added/removed between — none affect the
   2022-published van Zyl collection.
7. **V1 workbook text search**: the 30M workbook has no accession column (free-text Source);
   author/name substring searches can miss datasets cited by accession only. The "zero cornea"
   finding is robust (organ column is structured), but per-dataset exclusions from V1 rely on
   text search.
8. **Deletion/tombstone possibility**: a dataset could have been in the census/corpus and
   later removed; no tombstone evidence was checked (census table read reflects current
   release state).

---

## 7. Evidence inventory

| Item | Location |
|---|---|
| census 2023-05-15 datasets table (562 rows) | evidence/census_2023-05-15_datasets.csv (read via cellxgene_census 1.18.0, 2026-08-19) |
| Genecorpus-104M provenance table (2,904 rows) | evidence/43588_2026_972_MOESM3_ESM.xlsx (downloaded 2026-08-05) |
| Genecorpus-30M workbook (561 rows) | /tmp/opencode/41586_2023_6139_MOESM4_ESM.xlsx (re-downloaded 2026-08-19; verdicts in this report) |
| GSE GEO dates | NCBI GEO text endpoint (`form=text&view=quick`), 2026-08-19: GSE153515 2021-01-14; GSE155683 2021-04-19; GSE186433 2021-10-25; GSE299152 2025-11-18 |
| GSE299152 paper DOI | PubMed 41245088 → 10.1093/pnasnexus/pgaf348 (PNAS Nexus, 2025 Nov) |
| HF corpus metadata | theodoris-lab/Genecorpus-104M API: created 2026-02-17, lastModified 2026-03-27 |
| Cell-level Signal A GSE299152 | signalA_gse299152.json + signalA_gse299152.log (2026-08-19) |
| Cell-level Signal A GSE153515 (prior) | artifacts/paper1_data_audit/contamination_signalA.json (2026-08-05) |
| van Zyl h5ad provenance | uns.citation: DOI 10.1073/pnas.2200914119; collection 63d03351; is_primary_data=True |
| Prior audit | artifacts/paper1_data_audit/contamination_provenance.json (2026-08-05); P1-O1 report 2026-07-31 |

*Verification commands*: census read = `cellxgene_census.open_soma(census_version="2023-05-15")`
→ `census["census_info"]["datasets"].read().concat().to_pandas()`; corpus table parse =
openpyxl on MOESM3 sheet `genecorpus_104m`; GEO dates = NCBI `acc.cgi?acc=GSE…&targ=self&form=text&view=quick`.
