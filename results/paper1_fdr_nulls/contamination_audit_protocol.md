# Contamination-Audit Methods (ready-to-paste)

## Pre-training-corpus contamination audit and the paired external-test design

Because all fine-tuned models are initialised from the publicly released
Geneformer V2-104M (Genecorpus-104M; Chen & Venkatesh et al., *Nature
Computational Science* 2026, DOI 10.1038/s43588-026-00972-4), we audited
whether either external test set could have been part of the pre-training
corpus, and we deliberately paired one **same-corpus** and one
**post-cutoff, out-of-corpus** external test to bound the effect of any
residual contamination on the primary conclusions.

### Same-corpus external test (GSE153515)

GSE153515 (human limbal basal epithelium; Li et al., *The Ocular Surface*
2021, DOI 10.1016/j.jtos.2020.12.004, PMID 33388438; GEO public
2021-01-14) is the primary external test. A DOI-level provenance audit
(GEO record → PubMed ID → Crossref DOI → lookup in the official
Genecorpus-104M source table, MOESM3 sheet `genecorpus_104m` of the
*Nature Computational Science* SI, 2,903 dataset rows / 563 unique DOIs)
resolved the study's primary DOI **10.1016/j.jtos.2020.12.004** to a row
in the corpus table (53,290 cells, organ = eye). Because the corpus was
assembled in 2026 from datasets that all predate assembly, and GSE153515
was public from 2021-01-14, **no temporal exemption applies**: the frozen
model may have been exposed to GSE153515 cells during pre-training.
GSE153515 is therefore the *least conservative* external test, and we
report it as the same-corpus reference.

### Post-cutoff, out-of-corpus external test (GSE299152)

GSE299152 (human ocular-surface single-cell map; *PNAS Nexus* 2025, DOI
10.1093/pnasnexus/pgaf348, PMID 41245088; GEO public 2025-11-18) was
published and deposited **after** the Genecorpus-104M assembly window; its
DOI is absent from the corpus source table, and its donors are disjoint
from the 49 training donors. GSE299152 is the *most conservative*,
contamination-free external test. The paired design reports every endpoint
on both sets; conclusions that survive on GSE299152 are robust to
pre-training contamination, while discrepancies isolate contamination
effects.

### scContam Signal A bounded validation (corroborating, cell-level)

In addition to the DOI-level audit (decisive), we ran a bounded cell-level
validation in the spirit of scContam (Pervez & Wang, arXiv 2607.20572):
Geneformer V2 rank-encoding fingerprints (top-2048 tokens per cell, official
`token_dictionary_gc104M.pkl` + `gene_median_dictionary_gc104M.pkl`) were
compared against fingerprints of 7 uniformly sampled corpus shards
(≈0.6M cells; the full 1,202-shard / ≈600 GB corpus was too large to
download in this study) with vectorised MinHash (m=128, b=16, r=8) + banded
LSH + exact Jaccard confirmation. Positive control (200 injected GSE153515
cells must recall themselves) and negative control (200 random fingerprints
must yield zero hits) both passed. The DOI-level audit is the primary
evidence; the bounded Signal A run validates the pipeline.

### Secondary finding: training-atlas contamination

Three of the ten training-atlas studies (lako_GSE155683, 9,972 cells; and
li_GSE147979, 423 cells) resolved to DOIs also present in the corpus source
table (10,395/99,606 cells, 10.4%). This does not affect the external-test
conclusions (the external tests are disjoint from the training atlas at the
donor and study level) but is disclosed for completeness.

### Framing

Throughout the paper, "disjoint from the training atlas" refers to the
donor/study-level split used for fine-tuning; "disjoint from the
pre-training corpus" is **not** claimed for GSE153515 (it is same-corpus)
and **is** claimed for GSE299152 (post-cutoff, out-of-corpus). All
gradient-preservation comparisons are reported on both external tests so
that the reader can separate fine-tuning effects from pre-training
exposure.

## References

- Chen & Venkatesh et al. Geneformer V2-104M / Genecorpus-104M. *Nat. Comput.
  Sci.* 2026. DOI 10.1038/s43588-026-00972-4 (SI table MOESM3,
  `genecorpus_104m`).
- Li D-Q et al. Single-cell transcriptomics of limbal basal epithelium.
  *Ocul. Surf.* 2021;20:20-32. DOI 10.1016/j.jtos.2020.12.004. PMID 33388438.
- GSE153515. GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153515.
- *PNAS Nexus* 2025 glycocalyx ocular-surface map. DOI
  10.1093/pnasnexus/pgaf348. PMID 41245088.
- GSE299152. GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE299152.
- Pervez & Wang. scContam: single-cell contamination assessment.
  arXiv:2607.20572. https://github.com/sarwanpasha/scContam.
