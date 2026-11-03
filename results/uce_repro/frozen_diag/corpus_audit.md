# UCE corpus-status audit (GSE153515 vs UCE training corpus)

- UCE was pretrained on CELLxGENE census snapshot **2023-04-25**
  (verified: repo `data_proc/download_proc_czi_cxg.py` line 73, `VERSION = "2023-04-25"`).
- cellxgene-census API (v1.18.0) does NOT expose the 2023-04-25 snapshot anymore
  (`get_census_version_directory` lists 2025-11-17/10/08/04, 2025-01-30,
  2024-07-01, 2024-05-20, 2023-12-15, 2023-07-25, 2023-05-15; 2023-04-25 absent).
- Closest queryable snapshot **2023-05-15** (562 datasets): GSE153515 accession
  string "153515" -> **0 hits**; author/limbal-title search ("collin", "limbal
  stem") -> 0 hits. Only cornea datasets present are a different study
  (PNAS 2022, doi 10.1073/pnas.2200914119 "Cell atlas of the human ocular
  anterior segment").
- Verdict: **plausibly OUT-OF-CORPUS** — GSE153515 absent from census
  2023-05-15 (closest audited snapshot); 2023-04-25 itself not directly
  verifiable (snapshot retired from the API). GSE299152 (GEO 2025-11-18) is
  out-of-corpus by construction (predates all census snapshots queried).
