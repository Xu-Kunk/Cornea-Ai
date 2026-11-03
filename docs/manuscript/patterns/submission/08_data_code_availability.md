---
title: "Resource availability — lead contact, materials, data and code"
manuscript_version: v26
date: 2026-10-19
status: draft-for-author-review
source: latex/paper1.tex §RESOURCE AVAILABILITY (lines 458–476)
---

> **口径说明（不随件提交）**
> 正文逐字取自 v26 `\section*{RESOURCE AVAILABILITY}`（本节正文未改）。投稿系统的
> Data & Code Availability 表单可整段粘贴。三项**未兑现**的事实用 `[HOLD]` 标出。
> 阶段判读（官方依据见 `docs/changes/2026-10-20-patterns-coi-draft-and-readiness.md`）：
> 代码 DOI 或放入 SI 的硬时点是 **before acceptance**，不是首投；首投要求的是**可访问**
> （匿名免费访问链接 / reviewer token 如需，或随稿代码包），不是 public。
> 仓库 public 属**项目选定路线**（用户已明确后续转 public）；若上传时仍未 public，须改为真实
> 可用的声明并提供编辑/审稿人访问链接或随稿代码包，**不能仅以将来公开的承诺代替访问**。

---

## Lead contact

Further information and requests for resources should be directed to and will be fulfilled by
the lead contact, Qi Dai (dq@mail.eye.ac.cn).

---

## Materials availability

This study did not generate new unique reagents or biological materials; all data are public
(below).

---

## Data and code availability

- **Training data.** CELLxGENE Discover collection `0f7d022a-46c7-4e64-be4c-e34adbb78089`
  (Single cell atlas of the human ocular surface; Human Ocular Surface Atlas), current
  dataset-version of dataset `8ef06ea4-8d0f-4303-a652-7d4604402b1f`, distributed as file
  `30127f02-d0b9-4b56-bf4c-e5b77e4989bd.h5ad`; MD5 `39f098367af95bc3e1ee92528ceb251e`.

- **External test data.** GEO GSE153515; 17,587 cells from two donors (GSM4646295, GSM4646296).
  GEO GSE299152; 21,374 epithelial cells from three donors after QC.
  Cross-tissue: GEO GSE147424.

- **Cross-architecture probes.** scGPT-human checkpoint (`perturblab/scgpt-human`; scgpt 0.2.5,
  git commit `cebd6fae`); CellPLM v1.0.0; scFoundation 01B-resolution checkpoint
  (`genbio-ai/scFoundation` Hugging Face mirror); UCE 33-layer checkpoint
  (Figshare `10.6084/m9.figshare.24320806`); frozen-probe diagnostics in
  `results/scgpt_repro/frozen_diag/`, `results/scfoundation_repro/frozen_diag/`, and
  `results/uce_repro/frozen_diag/`.

- **Model.** Geneformer-V2-104M, Hugging Face model hub (`ctheodoris/Geneformer`; snapshot
  commit `04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5`).

- **Code.** Analysis code, evaluation scripts, and aggregated result files are available at
  https://github.com/Xu-Kunk/Cornea-Ai. **[HOLD — 仓库当前必须转为 public 后此句才成立]**

- **Evaluation scripts.** `scripts/run_tier_b_eval.py`; `scripts/run_tier_b_baselines.py`;
  `scripts/measure_layerwise_rho.py`; `scripts/bootstrap_layerwise_rho.py`;
  `scripts/analyze_trajectory_continuity.py`; `scripts/analyze_tac_sensitivity.py`;
  and additional scripts in `scripts/` for the v16 cross-architecture arms.

- **Results.** Main protocol: `results/paper1_tier_b_eval/*.json`; van Zyl arms:
  `results/paper1_vanzyl/corn_lora/summary.json`,
  `results/paper1_vanzyl/cellplm_layerwise/cellplm_layerwise_rho.json`,
  `results/paper1_vanzyl/head_init/head_init_summary.json`,
  `results/paper1_vanzyl/sensitivity/sensitivity_summary.json`; statistical controls:
  `results/paper1_vanzyl/fdr/bh_fdr_table.json`,
  `results/paper1_vanzyl/svd_validation/svd_axis_validation.json`; supplementary domains:
  `results/paper1_vanzyl/cross_tissue/final_test_eval.json`,
  `results/paper1_vanzyl/loo/final_test_eval.json`,
  `results/paper1_vanzyl/cornea2/summary_cornea2.json`,
  `results/paper1_vanzyl/cellplm_consistency/summary_all.json`.

- **Reproducibility.** All experiments are reproducible with fixed random seeds (42, 123, 456),
  locked configuration files, and SHA-256-locked fair-split manifests.

---

## 未兑现项（提交前必须清除）

| 项 | 现状 | 需要的动作 | 阶段 |
|---|---|---|---|
| GitHub public | `Xu-Kunk/Cornea-Ai` 当前未确认 public | 项目选定路线：后续转 public；上传前若未落实，须改为真实可用的 Code 句，并提供编辑/审稿人匿名访问链接或随稿代码包（不能仅承诺将来公开） | 首投前（项目路线） |
| Zenodo / figshare DOI | 未铸 | 建 release → 铸 DOI → 回填主稿 Code 句 → 重编译；DOI 或 SI 二选一 | **before acceptance**（非首投硬门） |
| Data S2（逐细胞预测与嵌入） | 未随包发布，SI 标为按需提供 | 若要随稿提交，需上传至仓库或仓库附件；否则在 SI 保持“按需提供” | 视编辑/生产需要 |
| 数据使用条款 | 原始数据受 GEO/CELLxGENE 原研究同意约束 | 不重新分发原始矩阵；仅提供 accession 与脚本 | 不变 |
| 预训练权重 | 受许可证与体积限制 | 仅给官方 hub 与 commit，不随包分发 | 不变 |

> 编译口径：改动本节任一句都需重编译主稿（`latex/build/paper1.pdf`），
> 并重新运行 `python3 scripts/assemble_patterns_submission.py` 刷新 manifest。
