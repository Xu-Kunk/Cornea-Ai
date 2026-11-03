# results/ — 聚合结果目录

> 生成: 2026-09-22 · 来源: `artifacts/` 手工提取 (含 `artifacts/archive/` 中已核实的干净聚合文件)
> 包含聚合 JSON/CSV/MD，排除原始数据、checkpoint、嵌入、tokenized 数据集、内部清单、日志。
> 每个条目映射到论文图表或结论。不确定的标注「待确认」。

---

## 总览

| 项目 | 值 |
|------|----|
| 文件总数 | 255 |
| 总大小 | 5.5 MB |
| 子目录数 | 109 |
| 来源 | `artifacts/` (排除 archive) + `artifacts/archive/` 干净子集 |

---

## paper1_vanzyl/ — van Zyl 角膜上皮数据集的主分析

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `paper1_vanzyl/corn_lora/summary.json` | 同上 | Fig 7 (CORN+LoRA three-seed ρ=0.924, F1=0.966) |
| `paper1_vanzyl/corn_lora/seed*_args.json` | 同上 | CORN+LoRA 训练参数配置 |
| `paper1_vanzyl/cellplm_lora/seed42/seed42.json` | 同上 | CellPLM LoRA ρ=0.873 (Table 2/3) |
| `paper1_vanzyl/cellplm_layerwise/cellplm_layerwise_rho.json` | 同上 | CellPLM layerwise ρ (Fig 3) |
| `paper1_vanzyl/cellplm/seed42/frozen_emb.npz` 等信息 → 已排除二进制 | — | — |
| `paper1_vanzyl/cellplm_consistency/summary_all.json` | 同上 | CellPLM consistency 跨种子/数据集汇总 |
| `paper1_vanzyl/cellplm_consistency/summary_epidermis.json` | 同上 | Epidermis 一致性汇总 |
| `paper1_vanzyl/cellplm_consistency/summary_loo220.json` | 同上 | LOO220 一致性汇总 |
| `paper1_vanzyl/cellplm_consistency/data_manifest_epidermis.json` | 同上 | Epidermis 数据划分 |
| `paper1_vanzyl/cellplm_consistency/data_manifest_loo220.json` | 同上 | LOO220 数据划分 |
| `paper1_vanzyl/cellplm_consistency/epidermis/seed42/seed42.json` | 同上 | Epidermis 一致性 (seed42) |
| `paper1_vanzyl/cellplm_consistency/epidermis/seed123/seed123.json` | 同上 | Epidermis 一致性 (seed123) |
| `paper1_vanzyl/cellplm_consistency/loo220/seed42/seed42.json` | 同上 | LOO220 (seed42) |
| `paper1_vanzyl/cellplm_consistency/loo220/seed123/seed123.json` | 同上 | LOO220 (seed123) |
| `paper1_vanzyl/head_init/head_init_summary.json` | 同上 | Head-init basin selection (ρ=0.572–0.804 vs default −0.091, Fig S?) |
| `paper1_vanzyl/head_init/headscale_0.01_best_val.json` | 同上 | Head scale 0.01 best-val |
| `paper1_vanzyl/head_init/headscale_0.01_final_epoch.json` | 同上 | Head scale 0.01 final epoch |
| `paper1_vanzyl/head_init/headscale_1.0_best_val.json` | 同上 | Head scale 1.0 best-val |
| `paper1_vanzyl/head_init/headscale_1.0_final_epoch.json` | 同上 | Head scale 1.0 final epoch |
| `paper1_vanzyl/head_init/headseed_999_best_val.json` | 同上 | Head seed 999 best-val |
| `paper1_vanzyl/head_init/headseed_999_final_epoch.json` | 同上 | Head seed 999 final epoch |
| `paper1_vanzyl/sensitivity/sensitivity_summary.json` | 同上 | Sensitivity (Fig S) |
| `paper1_vanzyl/sensitivity/rank_variants.json` | 同上 | Rank variant 汇总 |
| `paper1_vanzyl/sensitivity/verdict_*.json` (11 files) | 同上 | Sensitivity verdict A/M/R 各 variant |
| `paper1_vanzyl/sensitivity/sha_manifest.json` | 同上 | Sensitivity SHA 清单 |
| `paper1_vanzyl/svd_validation/svd_axis_validation.json` | 同上 | SVD-axis validation (frozen Spearman 1.0) |
| `paper1_vanzyl/svd_validation/monotonicity_stats.json` | 同上 | Monotonicity 统计 |
| `paper1_vanzyl/fdr/bh_fdr_table.json` | 同上 | BH-FDR 21 comparisons (20 significant, Table S?) |
| `paper1_vanzyl/fdr/full_grid.json` | 同上 | FDR 全网格 |
| `paper1_vanzyl/cross_tissue/final_test_eval.json` | 同上 | Cross-tissue per-seed ρ (Fig 6) |
| `paper1_vanzyl/cross_tissue/final_test_eval_ext.json` | 同上 | Cross-tissue extended eval |
| `paper1_vanzyl/cross_tissue/epidermis_test_frozen_diag.json` | 同上 | Epidermis frozen baseline 诊断 |
| `paper1_vanzyl/cross_tissue/runner_summary.json` | 同上 | Cross-tissue 运行汇总 |
| `paper1_vanzyl/cross_tissue/args.json` | 同上 | Cross-tissue 配置 |
| `paper1_vanzyl/loo/final_test_eval.json` | 同上 | LOO Hu0220 (ρ=0.456/0.447, 待确认) |
| `paper1_vanzyl/loo/frozen_baseline_loo220.json` | 同上 | LOO 冷冻基线 |
| `paper1_vanzyl/loo/runner_summary.json` | 同上 | LOO 运行汇总 |
| `paper1_vanzyl/loo/checkpoints/runner_summary.json` | 同上 | LOO checkpoint 汇总 |
| `paper1_vanzyl/cornea2/summary_cornea2.json` | 同上 | GSE186433/GSE155683 naive ρ (Table S?) |
| `paper1_vanzyl/cornea2/gse155683_frozen_baseline.json` | 同上 | GSE155683 frozen baseline |
| `paper1_vanzyl/cornea2/gse155683_marker_scores.csv` | 同上 | GSE155683 marker scores |
| `paper1_vanzyl/cornea2/gse155683_qc_decisions.json` | 同上 | GSE155683 QC decisions |
| `paper1_vanzyl/cornea2/gse155683_scgpt_panel_check.json` | 同上 | GSE155683 panel check |
| `paper1_vanzyl/cornea2/gse155683_annotation_report.md` | 同上 | GSE155683 annotation report |
| `paper1_vanzyl/cornea2/gse186433_frozen_baseline.json` | 同上 | GSE186433 frozen baseline |
| `paper1_vanzyl/cornea2/gse186433_marker_scores.csv` | 同上 | GSE186433 marker scores |
| `paper1_vanzyl/cornea2/gse186433_qc_decisions.json` | 同上 | GSE186433 QC decisions |
| `paper1_vanzyl/cornea2/gse186433_scgpt_panel_check.json` | 同上 | GSE186433 panel check |
| `paper1_vanzyl/cornea2/gse186433_annotation_report.md` | 同上 | GSE186433 annotation report |
| `paper1_vanzyl/cornea2/GSE155683/naive_seed42/eval.json` | 同上 | GSE155683 naive seed42 eval |
| `paper1_vanzyl/cornea2/GSE155683/naive_seed123/eval.json` | 同上 | GSE155683 naive seed123 eval |
| `paper1_vanzyl/cornea2/GSE186433/naive_seed42/eval.json` | 同上 | GSE186433 naive seed42 eval |
| `paper1_vanzyl/cornea2/GSE186433/naive_seed123/eval.json` | 同上 | GSE186433 naive seed123 eval |
| `paper1_vanzyl/cornea2/data_sha_manifest.json` | 同上 | corna2 data SHA |
| `paper1_vanzyl/layerwise/layerwise_rho.json` | 同上 | Layerwise ρ (Fig 3) |
| `paper1_vanzyl/g1_gf/gf_frozen_diag_vanzyl.json` | 同上 | Geneformer frozen vs fine-tuned 诊断 |
| `paper1_vanzyl/g1_scgpt/scgpt_frozen_diag_vanzyl.json` | 同上 | scGPT frozen 诊断 |
| `paper1_vanzyl/g1_scgpt/gse153515_frozen_diag_validation_v3.json` | 同上 | GSE153515 frozen 诊断验证 |
| `paper1_vanzyl/fixed_axis/fixed_axis_eval.json` | 同上 | Fixed axis eval |
| `paper1_vanzyl/baselines/baselines_rho.json` | 同上 | Baselines ρ (Table S?) |
| `paper1_vanzyl/baselines/harmony_nh2000_npc50.json` | 同上 | Harmony baseline |
| `paper1_vanzyl/baselines/hvg_pca_nh2000_npc50.json` | 同上 | HVG-PCA baseline |
| `paper1_vanzyl/baselines/hvg_pca_nh2000_npc100.json` | 同上 | HVG-PCA (npc100) |
| `paper1_vanzyl/baselines/hvg_pca_nh5000_npc50.json` | 同上 | HVG-PCA (nh5000) |
| `paper1_vanzyl/baselines/hvg_pca_nh5000_npc100.json` | 同上 | HVG-PCA (nh5000, npc100) |
| `paper1_vanzyl/baselines/scvi_nlatent128.json` | 同上 | scVI baseline |
| `paper1_vanzyl/contamination/contamination_audit.json` | 同上 | Contamination audit (Fig S, 待确认) |
| `paper1_vanzyl/contamination/contamination_audit.md` | 同上 | Contamination audit report |
| `paper1_vanzyl/contamination/evidence/census_2023-05-15_datasets.csv` | 同上 | Census dataset 证据 |
| `paper1_vanzyl/contamination/evidence/prior_provenance_2026-08-05.json` | 同上 | Prior provenance 证据 |
| `paper1_vanzyl/contamination/evidence/prior_signalA_gse153515.json` | 同上 | GSE153515 prior signal 证据 |
| `paper1_vanzyl/cluster0_forensics/cluster0_forensics.json` | 同上 | Cluster 0 法医分析 |
| `paper1_vanzyl/cluster0_forensics/report.md` | 同上 | Cluster 0 法医报告 |
| `paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_seed*.json` (6 files) | 同上 | GF official 1-epoch 各种子 |
| `paper1_vanzyl/gf_official_1epoch/gf_official_1epoch_summary.json` | 同上 | GF official 1-epoch 汇总 |
| `paper1_vanzyl/gf_official_1epoch/eval_armb_seed*.json` (6 files) | 同上 | GF official B-arm 评估 |
| `paper1_vanzyl/gf_official_1epoch/env_manifest.json` | 同上 | GF 1-epoch 环境清单 |
| `paper1_vanzyl/gf_official_1epoch/migration_sha_manifest.json` | 同上 | GF 1-epoch SHA 清单 |
| `paper1_vanzyl/gf_lora/pcs_frozen_spec.json` | 同上 | GF LoRA frozen spec |
| `paper1_vanzyl/umap_collapse/umap_manifest.json` | 同上 | UMAP collapse 图件清单 |
| `paper1_vanzyl/scgpt_panel_check.json` | 同上 | scGPT panel check |
| `paper1_vanzyl/stage2_prep_summary.json` | 同上 | Stage2 prep 汇总 |
| `paper1_vanzyl/stage2_prep/gf_tokenized_vanzyl_summary.json` | 同上 | GF tokenized 汇总 |
| `paper1_vanzyl/stage2_prep/stage2_prep_summary.json` | 同上 | Stage2 prep 汇总 |
| `paper1_vanzyl/run6/runner_summary.json` | 同上 | 6-run 运行汇总 |
| `paper1_vanzyl/run6/eval/naive_seed*.json` (3 files) | 同上 | 6-run naive 各种子 eval |
| `paper1_vanzyl/run6/eval/ordinal_seed*.json` (3 files) | 同上 | 6-run ordinal 各种子 eval |
| `paper1_vanzyl/run6/eval/frozen_baseline.json` | 同上 | 6-run frozen baseline |
| `paper1_vanzyl/run6/eval/summary_vanzyl_6run.json` | 同上 | 6-run 汇总 |
| `paper1_vanzyl/run6/convergence/runner_summary.json` | 同上 | 6-run convergence 汇总 |
| `paper1_vanzyl/run6/lora_v2/seed*_args.json` (3 files) | 同上 | 6-run LoRA v2 参数 |
| `paper1_vanzyl/run6/lora_v2/args.json` | 同上 | 6-run LoRA v2 配置 |
| `paper1_vanzyl/run6_ext/eval/frozen_baseline.json` | 同上 | 6-run ext frozen baseline |
| `paper1_vanzyl/run6_ext/eval/naive_seed2024.json` | 同上 | 6-run ext naive seed2024 |
| `paper1_vanzyl/run6_ext/eval/naive_seed789.json` | 同上 | 6-run ext naive seed789 |
| `paper1_vanzyl/convergence/final_test_eval.json` | 同上 | Convergence final test |
| `paper1_vanzyl/marker_scores.csv` | 同上 | vanZyl marker scores |
| `paper1_vanzyl/qc_decisions.json` | 同上 | vanZyl QC decisions |

---

## paper1_tier_b_eval/ & paper1_tier_b_eval_gse299152/ — Tier B 外部评估

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `paper1_tier_b_eval/results_*.json` (11 files) | 同上 | Tier B 各方法各种子 ρ/F1 (Table 4) |
| `paper1_tier_b_eval/summary.csv` | 同上 | Tier B 汇总 (Table 4) |
| `paper1_tier_b_eval/results_frozen_gf.json` | 同上 | Tier B frozen GF |
| `paper1_tier_b_eval/results_hvg_pca.json` | 同上 | Tier B HVG-PCA baseline |
| `paper1_tier_b_eval/percell_*.npz.meta.json` (6 files) | 同上 | 逐细胞 npz 元数据(≤1KB) |
| `paper1_tier_b_eval_gse299152/results_*.json` (9 files) | 同上 | GSE299152 外部评估 |
| `paper1_tier_b_eval_gse299152/summary.csv` | 同上 | GSE299152 汇总 |
| `paper1_tier_b_eval_gse299152/results_frozen_gf.json` | 同上 | GSE299152 frozen GF |

---

## paper1_layerwise_rho/ — 逐层 Spearman ρ

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `paper1_layerwise_rho/layerwise_rho_naive_s*.json` (3 files) | 同上 | 逐层 ρ naive 三种子 (Fig 3/4) |
| `paper1_layerwise_rho/layerwise_rho_caft_s*.json` (3 files) | 同上 | 逐层 ρ CAFT 三种子 |
| `paper1_layerwise_rho/layerwise_rho_frozen_gf.json` | 同上 | 逐层 ρ frozen GF |
| `paper1_layerwise_rho/causal_head_ablation_A.json` | 同上 | Causal head ablation A |
| `paper1_layerwise_rho/causal_head_ablation_B.json` | 同上 | Causal head ablation B |
| `paper1_layerwise_rho/causal_head_ablation_C.json` | 同上 | Causal head ablation C |
| `paper1_layerwise_rho/causal_head_ablation_summary.json` | 同上 | Causal head ablation 汇总 |
| `paper1_layerwise_rho/summary.json` | 同上 | Layerwise ρ 汇总 |

---

## paper1_ordinal_eval/ — 序数头评估

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `paper1_ordinal_eval/results_0.0_*.json` (3 files) | 同上 | Ordinal λ=0 三种子 |
| `paper1_ordinal_eval/results_0.5_*.json` (3 files) | 同上 | Ordinal λ=0.5 三种子 |
| `paper1_ordinal_eval/layerwise_rho_ordinal_naive_s*.json` (3 files) | 同上 | Ordinal naive 分层 ρ |
| `paper1_ordinal_eval/layerwise_rho_ordinal_caft_s*.json` (3 files) | 同上 | Ordinal CAFT 分层 ρ |
| `paper1_ordinal_eval/layerwise_summary.json` | 同上 | Ordinal 分层 ρ 汇总 |
| `paper1_ordinal_eval/summary.csv` | 同上 | Ordinal 汇总 |
| `paper1_ordinal_eval/runs_status.json` | 同上 | Ordinal 运行状态 |

---

## paper1_method_arms/ — 方法臂评估

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `paper1_method_arms/phase_a_summary.csv` | 同上 | Phase A 汇总 |
| `paper1_method_arms/phase_b_decision.json` | 同上 | Phase B 决策 |
| `paper1_method_arms/option_a_s42_summary.json` | 同上 | Option A seed42 汇总 |
| `paper1_method_arms/axis_ref.json` | 同上 | Axis reference |
| `paper1_method_arms/results_axis_ft_*.json` (4 files) | 同上 | Axis FT 各 λ 种子 |
| `paper1_method_arms/results_flow_0.5_42.json` | 同上 | Flow λ=0.5 seed42 |
| `paper1_method_arms/results_soft_spearman_*.json` (3 files) | 同上 | Soft-Spearman 三种子 |
| `paper1_method_arms/runs_status.json` | 同上 | 方法臂运行状态 |

---

## 其他 artifacts/ 子目录

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `paper1_corn_boundary/percell_lam*.json` (3 files) | 同上 | CORN 边界 per-cell λ (Fig 7b?) |
| `paper1_energy_drift/summary.json` | 同上 | Energy drift 汇总 |
| `paper1_energy_drift/summary_percell.json` | 同上 | Energy drift per-cell |
| `paper1_energy_drift/energy_drift_caft_s42.json` | 同上 | CAFT energy drift |
| `paper1_energy_drift/energy_drift_naive_s42.json` | 同上 | Naive energy drift |
| `paper1_energy_drift/energy_drift_caft_s42_percell.json` | 同上 | CAFT per-cell energy drift |
| `paper1_energy_drift/energy_drift_naive_s42_percell.json` | 同上 | Naive per-cell energy drift |
| `paper1_fdr_nulls/bh_fdr_table.json` | 同上 | BH-FDR 表 |
| `paper1_fdr_nulls/null_controls.json` | 同上 | Null controls |
| `paper1_fdr_nulls/contamination_audit_protocol.json` | 同上 | Contamination audit 协议 |
| `paper1_fdr_nulls/contamination_audit_protocol.md` | 同上 | Contamination audit 协议 |
| `paper1_fdr_nulls/donor_level_audit.json` | 同上 | Donor-level audit |
| `paper1_fdr_nulls/donor_level_audit.md` | 同上 | Donor-level audit 报告 |
| `paper1_fdr_nulls/gse299152_vs_gse153515_summary.csv` | 同上 | GSE299152 vs GSE153515 汇总 |
| `paper1_strict_fair/` (各 JSON，待确认) | 同上 | Strict fair comparison (待确认) |
| `paper1_pipeline_audit/ece_corrected.json` | 同上 | ECE 校正 |
| `paper1_pipeline_audit/metric_recompute.json` | 同上 | Metric 重算 |
| `paper1_pipeline_audit/caft_reframe_facts.json` | 同上 | CAFT reframe facts |
| `paper1_pipeline_audit/audit_report.md` | 同上 | Pipeline 审计报告 |
| `paper1_fair_split/donor_overlap_check.json` | 同上 | Donor overlap check |
| `paper1_fair_split/manifest.json` | 同上 | Fair split 清单 |
| `lambda_grid/gpr_lam*.json` (6 files) | 同上 | GPR λ 网格搜索 |

---

## artifacts/ 根级别 & 环境锁

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `cs_before_results.json` | 同上 | CS before 基线 (Fig 1b/Table 1) |
| `cellrefine_prototypes.json` | 同上 | CellRefine 原型 |
| `env_locks/pip_freeze_main_2026-08-21.txt` | 同上 | 主环境锁 (环境可复现性) |
| `env_locks/pip_freeze_cellplm_2026-08-21.txt` | 同上 | CellPLM 环境锁 |
| `ded_dryarm/manifest_v1.json` | 同上 | DED Dry ARM 清单 |
| `ded_dryarm/startup_gate_report.md` | 同上 | DED 启动门报告 |
| `gse153515_annotation/marker_scores.csv` | 同上 | GSE153515 marker scores |
| `gse299152_annotation/cluster_assignment.json` | 同上 | GSE299152 cluster 分配 |
| `gse299152_annotation/marker_scores.csv` | 同上 | GSE299152 marker scores |
| `gse299152_annotation/provenance.json` | 同上 | GSE299152 溯源 |
| `gse299152_annotation/annotation_report.md` | 同上 | GSE299152 注释报告 |
| `paper1_preprint/SUBMISSION_PACKAGE.md` | 同上 | 提交元数据与说明 |

---

## 跨架构 frozen 探针诊断 (来自 artifacts/archive/ 干净子集)

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `scgpt_repro/frozen_diag/scgpt_frozen_diag_gse153515.json` | `artifacts/archive/scgpt_repro/frozen_diag/` | scGPT-human frozen probe GSE153515 诊断 (Methods: Cross-architecture probes) |
| `scgpt_repro/frozen_diag/scgpt_frozen_diag_final.json` | 同上 | scGPT-human frozen probe 最终汇总 |
| `scfoundation_repro/frozen_diag/_model_config.json` | `artifacts/archive/scfoundation_repro/frozen_diag/` | scFoundation 模型配置 |
| `scfoundation_repro/frozen_diag/scfoundation_frozen_diag_gse153515.json` | 同上 | scFoundation frozen probe GSE153515 |
| `scfoundation_repro/frozen_diag/scfoundation_frozen_diag_gse299152.json` | 同上 | scFoundation frozen probe GSE299152 |
| `scfoundation_repro/frozen_diag/scfoundation_frozen_diag_final.json` | 同上 | scFoundation frozen probe 最终汇总 |
| `uce_repro/frozen_diag/uce_frozen_diag_final.json` | `artifacts/archive/uce_repro/frozen_diag/` | UCE frozen probe 最终汇总 |
| `uce_repro/frozen_diag/uce_frozen_diag_gse153515.json` | 同上 | UCE frozen probe GSE153515 |
| `uce_repro/frozen_diag/uce_frozen_diag_gse299152.json` | 同上 | UCE frozen probe GSE299152 |
| `uce_repro/frozen_diag/gse153515_forward_timing.json` | 同上 | UCE GSE153515 前向时序 |
| `uce_repro/frozen_diag/gse299152_forward_timing.json` | 同上 | UCE GSE299152 前向时序 |
| `uce_repro/frozen_diag/corpus_audit.md` | 同上 | UCE 语料审计报告 |

## Geneformer LoRA OOM 证据 (Limitations 第四点)

| results/ 相对路径 | artifacts/ 源路径 | 支撑图表/结论 |
|--------------------|-------------------|----------------|
| `paper1_vanzyl/gf_lora/oom_evidence/README.md` | 原始 `.log` 含内部路径已废弃，本文件为干净重述 | Geneformer LoRA 因显存 OOM 无法评估的证据 (Limitations, 4th point; identity check 6.45e-06 PASS, OOM @16GB) |

---

## 注意

1. 本目录仅为 `artifacts/` 中聚合结果的**副本**，不包含原始数据、checkpoint、嵌入文件、tokenized dataset。
2. 所有 `.pt`, `.ckpt`, `.h5ad`, `.npz`, `.pkl`, `.parquet`, `.arrow`, `.bin` 等二进制格式已排除；`*.log` 原始日志一律不发布（OOM 证据以重述 README 代替）。
3. 所有 `args.json` (训练参数)、`split_manifest.json` (数据划分)、`env_manifest*.json` (环境) 等内部管理文件已排除。
4. 标注「待确认」的条目需对照 v23 主稿 Results 节进行验证。
5. 如需完全复现论文结果，需先通过 `scripts/download_*.py` 获取原始数据并运行训练。