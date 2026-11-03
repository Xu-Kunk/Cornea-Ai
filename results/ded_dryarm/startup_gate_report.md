# DED Dry-Arm 启动门报告（Gate 1–4）

- 日期：2026-08-18
- 协议：DED-Frozen-Reference-Application-Protocol.md §3（2026-07-30，2026-08-12 修订）
- 前置 lane：docs/changes/2026-08-18-scgpt-vanzyl-arm-and-ded-datalane.md（并行 CPU lane）
- 产出：artifacts/ded_dryarm/manifest_v1.json + 本报告 + lane.log

## Gate 1 — GSE200097 归属核验：**PASS**

**证据**（data/ded_dryarm/gse200097/gse200097_verification.json，前任会话已跑通 verify_gse200097.py）：

- 40 GSM 样本 → 20 RNA-seq + 20 RiboSeq 正确区分（RiboSeq 排除，仅 RNA-seq 入主分析）。
- 本地 `GSE200097_rnaSeq_counts.txt.gz`（20 列）与 GEO `raw_counts` 的 RNA-seq 子集：**列序一致、基因序一致、全部数值相等**（`local_values_equal_geo_rna=true`）。
- 本地 metadata（20 行）与 series matrix 合并：both=20、condition/timepoint/GSM 三方一致均 20/20。
- 归属表 `gse200097_attribution.tsv`（20 行）：Control(4) / 1hOsm(4) / 6hOsm(4) 主分析 12 样本 + Torin1(4) / MeAIB(4) 扩展 8 样本；500 mOsm（100 mM NaCl）、Torin1 250 nM、MeAIB 5 mM、1h/6h 时点与系列矩阵一致。
- SHA-256 已锁定（raw_counts / rnaSeq_counts / series_matrix / sample_metadata 四文件，见 manifest）。
- 细胞系身份：10.014 pRSV-T（CRL-11515），Keratinocyte-SFM 淹没培养（protocol §8 决议：bulk 100% basal 映射为细胞系真实生物学信号）。

**开放问题**：无 donor/cell-line passage/batch 元数据 → 生物独立性与批次混杂未解析（同协议 §3 记录）。

## Gate 2 — GSE323164 命名空间核验：**PASS**（直接采纳前任产出）

**证据**（gse323164_namespace_verification.json + hgnc_spotcheck.jsonl + hgnc_notfound_resolution.json）：

- 34,962 基因 ×（6 FPKM + 6 count + 注释列），FPKM 列 cont_1..3 / HN_1..3 与 series matrix 3v3 对照一致；count 列同步。
- ENSG 主键：34,962 唯一、0 重复、100% 格式合法、0 版本后缀、0 空 → **以 gene_id 为权威主键**。
- gene_name 列：34,630 唯一 symbol，332 重复（51 个 symbol 映射 >1 ENSG，多为假基因/历史符号）；6 个非标准。
- HGNC 抽检 200 条（REST API）：蛋白质编码/lncRNA/假基因/snoRNA 全部 Approved；NOT_FOUND 集中在 **5S_rRNA、5_8S_rRNA、7SK（alias→RN7SK，已解析）、IQCA1、60 个 ENSG-as-symbol 直填项**；64 条未解析已完整记录于 hgnc_notfound_resolution.json。
- 正交映射能力（对 mouse/rat_human_orthologs.csv 人类侧）：symbol 并集可映射 **14,839/34,630 = 42.85%**；ENSG 级映射小鼠 **14,999/34,962 = 42.90%** —— 满足跨物种 scRNA-seq 参考映射的基因级需求。

**结论**：命名空间核验完整、可复现（脚本 scripts/ded_dryarm/verify_gse323164_namespace.py 在盘）。分析时以 ENSG 过滤 64 条未解析行。

## Gate 3 — PXD028922 评估：**PARTIAL（上下文层可用，无 raw 谱图）**

**结论**：**可获得 processed 蛋白定量矩阵，但不在 PRIDE 归档中** —— 位于论文开放获取补充材料。

**证据**：

- PRIDE 归档（PXD028922_README.txt，2021-10-04）：仅 9 个 `.raw` 谱图文件（5×PRG4 泪液 + 4×Sjögren）+ 搜索 FASTA（UP000005640）+ checksum.txt；**无 processed 定量矩阵**（submissionType=PARTIAL）。
- 关联论文：Frontiers in Pharmacology 2021，doi:10.3389/fphar.2021.787193，PMID 34950038（PMC8689002，开放获取；Dufour, Univ. Calgary；Orbitrap Fusion Lumos + MaxQuant v1.6.10.23 二甲基化标记定量）。
- 补充材料 EuropePMC supplementaryFiles 下载落盘 `data/ded_dryarm/pxd028922/suppl_pmc/`：**Table1.xlsx（779 KB）= Supplementary Table 1 Tears（1,626 蛋白行）+ Supplementary Table 2 Saliva，MaxQuant 格式（Protein IDs / Gene names / Ratio H/L normalized）**；另有 DataSheet2/3/6/7 等。
- 结论：processed 定量（蛋白级 Ratio H/L，SS vs 对照）可复用，无需下载 raw 谱图。

**处置**：纳入上下文证据层（PRG4/lubricin、蛋白酶/抑制剂等模块注释）；唾液层与角膜 DED 映射价值低，不进入核心管线；不做独立数据集。

## Gate 4 — 统一 manifest + mouse/rat h5ad 归属核验：**PASS**

- **manifest**：artifacts/ded_dryarm/manifest_v1.json（249 行，15 KB）——含协议 §3 全字段（5 数据集 accession/version/species/tissue/condition/samples/platform/cell count/gene namespace/inclusion-exclusion）+ reference lock 5 项。
- **mouse_ded.h5ad = GSE182582 派生**：obs 含 celltype_orig（11 类：LSC/LSC\*、TAC1/TAC1\*、TAC2/TAC2\*、Basal、Wing/Wing\*、Squam/Squam\*）、group（cntl/ded）、condition（control/DED）；1000 细胞 × 17,802 基因（var index+gene_name 为 mouse symbol）。系 GSE182582 forGEO 矩阵的 1,000-cell 子集（母矩阵 33,885×17,891，integrity OK；另两个 count 文件 integrity FAILED，永不使用）。
- **rat_ded.h5ad = OMIX008264**：40,911 细胞 × 19,580 基因，obs 仅 condition（control/DED），var index 为 rat symbol。源为 OMIX008264-01..09 tar 中 C1-C3/DE1-DE3 count 提取（PRJCA033622），与 lane 文档一致；h5ad 无 accession 字段，归属依据 lane changelog + DED-Public-Data-Manifest-2026-07-30。
- **reference lock**：ordinal-CAFT λ=0.5 三 seed checkpoint-7473 hash 已锁（config 一致 + safetensors 三哈希）；external-test（GSE153515）donor 级 macro-F1 0.641/0.647/0.656（均值 0.648）；tokenization = scGPT vocab（48,292 tokens）+ log1p→5 bins + seurat_v3 HVG；正交表 mouse 17,066 / rat 14,022 行；参考标签 = van Zyl PNAS 2022 标注（LSC→TAC→basal→wing→squamous 序数轴）；透明基线 = frozen Geneformer / HVG-PCA / ordinal-naive。

## 四门汇总

| Gate | 状态 | 证据位置 |
|---|---|---|
| 1 GSE200097 归属 | **PASS** | gse200097_verification.json + attribution.tsv（20 行）|
| 2 GSE323164 命名空间 | **PASS** | gse323164_namespace_verification.json + hgnc_spotcheck.jsonl（200 条）|
| 3 PXD028922 评估 | **PARTIAL→上下文层可用** | PXD028922_README.txt + suppl_pmc/Table1.xlsx（processed 定量）|
| 4 manifest + 归属 | **PASS** | manifest_v1.json（249 行）|

## 开放问题

1. **rat_ded.h5ad 缺 sample/donor 级标识**（obs 仅 condition）→ 动物级聚合与批次混杂不可控；若需动物级 CI，须回溯 OMIX008264 tar 元数据重建 sample 列。
2. **GSE182582**：母矩阵 forGEO 版本 OK；`GSE182582_count*.csv.gz` 两文件 integrity FAILED（永不使用）；1,000-cell 子集为派生资产，引用时须注明非独立 accession。
3. **GSE323164 64 条未解析 gene_name**：rRNA + 60 个 ENSG-as-symbol + IQCA1（疑为 HGNC 改名，未追查）；分析时以 ENSG 主键过滤即可，无碍 42.9% 映射。
4. **GSE200097 无批次/传代元数据**：生物独立性与批次混杂未解析（同协议 §3）。
5. **PXD028922 蛋白级 vs 细胞级差异**：Ratio H/L 为池化蛋白定量，与单细胞参考映射仅可做模块级对照，不可做细胞级位移。
6. **mouse_ded/rat_ded h5ad 无 accession 元数据字段**：归属靠 lane changelog + DED-Public-Data-Manifest-2026-07-30 佐证（celltype 层级/源文件组一致）；如需更强锁，可将 accession 回填 obs.uns。

## 下一步（不属本 lane）

- Stage A–D 分析（映射/位移/模块）在 manifest 与 reference lock 齐备后可启动；HCE-T bulk 100% basal 的解读按 protocol §8 决议执行。
- PXD028922 tears 层作上下文证据时引用 Dufour 2021（doi:10.3389/fphar.2021.787193）。
