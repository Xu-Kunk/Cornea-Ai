# Supplementary Material Index Audit — Paper 1 v23

> 生成: 2026-09-22
> 审计者: Fixer (机械核对)
> 范围: `paper1_final_candidate_v23.tex` SUPPLEMENTAL INFORMATION INDEX / 正文引用 ↔ 磁盘实物
> 约束: 只读不写（本报告为新建文件，不修改任何稿件/补充材料正文/tex）

---

## 1. 来源说明

| 来源 | 文件 |
|------|------|
| 声称清单 (INDEX) | `paper1_final_candidate_v23.tex` 第 496–513 行 |
| 正文引用集合 | 同 tex 正文中所有 `Figure~S*` / `Table~S*` / `Data~S*` / `Supplementary Figure~S*` 出现位置 |
| 补充材料正文 | `docs/manuscript/patterns/paper1_supplementary_materials.md`（222 行） |
| 图件实物 | `figures/paper1/`（含 `archive/` 归档） |
| 数据实物 | `artifacts/paper1_layerwise_rho/`、`artifacts/paper1_tier_b_eval/`、`artifacts/paper1_fair_split/` 等 |

---

## 2. INDEX 声称清单（逐条，共 13+9+3 = 25 项）

### 2.1 Figures S1–S13（INDEX 合为一行："Figures S1--S13 and their legends in a PDF"）

| # | INDEX 标题 | 正文引用? |
|---|-----------|----------|
| Figures S1–S13 | 合为一笔，未逐条列出 | 正文引用了 S3, S6, S7, S10, S11, S12, S13 |

### 2.2 Tables S1–S9

| # | INDEX 标题 | 正文引用? |
|---|-----------|----------|
| Table S1 | Related to Table 1: Per-seed external-test metrics by training epoch | ✓ (line 85) |
| Table S2 | Related to Table 1: Per-donor macro-F1 on external test (GSE153515) | ✗ 仅 INDEX 提及 |
| Table S3 | Related to Table 1: CAFT promotion rule evaluation | ✗ 仅 INDEX 提及 |
| Table S4 | Related to Table 1: Per-class precision, recall, F1, and support for fine-tuned arms (external test, 3-seed means) | ✗ 仅 INDEX 提及 |
| Table S5 | Related to Figure 2: Layer-resolved differentiation-gradient Spearman ρ (external test, n = 14,775 cells) | ✓ (lines 98, 186) |
| Table S6 | Related to Methods §2.1: Constituent studies of the training atlas | ✗ 仅 INDEX 提及 |
| Table S7 | Related to Methods §2.8: Provenance record (transcribed verbatim from figures/paper1/table6_provenance.csv) | ✗ 仅 INDEX 提及 |
| Table S8 | Related to Figure 2: Frozen-representation probes across single-cell foundation models | ✓ (line 393) |
| Table S9 | Related to Table 1: Alternative gradient-preserving recipes: per-seed external-test metrics (GSE153515) | ✓ (line 193) |

### 2.3 Data S1–S3

| # | INDEX 标题 | 正文引用? |
|---|-----------|----------|
| Data S1 | Layer-resolved differentiation-gradient ρ (raw values) | ✗ 仅 INDEX 提及 |
| Data S2 | Per-cell predictions and embeddings | ✗ 仅 INDEX 提及 |
| Data S3 | Locked fair-split manifest and SHA-256 checksums | ✗ 仅 INDEX 提及 |

---

## 3. 实际实物清单

### 3.1 补充图文件（`figures/paper1/`）

| 项目 | PDF | PNG | CSV 数据 | 补充材料 doc 描述 |
|------|:---:|:---:|:--------:|:-----------------:|
| Figure S1 | ✓ | ✓ | ✓ | ✓ |
| Figure S2 | ✓ | ✓ | ✓ (2 files) | ✓ |
| Figure S3 | ✓ | ✓ | ✓ | ✓ |
| Figure S4 | ✓ | ✓ | ✗ | ✓ |
| Figure S5 | ✓ | ✓ | ✓ | ✓ |
| Figure S6 | ✓ | ✓ | ✓ | ✓ |
| Figure S7 | ✓ | ✓ | ✓ | ✓ |
| Figure S8 | ✓ | ✓ | ✓ | ✓ |
| Figure S9 | ✓ | ✓ | ✗ | ✓ |
| Figure S10 | ✓ | ✓ | ✗ | ✓ |
| Figure S11 | ✓ | ✓ | ✓ | ✓ |
| Figure S12 (as figure3_trajectory_continuity) | ✓ | ✓ | ✗ | ✓ |
| Figure S13 (as figure4_annotation) | ✓ | ✓ | ✗ | ✓ |

### 3.2 补充表（在 `paper1_supplementary_materials.md` 中以 Markdown 表格形式存在）

| 项目 | 在 supp doc 中? | 引用 CSV 数据文件 |
|------|:--------------:|:-----------------:|
| Table S1 | ✓（行 58–86） | `figures/paper1/table_t3_primary_endpoint.csv` ✓ |
| Table S2 | ✓（行 87–102） | 内联 |
| Table S3 | ✓（行 104–113） | 内联 |
| Table S4 | ✓（行 114–127） | `figures/paper1/table_t4_per_class_metrics.csv` ✓ |
| Table S5 | ✓（行 129–141） | `artifacts/paper1_layerwise_rho/summary.json` ✓ |
| Table S6 | ✓（行 143–158） | 内联 |
| Table S7 | ✓（行 160–169） | `figures/paper1/table6_provenance.csv` ✓ |
| Table S8 | ✓（行 173–188） | 内联；引用 `artifacts/scgpt_repro/frozen_diag/` 等 |
| Table S9 | ✓（行 191–210） | `artifacts/paper1_method_arms/results_*.json` ✓ |

### 3.3 补充数据实物

| 项目 | 磁盘实物 | 路径 |
|------|---------|------|
| Data S1 | summary.json + 7 per-model JSONs ✓ | `artifacts/paper1_layerwise_rho/` |
| Data S2 | 7 × percell_{model}.npz + .meta.json + results JSONs ✓ | `artifacts/paper1_tier_b_eval/` |
| Data S3 | manifest.json + donor_overlap_check.json ✓ | `artifacts/paper1_fair_split/` |

---

## 4. 问题报告

### 4.1 MISSING（INDEX 声称但找不到实物）

**共 1 项：**

| 问题 | 详情 |
|------|------|
| **"Figures S1–S13 及其图例在一个 PDF 中"**（INDEX 第 1 行） | **不存在** 整合了所有补充图及其图例的单一 PDF。补充图以 13 个独立 PDF 文件存在于 `figures/paper1/`，图例以 Markdown 形式存在于 `paper1_supplementary_materials.md`。投稿包中的"补充材料"是 Markdown 文档 + 独立图文件，而非单一 PDF。INDEX 中的 "in a PDF" 表述不准确。 |

**其余项：** Tables S1–S9 全部存在于补充材料 Markdown 文档中；Data S1–S3 全部有对应的 artifacts 实物。无其他 MISSING。

### 4.2 NUMBERING_GAP（编号跳号或重复）

**无。** 核对结果：

| 集合 | 范围 | 状态 |
|------|------|------|
| Figures S1–S13 | 13 个编号连续无跳号 | ✓ 无跳号 |
| Tables S1–S9 | 9 个编号连续无跳号 | ✓ 无跳号 |
| Data S1–S3 | 3 个编号连续无跳号 | ✓ 无跳号 |

### 4.3 ORPHAN（有实物但 INDEX 与正文均未提及）

**共 3 项（均为 CSV 源数据文件，非投稿必须项，但值得记录）：**

| 文件 | 类型 | 说明 |
|------|------|------|
| `figures/paper1/table1_dataset_summary.csv` | CSV 数据 | 未被主稿 tex 或补充材料 doc 引用。可能是早期版本的训练数据集汇总表。 |
| `figures/paper1/table2_model_arms.csv` | CSV 数据 | 未被主稿 tex 或补充材料 doc 引用。可能是早期版本的模型臂汇总表。 |
| `figures/paper1/figure2_layerwise_rho_bootstrap.csv` | CSV 数据 | 主图 Figure 2 的 bootstrap 数据，未被主稿 tex 直接引用（tex 引用的是 PDF）。不属于 supplementary ORPHAN。 |

> 注：`figure2_layerwise_rho.csv`、`figure9_cross_architecture_spectrum.csv` 为主图的数据源，虽未在 supp doc 中列出但属于主图管线产物，不计入 ORPHAN。

### 4.4 TEXT_MISMATCH（INDEX 标题与补充材料正文标题不一致）

**无严重不匹配。** 逐项比对结果：

| 项目 | INDEX 标题摘要 | 补充材料 doc 标题 | 匹配? |
|------|--------------|-----------------|:-----:|
| Table S1 | Per-seed external-test metrics by training epoch | 一致 | ✓ |
| Table S2 | Per-donor macro-F1 on external test (GSE153515) | 一致 | ✓ |
| Table S3 | CAFT promotion rule evaluation | 一致 | ✓ |
| Table S4 | Per-class precision, recall, F1, and support... | 一致 | ✓ |
| Table S5 | Layer-resolved differentiation-gradient Spearman ρ... | 一致 | ✓ |
| Table S6 | Constituent studies of the training atlas | 一致 | ✓ |
| Table S7 | Provenance record (transcribed verbatim...) | 一致 | ✓ |
| Table S8 | Frozen-representation probes across single-cell foundation models | 一致 | ✓ |
| Table S9 | Alternative gradient-preserving recipes... | 一致 | ✓ |

补充材料 doc 中 Data S1–S3 的描述与 INDEX 表述一致。

---

## 5. 正文引用覆盖分析

| 项目 | INDEX 声称 | 正文实际引用 | 仅 INDEX 提及 |
|------|:---------:|:-----------:|:------------:|
| Figure S1 | ✓ | ✗ | ✓ |
| Figure S2 | ✓ | ✗ | ✓ |
| Figure S3 | ✓ | ✓ (Fig. 4 caption) | — |
| Figure S4 | ✓ | ✗ | ✓ |
| Figure S5 | ✓ | ✗ | ✓ |
| Figure S6 | ✓ | ✓ (Fig. 5 caption) | — |
| Figure S7 | ✓ | ✓ (Fig. 2 caption) | — |
| Figure S8 | ✓ | ✗ | ✓ |
| Figure S9 | ✓ | ✗ | ✓ |
| Figure S10 | ✓ | ✓ (正文 §R3 + Fig. 5 caption) | — |
| Figure S11 | ✓ | ✓ (正文 §R3) | — |
| Figure S12 | ✓ | ✓ (正文 §R3) | — |
| Figure S13 | ✓ | ✓ (正文 §R3) | — |
| Table S1 | ✓ | ✓ (正文 Table 1 caption) | — |
| Table S2 | ✓ | ✗ | ✓ |
| Table S3 | ✓ | ✗ | ✓ |
| Table S4 | ✓ | ✗ | ✓ |
| Table S5 | ✓ | ✓ (Fig. 2/Fig. 6 captions) | — |
| Table S6 | ✓ | ✗ | ✓ |
| Table S7 | ✓ | ✗ | ✓ |
| Table S8 | ✓ | ✓ (正文 §Methods) | — |
| Table S9 | ✓ | ✓ (正文 §Methods) | — |
| Data S1 | ✓ | ✗ | ✓ |
| Data S2 | ✓ | ✗ | ✓ |
| Data S3 | ✓ | ✗ | ✓ |

> **说明：** INDEX 中部分表格（S2/S3/S4/S6/S7）和所有 Data 项仅出现在 INDEX 中而未在正文中显式引用，这在 Cell Press 作者的投稿惯例中是常见的（INDEX 列举所有补充材料，正文选择性引用），不是问题。

---

## 6. 可执行补齐清单

以下项需要用户或后续会话处理：

| 优先级 | 项 | 建议操作 |
|--------|-----|---------|
| **高** | INDEX 第 1 行"Figures S1–S13 and their legends in a PDF" | 当前补充材料是 Markdown 文档 + 独立图 PDF，无整合 PDF。投稿前需确认：① 改为"Supplementary Figures S1–S13 (individual PDF files) and their legends in Doc S1"；② 或生成一个整合的 Supplementary PDF（将所有图及图例装入一个 PDF）。取决于 Cell Press Multi-Journal Submission 的补充材料格式要求。 |
| **低** | ORPHAN CSV 文件 `table1_dataset_summary.csv`、`table2_model_arms.csv` | 这两个文件可能是不再使用的历史文物。建议检查是否仍需要，若不需要可移入 `figures/paper1/archive/`。不影响投稿。 |
| **低** | Data S2 的 per-cell 预测文件 | 确认是否需要随投稿包一起压缩。当前以 `.npz` 格式存在，Cell Press 可能要求导出为更通用的格式。需用户决定。 |

---

## 7. 结论

**v23 补充材料已达到"可随投稿包提交"的状态，但有一项 INDEX 表述需要修正。**

- **声称项数：25**（Figures S1–S13 × 1 笔 + Tables S1–S9 × 9 + Data S1–S3 × 3）
- **实际找到：24/25**（所有 Tables 和 Data 项实物齐全；单独 Figure PDF 全部存在）
- **MISSING：1** — 整合图 PDF（INDEX 声称 "in a PDF"，实物为独立 PDF + Markdown 图例）
- **NUMBERING_GAP：0**
- **ORPHAN：2**（`table1_dataset_summary.csv`、`table2_model_arms.csv` — 残余文件）
- **TEXT_MISMATCH：0**

核心问题：INDEX 中"Figures S1–S13 and their legends in a PDF"是**不准确表述**。当前真实状态是：补充图以 13 个独立 PDF 文件存在，图例在 Markdown 补充材料文档中。投稿前需将这一行改为准确描述（取决于目标期刊要求的格式——是整合 PDF 还是分开提交）。除此以外，所有编号、标题、数据源均完全一致。