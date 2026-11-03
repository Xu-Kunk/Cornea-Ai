---
title: "Genome Biology 对标参考记录 — scFM 评估/基准类研究发表情况"
target_journal: "Patterns (Cell Press)"
date: 2026-08-12
status: RECORD
related: "paper1 (Fine-tuning collapses differentiation gradients in single-cell foundation models in a domain-specific manner)"
---

# Genome Biology 对标参考记录（2026-08-12）

> 背景：用户在打磨阶段要求记录 Genome Biology（GB）投稿指南要点（manusights.com 博客）+ 检索类似研究发表 GB 的情况，作为期刊定位的参考。结论：**维持 Patterns 目标**；本记录备查。
> 来源：GB 官方投稿指南博客（manusights.com/blog/genome-biology-submission-guide，2026-08-12 抓取）+ @librarian 检索（PubMed/Europe PMC 核验，2026-08-12）+ 用户提供 2 篇实例（SpringerLink 抓取）。

## 1. GB 投稿要求要点（与 Patterns 对照）

| 项目 | Genome Biology | Patterns | 对本稿影响 |
|---|---|---|---|
| 摘要 | ≤250 词非结构化 | ≤150 词单段 | 本稿 126 词，两刊均达标 |
| 结构 | Background（非 Introduction）→Results→Discussion→Conclusions→Methods | Introduction→Results→Discussion→Methods | 切 GB 需改章节名 |
| 正文字数 | 6,000–8,000 词典型（含参考文献，无硬上限） | ~5,000–7,000 词典型 | 均达标 |
| 图注 | **必须详解所有 panel、统计方法、样本量** | 同向（自明性要求） | 与本轮图注增强一致 ✅ |
| 数据可用性 | **投稿时即须入库**，"upon request" 拒稿信号 | acceptance 前 deposit，缺失可撤稿 | ⚠️ 现稿 "upon acceptance" 措辞为两刊风险点（用户自管公开时机） |
| 代码可用性 | 计算论文必须交代码 | 必需 | GitHub 仓库已有 |
| Cover letter | 1–2 页，须论证生物学发现；点名数据集但不给生物学洞见 = 拒稿模式 | 三锚点版已规划 | 切 GB 需重写 |
| 统计 | 多重校正、CI、负结果如实报告 | 同向 | 本稿 BH-FDR+置换零假设+honest negative ✅ |
| 拒稿模式 | ①基因组数据无生物学机制 ②统计严谨性不足 ③结论超出证据 ④工具/方法无生物学发现 ⑤cover letter 无洞见 | — | 本稿类型（表征几何负结果+基准）**命中 ①④ 风险** |

**GB 五大拒稿模式中最相关的两条**：#1（有基因组数据但无验证的生物学机制——本研究的"生物学"是已知先验测试系统，非新机制发现）、#4（工具/方法论文但结果中无生物学发现——本稿核心产出是负结果+基准资源）。GB 中 scFM 类论文几乎都以生物学洞见收尾（见下）。

## 2. 类似研究发表 GB 的实例

### 2a. 用户提供 2 篇（2026-08-12 SpringerLink 抓取，原文核对）

1. **Unlocking biological insight from single-cell data with an interpretable dual-stream foundation model**（scDMC）
   - Guo H, Cui Q, Zhang X, Chen C, Zheng W, Cai C, Wang X, Wang S. *Genome Biol* (2026-08-08). doi:10.1186/s13059-026-04193-w
   - 内容：新 scFM 预训练框架（双流编码：rank 序列 + 绝对表达序列；基因级 MLM + 细胞级 momentum contrastive）；仅 200 万细胞预训练即在细胞注释/聚类/整合基准上 SOTA。
   - 类型：**方法开发 + 生物学洞见收尾**（功能基因模块、细胞型特异调控网络、attention 生物学解释）——标题即 "biological insight"，符合 GB 拒稿模式规避策略。
2. **Impact of data quality on deep learning prediction of spatial transcriptomics from histology images**
   - Hallinan C, Lucas CHG, Fan J. *Genome Biol* (2026-08-11). doi:10.1186/s13059-026-04236-2
   - 内容：in silico 消融研究——分子数据稀疏度/噪声、图像分辨率对 H&E→空间转录组预测的影响；插补 rescue 收益有限且不外推；跨技术/组织/架构可复现。
   - 类型：**data-centric 评估/消融研究 + 方法论信息**（"数据质量是架构调优的正交策略"）——与本稿"评估/警示型"研究体裁最接近的 GB 实例。

### 2b. @librarian 检索（2026-08-12，PubMed/Europe PMC 核验，DOI 全部验证）

**Tier 1 — 直接相关：scFM 评估/基准/负结果**
- **Kedzierska KZ, Crawford L, Amini AP, Lu AX.** Zero-shot evaluation reveals limitations of single-cell foundation models. *Genome Biol* 26:101 (2025-04-18). doi:10.1186/s13059-025-03574-x — Geneformer/scGPT 零样本诚实负结果评估，**以 Brief Report 发表**，"may be outperformed by simpler methods"；已引 43 次。→ 本稿已引（kedzierska2025）。
- **Wu J 等.** Biology-driven insights into the power of single-cell foundation models. *Genome Biol* 26:334 (2025-10-03). doi:10.1186/s13059-025-03781-6 — 6 个 scFM vs 经典基线，2 基因级+4 细胞级任务、12 指标；"no single scFM consistently outperforms" → 本稿已引（wu2025）。

**Tier 2 — 邻近：微调/迁移分析**
- **Yuan H, Linder J, Kelley DR.** Parameter-efficient fine-tuning enables scalable transfer of regulatory sequence models to novel contexts. *Genome Biol* 27(1):60 (2026-01-30). doi:10.1186/s13059-026-03935-0 — Borzoi（DNA 调控模型）参数高效微调系统评估。**方法学上与本稿微调坍缩问题最接近的近亲**（序列模型而非 scFM 嵌入）。

**Tier 3 — GB 中 scFM/预训练模型开发（语境）**
- scKGBERT (Genome Biol 26:402, 2025, doi:10.1186/s13059-025-03862-6)、CellMemory (26:178, 2025, doi:10.1186/s13059-025-03638-y)、TMO-Net (25:149, 2024, doi:10.1186/s13059-024-03293-9)。

**Tier 4 — 基准基础设施与诚实负结果（编辑模式证据）**
- Liu Y 等. Benchmarking LLM-based agents for single-cell omics analysis. *Genome Biol* 27(1):123 (2026). doi:10.1186/s13059-026-03998-z — 50 任务 agent 基准，含失败归因分析。
- Radig J 等. scArchon: a scalable benchmarking framework for assessing single-cell perturbation models. *Genome Biol* 27(1):162 (2026). doi:10.1186/s13059-026-04104-z — **扰动模型基准含负结果**（"other tools occasionally underperform even compared to linear or control baselines"）。
- Yi C 等. Benchmarking deep learning methods for biologically conserved single-cell integration. *Genome Biol* 26:398 (2025). doi:10.1186/s13059-025-03869-z — 16 方法集成基准，含对 scIB 指标本身的批评。
- Ding J 等. DANCE: a deep learning library and benchmark platform for single-cell analysis. *Genome Biol* 25:72 (2024). doi:10.1186/s13059-024-03211-z。

## 3. 综合判断

1. **GB 会发 scFM 评估/负结果研究**：Kedzierska（Brief Report 诚实负结果，43 引）、Wu 2025（benchmark）、Yuan 2026（微调评估）、scArchon（基准+负结果）均为先例；2024–2026 仅 PubMed 就有 44 篇单细胞基准论文。负但严谨的结果在 GB 可发表。
2. **但 GB 中 scFM 论文以"方法+生物学驱动叙事"为主**（scDMC、scKGBERT、TMO-Net 均以生物学洞见收尾）。本稿"纯嵌入几何分析+角膜测试系统"位于评估端，最接近的同类（Kedzierska 零样本、Yuan 微调）均已发表，但**强生物学叙事 + 完整方法评估包**对 GB 的通过线重要。
3. **定位结论不变：Patterns 更优**。Patterns 有 benchmark 体裁直接先例（BioLLM、representation-quality 评估、honest-negative 文体——本稿引的 vanamsterdam2025/offlinellm2025 均在 Patterns），且本稿整个投稿包（elsarticle、Highlights/eTOC/Bigger Picture、presubmission inquiry）已按 Patterns 构建。GB 若要投，需改章节结构（Background/Conclusions）、重写 cover letter 为生物学发现论证、投稿时即 deposit 数据，成本高且命中拒稿模式 ①④ 风险。
4. **跨期刊行动项**（无论投哪家）：数据/代码可用性措辞 "upon acceptance" → 投稿时可用（GB 拒稿信号；Patterns 撤稿风险）。用户已确认自管公开时机（届时公开 GitHub 代码 + 预印本）。

## 附：GB 关键数字速查
- 接受率 ~15%；首决 30–45 天；desk 决定 1–2 周
- APC：Research Article ~£3,790/$5,690/€4,690（页内另有 ~$5,290 表述，两处不一致已记录）
- 提交流程：Editorial Manager / Springer Nature 统一后端；Word 或 LaTeX + 图单独上传（TIFF/EPS/高分辨率 PDF；照片 ≥300 dpi、线条 ≥600 dpi）
- 期刊 ID: s13059（DOI 前缀 10.1186/s13059-）
