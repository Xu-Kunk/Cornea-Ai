---
title: "Paper 1 期刊适配评估与专刊检索记录 (2026-08-12)"
target_journal: "Patterns (Cell Press)"
date: 2026-08-12
status: RECORD
related: "reference/genome_biology_scfm_landscape.md、paper1.tex v12"
---

# Paper 1 期刊适配评估与专刊检索

> 输入：本稿画像 + 本地 md 文献库 + Genome Biology 对标（上轮）+ @librarian 联网检索（期刊实态 2024–2026，DOI 全部经 Europe PMC/Crossref 核验）。
> 论文画像：Geneformer-V2-104M 分类微调坍缩分化序数几何（角膜上皮，三域边界）；CORN 唯一保梯度配方；五重诚实阴性目标；污染感知配对 benchmark 资源；BH-FDR 26 比较 + 置换零假设。形态 = findings.md 预判的 "CAFT null 但 benchmark 稳健暴露失败模式 → diagnostic/reliability paper"。

## 一、适配期刊清单（按适配度分层）

### 第一梯队（同型先例直接命中）
1. **Patterns（主投，维持）** — 直接同型：BioLLM scFM benchmark (2025, 10.1016/j.patter.2025.101326)、embeddings 单细胞表征评估 (2026, 10.1016/j.patter.2025.101431，即本稿引的 scelmo2026/Liu 2026)。honest-negative 在此是加分项；表征质量/序数几何正是 Patterns 已发的分析类型。
2. **Genome Biology** — 全领域最强先例集：Kedzierska 零样本负结果（Brief Report，43 引）、Wu 2025 六模型 benchmark、Yuan 2026 Borzoi 微调评估、scArchon 负结果 benchmark。Brief Report 格式可承载聚焦负结果。统计严谨性+基准资源加分；**单组织为主要减分**——需主打三域设计与 domain-specific 坍缩主张。

### 第二梯队（高度适配，先例密集）
3. **Bioinformatics** — 最直接同型聚集地：诚实负结果扰动基准（Wong 2025, btaf317）、scGPT 微调优化（Tay 2026, btag374）、细胞嵌入几何分析（Gilpin 2025, btaf595）、scFM 因果电路追踪（2026, btag379）。快、工具友好、单模型单组织为常规。
4. **PLOS Computational Biology** — Geneformer 尺度评估（Chen 2026, pcbi.1013701）+ 多个诚实负结果；可复现文化匹配 BH-FDR/置换零假设设计。

### 第三梯队（有条件适配）
5. **Nature Methods** — 2026 起接受 scFM 评估（DenAdel 等，10.1038/s41592-026-03120-y，Kedzierska 同组），但门槛为方法普适性；单模型单组织诚实负结果为减分，除非重构为"微调坍缩序数几何"普适原则。**修正：Geneformer 原文在 Nature (10.1038/s41586-023-06139-9)，非 Nature Methods。**
6. **npj Digital Medicine** — 仅当转临床/眼科框架（已有视网膜 FM 先例 10.1038/s41746-026-02496-7 + 健康 LLM 评估框架两篇）。
7. **NAR / GigaScience** — benchmark 资源拆分发表的家（NAR Database Issue / GigaScience 数据出版模型）。
8. **Briefings in Bioinformatics** — 有 LLM 细胞分型 benchmark（bbaf677/bbaf622），但综述占比高、流程慢。
9. **iScience** — 角膜-only 保底（findings.md 原判，非首选）。

### 不推荐
- **Cell Genomics / Cell Systems**：无 scFM 评估先例（ChromBERT/ProtoCloud/BulkFormer 均为新模型+生物学洞见型），honest-negative 偏离画像（findings.md 同判）。
- **eLife**：无 scFM 评估先例，不主投。
- **Nature Communications**：先例存在（scDrugMap benchmark 10.1038/s41467-025-67481-2、CellFM）但早前已内部否决（成本/门槛），除非补多模型普适性。
- **IOVS/TVST**：临床影像 AI scope 镜像错位（早前否决）。

## 二、专刊推荐（可核实项）

1. **NAR Database Issue（年度固定专刊）** — 已核实（2026 轮社论 10.1093/nar/gkaf1427 确认年度征集）；污染感知配对 benchmark 是典型 Database-issue 产物（同族有 CZ CELLxGENE Discover 10.1093/nar/gkae1142）。**与 Patterns 主投可拆分并行准备。** 当前轮截止日需浏览器查 academic.oup.com/nar。
2. **ISMB/ECCB Proceedings（Bioinformatics 年度专刊）** — 已核实存在（ISMB 2026 论文集 10.1093/bioinformatics/btag279）；评估可收但偏好新方法。
3. **无 scFM 专属开放专刊**（Europe PMC/Crossref 实证 absence）。趋势信号：Cell Virtual Cell Challenge（10.1016/j.cell.2025.06.008）、2026 评估预印本潮（PertEval-scFM 10.1101/2024.10.02.616248、MVCBench、"Signal, Bounds, and Baselines: Principles for Evaluating Virtual Cell Perturbation Models" bioRxiv 10.64898/2026.04.20.719650）——专刊可能从这些社群涌现，建议 watch。
4. **需浏览器人工核查**（自动化被 auth 墙拦截）：cell.com/collections、academic.oup.com/nar、biomedcentral.com/collections、nature.com 各刊 collections。**npj SBA collections 已于 2026-08-12 核验**（firecrawl 绕过 IDP；见下"npj SBA 开放 Collections 核验"）。

## 四、npj SBA 开放 Collections 核验（2026-08-12，nature.com/npjsba/calls-for-papers）

10 个开放 Collection 全部核实（deadline 均为确切日期）：
1. Systems AI: Integrating Systems Biology and AI for Mechanistic Drug Discovery — 2027-05-03
2. Towards generative biology: AI-enhanced synthetic biology — 2027-04-15
3. **Next-Generation Digital Twins for Drug Discovery** — 2027-03-22（Paper 2 候选）
4. Systems Biology for Psychiatric and Neurological Disorders — 2027-01-17
5. Spatial Biology Across Scales — 2027-01-15
6. Generative Models in Engineering Biology（跨刊）— 2027-01-13
7. Mechanistic and data-driven modeling in mathematical oncology — 2026-12-02
8. Nonlinear dynamics of living systems（跨刊）— 2026-11-09
9. Systems immunology: multi-omics + dynamical modeling + agentic AI — 2026-09-12
10. **Systems approaches for virtual clinical trials and digital twins** — 2026-10-31（Paper 2 候选，deadline 近）

**结论**：
- **Paper 1**：10 个 Collection 无一覆盖 scFM 评估/基准/表征质量——npj SBA 不推荐再添主源证据（其自身的 calls-for-papers 即证明 scope 在系统建模+机制发现端）。
- **Paper 2（DED-seq 虚拟细胞）**：#3 与 #10 两个 digital-twin Collection 真实相关（#10 deadline 2026-10-31 仅 ~2.5 月，若 Paper 2 处于设计期则偏紧；#3 至 2027-03-22 可行）。注意 Nature Portfolio Collection 模式 = 正常投稿 + 可路由入集合，入集合不等于保证接收。

## 五、npj Genomic Medicine 评估（2026-08-12，用户问询）

**结论：两篇论文均不适合。** 官方 calls-for-papers 页 5 个开放 Collection（PRS 与复杂疾病预测 2027-04-17 / Diploid Genome 2027-02-07 / **AI in Genomic Medicine 2026-09-23** / Non-coding Genome 2026-08-29 / microbiome 2026-09-28），唯一沾边的是 AI in Genomic Medicine（deadline 5 周内）。
**证据**（Europe PMC 85 条命中翻看 top 8）：该刊 scRNA-seq 论文全部绑定临床基因组学语境——癌症基因组（前列腺癌 lncRNA、多发性骨髓瘤免疫微环境、肺癌循环血标志物）、疾病相关变异（CHD 罕见变异外显子组）、临床表型（癫痫记忆表型）、精准医疗实施（量子计算）。**零 scFM 评估/基准先例**；single-cell 在此刊必须携带遗传学/临床基因组价值。
- Paper 1（scFM 表征评估）：无临床基因组学角度，不入 scope。
- DED 干湿代谢重编程：角膜上皮机制 + 转录组 + 代谢，无胚系/体细胞变异与基因型-表型，不入 scope；既有梯队（JTM / Redox Biology / Ocular Surface）明显更优。

## 六、BiB 与 MATHEMATICAL & COMPUTATIONAL BIOLOGY 类别评估（2026-08-12，lib-1）

### BiB 画像（已验证）
- **JIF**：2024 = 7.7 / 2025 = 7.3（Wikipedia + bioxbio 双源一致）；波动大：13.99(2021)→9.5→6.8→7.7→7.3。
- **体裁**：2024–26 样本 60% 原创研究，但综述是编辑身份（review-first legacy → 纯评估论文有 desk-reject 风险，须以"基准资源+配方"框架投）。
- **审稿**：DOAJ 确认 **14 周到发表**；首决时间未核实（OUP 页被墙）。
- **APC**：**USD 3,885** 全 OA（2024 起），waiver 可申请。
- **先例（DOI 核验）**：9+ 篇 benchmark/评估/局限研究（TCR-pMHC 统一评估 bbag289、AlphaFold 统计正确基准集 bbaf104、STGBench bbag354、局限研究 bbaf098 等）；**无 Geneformer 微调几何论文**；诚实阴性被接受的前提 = 负测量 + 可复用资源/配方——与本稿形态吻合。

### 类别全景（JIF 2024/2025，来源标注）
| 期刊 | 2024 | 2025 | 备注 |
|---|---|---|---|
| Briefings in Bioinformatics | 7.7 | 7.3 | 双源 ✓✓ |
| Bioinformatics | 5.4–5.6 | 5.5–6.2 | ⚠️ bioxbio vs scijournal 分歧，引用前需 JCR 确认 |
| PLoS Computational Biology | 3.5 | 3.5 | |
| BMC Bioinformatics | 3.3 | 4.4 | |
| GigaScience | 3.68 | 4.33 | 波动（2022 曾 8.36） |
| NAR Genomics and Bioinformatics | 2.69 | 2.99 | |
| Database (Oxford) | 3.6 | 2.7 | 下滑（2022 曾 5.8） |
| Frontiers in Genetics | 2.76 | 2.88 | |
| IEEE/ACM TCBB、CSBJ | 未核实 | 未核实 | |

### 类别内适配排名（本稿）
1. **Bioinformatics** — 三支柱先例全有（诚实负结果 btaf317、scGPT 微调 btag374、嵌入几何 btaf595、scFM 可解释 Kendiukhov）；最快（1–3 月）；单模型单组织常规。
2. **BiB** — 类别内最高 IF + 2025–26 benchmark 组合强；条件性（见时点规则）。
3. **PLOS CB** — Geneformer 评估（pcbi.1013701）+ 诚实负结果先例；IF 3.5、最慢（12–24 月）。
4. **BMC Bioinformatics** — **Kendiukhov 2026 残差流几何 scFM（10.1186/s12859-026-06538-5）= 全领域最近似先例**（本稿已引 kendiukhov2026genomics/bioinformatics）；便宜快，desk-reject 风险低。
5. **GigaScience / Database (Oxford)** — 若重框为 benchmark 资源；Database 有**泪液数据库先例**（2026 baaf091）+ GenAI 诚实评估（baaf011）。

### 时点决策规则（IF/THEN）
- 求 JIF/曝光 + 可吸收 $3,885 → **BiB**，且**仅在 Patterns/GB 路线耗尽后**（BiB = 安全高 IF 地板，非首选；resource+recipe 框架）。
- 求可靠接收 + 速度 → **Bioinformatics**（1–3 月，诚实负结果先例）。
- 重框为 benchmark 资源 → **Database (Oxford)** / **NAR Database Issue 2027**（年度周期已核实）。
- 可复现身份 + 时间灵活 → **PLOS CB**。
- 单组织负结果最小摩擦 → **BMC Bioinformatics**。
- **负条件**：不为 JIF 趋势选 BiB（13.99→6.8→7.7→7.3，等待不改善指标；2025 JIF 已计入）。

### 对本稿梯队的影响
- 维持 **Patterns 主投 → GB → Bioinformatics/PLOS CB**；新增 **BiB（高 IF 地板，条件性）**、**BMC Bioinformatics（被低估的稳妥选项）**、**Database/NAR（资源拆分通道）**。
- 未核实项（投稿前确认）：BiB 首决时间/字数上限、TCBB/CSBJ JIF、Bioinformatics JIF 分歧、精确 JCR 分区。



- **主投 Patterns**（投稿包已齐）→ 拒后按序 Genome Biology（可用 Brief Report 承载）→ Bioinformatics / PLOS CB。
- 并行拆分：benchmark 资源独立成文投 **NAR Database Issue**（与 Patterns 主投不冲突）。
- MJS 对冲仅 Springer Nature 组内适用（GB 三刊对冲，早前记录）。
- 若审稿反馈聚焦"单模型"，升级选项 = Nature Methods 普适原则重构 或 npj DM 临床转框。

## 七、v13 期刊再评估（2026-08-13，探针面板后）

**变化**：稿件新增 4 模型 frozen 探针面板（Geneformer ρ 0.81/0.71、scFoundation 0.75/0.72、UCE 0.58/0.71、scGPT 0.22/0.17 + 全模型细胞级 probe F1 + Table S8）——"序数编码架构依赖"从单点变多模型证据。

**影响**：
1. **Patterns（主投不变，概率上修 ~40–55%）**：representation-quality 体裁典型形态（多模型编码图谱 + 可分离性 + 诚实阴性 + 资源），直接命中其已发表分析类型；多模型证据回应"单模型"批评。
2. **Genome Biology（#2，微上修）**：跨架构探针部分缓解"单模型单组织"减分（Kedzierska/Wu 均为多模型）；RA/Brief Report 均可。
3. **Bioinformatics / BMC / PLOS CB（微上修）**：证据广度增加。
4. **Nature Methods（仍条件性）**：探针是冻结编码证据，**非微调坍缩证据**——坍缩测量仍 Geneformer 单模型（结构性限制），NM 需坍缩跨模型或通用方法。
5. **BiB**：resource+recipe 框架内容更足（多模型编码图谱入 benchmark 组合）。

**结论**：梯队排序不变（Patterns → GB → Bioinformatics/PLOS CB；BiB 高 IF 地板）；v13 上修"评估型期刊"接受概率，不改变"坍缩仍单模型"的结构边界。升级杠杆不变：第二模型的**同语料**坍缩测量（当前测试集对全部候选模型语料外）。

## 待办
- [ ] 浏览器人工核查四个 publishers 的 collections 页找专刊（自动化不可达）
- [ ] Patterns 主投流程推进（Graphical abstract、作者清单、数据 deposit）
