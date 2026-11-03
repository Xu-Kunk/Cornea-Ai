# Donor-Level Contamination Audit — GSE153515 vs Genecorpus-104M (审稿问题 M5)

**日期**: 2026-08-24（所有网络证据获取时间均为当日 CST，逐条标注）
**范围**: 判定外部测试集 GSE153515 的两名供体（GSM4646295 / GSM4646296，donor_1 / donor_2）是否出现在 Geneformer V2 预训练语料 Genecorpus-104M 中。
**前置事实（已确认）**: study 级重叠成立 — 语料官方来源表 MOESM3 `genecorpus_104m` 第 963 行 = DOI `10.1016/j.jtos.2020.12.004`，53,290 cells，organ=eye。
**只读约束**: 本审计未修改任何现有文件；仅新增本报告与配套 JSON。

---

## 1. 方法

按四条独立取证途径核查 donor 级粒度：

1. **语料源表粒度检查**：读取本地缓存的 NatCS 2026 SI 表（MOESM3 xlsx，两处副本校验一致），枚举全部列名与 eye-organ 行，确认该表的元数据粒度上限。
2. **语料发布物结构检查**：审查 HuggingFace `theodoris-lab/Genecorpus-104M` 数据卡与文件清单，确认发布 shard 是否携带任何 donor/样本级可匹配字段。
3. **GSE153515 供体构成核实**：解析 GEO series 与两个 GSM 样本记录、及该 series 关联的两篇论文（PMID 33388438、34297801），确定研究的供体总数与可匹配特征（年龄/性别等）。
4. **CELLxGENE 整合路线排查**：查询 CXG API 全量 collections/datasets，检查收藏 `0f7d022a`（Human Ocular Surface Atlas）的 dataset 元数据是否披露 donor_id、是否整合了 GSE153515 数据、及其时间线与语料装配窗口的关系。

判定逻辑采用三档标准：(a) donor ID 直接匹配 → confirmed；(b) 可验证地排除两名供体 → excluded；(c) 元数据粒度不足无法判定 → indeterminate。

---

## 2. 证据清单（URL + 获取时间）

| # | 证据 | 来源 URL / 路径 | 获取时间 | 内容要点 |
|---|------|----------------|---------|---------|
| E1 | NatCS SI MOESM3 源表 | https://static-content.springer.com/esm/art%3A10.1038%2Fs43588-026-00972-4/MediaObjects/43588_2026_972_MOESM3_ESM.xlsx （本地副本：`artifacts/paper1_vanzyl/contamination/evidence/` 与 `artifacts/archive/paper1_data_audit/evidence/`） | 2026-08-24 ~14:45（本地读取）；URL 存活验证 15:02（HTTP 206） | 全表仅 3 列：`source_DOI` / `n_cells_passed_filtering` / `organ`；第 963 行 = jtos DOI，53,290 cells，eye；eye 行共 136 行 / 26 个唯一 DOI，**无任何 donor 或 accession 字段** |
| E2 | Genecorpus-104M HF 数据卡 | https://huggingface.co/datasets/theodoris-lab/Genecorpus-104M | 2026-08-24 ~14:52 | 发布物仅 `genecorpus_104M_4096.dataset`（tokenized Arrow）+ lengths pkl + token dictionary；Data Fields 仅 `input_ids`；"does not contain annotations"；"no traceable sequencing reads"；来源含 GEO/SRA/CELLxGENE 等；2,903 datasets ≈ 104M cells |
| E3 | GEO GSE153515 series 记录 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153515 | 2026-08-24 | **全 series 仅 2 个样本**（GSM4646295、GSM4646296）；"isolated limbal basal cells from two young donors"；~18,000 cells pre-QC / 16,360 post-QC；关联 PMID 33388438 与 34297801 |
| E4 | GSM4646295 记录 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4646295 | 2026-08-24 | `LSC_scRNAseq_1`，Corneoscleral tissues-1，LEBT（Lions Eye Bank of Texas）眼库；characteristics 仅 `tissue: Corneoscleral tissues`——**无年龄、性别或任何供体标识符** |
| E5 | GSM4646296 记录 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4646296 | 2026-08-24 | `LSC_scRNAseq_2`，Corneoscleral tissues-2，同上，无供体标识符 |
| E6 | PubMed 33388438（Li et al., Ocul Surf 2021, DOI 10.1016/j.jtos.2020.12.004） | https://pubmed.ncbi.nlm.nih.gov/33388438/ | 2026-08-24 | 分析对象为 **16,360 limbal basal cells**（即上述两供体）；MeSH 含 Female+Male（两供体一男一女，年轻） |
| E7 | PubMed 34297801（Li et al., IOVS 2021, DOI 10.1167/iovs.62.9.36） | https://pubmed.ncbi.nlm.nih.gov/34297801/ | 2026-08-24 | 同一 GEO series 的第二篇论文，分析**同一批 16,360 细胞** → 不引入额外供体 |
| E8 | CELLxGENE API collection 0f7d022a | https://api.cellxgene.cziscience.com/curation/v1/collections/0f7d022a-46c7-4e64-be4c-e34adbb78089 | 2026-08-24 14:47 | "Single cell atlas of the human ocular surface"；联系人 Rui Chen（UCI；亦是 GSE153515 合作者）；collection version 创建于 **2026-06-10**，晚于语料发布（2026-03-27）；dataset 版本最早 published 2025-05-05 |
| E9 | CELLxGENE API 全量 datasets | https://api.cellxgene.cziscience.com/curation/v1/datasets | 2026-08-24 14:55 | HOSA all-cells dataset（61adb974…，577,857 cells）披露 **50 个 donor_id**：`shi_donor1–4`、`chen_donor1–2`、`li_donor72/73/75/76`、`chakravarti_Cornea1–3`、`dickman_Cornea01–08(+pooled)`、`BCM_*`(20)、`246Z/257T/283P/719I`、`Atlanta_A230245`、`LEI_22_49449` 等；fetal 版含 74 donors（数值型 fetal ID）。**无任何标签可对应到 GSM4646295/96 的两名供体**；assay 含 10x v2/v3（与 GSE153515 化学法兼容，但非判据）；tissue 含 corneo-scleral junction |
| E10 | CELLxGENE 收藏页（描述全文） | https://cellxgene.cziscience.com/collections/0f7d022a-46c7-4e64-be4c-e34adbb78089 | 2026-08-24 14:58 | "publicly available datasets were integrated to expand the scope of the atlas... over 1 million cells and nuclei from 102 donors" —— 收藏确有公开数据整合，但未公开 donor→source-study 映射表 |
| E11 | CAP 项目页 | https://celltype.info/project/565 | 2026-08-24 14:59 | 项目 DOI 为 CAP 内部 DOI `10.57772/cap.lr1qubc8lvq53byicflpmnedecs2.565`；无期刊 DOI；页面不含供体映射 |
| E12 | Crossref 10.1101/2023.11.07.566105 | https://api.crossref.org/works/10.1101/2023.11.07.566105 | 2026-08-24 | eye 行中最大的 bioRxiv DOI（1,378,557 cells）为视网膜 atlas（Li/Luecken/Monavarfeshani），**非** HOSA，排除混淆 |
| E13 | 有界细胞级 Signal A 结果（既有审计产物） | 本地 `artifacts/archive/paper1_data_audit/contamination_signalA.json`（+ 报告 §1.4） | 2026-08-24 读取 | 7 个均匀采样 shards（557,524 fingerprints，语料 0.54%）：GSE153515 15,827 个查询指纹 **0 命中**（max Jaccard=0.0，两种编码模式均然）；正对照召回 100%（median Jaccard=1.0）、负对照 0 误报。若整批数据位于单一 shard，被采样命中的概率仅 ~0.6% → **弱阴性，不构成排除证据** |

### 关于 E8/E9 的两点澄清（修正任务前提）

1. **"语料含 CELLxGENE 收藏 0f7d022a" 尚无可验证依据**：MOESM3 的 26 个 eye DOI 中无 HOSA 相关 DOI（其期刊 DOI 尚不存在，CAP DOI 10.57772/… 亦不在表中）；既有审计已记录"语料 23 行 cellxgene 来源行均无 eye organ"。因此 CXG 收藏进入预训练语料这条路线**未被证实也未被证伪**。
2. **HOSA 收藏本身不披露 donor→源研究映射**：50 个 scRNA-seq donor_id 中 `li_donor72/73/75/76` 的归属无法从公开渠道解析（GSE147979 经 GEO 核实为 Gautam et al. 多物种眼 compartment 研究，27 样本，与 "li_" 前缀的对应关系不明）。但即使 HOSA 曾整合 GSE153515，由于语料经 CXG 吸收 HOSA 这条路线本身未证实，它不改变主判定。

---

## 3. 判定

### 3.1 分档结论：**(c) indeterminate（donor ID 直接匹配不可得）— 但 donor 级暴露必须推定为存在**

| 档位 | 是否达成 | 理由 |
|------|---------|------|
| (a) donor 级重叠确认（donor ID 直接匹配） | ✗ 不可达 | 语料侧不存在任何 donor 标识：源表仅 DOI 级（E1）；发布 shard 仅含 `input_ids` rank 编码（E2）。匹配在数学上不可能，而非尚未尝试 |
| (b) donor 级排除（可验证地不含这两名供体） | ✗ 不可能成立 | 结构性论证：GSE153515 **只有** 两名供体（E3），其全部细胞必然属于 donor_1 或 donor_2；关联第二篇论文亦复用同一批 16,360 细胞（E7）。因此只要第 963 行对应真实摄入的数据，其中就不可能"不含这两名供体"。唯一逃逸路径是"该行系记账错误/错配"，无任何支持性证据但也无法公开证伪 |
| (c) indeterminate | ✓ 正式判定 | 元数据粒度不足以做 donor ID 直接核验；且 (b) 在本研究供体构成下逻辑不可达 |

### 3.2 为什么实际风险应按"重叠存在"处理（推定暴露，near-certain）

1. **研究级摄入已被决定性证据确认**（DOI 行，53,290 cells）。
2. **无选择性剔除机制的证据**：批量摄入管线以整库为单位获取公开 deposit；无任何文档表明 Genecorpus 按 GSM 样本或供体做了子集筛选。
3. **行内计数大于研究总量**：53,290 > 16,360（post-QC）> 18,000（pre-QC）。最合理的解释是摄入自未经空滴过滤的原始 barcode 矩阵或聚合器变体——这与"整库摄入（两个样本都进）"一致，而与"只挑了一个样本"矛盾（单样本最多 ~9k cells）。
4. **时间线无豁免**：GSE153515 公开 2021-01-14，远早于 2026 年语料装配。
5. **有界 Signal A 未命中不改变结论**：覆盖率 0.54%、整批数据大概率聚居于单一 shard，检测概率 ~0.6%，属预期内的弱阴性（E13）。

**残余不确定性**（诚实声明）：(i) 无法排除 MOESM3 该行存在元数据错配的可能（计数不一致提示摄入细节与论文 QC 口径不同）；(ii) 无 per-shard manifest 公开，完整 600 GB 语料下载在本项目资源约束下不可行。两者均指向同一个补救：**向语料作者索取 per-dataset/per-shard manifest 或直接询问**。

---

## 4. 对论文声称的影响（frozen 基线 ρ=0.806）

- ρ=0.806（`results_frozen_gf.json`: marker_gradient_rank_correlation = 0.80636…，GSE153515 frozen Geneformer 基线）产生于一个**应推定为被预训练见过的测试集**。零样本/frozen 端点在该集上的绝对值不能作 leakage-free 解释：模型可能在预训练中记忆了两名供体细胞的表达谱系结构，从而抬高梯度保持相关系数。
- **不影响的部分**：paired 设计的对照逻辑完好——GSE299152 为 post-cutoff、out-of-corpus 测试（DOI 不在源表、供体不相交），凡在 GSE299152 上成立的结论对预训练污染稳健；fine-tuning 引起的梯度塌缩是相对测量（frozen→fine-tuned 的变化方向），主要结论不依赖 GSE153515 的绝对值。
- **需要修改的部分**：
  1. 不得将 GSE153515 描述为 "unseen by pretraining"；现有 "same-corpus reference" 定性正确且应保留，并升级为明确包含 donor 级表述；
  2. Limitations 中应写明：donor 级排除不可达、按保守原则推定两名供体的细胞均在预训练语料内；
  3. 以 GSE153515 为唯一支撑的任何端点解释需加限定语。

---

## 5. 建议措辞（可直接粘贴）

> Because GSE153515 comprises exactly two donors (~16,360 post-QC cells), every cell of this study necessarily belongs to one of these two donors. The study's primary DOI is present in the official Genecorpus-104M source table (NatCS 2026 SI, MOESM3, sheet genecorpus_104m: 53,290 cells, organ=eye); the table provides DOI-level granularity only, and released corpus artifacts contain rank-value encodings without donor identifiers. Donor-level matching or exclusion is therefore not verifiable from public artifacts; by the conservative interpretation required here, we presume that cells from both donors were seen during pre-training. All endpoints on this same-corpus test are therefore reported alongside the post-cutoff, out-of-corpus test (GSE299152), and conclusions are conditioned on surviving there. A bounded cell-level fingerprint search over a uniform 0.54% corpus sample yielded no matches (positive/negative controls passed), which is expected under whole-dataset shard placement and does not constitute evidence of absence.
>
> （中文）GSE153515 仅由两名供体构成（QC 后约 16,360 细胞），故该研究任一细胞必属于这两名供体之一。其主 DOI 出现在 Genecorpus-104M 官方来源表中（DOI 级、53,290 cells）；源表仅提供 DOI 级粒度，且发布的语料 shard 不含供体标识，故 donor 级匹配与排除均无法经公开产物核验。按保守原则，我们推定两名供体的细胞均在预训练中被见过；同语料测试集的所有端点均与截止后、语料外测试集（GSE299152）并列报告，结论以在后者的存活为准。对均匀采样的 0.54% 语料的有界细胞级指纹检索未发现命中（正/负对照通过），该结果在整库分片放置模式下属预期，不构成"不存在"的证据。

**可选补强动作**（按性价比排序）：
1. 致信 Chen & Venkatesh / Theodoris lab，请求确认 `10.1016/j.jtos.2020.12.004` 行的摄入来源（GEO accession / 处理矩阵版本）与 per-shard manifest 的可用性——这是唯一能将判定升为 (a)/(b) 的途径。
2. 本地核查训练 atlas 的 `chen_*` 库 obs（81,108 cells，源自 HOSA 生态）是否存在 GSE153515 衍生条目（barcode 前缀 / LSC_scRNAseq 标签 / 16,360 总数指纹），以封堵训练侧同名混淆风险（超出 M5 范围，但成本极低）。
3. 若获得 GPU 窗口，可在 GSE299152 上补测 frozen 零样本端点，量化"同语料 vs 语料外"frozen 性能差，作为泄漏幅度的经验界。

---

## 6. 结论速览

- **正式判定**: (c) indeterminate —— donor ID 直接匹配不可得（语料产物无 donor 标识），donor 级排除逻辑上不可达（研究仅含这两名供体）。
- **操作判定**: donor 级暴露按"存在"处理；ρ=0.806 及一切 GSE153515 frozen 端点不得解读为无泄漏；结论承载转移至 GSE299152。
- **置信度**: 高（粒度不足这一事实本身由源表列结构与 HF 数据卡直接验证，不存在解释空间）。
