# Patterns (Cell Press) 投稿要求 — paper1 对标清单

> 来源：lib-1 (librarian) 2026-08-04 核实。官方 live pages（cell.com/patterns/information-for-authors + article-types）+ 2023 archive + Cell Press 官方 blog + ScienceDirect。
> 状态标注：[UNVERIFIED] = 未能当日复核。

## 1. 格式与字数

| 项目 | 要求 | 本稿现状 |
|------|------|---------|
| 正文字数 | ~5,000–7,000 词典型，无硬上限（"长度与科学内容匹配即可"） | 达标（见字数统计） |
| Summary/Abstract | 单段、≤150 词、非结构化、无引用、禁 "novel/first" 宣称 [UNVERIFIED 数字来自 2023 archive + Cell Press 全局标准] | 145 词 ✅ |
| 章节结构 | introduction → results → discussion → methods → references；首次投稿可放宽 | 一致 ✅ |
| Keywords | ≤10 | — |

## 2. 图表上限

- **正文无图/表/参考文献数量硬上限**（当前 article-types 页明确；比其他 Cell Press 期刊宽松，编辑有裁量权）
- 补充材料无明确上限；通用生产规格：彩色 ≥300 dpi、黑白 ≥500 dpi、线条图 ≥1000 dpi；TIFF/PDF 优先；单文件 ~20 MB 上限

## 3. 投稿包要求

| 项目 | 要求 | 状态 |
|------|------|------|
| Cover letter | 首次投稿必需；须论证与 Patterns scope 契合、含预印本链接、披露相关未发表论文；审稿人不可见 | 已规划三锚点版 |
| Bigger Picture | Patterns 研究文章特色项；**首次投稿不强制**，终稿强制；1–2 段、≤300 词 | 已规划 213 词 |
| Highlights | ≤4 条、每条 ≤85 字符（含空格）；终稿提交，与 eTOC 放同一 Word 文档 [UNVERIFIED 来自 2023 archive + blog] | 已规划 ≤4×85 |
| eTOC blurb | **≤80 词**、第三人称（"Corresponding author and colleagues…"）[UNVERIFIED] | 当前 39 词 ✅ |
| Graphical Abstract | Patterns 可选；1200px 方形 @300dpi，Arial 12–16pt [UNVERIFIED] | 待做 |
| Data & Code Availability | **必需**：lead contact / materials availability / data & code availability 三小节；数据/代码入 DOI 仓库，acceptance 前 deposit，缺失可致撤稿 | 稿件已有 4 组件 FAIR ✅ 需实际 deposit |
| 表单 | 数据报告 checklist、Declaration of Interests、CRediT、Inclusion & Diversity（2021-01 起强制）、generative AI 声明 | 待做 |
| STAR Methods | **Patterns 不使用 STAR Methods**——用集成式 Methods（"methods" 节） | 本稿 §2.1–2.10 集成式 ✅ |

## 4. 文章类型与 scope

- 类型：Research Article / Resource / Review (8,000–10,000 词, ≤150 refs) / Perspective / Tutorial / Preview / Opinion / People of Data（2023 版另有 Descriptor）
- Scope："data science in the broadest sense" —— 计算与数据密集型研究
- 流程：Editorial Manager (editorialmanager.com/patterns)；presubmission inquiry → patterns@cell.com（title+abstract+significance，2–5 工作日回复）；支持 Multi-Journal Submission；中位 5 天到首决、首轮审后 43 天、到接收 154 天
- **scFM benchmark 先例（重要）**：BioLLM, *Patterns* 6(10):101326 (2025), doi:10.1016/j.patter.2025.101326 —— 13 数据集 benchmark scGPT/Geneformer/scFoundation/scBERT；另有 2026-05-08 的 LLM 临床 QA benchmark 论文。→ 本稿"benchmark 资源"组件有直接先例

## 5. JIF

- 2024 JIF = **7.4**（确认：journalsimpactfactors.com）
- 2025 JIF = **10.8**（高可信，三独立追踪站一致 + 官方 JCR 2026-06-17 发布确认；JCR 本身付费墙 [UNVERIFIED 官方复核]）——frontmatter 所记 10.8 维持

## 6. APC 与预印本

- APC = **USD 4,900**（ScienceDirect 官方页；journalsearches.com 的 $8,900 判定为过时错误）
- 预印本：**bioRxiv 允许且鼓励**（"Authors are encouraged to share a preprint…"）；BioLLM 先例在 Patterns 发表前已挂 bioRxiv 2024.11.22.624786

## 与本稿对标结论

1. 形式五达标维持（字数/摘要/主图/主表/补充材料）
2. eTOC 字数上限 50→80（task_plan 修正）
3. Patterns 无 STAR Methods → 现有集成式 Methods 正确，勿改
4. 证据广度唯一缺口 = scGPT 基线（Lane fix-4 运行中）
5. 投稿包待办（inquiry 回复积极后做）：Cover letter 三锚点 / Bigger Picture / Highlights / eTOC / Graphical Abstract / 数据 deposit / 表单
