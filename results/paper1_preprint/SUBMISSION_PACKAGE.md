# Paper 1 — bioRxiv/Patterns Preprint Submission Package

> 注：本文件原为 v18 预印本口径，已于 2026-09-22 同步至 v23；bioRxiv 预印本计划已取消，当前目标为 Cell Press Multi-Journal Submission（Patterns + Cell Reports Methods）。
>
> 生成: 2026-08-27（v18 终稿）· 2026-09-22（v23 收尾更正）
> 稿件版本: **v23 final candidate**（唯一 active 主稿 `docs/manuscript/patterns/latex/paper1_final_candidate_v23.tex`）
> 稿件 PDF: `docs/manuscript/patterns/latex/paper1-current.pdf`（39 页，0 overfull，0 编译错误；主图 7、主表 5、补充图 S1-S13、补充表 S1-S9）
> 模板来源: Cell Press LaTeX article template v1.10 — https://www.cell.com/article-templates
> 历史版本: v3–v18 保留作溯源；v17 为上一预印本

## 提交元数据（Cell Press Multi-Journal Submission 表单字段）

| 字段 | 值 |
|------|-----|
| **Title** | Fine-tuning collapses differentiation gradients in single-cell foundation models in a corpus- and architecture-dependent manner |
| **Author 1** | Kunhui Xu (first author) — National Clinical Research Center for Ocular Diseases, Eye Hospital, Wenzhou Medical University |
| **Author 2** | Chenchen Wang — 同上机构 |
| **Author 3** | Li Fenfen (co-corresponding author) — 同上机构 |
| **Author 4** | Qi Dai (co-corresponding author) — 同上机构 |
| **Contact email** | lifenfen_wzmu@163.com（Li Fenfen）; dq@mail.eye.ac.cn（Qi Dai） |
| **Abstract** | 见下方（v23 口径，150 词，与主稿 SUMMARY 逐字一致） |
| **Keywords** | single-cell foundation models; classification fine-tuning; differentiation gradient; ordinal regression; contamination-aware benchmark; corneal epithelium; parameter-efficient adaptation; cross-architecture evaluation |
| **Subject category** | Bioinformatics / Genomics |
| **License** | CC-BY 4.0（建议；与 Patterns 政策兼容） |

## Abstract（提交用）

> 口径：v23；与主稿 `\section*{SUMMARY}` 逐字一致（150 词，为 Patterns 与 CRM 的 ≤150 词上限）。LaTeX 记号已转纯文本（`$\rho$` → ρ，`$-0.008$` → −0.008）。摘要为单段、无引用、无 "novel/first" 宣称，符合 Cell Press 两刊要求。

Supervised fine-tuning of single-cell foundation models is common, but whether it preserves the continuous differentiation geometry of pretrained representations remains unknown. Using the human corneal epithelium as a four-stage differentiation continuum, we find that classification fine-tuning collapses the differentiation gradient across architectures. Collapse is corpus-determined: on same-corpus tissue, Geneformer degrades the gradient from a frozen Spearman ρ of 0.806 to −0.008 under the main three-epoch protocol and to 0.096 under the official one-epoch budget, and scGPT-human shows moderate loss (0.789 to 0.44, best-val), whereas out-of-corpus and cross-tissue effects remain mild or reversed. Collapse determinism is architecture-determined: Geneformer collapses deterministically, whereas scGPT-human shows pronounced seed dependence, its outcome ranging from collapse to full gradient retention. A cumulative ordinal regression network (CORN) head rescues the gradient in both architectures within the same corpus; combined with LoRA, CORN+LoRA achieves a three-seed mean ρ of 0.924. We release a contamination-aware benchmark pairing same-corpus and out-of-corpus tests.

## v17 文风润色（2026-08-22）

- **Summary 重写**：gap-first，零数字，claim 动词 demonstrate/reveal。
- **Results 小节标题**：全部改为陈述句信息型（8–12 词），无编号。
- **术语统一**：scFMs 首次出现标注；ρ 首次出现加定义 "Spearman's rank correlation coefficient (ρ)"。
- **去 AI 味**：Notably/Importantly/"It is worth noting"/"In conclusion"/delve/remarkably/strikingly 等禁用词清零；em-dash 仅保留括号插入式。
- **阴性结果话术**：CAFT/AA-FT/soft-Spearman/flow/GF LoRA OOM 各节按四句式改写。
- **Bigger picture 重写**：四段式白话，零术语零数字，约 213 词。
- **Methods 轻度润色**：仅去 AI 味与时态规范，技术内容/参数/文件名不动。

## v16 核心结果摘要

- **CORN + LoRA（scGPT-human, van Zyl）**: three-seed mean ρ = 0.924 ± 0.001, macro-F1 = 0.966 ± 0.002 (source: `artifacts/paper1_vanzyl/corn_lora/summary.json`; seed456 extracted from `checkpoints/corn_lora_seed456/run.log`).
- **CellPLM LoRA**: ρ = 0.873, z = 132.4, macro-F1 = 0.958 (source: `artifacts/paper1_vanzyl/cellplm_lora/seed42/seed42.json`).
- **CellPLM layerwise**: frozen LAT ρ = 0.865; naive LAT ρ = −0.046; CORN LAT ρ = 0.920; naive L1 collapse 0.906 → 0.118 (source: `artifacts/paper1_vanzyl/cellplm_layerwise/cellplm_layerwise_rho.json`).
- **Head-init basin selection**: three variants retained ρ = 0.572–0.804 vs default seed42 collapse ρ = −0.091 (source: `artifacts/paper1_vanzyl/head_init/head_init_summary.json`).
- **Sensitivity**: pc1_centroid robust across seeds/architectures; marker panels robust; rank-invariance gates all pass (source: `artifacts/paper1_vanzyl/sensitivity/sensitivity_summary.json`).
- **BH-FDR**: 21 comparisons, 20 significant; only GF vanZyl naive non-significant (q = 1.00) (source: `artifacts/paper1_vanzyl/fdr/bh_fdr_table.json`).
- **SVD-axis validation**: frozen Spearman layer-ordering = 1.0; ρ axis not PC artifact (source: `artifacts/paper1_vanzyl/svd_validation/svd_axis_validation.json`).
- **Cross-tissue/LOO**: epidermis final-epoch per-seed ρ = 0.350/0.191; LOO Hu0220 final-epoch ρ = 0.456/0.447 (sources: `cross_tissue/final_test_eval.json`, `loo/final_test_eval.json`).
- **Second corneal datasets**: GSE186433 naive ρ = 0.816/0.849; GSE155683 naive ρ = 0.836/0.877 (best-val, seeds 42/123) (source: `artifacts/paper1_vanzyl/cornea2/summary_cornea2.json`).
- **GF LoRA OOM**: identity check pass (max|Δ| = 6.45e-06); first forward OOM on 16 GB GPU, so GF LoRA arm absent (source: `artifacts/paper1_vanzyl/gf_lora/oom_evidence/`).

## 文件清单

- `docs/manuscript/patterns/latex/paper1_final_candidate_v23.tex` — v23 主 LaTeX 源文件（Cell Press template v1.10；唯一 active 入口）
- `docs/manuscript/patterns/latex/numbered.bst` — Cell Press 数字上标 AMA 参考文献样式
- `docs/manuscript/patterns/latex/numcompress.sty` — Cell Press 参考文献压缩包
- `docs/manuscript/patterns/latex/paper1.bib` — 80 条参考文献
- 历史副本：`docs/manuscript/latex/`（superseded）；v18 归档：`docs/manuscript/biorxiv/archive-v18/`
- `docs/manuscript/patterns/latex/paper1-current.pdf` — v23 PDF（39 页，0 错误，0 overfull）
- `docs/manuscript/patterns/patterns_submission_extras.md` — Highlights / eTOC / Bigger Picture
- `docs/manuscript/paper1_v17_polish_report.md` — 润色报告
- `docs/changes/2026-08-22-paper1-v17-style-polish.md` — 变更日志

## Patterns 投稿附加件状态

- **Highlights**: 4 条，每条 ≤85 字符（已复核）
- **eTOC blurb**: 39 词，≤80 词
- **Bigger Picture**: 约 213 词，≤300 词；零术语零数字
- **Graphical Abstract**: Patterns 可选，投稿后视编辑要求补

## 编译信息

```text
pdflatex -> bibtex -> pdflatex x2
模板: Cell Press LaTeX article template v1.10
页数: 39
bib 文件条目: 80；正文中实际引用 43 条
摘要词数: 145（v18 口径，待 v23 摘要替换）
编译错误: 0
Overfull \hbox: 0
Underfull \hbox: 少量（itemize 长路径，已尽量拆分）
```

## 未完成项 / %TODO

- ~~作者清单 PLACEHOLDER~~ ✅ 2026-08-24 定稿：Kunhui Xu / Chenchen Wang / Qi Dai（通讯）；四处声明节已起草织入（Acknowledgments 未含基金号——如有资助需用户补充；AI 声明按 Cell Press 模板句式如实披露 LLM 辅助代码/统计/语言润色）。
- ~~模拟审稿修复~~ ✅ 全部 Critical/Major/Minor 项已处理（C1-C5/M1-M8/m1-m10），详见 docs/changes/2026-08-24-paper1-v18-review-response-weave.md
- Graphical abstract：用户决策暂缓，投稿后再补（Patterns 不强制）。
- **v23 冻结**：`paper1-current.pdf`（39 页 / 0 错误 / 0 overfull）为投稿版本；bioRxiv 预印本已取消，目标为 Cell Press Multi-Journal Submission（Patterns + Cell Reports Methods）。

## 2026-08-22 终稿重建
- 图件全部定稿后重编译：Figure 1 v4（draw.io 三列流水线）+ fig7_corn_lora_umap 9/9 + fig_umap_collapse 9/9 + figureS10 真实数据 + 全部图件 Patterns 美学标准
- bib 修正：wen2024cellplm 补 OpenReview URL (BKXvPDekud)
- 编译：0 错误 / 0 overfull / 27 页

## 2026-08-22 五轮图件迭代定稿（R1–R5）
- 流程：5 轮 "对标 Patterns 已发表图（scELMo/BioLLM/GSFM/NESS 8 张参考）→ observer 审查 → 修复" 循环；主图+补充图双组并行，共 13 个审查/修复专线
- 主图终态：fig1(draw.io v8, 9 分)/fig2–9 全部达标（R5 终审 9/11 达标 + micro-pass 定点修复 3 图）；fig7_umap 3×3 等大 UMAP；fig9 标签/注解重组；fig4 图例外置；bistable 4 panel；collapse 紧凑网格
- 补充图终态：S1–S11 + sensitivity summary 全部按 Patterns 标准定稿（R5 终审 8/12 达标 + micro-pass 修复 S2/S3/S9/S11 四图）
- 全局规范：纯白底（轴内像素采样验证）/ Liberation Sans（pdffonts 验证）/ 锁定色板 / 图例全小写无框 / panel 字母轴外左上 / 300dpi / ≤6.5×8in / 最小字号 ≥6.5pt / seed 三蓝阶统一（#0072B2/#56B4E9/#003C66）
- 重编译：27 页 / 0 错误 / 0 overfull（2026-08-22 12:22）
- changelog：docs/changes/2026-08-22-figure-iteration-round{1,2,3,4-main,4-supplementary,5-micropass,5-supplementary-micropass}.md

## 2026-08-24 图/表编号顺序修复 + 版式审计修复
- 重编号主图/主表以匹配正文首次引用顺序：Figure 1–7 = pipeline → layerwise → bio-validation → continuity metrics → external eval → domain boundary → cross-architecture spectrum；Table 1–5 = external → domains → recipes → convergence → gfbudget。
- 修复 p.14（原 Figure 7 / bio-validation）图片宽度 7.57in → 6.5in。
- 重新编译 pdflatex→bibtex→pdflatex×2，0 错误；pdf_layout_audit.json 的 order_issues 已清空。
- 审计脚本仍提示 low-coverage 页 14/22/23/24（占位符索引页/参考文献页为主，p.14 为浮动体排版导致的边缘低覆盖，可接受）。
