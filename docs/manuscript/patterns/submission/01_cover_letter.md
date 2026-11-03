---
title: "Cover letter — Patterns (direct submission)"
manuscript_version: v26
date: 2026-10-19
status: draft-for-author-review
supersedes: ../archive/docs/cover_letter_patterns_2026-10-09.md
---

> **提交前必读（本段不随信提交）**
> 1. **仓库 public 仍待落实**：`https://github.com/Xu-Kunk/Cornea-Ai` 的项目选定路线是后续转为
>    public（用户已明确）。期刊首投要求“可访问”（官方 2021 指南：code location、匿名免费访问、
>    reviewer token 如需），并不强制 public。若上传时仍未 public，**不能**只写将来公开的承诺；
>    须同步改为真实可用的 Availability 声明，并提供编辑/审稿人可用的匿名访问链接或随稿代码包。
> 2. **基金信息（2026-10-22 更新，取代 2026-10-21 口径）**：作者确认恢复为两条资助——
>    National Natural Science Foundation of China (82571251) 与 Medical and Health Science Program of
>    Zhejiang Province (WKJ-ZJ-26067)。正文 Acknowledgments 已同步。
> 3. 本信不使用 “novel / first / unprecedented”；不重复摘要句式；一页以内。
> 4. 数字口径：GSE153515 = 14,775 epithelial cells / 2 donors（主 holdout）；
>    49 donors / 99,606 cells 是 adaptation-atlas 内部分区（Methods §1），
>    二者不可混写为同一协议。
> 5. 旧版（2026-10-09，旧标题 + v18 口径）已归档，勿再使用。

---

**To:** The Editors, *Patterns*
**Submission type:** Research article (direct submission)

Dear Editors,

We submit our manuscript, **“Annotation accuracy is not a proxy for differentiation order in fine-tuned single-cell foundation models,”** for consideration as a research article in *Patterns*.

**What was known.** Single-cell foundation models such as Geneformer, scGPT, and CellPLM are now routinely fine-tuned for cell-type annotation, and the field has measured what adaptation improves: label accuracy, transfer, and parameter efficiency. That literature has also documented label-level forgetting and has shown that frozen embeddings can underperform simple baselines on some tasks. What has not been measured is whether the continuous ordering of differentiation states survives adaptation while discrete classification remains accurate. Annotation metrics cannot answer this, because a model can separate cell types while losing the order in which those types arise.

**What we did.** We measured the differentiation gradient — the Spearman correlation between a representation-derived ordering and a prespecified stage order — before and after classification fine-tuning, in a paired, donor-disjoint design. We crossed three evaluation domains chosen to separate pretraining-corpus membership from tissue: a same-corpus corneal holdout (GSE153515; 14,775 epithelial cells, two donors), a post-cutoff out-of-corpus corneal test (GSE299152; 21,374 cells, three donors), and two epidermal controls.

**What we found.** In Geneformer, donor-level macro-F1 rose from 0.4421 to 0.6492 while the gradient fell from 0.8064 to −0.0082; local layer adjacency was essentially unchanged (0.9665 to 0.9768). The loss was therefore specific to continuous order, not to classification structure. Out of corpus the change was mild (0.707 to 0.641 ± 0.030), and in epidermis the direction reversed (0.239 to 0.455; 0.518 to 0.714). The depth of the change differed across architectures: Geneformer lost the gradient in upper layers, CellPLM at its entry layer (0.906 to 0.118), and scGPT-human ranged from collapse to retention across seeds and checkpoints.

**The intervention and its boundary.** An unconditional ordinal head preserved the Geneformer gradient (ρ 0.833) at macro-F1 0.649, whereas a cross-entropy head given the same four stage labels over the same cells reached only 0.254 — so the retention tracks the objective, not the label set. In scGPT-human, ordinal supervision with low-rank adaptation reached ρ 0.9241 ± 0.0006 and macro-F1 0.9659 ± 0.0024, against 0.4395 ± 0.1240 and 0.9311 ± 0.0109 for cross-entropy under a matched longer budget. We also report where this stops: the separate ordinal-head recipe fell below cross-entropy on the post-cutoff test (0.350 and 0.539 versus 0.641). Because six classical baselines already recover the axis at 0.840–0.865, we do not claim that foundation models are superior on this readout; we claim that the readout exposes something annotation metrics miss.

**Why *Patterns*.** The contribution is an evaluation methodology and a reusable benchmark rather than a domain application: the effect is measured directly in embedding space, with a preregistered gate, fixed-axis controls, permutation nulls, and false-discovery-rate accounting. The journal has published critical work on model selection and on the need for prospective validation, and this manuscript is built in that spirit, boundary results included.

**Availability and declarations.** All data are public. Analysis code and aggregated results are available at https://github.com/Xu-Kunk/Cornea-Ai. The manuscript is original, has not been posted as a preprint, and is not under consideration elsewhere. All authors approved the submission and declare no competing interests.

We would welcome your assessment of fit, and thank you for your consideration.

Sincerely,

Qi Dai, Ph.D. (lead contact)
The First Affiliated Hospital, Zhejiang University School of Medicine
1367 Wenyi West Road, Yuhang District, Hangzhou, Zhejiang Province 311121, PR China
dq@mail.eye.ac.cn

Corresponding authors: Hongguang Cui (1189002@zju.edu.cn), Peng Guo (guopeng@ucas.ac.cn), and Qi Dai (dq@mail.eye.ac.cn).

---

## 作者待确认

- [x] 基金信息：2026-10-22 更新为 NSFC 82571251 与 Zhejiang 省级 WKJ-ZJ-26067 两条资助（正文已同步）
- [x] 作者批准与通讯信息：全体作者同意投稿；Qi Dai 为 lead contact（按当前十一位作者名单）
- [ ] GitHub 仓库转为 public（项目选定路线；若未落实，须改为真实可用的可用性声明并提供编辑/审稿人访问链接或随稿代码包）
- [x] 通讯作者口径：Hongguang Cui、Peng Guo、Qi Dai 为共同通讯，Qi Dai 兼任 lead contact（已确认）
- [ ] 是否需要提及 presubmission inquiry 的回复（如已获积极回复，可在首段加一句）
