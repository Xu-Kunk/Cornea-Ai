---
title: "Highlights / eTOC / Bigger Picture — v26"
manuscript_version: v26
date: 2026-10-19
status: draft-for-author-review
supersedes: ../archive/docs/patterns_submission_extras.md
---

> **口径说明（不随件提交）**
> 本文件全部按 v26 结论重写。旧版（2026-08-12，v25 前）含三处与 v26 冲突的表述，已废止：
> ① “erases its differentiation gradient” → 语外仍有 0.641、表皮反向升高；
> ② “corpus- and architecture-dependent manner” → 标题已改为 “coexist with”；
> ③ “preserved the geometry across different model architectures” → 有序头在语外
>    0.350/0.539 低于 CE 0.641，CellPLM 仅单 seed。
> 字数按 Patterns 规范自检：Highlights ≤4×85 字符；eTOC ≤80 词；Bigger Picture ≤300 词。

---

## Highlights（4 条，每条 ≤85 字符，含空格）

1. Classification fine-tuning can keep labels while losing differentiation order
2. Gradient loss depends on corpus overlap, architecture, seed, and checkpoint
3. An ordinal head keeps the gradient in every in-corpus test
4. Out of corpus it falls below cross-entropy on a post-cutoff dataset

> 字符数自检：77 / 75 / 69 / 64。

---

## eTOC Blurb（≤80 词，第三人称）

Fine-tuning single-cell foundation models for cell-type classification can raise accuracy while weakening the continuous order of differentiation states. Across donor-disjoint corneal and epidermal tests, the change depends on corpus overlap, model, seed, and checkpoint. A cross-entropy head given the same four stage labels keeps only a quarter of the frozen ordering, whereas an ordinal head preserves it in every in-corpus test and falls below cross-entropy out of corpus. The study offers a benchmark for reading biological structure alongside annotation performance.

> 词数自检：77。

---

## Bigger Picture（≤300 词，非专家向，零术语零数字）

Cells in a tissue are not all in the same state. They move from stem cells that renew the tissue, through dividing intermediates, to mature cells. The order of that progression is part of what we know about a tissue: it tells us how repair works and where it can fail.

Most machine-learning models for cells are trained to put each cell into a named category. That is useful, but it throws away the in-between. A model can become very good at naming categories while its internal map stops preserving the order in which cells mature. The accuracy score will not show this, because accuracy only asks whether the label is right.

Our study asks a different question: after a model is adapted for labeling, can we still read the continuous progression from its internal representation? We measure this before and after adaptation, in corneal tissue and in skin, using several model families. We find that the answer is not the same everywhere. In one setting the progression nearly disappears while labeling improves. In another the change is small. In a third the progression becomes clearer than before. Where it is lost, changing how the task is framed — asking the model to respect the order of stages rather than treating them as unrelated categories — can keep the progression at similar labeling performance. But on a newer corneal dataset that advantage did not hold.

The practical point is simple. A good label score is not evidence that a model still represents biology. Evaluating what adaptation preserves, not only what it predicts, should become routine when these models are used to study development, regeneration, or disease.

> 词数自检：278（连字符复合词计一词；破折号不计）。

---

## 提交形态

| 件 | 要求 | 现状 |
|---|---|---|
| Highlights | ≤4 条 × ≤85 字符；与 eTOC 同一 Word 文档 | 4 条，77/75/69/64 ✅ |
| eTOC | ≤80 词，第三人称 | 77 词 ✅ |
| Bigger Picture | ≤300 词，非专家向 | 278 词 ✅ |
| Graphical Abstract | Patterns 可选 | 暂缓（投稿后视编辑要求） |
| 字体说明 | 图用 Liberation Sans（Arial 度量等价） | 已在 cover note 口径中准备 |

> 待确认：Patterns 首投是否强制 Bigger Picture。本地两份要求文档结论冲突
> （`reference/patterns_submission_requirements.md` 记“首次投稿不强制”，旧 extras 记“必需”）。
> 若编辑系统未要求，可只上传 Highlights + eTOC。
