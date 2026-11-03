---
title: "Patterns submission checklist — Paper 1 v30d（2026-10-22 封版）"
manuscript_version: v30d（2026-10-22）
date: 2026-10-21
status: active
---

# Patterns 投稿清单 — Paper 1 v30d（2026-10-22）

> `submission/` 中 `03`–`06` 与 `MANIFEST.sha256` 由
> `scripts/assemble_patterns_submission.py` **生成**，禁止手改。
> `00`、`01`、`02`、`07`、`08` 为手写件。
> 校验：`python3 scripts/assemble_patterns_submission.py --check`（当前 exit 0）。

---

## 1. 交付物状态

| # | 件 | 文件 | 规格 | 状态 |
|---|---|---|---|---|
| 00 | 本清单 | `00_CHECKLIST.md` | — | ✅ |
| 01 | Cover letter | `01_cover_letter.md` | ≤1 页 | 🟡 待仓库 public |
| 02 | Highlights / eTOC / Bigger Picture | `02_highlights_etoc_bigpicture.md` | Highlights 4×≤85 字符 / eTOC ≤50 词（终稿）/ Bigger Picture ≤300 词 | ✅ 首投非必需；当前 eTOC 77 词，终稿时压缩 |
| 03 | 主稿 | `03_manuscript.pdf` | **29 页** A4，`39599233…`（v31） | ✅ |
| 04 | 补充材料 | `04_supplementary.pdf` | **30 页** A4，`86d37073…`（v30v，Figure S1–S13 + Table S1–S12 + SM1/SM2） | ✅ |
| 05 | 主图 | `05_figures_main/` | Figure 1–6，PDF + 600 dpi PNG | ✅ 12 文件 |
| 06 | 补充图 | `06_figures_supp/` | Figure **S1–S13**，PDF + 600 dpi PNG | ✅ 26 文件 |
| 07 | 声明 | `07_declarations.md` | CRediT / COI / AI / I&D / Funding | 🟢 草稿就绪；I&D 属 accepted-in-principle 后自愿项 |
| 08 | 数据与代码可用性 | `08_data_code_availability.md` | Lead contact / materials / data / code | 🟢 声明就绪；仓库 public 与 DOI 按阶段落实 |
| 09 | Cell Press Declaration of Interests | `09_declaration_of_interests_draft.pdf` | 官方 3 页 AcroForm 原版式 | 🟡 草稿；final checkbox 留空待作者 |
| — | Graphical abstract | — | 1200×1200 px @300 dpi | ⏸ 暂缓（Patterns 可选） |
| — | Data reporting checklist | — | Cell Press 仅 battery / photovoltaic / catalysis 类目 | ➖ 本稿无适用通用表 |

> `09` 为作者提供类额外文件，**不计入** assembler 机器生成的 42 件，不写入 `MANIFEST.sha256`。

---

## 2. 首投就绪度（按阶段区分，非“任一未清即不可投”）

官方依据（2026-10-20 读取；详见 `docs/changes/2026-10-20-patterns-coi-draft-and-readiness.md`）：
journal-policies `https://www.cell.com/patterns/information-for-authors/journal-policies`；
article-types `https://www.cell.com/patterns/information-for-authors/article-types`；
submit-manuscript `https://www.cell.com/patterns/information-for-authors/submit-manuscript`；
final-submission `https://www.cell.com/patterns/information-for-authors/final-submission`；
inclusion & diversity FAQ `https://www.cell.com/inclusion-diversity-statement-faqs`。

| # | 事项 | 期刊阶段（规则） | 本稿状态（ready） |
|---|---|---|---|
| R1 | Cell Press Declaration of Interests 官方表 | **首投必交** | 🟡 草稿已备 `09_declaration_of_interests_draft.pdf`；无冲突选项已勾，final “On behalf of all authors” 留空待作者 |
| R2 | 数据与代码可用性三段式声明（准确） | **首投必须存在** | 🟢 结构已备（主稿与 `08`）；public 表述随 R3 落实 |
| R3 | 代码可访问 / public / DOI | 首投要求**审稿访问**（官方措辞 “should provide … access”，非 must；提供匿名免费访问路径） | 🟡 项目选定路线：仓库后续转 public（用户已明确）；若不 public 须给编辑/审稿人访问链接或随稿代码包；acceptance 前落实 DOI 或 SI |
| R4 | 基金编号 | 首投无强制 | ✅ 两条资助（NSFC 82571251；Zhejiang WKJ-ZJ-26067），正文已同步；COI 表 Funding 字段待作者填 |
| R5 | 作者目检 PDF | 内部 QA，**非**期刊 gate | ⚪ 未单独记录（不因作者整体批准而记为已目检） |
| R6 | Bigger Picture | 首投不要求（终稿必需，≤300 词） | ✅ 已就绪（`02`） |
| R7 | Standardized data reporting checklist | 官方仅列 battery / photovoltaic / catalysis；无 scRNA 通用表 | ➖ 无适用表；按 Editorial Manager 实际要求提交，不套用其它类目表单 |
| R8 | Inclusion & Diversity | **accepted in principle 之后**、通讯作者表单、自愿 | ⚪ 不作首投项；不索取敏感身份信息；Patterns 是否纳入该计划官方未明示 |
| R9 | Graphical abstract | Patterns 可选 | ⏸ 暂缓 |

> 与旧 G1–G4 的对应：G1→R3、G2→R4（已 clear）、G3→R3（acceptance 阶段）、G4→R5（内部 QA）。
> **规则允许 ≠ 本稿 ready**：R1 的 final 勾选、R3 的仓库可访问（及后续 DOI）仍需落实。
> 作者已确认：全体作者同意当前稿投稿、署名与贡献、无利益冲突。资助信息见 §6（2026-10-21 更新为两条含编号的基金）。

---

## 3. 阻断项

### 已修正（2026-10-19，授权后执行）

| # | 问题 | 处理 | 验证 |
|---|---|---|---|
| B1 | 主稿 PDF 第 1 页印有 “Review copy --- 2026-10-17.”（`paper1.tex:61`） | 删除该行 | `pdftotext` 首页 “review copy” 命中 0 |
| B2 | `paper1.tex:91` TAC 敏感性指向 Figure S8（实为 S10） | 改为 Figure S10 | 正文出现 `Figure S10`；S8 另指 internal/external inversion |

两项均为行级改动，无科学内容变更。重编译结果：**35 页 / 0 error / 0 undefined / 0 overfull**
（11 underfull，与改动前一致）。`CHECKSUMS.sha256` 已刷新，`submission/` 已按新 PDF 重建：

- `latex/paper1.tex` → `efd23d11…`
- `latex/build/paper1.pdf` → `c3d1b0fd…`
- `submission/03_manuscript.pdf` → `c3d1b0fd…`（与 build 逐字节一致）
- `submission/04_supplementary.pdf` → `ce869b72…`（与 build 逐字节一致）
- `submission/MANIFEST.sha256` → `e09c7a12…`

### B3. CellPLM frozen “三值不一致” —— 已由后续记录纠正（历史文档错误，非现有阻断）

本清单早前登记“主稿写 0.865，但项目记录三值为 **0.8652 / 0.8026 / 0.8600**”。
后续记录核实该“三值”本身是**文档错误**：0.8026 实为
`artifacts/paper1_vanzyl/cornea2/GSE186433/naive_seed42/args.json` 的 `val_f1`；0.8600 未检出；
0.8652 在 10+ 处独立分析中复现。`findings.md` 同时记录主稿 `paper1.tex:119` 的 0.865 为
seed-42 reference。此处仅记录该次核实，不对 frozen 运行的变动作一般性判断。

- **结论**：本项**已由后续记录纠正，不作为现有阻断**或当前未解科学风险保留。
- 不再就该“三值”延伸任何数值分析；如需改动措辞，属作者编辑决定，与本清单无关。

### B4. 其他已登记 holds（不影响装配，但影响审稿）

| 项 | 状态 |
|---|---|
| 概率解码 / ECE 口径 | 未测 threshold-count 与修复概率 argmax 的分歧率 |
| bib 重复 DOI | `ding2025scgptretina` 与 `ding2026protocol` 共用同一 DOI |
| FDR 家族范围 | 21 项冻结家族 + 6 项 freeze-time pending；后续完成的 run 不在家族内 |
| Figure 8/9 legacy 编号 | v26 已改为 6 主图；旧 “Figure 8 B–D 渲染器绑定” 需重新映射确认 |

---

## 4. 本轮已修正的不一致

| # | 问题 | 处理 |
|---|---|---|
| 1 | `PATTERNS_ACTIVE_PACKAGE.md` 哈希过期（tex `61556542…`、PDF `f4ee5359…`） | 按 `CHECKSUMS.sha256` 重写（随后 B1/B2 重编译再次更新，见 §3） |
| 2 | `patterns_submission_extras.md` 三处与 v26 冲突（“erases”、“corpus- and architecture-dependent”、“preserved across architectures”） | 按 v26 重写 → `02_highlights_etoc_bigpicture.md`；旧件归档 |
| 3 | Cover letter 旧标题（“Fine-tuning collapses…determined”） | 按 v26 标题重写 → `01_cover_letter.md`；旧件归档 |
| 4 | 49 donors / 99,606 cells 与 14,775 cells / 2 donors 混写 | 新信明确分开：atlas 分区 vs GSE153515 holdout |
| 5 | 两份要求文档并存 | 旧 `submission_requirements.md` 归档；保留 `reference/patterns_submission_requirements.md` |
| 6 | 投稿件散落 5 处，无投递目录 | 新建 `submission/`，脚本装配 + MANIFEST |
| 7 | `artifacts/paper1_preprint/SUBMISSION_PACKAGE.md` 被误当清单 | 顶部加 SUPERSEDED 横幅，指向本文件 |
| 8 | Inclusion & Diversity 缺失 | 新增于 `07_declarations.md`（含待作者确认段） |

---

## 5. 重建与校验

```bash
cd /home/dq/Codex/Ai

# 1) 主稿（若改了 tex/bib）
cd docs/manuscript/patterns/latex
pdflatex -output-directory=build paper1.tex
( cd build && BSTINPUTS=.. BIBINPUTS=.. bibtex paper1 )
pdflatex -output-directory=build paper1.tex
pdflatex -output-directory=build paper1.tex

# 2) 刷新包哈希
cd .. && sha256sum latex/paper1.tex latex/paper1.bib latex/build/paper1.pdf \
  supplementary/supplementary_information.tex \
  supplementary/build/supplementary_information.pdf \
  paper1_supplementary_information.pdf \
  ../../../figures/paper1/figure_manifest.json > CHECKSUMS.sha256

# 3) 重新装配投稿包
cd /home/dq/Codex/Ai
python3 scripts/assemble_patterns_submission.py
python3 scripts/assemble_patterns_submission.py --check
```

> `paper1_supplementary_information.pdf` 必须与 `supplementary/build/…pdf` 逐字节一致。
> 任一正文改动都会使 `03_manuscript.pdf` 与 `MANIFEST.sha256` 失效，必须重跑第 3 步。

---

## 6. 待作者确认

- [x] **B1**：删除 “Review copy --- 2026-10-17.” 并重编译（2026-10-19 完成）
- [x] **B2**：`:91` 的 Figure S8 → Figure S10（2026-10-19 完成）
- [x] **B3**：CellPLM frozen “三值”经核实为文档错误，已由后续记录纠正（见 §3）
- [x] **G2**：基金信息 — 2026-10-22 更新为 NSFC 82571251 与 Zhejiang WKJ-ZJ-26067 两条（含编号），正文已同步
- [x] **Bigger Picture**：官方核实首投非必需（终稿必需），`02` 已就绪
- [x] 作者全体同意：当前稿投稿、署名与贡献、无利益冲突
- [ ] **09 COI 表 final 勾选**：`09_declaration_of_interests_draft.pdf` 的 “On behalf of all authors”
      须由作者本人勾选（本会话不代勾/代签）
- [ ] 仓库 `Xu-Kunk/Cornea-Ai` 转 public（项目选定路线，上传前落实；否则改 Availability 句）
- [ ] Bigger Picture 首投是否随附（非必需，可自愿附上）
- [ ] 通讯作者签名口径：Qi Dai 为 lead contact；Hongguang Cui、Peng Guo 与 Qi Dai 共同通讯
- [ ] 是否在 cover letter 首段提及 presubmission inquiry 的回复
- [ ] 07 的 Inclusion & Diversity 第 2 段（自愿项；作者若不填则保留数据层面表述）

---

## 7. v28 post-check corrections（2026-10-20 核对修订）

> **历史块**：§7/§7.1 记录 2026-10-20 的两次构建；其“当前哈希”已被 **§9（2026-10-21）** 取代。

本块为**当前**状态，覆盖 §1 与 §3 中的 2026-10-19 历史哈希值（后者保留为历史记录，不再代表当前包）。本轮无新实验；除下列正文/图注改动外未改文案。

**本轮改动**

- `latex/paper1.tex:79`（Introduction 小结）：把同一语料与 separate Geneformer 序数头两件事分句，明确 separate GF ordinal-head recipe 在 post-cutoff 测试上失去优势。
- `latex/paper1.tex:163`（Figure 1 图注）：(C) 覆盖范围随模型与域变化；(D) 中间 readout 明确为 layer adjacency accuracy (LAA)；(E) 为在 BH–FDR 家族内的正关联检验，明确**不是**臂间差异检验。
- `submission/02_highlights_etoc_bigpicture.md`：Bigger Picture 末句改为新角膜数据（Bigger Picture 278 词）。
- `figures/paper1/figure1_pipeline_overview.{pdf,png}`、`design/figure1_final.{fodp,pptx}` 与 `figure_manifest.json` 的 `figure1` 条目重跑更新（producer `scripts/figure1_layout_pptx.py`）；其余 19 图与 manifest 其余条目逐字节未变。

**编译**（`pdflatex → bibtex → pdflatex ×2`，日志 `/tmp/opencode/paper1-postcheck-build-2026-10-20/`）

- 主稿：**35 页 / 0 error / 0 undefined / 0 overfull**（11 underfull，未变）。
- SI：未改动、未重编译，`ce869b72…` 维持不变。

**当前哈希（2026-10-20，第一轮 post-check）** —— 见下方 conclusion pass，已非当前

| 文件 | SHA-256（abridged） |
|---|---|
| `latex/paper1.tex` | `594013ad…` |
| `latex/build/paper1.pdf` | `198ea832…` |
| `submission/03_manuscript.pdf` | `198ea832…`（与 build 逐字节一致） |
| `submission/04_supplementary.pdf` | `ce869b72…`（与 SI build 逐字节一致） |
| `submission/MANIFEST.sha256` | `f7cde83f…` |
| `latex/paper1.bib` | `912b9c27…`（未变） |
| `supplementary/supplementary_information.tex` | `45f4f2fd…`（未变） |
| `paper1_supplementary_information.pdf` | `ce869b72…`（未变） |
| `figures/paper1/figure1_pipeline_overview.pdf` / `.png` | `62aa3bba…` / `54cfab78…` |
| `figures/paper1/figure_manifest.json` | `f4ebe5e1…` |

**装配校验**：`python3 scripts/assemble_patterns_submission.py` 与 `--check` 均 exit 0；`sha256sum -c MANIFEST.sha256` 42/42 通过；6 主图 + 14 补图共 40 图文件，图 1 两份更新、其余 38 份 sha 不变；手写 `01`/`02`/`07`/`08` 与构建前逐字节一致。硬门 G1–G4 及其余 holds **仍未清**，不因本轮装配而解除。

---

### 7.1 Conclusion pass — Discussion 末段两句（2026-10-20）

Discussion 收尾段（`latex/paper1.tex:267`）只改两句，**Summary 及所有其它 source 未改**：

- 旧 “Stage-ordered supervision was the cheapest intervention that worked in our same-corpus tests.” → 新 “Stage-ordered supervision retained the gradient in our same-corpus tests.”
- 旧 “However, it did not transfer out of corpus, so it repairs a known corpus rather than safeguarding an unseen one.” → 新 “However, the separate Geneformer ordinal-head recipe lost its advantage on the post-cutoff test; the combined ordinal+LoRA recipe has not been evaluated out of corpus.”

仅此两句变化：逆替换可逐字节复原上一轮 `594013ad…` 的 tex，故 Summary 与其它 source 未改。同流程重编译（日志 `/tmp/opencode/paper1-conclusion-build-2026-10-20/`）：**35 页 / 0 error / 0 undefined / 0 overfull**（11 underfull 未变）；内容变化仅落在 **PDF 第 15 页**，无浮动分页变化。

**当前哈希（2026-10-20 conclusion pass）**

| 文件 | SHA-256（abridged） |
|---|---|
| `latex/paper1.tex` | `744dd81f…` |
| `latex/build/paper1.pdf` | `adc1b703…` |
| `submission/03_manuscript.pdf` | `adc1b703…`（与 build 逐字节一致） |
| `submission/MANIFEST.sha256` | `6f739ed7…` |
| `latex/paper1.bib` | `912b9c27…`（未变） |
| `supplementary/supplementary_information.tex` | `45f4f2fd…`（未变） |
| `submission/04_supplementary.pdf` / `paper1_supplementary_information.pdf` | `ce869b72…`（未变） |
| `figures/paper1/figure1_pipeline_overview.pdf` / `.png` | `62aa3bba…` / `54cfab78…`（未变） |
| `figures/paper1/figure_manifest.json` | `f4ebe5e1…`（未变） |

§7 上方第一轮表（`594013ad…` / `198ea832…` / `f7cde83f…`）为已覆盖的中间快照。装配复核：`assemble_patterns_submission.py` 与 `--check` exit 0；`sha256sum -c MANIFEST.sha256` 42/42；`03` 与 build、`04`/SI copy 与 SI build 均逐字节一致；20 图源未变。（首投就绪度改见 §2 的分阶段表；R1/R3 等待落实。）

---

## 8. Cell Press COI 官方表草稿（2026-10-20）

官方空白模板（原始 hash 与出处见 `docs/changes/2026-10-20-patterns-coi-draft-and-readiness.md`）：

- URL（经官方 submit 页 `## Other files` 的 live href 取得）：
  `https://www.cell.com/pb/assets/raw/shared/forms/di_form-1713192433827.pdf`
- 原始模板 SHA-256：`a14d543fb2f153e55e5da9d691d18f518a02c65afd37cdba9de16abc53a14351`（3 页，未改版式）

草稿：`09_declaration_of_interests_draft.pdf`（直接填写官方 AcroForm 的 11 个字段，保留 3 页原版式；未重造表、未加水印、未签名）。

| 节 | 勾选 | 字段数 |
|---|---|---|
| Institutional affiliations | ✅ “All affiliations are listed…”（对立项 Off） | 1 On / 1 Off |
| Funding sources | ✅ “All funding sources…” | 1 On |
| Competing financial interests | ✅ “no financial interests”（对立项 Off） | 1 On / 1 Off |
| Advisory/management positions | ✅ “no positions”（对立项 Off） | 1 On / 1 Off |
| Patents | ✅ “no related patent applications”（对立项 Off） | 1 On / 1 Off |
| 文本框 | “The authors declare no competing interests.” | 1 Tx |
| Final attestation | ☐ **Off（留待作者勾选）** | 1 Off |

合计 10 复选框（5 On / 5 Off）+ 1 文本框。无签名/日期字段，故未伪造姓名或日期。

---

## 9. 2026-10-21 构建收口（SI layout + guidance corrections）

两条已授权 source lane 合并构建：SI 版式（designer：`supplementary_information.tex`）与主稿图注/指引（guidance：`latex/paper1.tex`）。无原始数据/训练/图件重出，无科学数值改动。

**本轮修**

- SI：自动 S 编号（不再手写重复 “Figure Sx.”）；178 mm 文本块使 178 mm 图源 **1:1** 印刷；Table S2/S4 列重排（约 10 pt）；Table S5 标题 `needspace`；补一个 DOI 的 “j.”；Related anchors 与主稿 SI index 同步；Table S11 明确 logit 口径；S12 caption 机械校正后在本轮重新编译（左：DPT vs frozen stage-centroid reference；右：三 seed 均线 + per-decile min–max band）。
- 主稿：Fig6 A caption 与图面一致 —— GF official-budget 行绘 **三** seed 42/123/456（mean 0.096；六 seed 0.058 仍在正文与 Table S8），五色键（frozen 深灰；CE-naive 蓝且所有 CE 行同蓝；ordinal head 绿；LoRA 粉紫；ordinal+LoRA 橙红）+ 点/黑均值 tick/ρ=0 点虚线语义，逐项表指针（S8；S6–S7；S9；S11）；Fig3C 删 “Wang 2020” 未核实归属（保留 GSE147424 与 18,355 cells）；SI index：S4→Table 1、S9→Figure 6、S10→Table 3。

**当前哈希与页数（2026-10-21）**

| 文件 | SHA-256（abridged） |
|---|---|
| `latex/paper1.tex` | `99c6b236…` |
| `latex/build/paper1.pdf` / `submission/03_manuscript.pdf` | `7d55fbab…`（35 页，与 build 逐字节一致） |
| `supplementary/supplementary_information.tex` | `15208dab…` |
| `supplementary/build/supplementary_information.pdf` / `paper1_supplementary_information.pdf` / `submission/04_supplementary.pdf` | `99bb6412…`（27 页，三者逐字节一致） |
| `submission/MANIFEST.sha256` | `dc74942d…` |
| `latex/paper1.bib` | `912b9c27…`（未变） |
| `figures/paper1/figure_manifest.json` | `f4ebe5e1…`（未变；40 图文件亦未变） |

- 编译：主稿 `pdflatex → bibtex → pdflatex ×2` → **35 页 / 0 error / 0 undefined / 0 overfull**（11 underfull）；SI 纯 `pdflatex` 三遍（aux 自第二遍稳定）→ **27 页 / 0 error / 0 undefined / 0 overfull**（69 underfull；该计数取自重编译前读到的 designer 试编译日志，未对更早的窄版式作比较断言）。
- 主稿内容 diff 相对 2026-10-20 版：仅第 10、13、28 页（三处编辑处）；页数不变，无浮动重排。
- 装配：`assemble_patterns_submission.py` 与 `--check` exit 0；`sha256sum -c MANIFEST.sha256` 42/42；40 图拷贝逐字节一致；手写 `01`/`02`/`07`/`08` 未变；`09` 草稿在机器 42 件之外、final attestation 仍为 Off。
- CHECKSUMS header 保持 **v28** 并加 “SI layout and guidance post-check 2026-10-21”，未升 v29。
- 2026-10-21（后段）**S2/S4 wording pass（作者确认）**：仅 4 处文字替换 —— S2 row 2 → “Classification safeguard / Historical CI-overlap screen (descriptive) / PASS (screen)”；S2 note → observed macro-F1 接近 naive、CI overlap 仅描述，不构成 equivalence 或 non-inferiority；S4 scFoundation → “Unknown (pretraining corpus not publicly enumerated)”；UCE → “Inferred out-of-corpus (census 2023-04-25)”。数字未动：774 个测量值 token 多重集一致（仅 “Macro-F1” 的 F1 与 “95% CIs” 的 95 随改写消失）；SI 重编 **27 页 / 0-0-0 / 69 underfull / 26 headheight（非阻断）**；aux 未变，26 条目录 pageref 不变；内容 diff 仅 SI p17–18。未做正式 NI 检验、未新增 corpus 审计。

> 状态：**built; authorized layout/guidance visual verification PASS**（针对 `7eb9469f…`；未变页可复用）+ **built; S2/S4 wording-pass visual verification PASS**（覆盖 SI p16–19 全页 150 dpi、S2/S4 放大核对；邻页 16/19 与上版像素 diff 0，无新增断页/表头孤悬；研究数据无变，未新增 NI 或 corpus 审计）。observer 已对最终 main `7d55fbab…` p10/p13/p28 与 SI `7eb9469f…` **全部 27/27 页**逐页 150 dpi 复核并判 PASS；主会话独立核过 7 source + 42 包 hash、SI labels 与 PDF copy。
> PASS 范围**仅限本轮 SI 排版与指引图注**：不证明统计结论已全部核实，也不保证 Patterns 接收。
> 记录、不再微调：SI p19/p24 note 尾页留白偏大；p23 “Notes.” 标题与说明跨页；个别长路径词间空隙偏大。`\headheight` warning 非阻断（实测 header/body 最小间距 13.8 pt、无重叠），不声称其消失。
> 原两项科学待讨论**已按作者确认收紧措辞**（非新证据）：S2 CI overlap 现标 “descriptive / PASS (screen)”，不构成 equivalence 或 non-inferiority；S4 scFoundation 标 Unknown、UCE 标 Inferred out-of-corpus。**未**做正式 NI 检验、**未**新增 corpus 审计。

---

## 10. v29.1 快照（2026-10-21，投稿准备修订）

> 取代 §1 的 v28 哈希与页数。§7–§9 为历史块，保留溯源。

**本轮改动**

- **统计补齐**：Results §1 增加同细胞配对、复现内重拟合轴的 Δρ = 0.815（95% CI 0.737–0.898，0/2000 ≤ 0）；LAA 改为幅度表述 +0.010（95% CI 0.007–0.013），对照置换零分布 0.642（间距 0.316）。Methods §14 增配对 bootstrap 与置换零分布说明。来源 `results/paper1_vanzyl/claim_audit/delta_rho_laa.json`（门禁复现最大偏差 3.9e-05）。
- **措辞**：Summary 重写（145 词，gap 开头 + 配对数字）；删 Results §1 与旧句重复的 LAA 句；Discussion ¶1 "weakened substantially" → "fell to near zero"；三处"未检验差异"免责合并为 Methods 一句 + 两处指针。
- **合规**：AI 声明标题改 Elsevier 标准 "in the writing process"，正文 48 词（DeepSeek-V4 系列，本地部署，润色与代码；无 AI 生成数据/结果/图像）；Methods §14 增对应一句；Resource Availability 补官方固定收尾句；Code 条目不再断言已 public；Qi Dai 加 Lead contact 脚注；贡献改缩写。
- **基金**：NSFC (82571251) 单条；Zhejiang 省级项目按作者指示删除。`07_declarations.md`、`01_cover_letter.md` 同步。

**当前哈希**

| 文件 | SHA-256（abridged） |
|---|---|
| `latex/paper1.tex` | `92281f76…` |
| `latex/build/paper1.pdf` / `submission/03_manuscript.pdf` | `c59f4af6…`（逐字节一致，36 页） |
| `supplementary/build/supplementary_information.pdf` / `04_supplementary.pdf` | `99bb6412…`（27 页，未改） |
| `latex/paper1.bib` | `912b9c27…`（未改） |
| `figures/paper1/figure_manifest.json` | `f4ebe5e1…`（未改） |
| `submission/MANIFEST.sha256` | `e4315fde…` |

**校验**：`assemble_patterns_submission.py --check` exit 0；`MANIFEST.sha256` 42/42；`CHECKSUMS.sha256` 7/7。

**仍开放**：仓库 public + DOI；`09_declaration_of_interests_draft.pdf` 的 Funding 字段与 "On behalf of all authors" 勾选（作者本人）；Bigger Picture 是否随附；外部评审提出但未处理项见 `docs/changes/2026-10-21-paper1-external-review-triage.md`。

---

## 11. v30d 封版快照（2026-10-22）

取代 §10 的 v29.1 数值。本轮按外部评审意见完成"故事线一致性"改造。

**内容变化**
- 删除一切失败/未生效方法材料（CAFT-intended、axis-anchored、soft-Spearman、flow、GF-LoRA OOM、basin 判据），Methods 小节重编 1–13、臂列表编号 1–7；
- 新增两个 4 类上皮 CE 对照臂（Arm 6：27,878 上皮细胞、2,616 步；Arm 7：保留 79,684 行但仅上皮行计损、7,473 步与序数臂同预算），Table 1 增两行，SI Table S2 改为 "Label-set and step-budget controls"（逐种子 + mean±SD）；
- 补齐配对统计：Δρ(frozen−naive)=0.815 [0.692, 0.908]（pooled，与 Methods 口径统一）、Δρ(epi4−ordinal)=−0.579 [−0.631, −0.541]、Δρ(frozen−mask)=+0.552 [0.514, 0.604]；`paired_drho_*.json` 内 `pooled` 字段已注明算法（三种子 × 2,000 复现拼接后取百分位）；
- 摘要换为 V1（130 词，引 0.254）；LAA 标为 low-sensitivity companion metric（+0.010 [0.007, 0.013]，为零分布间距的 3%）；术语首现定义（CE/LAA/ECE/PCS/DPT/三域）；Limitations 改四条；
- CellPLM LoRA-only 三种子如实报告 ρ 0.551 ± 0.365（0.154–0.874，epochs 184/366/80）；
- 图件：Figure 3 与 9 张补充图重生成（去 CAFT 系列、修正 Wang 2020 → GSE147424）；S5 改为 naive vs 序数头训练轨迹；S12 面板字母移入可见区；SI 七条图注与 S1–S3 表行清理。

**当前哈希**

| 文件 | SHA-256（abridged） |
|---|---|
| `latex/paper1.tex` | `788f1ded…` |
| `latex/build/paper1.pdf` / `submission/03_manuscript.pdf` | `db69e72d…`（逐字节一致，34 页） |
| `supplementary/supplementary_information.tex` | `aa5b1975…` |
| `supplementary/build/supplementary_information.pdf` / `paper1_supplementary_information.pdf` / `04_supplementary.pdf` | `639ac88b…`（逐字节一致，26 页） |
| `figures/paper1/figure_manifest.json` | `280759b9…` |
| `CHECKSUMS.sha256`（v30d） | 7/7 通过 |
| `submission/MANIFEST.sha256` | `--check` OK |

**零命中校验**：20 张图 + SI 的 `CAFT|Axis-anchored|Soft-Spearman|gate-refused|AA-FT|flow consistency` 全为 0；主稿 PDF 唯一命中为真实模块路径 `corneal_fm/caft.py`（代码可用性清单，保留）。

**仍开放**：仓库 public + DOI；COI 表 Funding 字段与 "On behalf of all authors" 勾选（作者本人）；Bigger Picture 是否随附；主图顺序重排（评审建议 F4→F2→F5→F6…，尚未执行，详见 v30 工作单 §E）。

---

## 12. v30f — 作者名单与资助（2026-10-22，取代 §11 的相关行；作者/资助口径由 §15 v30k 更新）

**作者（11 位，三通讯）**：Kunhui Xu¹˒² · Chenchen Wang¹ · Fenfen Li¹ · Xiaoyu Chen² · Xinyu Wang² · Yi Zhang² · Tiancheng Pu² · Jiahao Tang² · Hongguang Cui¹˒\* · Peng Guo²˒\* · Qi Dai¹˒²˒\*˒†（lead contact）

**单位**：¹ The First Affiliated Hospital, Zhejiang University School of Medicine, Hangzhou 311121；² Eye Research Center, Hangzhou Institute of Medicine, Chinese Academy of Sciences, Eye Hospital, Wenzhou Medical University, Hangzhou 310018。原"National Clinical Research Center for Ocular Diseases, Eye Hospital, WMU, Wenzhou"已整条替换。

**通讯邮箱**（正文与 SI 封面一致，仅此三个）：1189002@zju.edu.cn · guopeng@ucas.ac.cn · dq@mail.eye.ac.cn

**资助（两条）**：National Natural Science Foundation of China (82571251) 与 Medical and Health Science Program of Zhejiang Province (WKJ-ZJ-26067)。

**当前哈希**：`latex/paper1.tex` `ef07a568…`；`latex/build/paper1.pdf` = `03` = `1d559137…`（34 页）；SI tex `840e5d95…`；SI PDF（build/同步/`04`）`2cbc81d8…`（26 页）；`CHECKSUMS` v30f 7/7；`MANIFEST` `7d1b9b57…`，`--check` OK。

**须作者本人更新**：`09_declaration_of_interests_draft.pdf` 系按旧名单（4 位、温州单位）勾选，机构、资助与最终勾选均需重填。

---

## 13. v30h — 图注压缩、账目语清理（2026-10-22，取代 §12 的哈希行）

**图注词数（语料中位 136、每面板约 33 词）**

| 件 | 原 | 现 |
|---|---|---|
| Fig 1 pipeline | 155 | **119** |
| Fig 2 layerwise | 101 | **96** |
| Fig 3 readouts | 187 | **119** |
| Fig 4 external eval | 123 | **110** |
| Fig 5 domain | 125 | **111** |
| Fig 6 spectrum | 297 | **121** |
| Table 1 / 2 / 3 | 51 / 82 / 58 | 48 / **61** / 57 |

内部预算 ≤120（图）/ ≤100（表），**不写入稿件**。压缩顺序：防御语 → 重复区间定义 → 长表指针串 → 面板细节；编码键与面板字母全程保留。

**账目语清理**：Methods 6 句（去 archived/recorded 形容词、删 "retained for traceability" ×2、删环境锁说明从句、`were retained with` → `are reported with`、删 "record of record" 尾句），另删 1 句孤立的元评论（`Artifact names containing historical method abbreviations…`——因 `corn_lora` 已不出现在正文，该句失去指代对象）。全文防御语仅 4 处，全部保留（范围限定，非防御）。

**已知取舍（已接受）**：Fig 6 去掉了行语义、0.8652 与 0.058 两个重述数字、四处表指针并为一句；Fig 3 去掉了 resample 计数（Methods 有）、面板 B 的 checkpoint 身份、z-score 定义（Methods 有）。

**当前哈希**：`latex/paper1.tex` `d9653514…`；`latex/build/paper1.pdf` = `03_manuscript.pdf` `13de5fde…`（34 页）；SI 未改（`840e5d95…` / `2cbc81d8…`，26 页）；`CHECKSUMS` v30h 7/7；`assemble --check` OK。

---

## 14. v30i — Figure 1 重绘（2026-10-22，取代 §13 的哈希行）

Figure 1 由 `design/figure1_final.pptx` **手工维护**（python-pptx 编辑），导出 PDF/PNG；`scripts/figure1_layout_pptx.py` 已标记为"历史生成器，勿重跑"，`figure1_final_layout_spec_2026-10-19.md` 与 `.fodp` 退为历史记录。manifest 的权威输入改为 PPTX。

**改动**：删除 E 面板（其观测点落在零分布包络内，读作"不显著"，与图注矛盾）；A–D 重新分配空间；C 补卡片边框；D 补表头带；全图统一 7.0 mm 栏距（原为 19.9/8.5/22.7 pt 三个值且竖向错位 14.2 pt）；D 分栏线 0.51→0.8 pt、1.41:1→3.36:1；D 第三栏 10 轮廓形 → 10 实心形（原 9）且形间距 0.85→2.83 pt；C 三色 hatch（0.1 pt）→ 统一实心浅色 + 0.6 pt 深边（7:1）；B 中画错颜色的粗横条删除并改为 `HVG/PCA` 文字；占位语 `as specified`、`evaluated configurations`、`not all recipes on all models` 全部清除；最小字号 7.0→7.5 pt；A→B、B→C 箭头重新锚定并新增 B/C→D 竖箭头。

**图注**：删 (E) 句，119 → **93 词**。

**当前哈希**：`latex/paper1.tex` 见 `CHECKSUMS.sha256`；`latex/build/paper1.pdf` = `03_manuscript.pdf`（34 页）；Fig 1 PDF `2efd5e74…`、PNG `e4d67f0e…`、PPTX `e862ad53…`；SI 未改；`CHECKSUMS` v30i 7/7；`assemble --check` OK。

---

## 15. v30k — 第十一位作者与第二条资助（2026-10-22）

作者/资助口径更新；无科学文本、数值、表或图改动。

**作者（11 位，三通讯）**：Xiaoyu Chen（ORCID 0000-0002-9980-5524，单位 2）插在 Fenfen Li 之后；顺序为
Kunhui Xu · Chenchen Wang · Fenfen Li · Xiaoyu Chen · Xinyu Wang · Yi Zhang · Tiancheng Pu · Jiahao Tang · Hongguang Cui · Peng Guo · Qi Dai。Hongguang Cui、Peng Guo、Qi Dai 仍为共同通讯（仅此三个邮箱出现），Qi Dai 仍为 lead contact。贡献节新增 `X.C.: investigation, data curation.`（在 F.L. 之后），其余角色未变。主稿首页与 SI 封面同序、同十一人。

**资助（两条）**：`This work was supported by the National Natural Science Foundation of China (82571251) and the Medical and Health Science Program of Zhejiang Province (WKJ-ZJ-26067).` 已同步 `07_declarations.md`、`01_cover_letter.md`、本表 R4/§12、COI 填表指引。

**当前哈希**：`latex/paper1.tex` `ce0dcb36…`；`latex/build/paper1.pdf` = `03_manuscript.pdf` `e2e3fa07…`（34 页，0/0/0，14 underfull）；SI tex `c3c8f0d7…`，SI PDF（build = 同步件 = `04`）`60f6ebc8…`（26 页，0/0/0/63 underfull）；`figure_manifest.json` 未改；`CHECKSUMS` **v30k** 7/7；`MANIFEST.sha256` `b06604d4…`，42/42；`assemble --check` OK。

**历史块说明**：§10（v29.1）仍保留其"NSFC 单条"的历史快照文字，属旧版快照，非当前口径。

---

## 16. v30l — 作者增至 11 位、资助两条、Fig 1 收尾（2026-10-22）

**作者（11 位，三通讯）**：Kunhui Xu¹˒² · Chenchen Wang¹ · Fenfen Li¹ · **Xiaoyu Chen²** · Xinyu Wang² · Yi Zhang² · Tiancheng Pu² · Jiahao Tang² · Hongguang Cui¹˒\* · Peng Guo²˒\* · Qi Dai¹˒²˒\*˒†（lead contact）

**资助（两条）**：National Natural Science Foundation of China (82571251) + Medical and Health Science Program of Zhejiang Province (WKJ-ZJ-26067)。正文致谢、`07_declarations`、`01_cover_letter`、R4、COI 对照清单均已同步。

**Fig 1 收尾**：`B26` 上移 0.4 mm（消除全图最紧行距）；两条横向箭头重锚，A→B 现覆盖整条 45.0–52.0 mm 栏间、B→C 覆盖 122.0–129.0 mm，两端均贴卡片边框（像素实测 44.99→51.98 / 121.98→128.97）。observer 终审结论：**无阻断性缺陷，可付印**。

**当前哈希**：`latex/paper1.tex` `ce0dcb36…`；`latex/build/paper1.pdf` = `03_manuscript.pdf` `e2e3fa07…`（34 页）；SI `c3c8f0d7…` / `60f6ebc8…`（26 页）；Fig 1 PPTX `694ae88d…`、PDF `666f8839…`、PNG `e8aeb49f…`；`figure_manifest.json` `002123ff…`；`CHECKSUMS` v30l 7/7；`assemble --check` OK。

**仅剩作者本人事项**：仓库 public + Zenodo DOI；COI 表按 11 人名单重填（对照清单已更新至 v30k）；Bigger Picture 是否随附。


---

## 17. v30r — 终态（2026-10-22）

**结构**：正文 31 页 / 0 error / 0 undefined / 0 overfull（14 underfull）；SI 25 页；投递副本逐字节一致（`03` = `3d70e609…`、`04` = `6b2008a6…`）。
**SI**：Figure S8（internal/external inversion）已删除并重编号，现为 **S1–S13**；六个产生器脚本同步改名；装配清单 6 主图 / 13 补充图；`S14` 与全部旧文件名在 tex 与 PDF 中零命中。
**主稿**：Kendiukhov 电路映射预印本已引；P0 九项修复落地（LAA 余量 0.32/0.33、80/20 划分补齐 79,684+19,922=99,606、饱和宣称限定于 CellPLM、"same arm" 措辞、Fig 3 图注分面板、指纹 0.47%）；句级润色达标（均长 24.4 词、分号 0.07/100 词、`rather than` 3 处），数字多重集不变。
**哈希**：`paper1.tex` `287b5217…`；SI tex `0cebbcfd…`；`CHECKSUMS` v30r 7/7；`assemble --check` OK。

**遗留（非阻断）**：改名的六张图件 PDF 内部元数据仍带旧文件名（字节搬移而非重生成）；`00_CHECKLIST` §1/§10 的历史快照保留旧数字（已标注为历史）。

**仅剩作者本人**：仓库转 public + Zenodo DOI；COI 表按十一位作者与两条资助重填；Bigger Picture 是否随附。


---

## 18. v30v — 终态（2026-10-22 晚）

**主稿 29 页**（31→29：删除模板占位板块 + Methods 细节外移），**SI 30 页**（25→31→30：搬入方法细节与 Table S12，再经编辑清理）。
- Methods 占比 50.4% → **42.9%**（语料中位 35%）；Results 35.5%（中位 38%）。
- 句级润色：主体均句长 24.4 词、分号 0.07/100 词、`rather than` 3 处（语料中位 136 词图注，本稿图注 92–119 词）。
- 图件：6 主图（Fig 1 臂清单修正 + 跨架构独立块；Fig 6C 拆双轴；Fig 3C 区间可见；配色统一）、13 补充图；主表 2（Table 3 移入 SI 成 Table S12）。
- SI 编辑：删变更日志与过程事故叙述（changelog ×6、ECE 分箱修正史、corrupt file、auto-tuned、lacked MKI67、two deviations），配置改为 Table SM1/SM2；prose 分号密度 2.65 → 0.26/100 词。
- 臂名统一：CE-naive, 31-class / Ordinal head（表、正文、SI 一致）；P0 九项修复（LAA 余量 0.32/0.33、80/20 划分 79,684+19,922、饱和宣称限定 CellPLM、`−0.579` 方向明示、指纹 0.47%）。
- 引文：Kendiukhov 电路映射预印本（arXiv:2603.11940）已引并标注 preprint。

**哈希**：`latex/paper1.tex` `70409943…`；`latex/build/paper1.pdf` = `03` `5e261c71…`；SI tex `5bb338e3…`；SI PDF（build/同步/`04`）`86d37073…`；`CHECKSUMS` v30v 7/7；`assemble --check` OK。

**仅剩作者本人**：仓库转 public + Zenodo DOI；COI 表按 11 位作者与两条资助重填；Bigger Picture 是否随附。

---

## 19. 投稿前定位评估与作者裁定（2026-10-22）

**结论**：可发 Patterns，预计大修；接受与否约五五开。最强支撑是同细胞/同标签/同步数的 CE vs ordinal 配对对照（Δρ 0.579，CI 0.541–0.631）；最强风险是头条效应只占设计矩阵一格（Geneformer × 同语料角膜），Table 2 的语外/表皮结果反过来削弱标题的强度。

**查重结论**：该解耦（同一嵌入上分类提升而连续序下降 + 序数头对照）截至 2026-10 未见发表。最近三篇各差一步：Zhou 2026（零样本）、VCBench 2026（架构固有 vs 任务）、Liu 2026 Adv Sci（逐任务效用）。最强反证为 Gene-Chronos（Brief Bioinform 2026, `10.1093/bib/bbag469`）——冻结模型 + 时序适配器可恢复发育排序。

**作者裁定（均记录为"已知并接受的风险"）**：
1. **不补**"微调后嵌入用于分化分析"这一前提的引用（Introduction 该句现为零引用）→ 保留"前提真空"暴露面；
2. **不引** Gene-Chronos 反证；
3. **不提级** CellPLM"精度饱和后才丢序"这一发现；
4. **不跑**新增实验（语外 ordinal+LoRA、扩供体）；
5. **不动** cover letter 定位句。

**处置**：以 v30v 直投 Patterns。上述五项若日后返修被问及，随时可低成本加入。

---

## 20. 可读性评估与作者裁定（2026-10-22）

**实测**：正文 4,509 词；Results 数字密度 **10.2/100 词**（一般研究 2–4）；均句长 24.4→**23.2**；>35 词长句 22→**18**。臂名已统一拼写（29 处），`ECE` 已在 Methods 首次出现处展开。

**读者视角判定（独立评估）**：编辑 10 分钟扫读＝**难**；外审专家＝**中等**（两处死路）；应用型读者＝**难**。

**已实施**：臂名拼写统一 29 处、`Expected calibration error (ECE)` 展开、6 句最长句拆分（数字多重集零变化）。

**作者裁定不改（已记录为已知风险）**：
1. 五处"读不通/读错"项：`that earlier subset` 无前指、**continuity 项从未定义**、`ROBUST/PARTIAL` 判据未定义、Discussion 将 post-cutoff 精度代价写成 "lost its advantage"（Results 实为 macro-F1 0.452→约 0.29）、`q = 1.0` 缺注；
2. 三处指代混乱：`frozen reference` 三个所指、`same-corpus` 为模型相对义、GSE186433/GSE155683/Hu0220 空降无说明；
3. Introduction 末尾 59 词阅读地图。

> 其中第 1 组的第 2、4 两条超出"可读性"范畴：前者使审稿人**无法评估** Table 1/2 的 λ=0.5 各行，后者可能导致应用型读者**选错配方**。若返修被问及，这两条是最可能的触发点，届时按本文件补入即可。

---

## 21. 文风与可读性对标裁定（2026-10-22）

**对标样本**：7 篇已发表 Patterns 论文（MinerU：scELMo / BioLLM / GSFM；JATS 纯正文：NESS / EmmaEmb / scVIDR / DestinyNet）。

**AI 味：不成立。** AI 高频词密度 **0.8/1000**，语料 1.3–11.2（中位 6.5）——本稿为八份文档最低；em-dash 0/1000（语料 0–1.2）；被动语态 4.4（语料 4.3–14.8）。

**句法与语域：已改善。** 第一人称 3.01 → **4.47/1000**（越语料下限 3.4）；7 处句法性名词短语中 4 处改动词式；剩余 3 处经核对非名词堆叠或属主题术语。均句长 25.8、>30 词占比 25.7% 均在语料区间内。

**数字密度：接受现状（实测不可降）。** Results 数字密度 **10.3/100 词**，JATS 纯正文语料 2.7–4.0。已执行 5 处"范围+SI 指针"改造，但仅含 10 个数（Results 共 312 个），密度 10.4→10.3 无实质变化。降至语料水平需删约 160 个数，即删除配对 CI、逐种子值、多臂对照本身——**密度是设计的代价，与本文最强的证据（同细胞配对对照）同源**。作者裁定：接受。

**执行纪律留档**：两处删除因"被删值在 SI 任何表中都不存在"而被主动跳过（`0.4180→0.8158/0.8192`、`0.6887/0.5668/0.7126`）——宁可保留也不丢值。已删的 10 个值经核实在 Table S3/S8/S7 中。

**当前哈希**：`latex/paper1.tex` `f1eca6d3…`；`latex/build/paper1.pdf` = `03_manuscript.pdf` `39599233…`（29 页）；SI 未改（`5bb338e3…` / `86d37073…`，30 页）；`CHECKSUMS` v31 7/7；`assemble --check` OK。

---

## 22. 参考文献核验与整理（2026-10-23，v32）

**回源核验**：63 个 DOI 送 Crossref 逐条比对（标题/首作者/年份），**60 对、2 错配、1 需换库**；18 个 arXiv 条目送 DataCite 核 `10.48550/arXiv.<id>`，**18/18 通过**。

**已修正（均有回源依据）**
| 条目 | 问题 | 处理 |
|---|---|---|
| `merchant2020` | DOI 指向另一篇 | `…blackboxnlp-1.3` → **`…-1.4`**（Crossref 标题核验） |
| `finetuningmasking2026` | 论文被改名，bib 标题过期 | 标题改为 CrossRef 现标题 *Re-evaluating the benefit of training scale in cell foundation models*，并加 retitled 注 |
| `ding2025scgptretina` / `ding2026protocol` | 同篇重复、年份与首作者互相矛盾、均未被引 | 合并为一条（Shanli Ding；Nature Protocols 21:873–893；2025-07-15），删除重复键 |
| `suprank2025` | 正式期刊论文缺 DOI | 补 `10.1109/tpami.2025.3543846`（IEEE TPAMI） |
| 17 条 arXiv 预印本 | 缺 DOI | 各补已核实的 `10.48550/arXiv.<id>` |

**结构整理**：`latex/paper1.bib` 只保留**被正文引用的 48 条**（编译出的参考文献表即 48 条，0 undefined citation）；未被引用的 39 条移入 `latex/paper1_unused.bib`（保留、未删，供后续引用或删除决策）。

**当前哈希**：`latex/paper1.tex` 未改；`latex/paper1.bib` 见 `CHECKSUMS.sha256`；`latex/build/paper1.pdf` = `03_manuscript.pdf`（29 页）；`CHECKSUMS` v32 7/7；`assemble --check` OK。
