# Paper 1 — Patterns 投稿包（v31, 2026-10-23）

当前方向：**Cell Press *Patterns* 直投**（Research Article）。提交件以 `submission/` 为唯一投递目录，由 `scripts/assemble_patterns_submission.py` 生成，禁止手改。

## 1. 入口

| 用途 | 路径 | 状态 |
|---|---|---|
| 主稿源（唯一权威文本） | `latex/paper1.tex` | v31 |
| 书目 | `latex/paper1.bib` | 88 条 |
| 主稿编译件 | `latex/build/paper1.pdf` | **29 页**，0 error / 0 undefined / 0 overfull（12 underfull） |
| 补充材料源 | `supplementary/supplementary_information.tex` | v31 |
| 补充材料编译件 | `supplementary/build/supplementary_information.pdf` | **30 页**；Figure S1–S13、Table S1–S12、Supplementary Methods（SM1–SM12 + Table SM1/SM2）、Data S1–S3 |
| 投稿用补充材料单文件 | `paper1_supplementary_information.pdf` | 与 build 逐字节一致 |
| 包元数据与版本历史 | `PATTERNS_ACTIVE_PACKAGE.md` | 记录 v25→v31 |
| 投稿清单与开放项 | `submission/00_CHECKLIST.md` | §19–§21 为最新裁定 |
| 投递目录（生成物） | `submission/` | `assemble --check` 通过 |
| Cover letter | `submission/01_cover_letter.md` | 手写 |
| Highlights / eTOC / Bigger Picture | `submission/02_highlights_etoc_bigpicture.md` | 手写；Bigger Picture 首投非必需 |
| 声明（CRediT / COI / AI） | `submission/07_declarations.md` | 手写 |
| 数据与代码可用性 | `submission/08_data_code_availability.md` | 手写 |
| COI 官方表 | `submission/09_declaration_of_interests_draft.pdf` | **待作者重填**（按 11 位作者、两条资助） |
| 参考材料 | `reference/` | 图件规范、文风指南、投稿要求、期刊定位、GB 对标 |
| 图件系统 | `figures/paper1/`（`FIGURE_SYSTEM.md`、`figure_manifest.json`、`CAPTION_DEBT.md`） | 6 主图 + 13 补充图 |

## 2. 编译与装配

```bash
# 主稿
cd docs/manuscript/patterns/latex
pdflatex -output-directory=build paper1.tex
( cd build && BSTINPUTS=.. BIBINPUTS=.. bibtex paper1 )
pdflatex -output-directory=build paper1.tex && pdflatex -output-directory=build paper1.tex

# 补充材料
cd ../supplementary && pdflatex -interaction=nonstopmode supplementary_information.tex   # ×3
cp supplementary_information.pdf ../paper1_supplementary_information.pdf                 # 若产出在 supplementary/

# 哈希 → 装配 → 校验（顺序不可颠倒：装配脚本先验 CHECKSUMS）
cd /home/dq/Codex/Ai && python3 scripts/assemble_patterns_submission.py && python3 scripts/assemble_patterns_submission.py --check
cd docs/manuscript/patterns && sha256sum -c CHECKSUMS.sha256
```

## 3. 当前内容要点（v31）

- 标题：*Annotation accuracy is not a proxy for differentiation order in fine-tuned single-cell foundation models*
- 11 位作者，三位通讯（Hongguang Cui / Peng Guo / Qi Dai），Qi Dai 为 lead contact；两个单位（ZJU 一附院、HIOM-CAS/WMU 眼科医院）
- 资助：NSFC 82571251 + Medical and Health Science Program of Zhejiang Province (WKJ-ZJ-26067)
- 七个 Results 小节均为命题式标题；Methods 13 小节（配置细节已移入 Supplementary Methods）
- 关键数字：Geneformer 同语料 macro-F1 0.4421→0.6492、ρ 0.8064→−0.0082（配对 Δρ 0.815，CI 0.692–0.908）；4 类 CE 对照 0.254（7,473 步）/0.158（2,616 步）vs 序数头 0.833（Δρ 0.579 低，CI 0.541–0.631）；语外 0.707→0.641；表皮反向；CellPLM 精度饱和后才丢序
- 文风对标（7 篇已发表 Patterns）：AI 高频词密度 0.8/1000（最低）；第一人称 4.47/1000；数字密度 10.3/100 词（设计代价，已裁定接受）

## 4. 投稿前仅剩作者事项

1. GitHub `Xu-Kunk/Cornea-Ai` 转 public + Zenodo 铸 DOI（现声明为"审稿人可索取、发表前公开并铸 DOI"）
2. Cell Press COI 官方表按当前 11 位作者与两条资助重填，并勾选 "On behalf of all authors"（对照清单：`docs/changes/2026-10-22-paper1-coi-fill-guide.md`）
3. Bigger Picture 是否随附首投（非必需）

## 5. 版本说明

v25（2026-10-16）及更早归档于 `latex/archive/`、`supplementary/archive/`、`archive/`。v26–v31 的逐次变更见 `PATTERNS_ACTIVE_PACKAGE.md` 与 `docs/changes/2026-10-2*.md`。
