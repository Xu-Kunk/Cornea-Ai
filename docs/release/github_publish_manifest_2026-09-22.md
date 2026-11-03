# GitHub 发布清单 — github_publish_manifest_2026-09-22

> 生成日期: 2026-09-22
> 项目: Cornea-Ai (Paper 1 — "Fine-tuning collapses differentiation gradients in single-cell foundation models")
> 目标仓库: `https://github.com/Xu-Kunk/Cornea-Ai`
> 边界依据: `docs/manuscript/patterns/README.md` 公开边界 + `SUBMISSION_PACKAGE.md`

---

## 一、三档分类总表

| # | 路径 | 类型 | 文件数 | 大小 | 归类 | 理由 |
|---|------|------|--------|------|------|------|
| 1 | `corneal_fm/` | 源码包(Python) | 24 | 312K | **PUBLISH** | 核心代码包，边界明确列入「代码」 |
| 2 | `scripts/` | 实验脚本 | 151 | 2.3M | **PUBLISH** ⚠ 见下注 | 边界列入「scripts」，但内部运维脚本(gx10\*.sh, auto_recover.sh, ded_dryarm/)需剔除 |
| 3 | `config/` | 配置文件 | 7 | 44K | **PUBLISH** ⚠ 见下注 | 边界列入「config」；`requirements-gx10.*` 含主机名 gx10，建议改名 |
| 4 | `tests/` | 测试套件 | 20 | 204K | **PUBLISH** | 最小可复现包组成部分 |
| 5 | `tables/paper1/` | 聚合 CSV | 2 | 12K | **PUBLISH** | 聚合 CSV，边界明确列入 |
| 6 | `artifacts/env_locks/` | 环境锁 | 2 | 4.4K | **PUBLISH** | 边界明确列入「环境锁」 |
| 7 | `README.md` | 说明 | 1 | 6.7K | **PUBLISH** | 边界明确列入「README」 |
| 8 | `pyproject.toml` | 项目配置 | 1 | 2.0K | **PUBLISH** | 项目元数据/依赖声明 |
| 9 | `requirements-ci.txt` | 依赖(CI) | 1 | 539 | **PUBLISH** | CI 依赖 |
| 10 | `requirements-core.txt` | 依赖(核心) | 1 | 1.6K | **PUBLISH** | 核心依赖 |
| 11 | `Dockerfile` | 容器配置 | 1 | 1.5K | **PUBLISH** | 可复现环境 |
| 12 | `.ruff.toml` | lint 配置 | 1 | 136 | **PUBLISH** | 代码风格 |
| 13 | `code-style/` | lint 工具配置 | 5 | 24K | **REVIEW → PUBLISH** | lint tooling，无害，建议包含到 `code-style/` |
| 14 | `.github/workflows/ci.yml` | CI 配置 | 1 | 1.2K | **REVIEW → PUBLISH** | CI/CD 标准配置，建议包含 |
| 15 | `notebooks/corneal_fm_walkthrough.ipynb` | 交互式笔记本 | 1 | 8.7K | **REVIEW → PUBLISH** | 用户 Walkthrough，建议包含 |
| 16 | `figures/paper1/` | 论文定稿图件 | 155 | 22M | **REVIEW → PUBLISH** | 论文最终正式图件(300dpi, 300dpi 验证通过)，建议包含 |
| 17 | `figures/cs_before_pca.*` | PCA 图件 | 2 | 180K | **REVIEW → PUBLISH** | 论文辅助图件，建议包含 |
| 18 | `docs/manuscript/patterns/latex/` | LaTeX 稿件源码 | ~15 | 6.1M | **REVIEW → PUBLISH** | 论文 LaTeX 源码 + bib，监督代码+稿件联合复现的关键版本(v23)；作者邮箱需保留(论文内容) |
| 19 | `docs/manuscript/patterns/*.md` | 投稿附加件 | ~10 | 500K | **REVIEW → PUBLISH** | highlights/eTOC/Bigger Picture/补充材料，建议仅留 active 最新版，归档剔除 |
| 20 | `artifacts/` (聚合 JSON/CSV 子集) | 聚合结果 | ~290* | ~50M* | **REVIEW → PUBLISH** | 边界明确列入「聚合 JSON/CSV」，但 artifacts/ 层级混杂二进制(见下节) |
| 21 | `artifacts/paper1_preprint/SUBMISSION_PACKAGE.md` | 提交元数据 | 1 | 9.6K | **REVIEW → PUBLISH** | 投稿说明，含公开作者邮箱 |
| 22 | `context.yaml` | Agent 上下文配置 | 1 | 235 | **REVIEW → EXCLUDE** | 内部 agent 工作区配置，不属于论文可复现包 |
| 23 | `AGENTS.md` | Agent 纪律文件 | 1 | 4.5K | **REVIEW → EXCLUDE** | 内部 agent 工作纪律，不属于论文可复现包 |
| 24 | `findings.md` | 研究发现笔记 | 1 | 6.0K | **REVIEW → EXCLUDE** | 内部研究笔记 |
| 25 | `progress.md` | 项目进度追踪 | 1 | 34K | **REVIEW → EXCLUDE** | 内部进度管理 |
| 26 | `task_plan.md` | 任务计划 | 1 | 17K | **REVIEW → EXCLUDE** | 内部任务规划，含内部路径 /home/dq/ |
| 27 | `docs/research/` | 研究笔记/开题报告 | ~25 | 6.4M | **EXCLUDE** | 内部设计文档 |
| 28 | `docs/changes/` | 变更日志 | ~20 | 732K | **EXCLUDE** | 内部开发日志，含内部路径和主机名 |
| 29 | `docs/manuscript/archive/` | 旧稿件存档 | ~15 | 3.5M | **EXCLUDE** | 历史草案，含内部邮箱 |
| 30 | `docs/SCI-reference/` | 参考文献 PDF | ~300 | 391M | **EXCLUDE** | 第三方受版权保护 PDF，不可再分发 |
| 31 | `docs/GX10-deployment-checklist.md` | 部署清单 | 1 | 16K | **EXCLUDE** | 含内部主机名/SSH 配置/绝对路径 /home/dq/ |
| 32 | `docs/preprocessing_report.md` | 预处理报告 | 1 | 4.0K | **EXCLUDE** | 内部报告 |
| 33 | `notes/` | 思想笔记 | 20 | 3.2M | **EXCLUDE** | 内部笔记/灵感，含 jpg 引用图 |
| 34 | `tasks/` | 任务跟踪 | 12 | 136K | **EXCLUDE** | 内部任务管理，含内网 IP/主机名 |
| 35 | `data/` | 原始/处理数据 | 270 | 81G | **EXCLUDE** | 原始数据，边界明确排除；通过 download scripts 获取 |
| 36 | `save/` | 训练 checkpoint | 407 | 35G | **EXCLUDE** | fine-tuned checkpoint 与训练日志，边界明确排除 |
| 37 | `.venv/` | Python 虚拟环境 | 51,707 | 5.8G | **EXCLUDE** | 虚拟环境，非项目内容 |
| 38 | `.venv-cellplm/` | Python 虚拟环境 | 41,169 | 5.5G | **EXCLUDE** | 虚拟环境，非项目内容 |
| 39 | `.opencode/` | Agent 工具缓存 | 3,660 | 63M | **EXCLUDE** | Agent 运行缓存/日志/技能，非项目内容 |
| 40 | `.omo/` | Agent 会话缓存 | 97 | 404K | **EXCLUDE** | Agent 运行缓存(已在 .gitignore) |
| 41 | `.codegraph/` | 代码图数据库 | 5 | 8.3M | **EXCLUDE** | Agent 缓存 DB(已在 .gitignore) |
| 42 | `artifacts/archive/` | 历史实验产物 | ~500 | 27G | **EXCLUDE** | 历史归档实验产物 |
| 43 | `artifacts/`(二进制子集) | 模型/嵌入/数据集 | ~512 | ~3.6G | **EXCLUDE** | fine-tuned checkpoint、tokenized datasets、npz/cache 文件 |
| 44 | `artifacts/paper1_preprint/*.pdf` (历史) | 稿件 PDF | 11 | 9.7M | **EXCLUDE** | 历史版本(v3-v17)，v18 可通过 bioRxiv 获取；保留 archive/ 中的 v18 供审查 |
| 45 | `artifacts/paper1_preprint/archive/` | 稿件 PDF | 5 | 4.8M | **EXCLUDE** | 更早历史版本 |

> ※ scripts 注: 151 个文件中含 gx10\*.sh (4 个运维脚本) 和 auto_recover.sh (集群恢复脚本) 应排除；ded_dryarm/ (2 个验证脚本) 可保留但极小。剔除后约 ~145 个脚本可 PUBLISH。
>
> ※ config 注: `requirements-gx10.txt` 和 `requirements-gx10.lock.txt` 含主机名 "gx10"，建议重命名为 `requirements-cellplm.txt` / `requirements-cellplm.lock.txt` 后再 PUBLISH。
>
> ※ artifacts 聚合 JSON/CSV 注: 约 290 个 JSON/CSV 文件 (<2MB) 散布在各子目录中，与 .pt/.npz/.arrow 等大二进制文件混合存放。用户需手动提取 (见下节 REVIEW 建议)。

---

## 二、安全扫描结果

### 2.1 密钥/凭证 (API Key / Token / Password / Secret / 私钥)

| 等级 | 命中 | 位置 | 说明 |
|------|------|------|------|
| 🟡 LOW | `ZOTERO_API_KEY` (引用,非实际值) | `docs/changes/archive/2026-06/2026-06-10-sop-v4.1-zotero-mineru.md:20` | 变更日志提及环境变量名，未包含实际 key 值。但该文件属于 **EXCLUDE** 范围 (docs/changes/ 历史归档) |
| 🟡 LOW | `Bearer {TOKEN}` (占位符) | `scripts/mineru_convert_missing.py:41`, `scripts/mineru_batch_convert.py:29` | 这两个脚本使用 MinerU API token，但在源码中仅出现 `Bearer {TOKEN}` 占位符，无实际值运行。但脚本引用 token 模式——如果用户想 PUBLISH 这些脚本，需确认移除 token 引用或提醒用户清理。**建议**: 检查这两个脚本是否必需；若 PUBLISH 则清理 token 相关行 |
| ✅ 无 | 私钥文件 | — | 未发现 `BEGIN *PRIVATE KEY`、`.pem`、`.key`、`.netrc` 文件 |
| ✅ 无 | `.env` / `credentials` | — | 未发现 `.env*`、`credentials*`、`*.p12`、`*.pfx` 文件 |

**结论**: 未发现实际有效的密钥/凭证泄露风险。

### 2.2 邮箱/个人信息 (PII)

| 等级 | 邮箱 | 位置 | 说明 |
|------|------|------|------|
| 🟡 MED | `dq@mail.eye.ac.cn` | `paper1_final_candidate_v23.tex` + `paper1_supplementary_materials.md` + `SUBMISSION_PACKAGE.md` + 多处备份 | **通讯作者邮箱**，属于论文投稿必需的公开信息，Safe。但多处历史备份中存在冗余副本(5+ 处旧 .tex 备份) |
| 🟢 OK | `lifenfen_wzmu@163.com` | `paper1_supplementary_materials.md:10` | **作者邮箱**，已在正文中公开属于论文法定内容，Safe |
| 🟢 OK | `patterns@cell.com` | `patterns_submission_requirements.md:37` | Patterns 期刊官方邮箱，Safe |
| 🟢 OK | `latex.support@river-valley.com` | `numcompress.sty:5` | LaTeX 模板供应商邮箱，Safe |

**结论**: 论文对应的公开邮箱 (`dq@mail.eye.ac.cn`, `lifenfen_wzmu@163.com`) 是投稿元数据的正常组成部分，无需清除。建议仅在 PUBLISH 前确认只保留 v23 定稿 .tex 中的邮箱、移除所有备份/归档文件中的副本。

### 2.3 内部路径 (`/home/dq/`) 与服务器主机名

| 等级 | 命中 | 位置 | 说明 |
|------|------|------|------|
| 🔴 **HIGH** | `/home/dq/Codex/Ai/...` | `docs/GX10-deployment-checklist.md` (12 处) | 内网部署文档包含**完整绝对路径** (`/home/dq/Codex/Ai/save/...`, `/home/dq/venv-ds/`, `/home/dq/.cache/huggingface/`) 和 SSH 部署指令。该文件应 **EXCLUDE** |
| 🔴 **HIGH** | `/home/dq/Codex/Ai` | `progress.md:334`, `task_plan.md:4,71,135,137,168` | 内部进度/任务跟踪文档。该文件应 **EXCLUDE** |
| 🟡 MED | `/home/dq/venv-ds` | `config/environment-provenance.md:8`, `docs/GX10-deployment-checklist.md` | 内部环境路径 |
| 🟡 MED | `p5070` (主机名) | `tasks/train-ops-headinit-cellplm-layerwise.yaml:3`, `tasks/train-ops-cellplm-vanzyl.yaml:2`, `docs/changes/archive/...` | 内部训练服务器名。tasks/ 应 EXCLUDE；docs/changes 应 EXCLUDE |
| 🟡 MED | `GX10` / `gx10` (主机名) | `docs/GX10-deployment-checklist.md` (30+ 处) + `config/requirements-gx10.*` + `scripts/gx10*.sh` (4 files) | 内部 Blackwell 服务器名(ASUS Ascent GX10)。`docs/GX10-deployment-checklist.md` 已 EXCLUDE；文件名中的 gx10 需改名 |
| 🔴 **HIGH** | `192.168.50.2`, `192.168.100.1`, `192.168.100.2` | `tasks/train-ops-*.yaml`, `docs/changes/archive/...` | **内网 IP 地址**全部位于 EXCLUDE 范围的文档中 |

**结论**: 内网路径/IP/主机名命中均位于 EXCLUDE 文件中(任务跟踪、变更日志、部署清单)。PUBLISH 范围内除 `config/requirements-gx10.*` 文件名外，**未发现内网路径泄露**。建议将 `requirements-gx10.*` 重命名。

### 2.4 聚合 JSON/CSV 内部混合情况

**关键冲突**: artifacts/ 下的聚合 JSON/CSV (PUBLISH 候选) 与 Checkpoint/嵌入文件 (EXCLUDE) 位于同一子目录内，无法简单区分，需要手动提取。

示例:
| artifacts 子目录 | 聚合 JSON/CSV (PUBLISH 候选) | 二进制/模型文件 (EXCLUDE) |
|-----------------|-----------------------------|--------------------------|
| `paper1_vanzyl/corn_lora/` | `summary.json` | `checkpoints/` (6×198MB .pt), `model.pt`, `run.log` |
| `paper1_vanzyl/sensitivity/` | `sensitivity_summary.json` | `_caches/*.npz` (56M ×4) |
| `paper1_vanzyl/cellplm/` | — | `seed42/frozen_emb.npz` (40M) |
| `paper1_tier_b_eval/` | `summary.csv`, `results_*.json` | `external_tokenized/*.arrow` (167M), `percell_frozen_gf.npz` (60M) |
| `paper1_vanzyl/cross_tissue/` | `final_test_eval.json` | `scgpt_epidermis_train.h5ad` (246M), .pt checkpoints |

### 2.5 未发现的安全类目

- 手机号: 未发现
- 云盘/集群路径(非 /home/dq/): 未发现
- SSH private key / authorized_keys: 未发现
- AWS/GCP/Azure 凭证: 未发现
- GitHub Token / Personal Access Token: 未发现

---

## 三、大文件清单 (>10 MB)

以下列出 PUBLISH + REVIEW 候选范围中 >10 MB 的文件。**绝大多数位于 artifacts/ 中且属于 EXCLUDE 类型**。

### EXCLUDE 类 (不应上传)

| 文件路径 | 大小 | 类型 |
|----------|------|------|
| `artifacts/paper1_vanzyl/cellplm_consistency/loo220/seed42/model_best_valid.pt` | 249 MB | Fine-tuned checkpoint |
| `artifacts/paper1_vanzyl/cellplm_consistency/loo220/seed123/model_best_valid.pt` | 249 MB | Fine-tuned checkpoint |
| `artifacts/paper1_vanzyl/cellplm_consistency/epidermis/seed42/model_best_valid.pt` | 249 MB | Fine-tuned checkpoint |
| `artifacts/paper1_vanzyl/cellplm_consistency/epidermis/seed123/model_best_valid.pt` | 249 MB | Fine-tuned checkpoint |
| `artifacts/paper1_vanzyl/cross_tissue/scgpt_epidermis_train.h5ad` | 246 MB | Tokenized dataset |
| `artifacts/paper1_tier_b_eval_gse299152/external_tokenized/cell.dataset/data-00000-of-00001.arrow` | 237 MB | Tokenized dataset |
| `artifacts/paper1_vanzyl/corn_lora/checkpoints/*.pt` (×6) | 198 MB each | Fine-tuned checkpoint |
| `artifacts/paper1_vanzyl/g1_gf/gf_vanzyl_cell_emb.npz` | 187 MB | Embeddings |
| `artifacts/paper1_tier_b_eval/external_tokenized/cell.dataset/data-00000-of-00001.arrow` | 167 MB | Tokenized dataset |
| `artifacts/paper1_vanzyl/g1_scgpt/vanzyl_cell_emb.npz` | 126 MB | Embeddings |
| `artifacts/paper1_tier_b_eval/percell_frozen_gf.npz` | 60 MB | Embeddings |
| `artifacts/paper1_vanzyl/sensitivity/_caches/*.npz` (×4) | 40–58 MB each | Cache |
| `docs/SCI-reference/**/*.pdf` (multiple) | 13–40 MB | Reference PDFs |
| `artifacts/paper1_vanzyl/cellplm/seed42/*.npz` (×3) | 33–40 MB | Embeddings |
| `artifacts/paper1_vanzyl/cellplm_lora/seed42/*.npz` (×3) | 37–40 MB | Embeddings |

**PUBLISH 范围内最大文件**: `figures/paper1/` 中的 PDF/PNG 图件，单个 < 2MB，总计 22MB (155 files)。**无 >10MB 文件在 PUBLISH 范围内**。

---

## 四、需用户裁决项 (REVIEW) — 共 14 项

| # | 路径 | 建议 | 理由 |
|---|------|------|------|
| 1 | `figures/paper1/` (155 文件, 22M) | ✅ **PUBLISH** | 论文正式定稿图件(300dpi 验证通过)，无原始数据。22M 对 GitHub 仓库可接受。读者需图件验证结果 |
| 2 | `docs/manuscript/patterns/latex/` (v23 定稿) | ✅ **PUBLISH** | 论文 LaTeX 源码+PDF。提升可复现性(代码指向的图/表路径)。建议仅保留 `paper1_final_candidate_v23.tex` + `paper1.bib` + `paper1-current.pdf` + `elsarticle.cls` + `numbered.bst` + `numcompress.sty`，剔除 build/ 和 archive/ 子目录 |
| 3 | `docs/manuscript/patterns/*.md` (附加件) | ✅ **PUBLISH** (精选) | 仅保留 highlights/eTOC/Bigger Picture/补充材料 (SUBMISSION_PACKAGE.md 定义的最新版) ≈ 5 个 .md；归档历史如 `provenance/`、`archive/` 内旧版应剔除 |
| 4 | `notebooks/corneal_fm_walkthrough.ipynb` (8.7K) | ✅ **PUBLISH** | 单文件无输出(8.7K 纯代码)，值为 Walkthrough 示例。极小，建议包含 |
| 5 | `tables/paper1/*.csv` (2 文件, 12K) | ✅ **PUBLISH** | 聚合 CSV，边界已列入。建议包含 |
| 6 | `artifacts/` 聚合 JSON/CSV (~290 文件, ~50M) | ✅ **PUBLISH** (精选) | 边界明确列入「聚合 JSON/CSV」；但散布在 artifacts/ 各目录中。**建议操作**: 手动从各子目录提取 `summary.json`, `summary.csv`, `*_summary_*.json`, `*_layerwise_rho_*.json`, `results_*.json`, `sensitivity_summary.json`, `head_init_summary.json`, `bh_fdr_table.json`, `svd_axis_validation.json`, `final_test_eval.json`, `percell_*.json`, `cs_before_results.json` 等集中放入 `results/` 目录 |
| 7 | `artifacts/paper1_preprint/SUBMISSION_PACKAGE.md` | ✅ **PUBLISH** | 投稿说明，含公开作者邮箱。可随 repo 发布 |
| 8 | `artifacts/paper1_preprint/paper1_v18_preprint.pdf` (517K) | ⚠️ **可选** | v18 PDF 应上传至 bioRxiv；GitHub 上可保留一个指向 bioRxiv 的链接，也可存放该 PDF 供审稿人便利 |
| 9 | `code-style/` (5 文件, 24K) | ✅ **PUBLISH** | Lint/tooling 配置，无害，有助于代码风格一致 |
| 10 | `.github/workflows/ci.yml` | ✅ **PUBLISH** | CI/CD 标准配置，有助于审稿人运行测试 |
| 11 | `context.yaml` (235B) | ❌ **EXCLUDE** | Agent 内部上下文配置，非论文内容 |
| 12 | `AGENTS.md` (4.5K) | ❌ **EXCLUDE** | Agent 工作纪律，非论文内容 |
| 13 | `code-style/` 的 README 等附注 | ✅ **PUBLISH** | 同 #9，无害 |
| 14 | `artifacts/paper1_vanzyl/layerwise/` 等混合目录 | ⚠️ 需手动 | 用户需手工挑选 JSON 聚合结果、排除 .pt/.npz |

---

## 五、建议的 GitHub 上传步骤

以下仅为文字建议，不执行操作：

### 准备阶段
1. 在 `docs/release/` 中审查本清单，确认每项 REVIEW 决策
2. 手动从 `artifacts/` 中提取聚合 JSON/CSV 到 `results/` 目录(见第四节 #6 建议)
3. 清理 `scripts/` — 移除 `gx10_*.sh`、`auto_recover.sh` 等内部运维脚本
4. 重命名 `config/requirements-gx10.*` → `config/requirements-cellplm.*`
5. 从 `notebooks/` 中确认 `.ipynb` 不包含输出单元格中的敏感内容
6. 清理 `docs/manuscript/patterns/latex/` — 仅保留 v23 定稿文件，移除 `archive/` 和 `build_v23/`

### .gitignore 更新
7. 应用 `gitignore_proposed_additions_2026-09-22.md` 中的建议规则

### 上传
8. 用户自行在 GitHub 创建仓库 `Xu-Kunk/Cornea-Ai`(不设 .gitignore 或选 Python 模板)
9. `git init` (由用户在本地执行)
10. 复制/移动 PUBLISH 文件到工作副本，确认 EXCLUDE 文件不被追踪
11. `git add .` → `git status` 确认无意外文件
12. `git commit -m "Initial release: reproducible package for Paper 1"`
13. `git remote add origin https://github.com/Xu-Kunk/Cornea-Ai.git`
14. `git push -u origin main`

### 发布后核对
15. 在 GitHub 上检查文件列表，确保无 `.venv/`、`data/`、`save/` 等上传
16. 在 `README.md` 中更新下载脚本链接指向原始数据来源
17. 在 `README.md` 中加入指向 bioRxiv 的 DOI 链接 (发布后)