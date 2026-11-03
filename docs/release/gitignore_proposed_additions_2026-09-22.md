# .gitignore 建议追加规则 — gitignore_proposed_additions_2026-09-22

> 基于 `/home/dq/Codex/Ai/.gitignore` 现行版本 (61 行, 2026-09-22) 的审计与改进建议。
> 目标: 确保 GitHub 最小可复现包不泄露任何敏感文件，同时不误伤 PUBLISH 内容。

---

## 一、现有保留规则 (建议保留，不变)

以下现有规则覆盖充分，无须修改:

```gitignore
# 虚拟环境 (已有 .venv/，需追加 .venv-cellplm/)
.venv/

# Python 缓存
__pycache__/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*.pyc
*.pyo

# Agent 缓存 / 非项目内容
.codegraph/
.omo/

# 原始数据规则 (覆盖 h5ad/csv/gz/tar/epidermis/gf_tokenized)
data/*.h5ad
data/*.csv
data/*.gz
data/*.tar
data/epidermis/
data/gf_tokenized_*/
data/corneal_*.h5ad
data/gse299152/
data/cornea2/
data/cornea2_download/
data/ded_dryarm/

# checkpoint 与模型权重
checkpoints/
*.pt
*.ckpt
*.safetensors
*.pkl
*.joblib
*.bin
*.onnx
*.h5
*.hdf5
*.npy
*.npz
*.model
*.weights

# 运行输出与大体积结果
save/
artifacts/
releases/

# 第三方 / IDE
node_modules/
dist/
build/
.DS_Store
Thumbs.db
*.tmp
*.bak
.frontend-slides/

# HTML 生成物
*.html
```

---

## 二、关键缺口 — 必须追加

### 缺口 1: `.venv-cellplm/` — 虚拟环境 (HIGH 风险)

```gitignore
.venv-cellplm/
```

**理由**: 第二个 Python 虚拟环境，5.5G / 41,169 文件。包含 torch、cuda 库等。当前完全不拦截。
**风险**: 未拦截情况下 `git add .` 会上传 ~5.5G 的无用文件，且可能包含本地构建 CUDA .so 库的路径泄露。

### 缺口 2: `.opencode/` — Agent 工具缓存 (MED 风险)

```gitignore
.opencode/
```

**理由**: OpenCode Agent 工具缓存: 63M / 3,660 文件，含 `node_modules/` (npm 包)、skill 配置、会话日志。不是项目代码。
**风险**: `node_modules/` 已混淆在 `.opencode/` 目录中；即使根目录 `node_modules/` 被忽略，此子目录仍被 git 追踪。技能配置 SKILL.md 可能含内部路径。

### 缺口 3: 内部工作笔记 (HIGH 风险)

```gitignore
# 内部工作笔记与任务跟踪 (非论文可复现包)
notes/
tasks/
```

**理由**: `notes/` (3.2M, 20 文件) 含内部研究灵感和引用图 jpg；`tasks/` (136K, 12 文件) 含 train-ops YAML 文件，其中记录内网 IP (`192.168.50.2`, `192.168.100.1`) 和内部服务器主机名 (`p5070`)。一旦上传将泄露内部网络拓扑。
**风险**: **实际已存在内网 IP 泄露源证据**——若不拦截，上传后攻击者可利用内部网络架构信息。

### 缺口 4: 内部根文档 (MED 风险)

```gitignore
# 内部进度/任务/发现文档 (非论文产出)
findings.md
progress.md
task_plan.md
context.yaml
AGENTS.md
```

**理由**: 
- `progress.md` 含 `/home/dq/Codex/Ai` 绝对路径
- `task_plan.md` 含 `/home/dq/Codex/Ai` 及内部环境描述
- `context.yaml` + `AGENTS.md` 是 agent 工作区配置，非论文内容

**风险**: 尚无外网泄露，但 EXPOSE `/home/dq/` 路径。

### 缺口 5: `docs/GX10-deployment-checklist.md` — 服务器部署文档 (HIGH 风险)

```gitignore
docs/GX10-deployment-checklist.md
```

**理由**: 该文件含 30+ 处内网服务器配置细节：
- `/home/dq/Codex/Ai/...` 完整文件系统路径
- `/home/dq/venv-ds/bin/python` 内部 Python 解释器路径
- `/home/dq/.cache/huggingface/hub/...` HuggingFace 缓存路径
- `GX10` 服务器主机名及 SSH 部署命令
- 内部 IP 地址

**风险**: 这是本次扫描发现的**最高风险文件**，暴露完整的内部基础设施拓扑。

---

## 三、建议优化 — 降低误伤

### 优化建议 1: `artifacts/` 全忽略太宽

当前 `.gitignore` 用 `artifacts/` 全忽略，但边界要求 PUBLISH 聚合 JSON/CSV。两个方案:

**方案 A (推荐)**: 保留 artifacts/ 忽略。将聚合 JSON/CSV 手动拷贝到 `results/` 目录(PUBLISH)，然后在 `.gitignore` 中追加:

```gitignore
results/
```

(文件已从 artifacts 复制到 results 后，不再需要额外 ignore)。

**方案 B**: 精确忽略 artifacts/ 内的大文件类型，但不是推荐做法(JSON/CSV 仍散布在二进制文件中):

```gitignore
# 可选: 更精细控制 — 不推荐，容易遗漏
# artifacts/*.zip
# artifacts/archive/
```

### 优化建议 2: `config/` 不应全拦截

当前 `config/` 没有被 `.gitignore` 覆盖(即默认会 git add)，这是正确的。但 `config/requirements-gx10.*` 文件名含内部主机名。建议**不改 `.gitignore`，改文件名为** `requirements-cellplm.*`。

### 优化建议 3: `docs/manuscript/` 手工挑选

`docs/manuscript/archive/`、`docs/manuscript/patterns/archive/`、`docs/manuscript/patterns/latex/archive/`、`docs/manuscript/patterns/latex/build_v23/` 均为内部历史/构建产物。可在 `.gitignore` 中追加:

```gitignore
# LaTeX 构建产物与稿件历史档案
docs/manuscript/archive/
docs/manuscript/patterns/archive/
docs/manuscript/patterns/latex/archive/
docs/manuscript/patterns/latex/build_v23/
```

### 优化建议 4: `figures/` 中的 PCA 中间产物

`figures/cs_before_pca.png` (152K) 和 `figures/cs_before_pca.pdf` (28K) 是预处理 PCA 图件。如果不想发布，可追加:

```gitignore
# 预处理中间图件 (非论文定稿图)
figures/cs_before_pca.*
```

### 优化建议 5: `*.html` 过于宽泛会误伤

当前 `*.html` 全忽略 (line 61)。检查 PUBLISH 范围内是否有 .html 文件:
- `docs/manuscript/patterns/latex/` — 无 .html
- 其他 PUBLISH 目录 — 无 .html
结论: 当前 `*.html` 规则无害，但可添加例外注释。

---

## 四、完整建议 $`.gitignore` 追加块

将以下块追加到现有 `.gitignore` 末尾:

```gitignore
# === 追加规则 (2026-09-22 审计) ===

# 第二个虚拟环境 (5.5G, 41k 文件)
.venv-cellplm/

# Agent / IDE 工具缓存
.opencode/
.omo/

# 内部工作笔记与任务跟踪 (含内网 IP 和主机名)
notes/
tasks/
findings.md
progress.md
task_plan.md
context.yaml
AGENTS.md

# 内部服务器部署文档 (含绝对路径与 SSH 配置)
docs/GX10-deployment-checklist.md

# LaTeX 构建产物与稿件历史档案 (只保留 v23 定稿)
docs/manuscript/archive/
docs/manuscript/patterns/archive/
docs/manuscript/patterns/latex/build_v23/

# 预处理中间图件 (非论文定稿图)
figures/cs_before_pca.*
```

---

## 五、.gitignore 覆盖总结

| 范围 | 当前状态 | 建议 | 优先级 |
|------|----------|------|--------|
| `.venv-cellplm/` | ❌ 未覆盖 | 追加 | HIGH |
| `.opencode/` | ❌ 未覆盖 | 追加 | HIGH |
| `notes/` | ❌ 未覆盖 | 追加 | HIGH (含内网 IP) |
| `tasks/` | ❌ 未覆盖 | 追加 | HIGH (含内网 IP) |
| `docs/GX10-deployment-checklist.md` | ❌ 未覆盖 | 追加 | HIGH |
| `findings.md` / `progress.md` / `task_plan.md` | ❌ 未覆盖 | 追加 | MED |
| `AGENTS.md` / `context.yaml` | ❌ 未覆盖 | 追加 | LOW |
| `artifacts/` | ✅ 已覆盖 | 保留但建议手工提取 JSON | — |
| `save/` | ✅ 已覆盖 | 保留 | — |
| `data/` | ✅ 已覆盖 | 保留 | — |
| `.venv/` | ✅ 已覆盖 | 保留 | — |
| `docs/` 下内部子目录 | ❌ 未覆盖 | 追加(建议) | LOW (不影响安全,仅整洁) |
| `config/requirements-gx10.*` | ❌ 未覆盖 | 不改 .gitignore，改文件名 | MED |