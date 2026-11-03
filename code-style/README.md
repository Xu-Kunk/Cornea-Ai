# code-style 模板 — R(tidyverse) + Python(PEP8) 统一代码风格

基于 [tidyverse style guide](https://style.tidyverse.org/) 与 [PEP8](https://peps.python.org/pep-0008/) 官方规范，为多项目工作流提供的可复制风格配置。

## 文件清单

| 文件 | 用途 | 复制到 |
|------|------|--------|
| `.lintr.R` | R 风格检查（lintr，tidyverse 规则） | R 项目根目录（**用 `.lintr.R`，勿用 `.lintr`**——DCF 格式在本机 locale 下解析失败） |
| `pyproject.toml.ruff` | Python 风格配置（ruff，PEP8+black 88 行宽） | 合并进项目 `pyproject.toml` 的 `[tool.ruff]` 段，或复制为 `ruff.toml` |
| `pre-commit-style.yaml` | pre-commit 风格钩子片段（ruff + lintr + styler） | 合并进项目 `.pre-commit-config.yaml` 的 `repos:` 末尾 |
| `iron-rules.md` | agent 自查铁律清单（10 条中英双语） | 放入 AGENTS.md 或 agent 系统提示 |

## 快速开始

**R 项目（如 DED-seq）**：
```bash
cp .lintr.R <项目根>/.lintr.R
cd <项目根> && Rscript -e 'lintr::lint_package()'   # 检查
Rscript -e 'styler::style_pkg()'                     # 自动格式化
```

**Python 项目**：
```bash
ruff check .          # lint（读项目 pyproject.toml 的 [tool.ruff]）
ruff check . --fix    # 自动修复
ruff format .         # 格式化
```

**pre-commit 一键集成**：
```yaml
# 将 pre-commit-style.yaml 内容并入 .pre-commit-config.yaml 后：
pre-commit install && pre-commit run --all-files
```

## 风格铁律速查（来自官方规范）

1. 两语言统一 **snake_case**（类名除外用 PascalCase）
2. R 用 `<-` 赋值、`TRUE/FALSE`、双引号、`|>` 原生管道一步一行
3. Python 用双引号、`is None`（禁 `== None`）、类型注解
4. 行宽：R 80 / Python 88（black 默认）
5. 导入置顶集中：R 的 `library()` 在文件头，Python 按 isort 三组排序
6. 注释 `# ` 后加空格；Python 公共对象写 PEP257 docstring
7. 文件命名：小写、`-`/`_` 分隔、日期 ISO8601、序号前缀，禁用 "final" 字样

> 维护说明：本模板由 2026-08-12 会话创建（tidyverse+PEP8 调研），配置变更请同步更新本 README。
> 维护说明（2026-08-13）：续搜增补了 ruff 规则集（S/RUF/PD/NPY/ANN/D/TCH/FA/DTZ/ICN/PTH/C90/ERA/INP）与 lintr 增补 linter；完整 agent 自查清单见 [iron-rules.md](iron-rules.md)。
