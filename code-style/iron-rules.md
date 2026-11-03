# 代码风格铁律清单 Code Style Iron Rules (Python + R)

> 用途：放入 opencode 的 AGENTS.md 或 agent 系统提示。
> 用法：agent 写码时自查；提交前以 ruff/lintr 工具输出为准（机器可验证项不依赖 agent 记忆）。

1. **命名 Naming** — snake_case 一切（Python 类名 PascalCase；R 禁点号）；文件小写、`_`/`-` 分隔、无空格；日期 ISO8601、序号补零、禁 "final"。检查：R 过 object_name_linter；Python 过 N 类；变量名有意义（pandas 禁 `df`）。

2. **行宽 Line width** — Python 88（ruff/black 默认；docstring/注释 ≤72）；R 80。检查：E501 / line_length_linter。超宽先断行，不硬压。

3. **赋值 Assignment** — R 用 `<-`，禁 `=`（仅命名参数）、禁 `<<-`、`->`；Python 用 `=`。检查：assignment_linter(allow_trailing = FALSE) / undesirable_operator_linter。

4. **布尔与比较 Booleans** — R 用 TRUE/FALSE（禁 T/F）、if 内用 &&/||、`x == NA` 写 `is.na(x)`、`length(x) > 0` 显式；Python 用 `is None`（禁 `== None`）。检查：T_and_F_symbol / vector_logic / equals_na / E711。

5. **空格与括号 Spacing** — 逗号后一空格；二元运算符两侧空格；`(` 前无空格；R 的 if/for/while 后空格。格式交给 formatter（ruff format / styler），不要手排；禁行尾空格。

6. **引号与标点 Quotes** — 统一双引号；R 禁分号、禁注释掉的代码、注释 `# ` 后空格。检查：quotes / semicolon / commented_code / ERA001。

7. **导入 Imports** — 文件头集中；Python isort 三组（stdlib/第三方/本地）+ `import numpy as np`、`import pandas as pd`、`import matplotlib.pyplot as plt`；仅类型用的导入放 `TYPE_CHECKING` 块；R 的 library() 置顶、禁未用导入。检查：I / ICN / TCH / library_call / unused_import。

8. **函数与注解 Functions** — Python 函数参数与返回值全量类型注解；禁可变默认参数（`def f(x=[])` → `x=None`）；禁默认参数内调函数；R 函数名用动词、长调用一行一参数。检查：ANN201/ANN001 / B006 / B008 / function_argument_linter。

9. **文档 Docstrings** — 公共对象写 PEP257 docstring，用 NumPy 风格（Parameters / Returns / Raises 节）；注释解释"为什么"而非"是什么"。检查：D 系列（pydocstyle convention = "numpy"）。

10. **管道与数据科学陷阱 Pipes & DS pitfalls** — R 用原生 `|>` 一步一行、续行缩进 2、管道在行尾，多对象/需命名中间结果时不硬套管道；Python 禁 >3 层嵌套（提炼函数）、pandas 用 .loc 禁链式索引、datetime 必须带时区、路径用 pathlib、NumPy 2.0 禁已删别名（np.float 等）。检查：pipe_consistency / pipe_continuation / PD015 / DTZ / PTH / NPY201 / C90。
