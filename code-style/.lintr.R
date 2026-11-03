# .lintr.R - R code style config (aligned with tidyverse style guide)
# Usage: copy to R project root as ".lintr.R", then run lintr::lint_package() or pre-commit
# Note: use .lintr.R (R expression) not .lintr (DCF) - DCF parsing is locale-fragile
# Reference: https://style.tidyverse.org/ | https://lintr.r-lib.org/
linters <- linters_with_defaults(
  # linters_with_defaults() 保留 lintr 3.4.0 全部 26 个默认 linter，
  # 下列参数是在默认之上新增/覆盖（同名参数覆盖默认值，新 linter 为新增）。
  # Line length 80 (tidyverse official)
  line_length_linter(80L),
  # Naming: snake_case (lowercase with underscores)
  object_name_linter(styles = "snake_case"),
  # Assignment must use <- (no = assignment; allow_trailing = FALSE bans x <- y <- z)
  assignment_linter(allow_trailing = FALSE),
  # Spaces around infix operators
  infix_spaces_linter(),
  # Pipe |>/%>% continuation: indent 2 spaces after line break
  pipe_continuation_linter(),
  # Space after comma, no leading commas
  commas_linter(),
  # library() calls at top of file
  library_call_linter(),
  # ── 增补（默认之外，tidyverse 强化）──
  # 禁 sapply(类型不稳应用 vapply/purrr::map_*)、setwd 等；modify_defaults 移除 library/require（脚本项目需显式 library()）
  undesirable_function_linter(
    fun = modify_defaults(default_undesirable_functions, library = NULL, require = NULL)
  ),
  # 禁 <<-、:::、>> 及右赋值 -> / ->>
  undesirable_operator_linter(
    op = modify_defaults(default_undesirable_operators, "->", "->>")
  ),
  # 未使用的 library()
  unused_import_linter(),
  # 圈复杂度门（阈值 15，防过深嵌套）
  cyclocomp_linter(complexity_limit = 15L),
  # 0.5 不写 .5（官方规范）
  numeric_leading_zero_linter(),
  # class(x) == "y" → inherits(x, "y")
  class_equals_linter(),
  # if (x <- ...) 隐式赋值
  implicit_assignment_linter(),
  # 禁 stringsAsFactors = TRUE 等坑参数
  function_argument_linter(),
  # stop("字面量") 非 stop(paste(...))
  condition_message_linter(),
  # sprintf / paste 字符串构建规范
  sprintf_linter(),
  paste_linter(),
  # length(x) > 0 显式判定（官方规范）
  length_test_linter(),
  # 覆盖基础函数名（如 dir <-）
  object_overwrite_linter(),
  # TODO 注释管理
  todo_comment_linter()
)

exclusions <- list(
  "renv" = list(),
  "packrat" = list(),
  ".Rhistory" = list(),
  ".RData" = list()
)
