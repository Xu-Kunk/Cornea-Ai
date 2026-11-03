# Patterns 文风基准指南（2026-08-22，lib-3 调研落盘）

> 对标 4 篇 Patterns 2025–2026 高影响力文章全文提取：scELMo (10.1016/j.patter.2025.101431)、BioLLM (10.1016/j.patter.2025.101326)、GSFM (10.1016/j.patter.2026.101565)、NESS (10.1016/j.patter.2026.101621)。
> 用途：Paper 1 v17 去 AI 味 + 润色的执行基准。用户已确认"完全对标 Patterns 实态"。

## 1. Summary（摘要）结构
- **gap-first**：背景/问题开场 → "Here/To address this, we present…" → 主张句 → 结尾三选一（含义+hedge / 愿景 / 资源交付）
- **摘要内零数字**（四篇实态全部如此）；主张用 demonstrate/reveal/achieves
- 阴性结果可入摘要但带因果解释（"…, likely due to…"）
- ≤150 词硬约束

## 2. Results 小节标题
- **陈述句信息型**（Cell Press 更推崇）：方法名作主语 + 现在时动词，8–12 词
  - 实例："BioLLM provides a unified framework for scalable scFM analysis" / "PCS-guided assessment reveals limitations of NE algorithms…"
- 备选名词短语型（3–6 词）："scELMo for clustering and batch effect correction"
- 禁编号；子标题可斜体陈述句

## 3. 段落逻辑
- **claim-first** 为主（主张先行，证据随后）；evidence-first 仅用于图表细节段
- 图表引用句式（按频率）："As shown in Figure X, …" / "Figure X shows…" / "Table X demonstrates…" / 句末括号 "(Figure 4a)"
- 方法引入段固定 "To test/assess/evaluate X, we …"

## 4. 时态与 claim 动词分工（严格）
- 方法/实验过去时（we trained/evaluated）；已知事实现在时
- **demonstrate**=自家已证主张 / **show**=图表呈现 / **reveal**=揭示局限或发现 / **indicate**=数据驱动观察 / **suggest**=推断（最弱）——不混用

## 5. Hedging 与域限定
- 因果归因："likely due to…"；推断用 suggest/may；比较弱化 "do not appear to"
- 域限定用**具体框架名**："in this benchmarking study" / "across these settings" / "under our fine-tuning protocol"（不用空洞的 "in this setting"）

## 6. 阴性结果话术（四种句式）
1. 因果归因："X lagged behind, likely due to …"
2. 例外框架："…generally improved, except for X, which did not…"
3. 趋势正常化："X failed to improve …, a trend consistent across …"
4. 比较反转："In contrast, Y consistently …"

## 7. 术语惯例
- **fine-tuning**（连字符）、**zero-shot**（连字符）、*in silico*（斜体）、foundation model 小写（句首大写）
- 首次出现给缩写："single-cell foundation models (scFMs)"
- ρ 保留（用户决策）：首次出现定义 "Spearman's rank correlation coefficient (ρ)"，之后全用 ρ
- 模型名按原名：Geneformer / scGPT / CellPLM / scFoundation / UCE

## 8. Bigger picture（Patterns 特有，≤300 词）四段式
1. 白话背景（零术语）："Single-cell technologies allow scientists to measure…"
2. "Yet," + 痛点："Yet, analyzing these massive datasets often demands…"
3. "Our method/study … offers/simplifies"（白话讲方法，允许一个类比，如"the same AI systems behind modern chatbots"）
4. 收益/愿景收尾（对社会价值）
- 零术语、零数字、≤300 词

## 9. 去 AI 味禁用清单
**禁用**："It is worth noting" / "In conclusion"（作全文收尾）/ delve / tapestry / pivotal / remarkably / strikingly / "a testament to" / 戏剧化 em-dash（仅限括号插入式）/ 修辞性空排比（仅实质内容列举可用）/ 摘要数字堆砌与 result-first 开场
**限用**：Notably ≤4 次/全文；Importantly 慎用；leverage/harness 仅技术义
**允许**：Overall/Moreover/However 正常段落衔接；"seamless/streamlined" 类词低密度不堆叠

## 10. 速查：本稿执行规则
1. Summary 重写为 gap-first + 零数字（当前 v16 摘要 149 词含数字，需重写）
2. Results 标题全部改陈述句信息型（方法名主语 + 现在时）
3. 段落 claim-first 重梳；图表引用统一句式
4. claim 动词按第 4 节分工全文重梳
5. 阴性结果节（CAFT/AA-FT/soft-Spearman/flow/GF LoRA OOM）按第 6 节四句式改写
6. 术语按第 7 节统一（fine-tuning 连字符全局替换核查）
7. Bigger picture 按第 8 节四段式重写
8. 禁用清单全文扫描清零
9. ρ 首次出现加定义句
10. Methods 轻度润色（仅去 AI 味，不改技术内容与数字）
