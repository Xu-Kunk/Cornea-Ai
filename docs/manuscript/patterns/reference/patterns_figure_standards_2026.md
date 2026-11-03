# Patterns 图件审美与排版基准（2026-08-18，lib-1 调研 + obs-1 审计合成）

> 用途：Paper 1 图件改造的执行基准清单。来源可验证（官方 URL + 4 篇对标论文 DOI）。

## 一、官方硬约束（Cell Press Figure Guidelines + Patterns FFC）

1. 画布 **178 mm 满宽**（Cell Press 双栏宽，=7.01 in；见 `FIGURE_SYSTEM.md` 的构建契约）× 高度 ≤8 in，全部 panel 单页，按期刊排版顺序组织
   > 2026-10-22 更正：此前写的 ≤6.5 in 系保守值，与 `FIGURE_SYSTEM.md` 冲突；以 178 mm 为准，本行已改。
2. 分辨率：彩色 ≥300 dpi / 黑白 ≥500 dpi / 线稿 ≥1000 dpi（按印刷尺寸）
3. 字体仅 Arial（本项目用 Liberation Sans，度量等价，cover note 已说明）；正文文字 6–8 pt（印刷尺寸）
4. Panel 标签大写字母（A, B, C），同字号同位置（左上）
5. RGB；禁红+绿并列；灰度填充 10–80% 且相邻 ≥20% 差异
6. 线宽 0.5–1.5 pt
7. 每图 ≤20MB；TIFF/PDF 优先；图注/标题不入图文件；最终命名 Figure 1.tif 式
8. 误差棒含义 + 统计检验必须写入图注；显著性标注全图统一
9. 主图 5–6 张，Fig1 = 方法总览 schematic；补充图建议 ≤20 张，标题关联主图（"Related to Figure X"）
10. 图注：描述性标题 + panel 逐项说明 + 统计细节 + See also 交叉引用

## 二、审美升级基准（4 篇 Patterns 2025–26 单细胞论文实态：scELMo 10.1016/j.patter.2025.101431 / BioLLM 10.1016/j.patter.2025.101326 / GSFM 10.1016/j.patter.2026.101565 / NESS 10.1016/j.patter.2026.101621）

11. 白底 + Okabe–Ito 色盲安全色板（强调色高饱和是设计选择；整幅白底印象的中位饱和度为 0.00–0.10）；禁深色背景、禁霓虹堆叠、禁红绿并列
   > 2026-10-22 更正："低饱和" 指整幅印象，不指强调色；臂色固定为 frozen 灰 / CE-naive 蓝 / ordinal head 绿 / LoRA 粉紫 / ordinal + LoRA 橙红 / ordinal + continuity 浅蓝。
12. 统一图标语言与色板：schematic 与数据图共享色板；同一生物学概念跨图颜色一致
13. 方法总览图：扁平方框-箭头、图标化、≤6 步骤块、单向视觉流、每块 ≤5 词标签
14. Panel 间距均匀（≥0.15 in），垂直留白克制
15. 字体层级：panel 标签 8pt 粗体 > 轴标题 7pt > 刻度/图例 6pt，全稿统一
16. UMAP/t-SNE：点大小/透明度统一，大样本低透明度防过绘；连续变量 viridis 类，禁 rainbow/jet；图例不遮数据
17. 柱图填充色 + 细边框（0.5–1pt）；误差棒细线 + 端帽；显著性标注在柱顶
18. 图例置图内角落或图上方一行，与 panel 对齐
19. Graphical Abstract：1200×1200 px @300dpi 方形、Arial 8–12pt、单 panel、无数据项、单一信息点、柔和配色、禁通用生成式 AI（Patterns 可选但建议做）
20. GA 与主图均做色盲复核（禁红/绿、红/黑对比）

## 三、备选期刊差异（转投适配）

- Genome Biology：无 Arial 强制、panel 大写、图宽 600/1200px、无 GA 要求 → 现有图件基本可复用
- Nature Methods/Comms：panel 小写粗体、5–7pt、需 Extended Data 体系 → 改造量大

## 四、本项目锁定语义（改造不得破坏）

- 臂色：frozen=灰、CE-naive=蓝、ordinal head=绿、LoRA=粉紫、ordinal + LoRA=橙红（2026-10-22 更新：CAFT/axis-anchored/soft-Spearman 等已从全稿删除，原 `CAFT=橙` 与 hatch 区分规则作废；hatch 亦因 0.1 pt 印刷不可复现被取消，改为实心浅色 + ≥0.5 pt 描边）
- 细胞类型序：LSC → TAC → basal → differentiated（全稿 UMAP legend 统一）
- donor 色：blue/orange-yellow 系（待全稿统一钉死）
- QA 锁定数字：frozen 同语料 ρ=0.806、GSE153515=14,775 上皮细胞、最小字号 ≥6.5pt
