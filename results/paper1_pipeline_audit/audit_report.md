# Paper1 全部计算管线系统性 Bug 审计报告

日期: 2026-08-05 · 审计方式: 只读 + CPU 独立数值复算 (不调用被测代码) · 解释器 /home/dq/venv-ds/bin/python3
范围: Paper1 已锁结果全部计算管线。未改任何代码/数据/结果。新文件仅写入本目录。

审计方法: 对每个脚本读代码 → 关键数值用独立实现重算 → 与已锁 JSON 比对 (容差 1e-6 / 按指标约定)；
对"训练时做了什么"类问题查 run 日志目标损失痕迹 + callback 注册 + 框架源码。

---

## 0. 核心结论 (摘要)

| # | 结论 | 严重度 |
|---|------|--------|
| 1 | **ECE 分箱 bug (新发现)**: `evaluation.expected_calibration_error` bin 0 用 `conf > 0.0` (无上界) → bin 0 覆盖 (0, 1.0] 全部细胞 → 权重和 ≈ 2 → **所有已锁 ECE 数值被系统性高估** (frozen 0.5967 → 真值 ≈ 0.301; 微调臂 ≈ +0.03~0.06 高估)。影响: paper1.tex L106/L124、Table S1、图 S1/S2/S3 全部 ECE。 | **P0** |
| 2 | **HPA-FT 缺 callback (新发现, CAFT 式 bug #3)**: `HPATrainer` 继承 CAFTTrainer 的死代码 `on_epoch_begin`, `build_hpa_trainer`/`run_hp_aft.py` **均未注册 callback 也未调用 `_precompute_centroids`** → 真实 HPA-FT 训练 (mode=lwc/both, λ>0) 的 LWC 连续性项**永不执行** (λ 只对 freeze 策略有效)。feasibility 脚本因显式调用 `_precompute_centroids` 而"通过", 掩盖了该缺陷。目前尚无真实 HPA 网格 run, 属**潜伏 bug**。 | **P0 (启动前必修)** |
| 3 | **稿件"CAFT 部分恢复 ρ"因果归因不成立** (已知 fix-7 的后果延伸): λ=0 与 λ=0.5 的 6 个 run 代码/数据/seed 逐字节相同 (已核 md5 + 全数据集 hash 一致), 连续性项从未运行 → naive ρ (−0.042/−0.036/+0.054) 与 "CAFT" ρ (+0.191/+0.230/+0.210) 的差异是**运行间 bf16 非确定性/环境差异**, 不是 λ 的因果效应。paper1.tex L48/L82/L90/L114/L176 的"partial restoration / continuity objective acted on..."表述需重写 (诚实阴性结论不受影响, 但"部分恢复"叙述不可再归因于 CAFT 目标)。 | **P1** |
| 4 | 其余核心指标 (donor-level macro-F1、bootstrap CI、LAA、marker ρ、L12 分层 ρ、TAC 敏感性、percell npz 一致性、MMD、因果头消融) 全部经独立复算**位级一致 (diff=0.0)**。 | — |

---

## 1. 完整审计表

| 管线 | 声称 | 实际 | 判定 | 证据 |
|------|------|------|------|------|
| **1. frozen 评估** (run_tier_b_baselines arm2 + run_tier_b_eval) | | | | |
| CLS 提取: position-0 CLS + L2 归一化 fp16 | 3 个脚本 (e0_axis1_baseline / run_tier_b_baselines / measure_layerwise_rho) 实现一致: `last_hidden_state[:,0]` + `F.normalize` + fp16 autocast + right-pad/mask | percell_frozen_gf.npz emb_ext L2 范数 = 1.0 (min 0.9999999, max 1.0000001) 位级确认 | **OK** | npz 范数检查; 三脚本 diff 一致 |
| linear probe 只训 head | LogisticRegression 在冻结嵌入上 (StandardScaler + lbfgs, max_iter=5000, seed 42) | 模型参数从未更新, 仅拟合探针 | **OK** | 代码审查 (arm2_frozen_gf) |
| frozen macro-F1 0.4421 (CI 0.3749–0.5093) | 独立实现 (per-donor macro-F1 + bootstrap n_boot=1000 seed=42) 重算 | macro_f1=0.442104..., CI=(0.374902, 0.509306), **diff vs 已锁 = 0.0** | **OK** | metric_recompute.json; 手算 per_donor {donor_1: 0.5093, donor_2: 0.3749} |
| frozen ρ 0.8064 | 独立实现 (centroid PCA 伪时间 + Spearman) 重算 | ρ=0.806361, **diff vs 已锁 = 0.0**; 且与另一次 GPU 推理 (layerwise L12) 交叉一致 diff=0.0 | **OK** | 同上 + layerwise_rho_frozen_gf.json sanity_gate diff=0.0 |
| frozen ECE 0.5967 | 独立实现两种分箱 | 项目分箱复算=0.596687 (一致); 标准分箱 [lo,hi) = **0.300961** → 报告值高估 +0.2957 | **BUG** | ece 分箱 bug 见 §3 |
| 探针/嵌入全量复跑 | — | 需要 GPU (99606+14775 cells 前向) | **需 GPU 后续验证** | 未重跑; 一致性链 (percell gate diff=0.0, layerwise 二次推理 diff=0.0) 已闭环 |
| **2. 分层测量** (measure_layerwise_rho.py) | | | | |
| sanity gate 覆盖全部 7 模型 | summary.json: 7/7 模型 sanity_gate passed=True, abs_diff=0.0 (bit-exact) | 与 results JSON (frozen + 6 FT) 全部 diff=0.0 | **OK** | summary.json sanity_gate |
| 脚本未被改坏 | 脚本 mtime 2026-08-03 11:44 < 结果生成 12:47; 结果内含 gate 记录 | 无 git 仓库可 diff, 但时间戳证明脚本先于结果存在 | **OK** | stat mtime |
| L0 NaN 处理 | L0 (embedding output) CLS 对全部细胞恒等 (特殊 token 嵌入) → 质心零方差 → rho=NaN | 7/7 模型 L0=NaN, 正确行为 | **OK** | summary.json L0 全 NaN |
| L1–L11 中间层 ρ | 同一 metric 函数 (marker_gradient_correlation), 未逐层独立复算 (需 GPU hidden states) | gate 只覆盖 L12; L1–L11 无独立交叉验证 | **风险(低)** | 需 GPU; 函数已在 L12 位级验证, 层间同函数同标签 |
| **3. evaluation.py 指标** | | | | |
| donor_level_macro_f1 (bootstrap 1000, seed 42) | 独立实现: per-donor macro-F1 → mean; bootstrap 重抽样 donor id → percentile 2.5/97.5 | 7/7 模型 macro_f1/ci 全部 diff=0.0 | **OK** | metric_recompute.json |
| 2-donor 统计问题 | CI = [min, max] of 2 个 per-donor 值 (bootstrap 分布退化) | 实现正确但统计意义弱: **CI 不能视为不确定性区间** | **风险(统计报告)** | frozen ci=[0.3749,0.5093] == per_donor 两值 |
| LAA (NN 邻接或同层) | 与稿件定义一致: NN(k=2) 标签 |Δ|<=1 比例 (含同层) | 独立复算 7/7 diff=0.0 | **OK** | paper1.tex L239 vs evaluation.py L431-458 |
| ECE 分箱 | 见 §3 | | **BUG** | 见 §3 |
| **4. percell npz** (measure_percell_probs.py) | | | | |
| meta.json macro_f1 0.654186 (naive_s42) | npz 数组 → 独立重算 macro=0.6541863606043621 | 与 results_0.0_42.json (epoch-3) 与 meta 三方完全一致 (diff=0.0) | **OK** | metric_recompute.json |
| emb_ext/probs_ext 与 sanity_gate 一致 | 7/7 模型全部 6 项指标 (macro/ci/laa/rho/ece) 重算 diff=0.0 | 行对齐: npz y_true_ext == h5ad 上皮标签 (np.array_equal=True) | **OK** | verify + alignment 检查 |
| --epoch2 门禁 | epoch-3 checkpoint 重跑必须位级一致 | epoch2_run.log 全部 PASS (diff=0.000e+00) | **OK** | epoch2_run.log |
| **5. 新代码** (fix-7/10 产物) | | | | |
| CAFT 连续性 (已知 fix-7) | transformers 4.57 `Trainer` **无 on_epoch_begin 方法** (框架源码证实), CallbackHandler 只派发 callback 对象 | CAFTTrainer.on_epoch_begin 死代码; 6 run 无 L_cont 记录; layer_centroids 恒 None → λ=0.5 ≡ λ=0 (纯 CE) | **BUG (已知)** | CallbackHandler 源码 + trainer_state 无 L_cont |
| ordinal fix | `OrdinalEpochBeginCallback` 经 add_callback 注册 → 框架正常派发; ragged 安全 padding; epi-only 转发; bf16 质心 | 结构正确; 但当前仅有 λ=0 run 完成 (λ=0.5 在 GPU 队列等待) | **OK (结构) / 待实证** | ordinal.py L297-358; ordinal_lam0.0_seed42.log 只有 loss (λ=0 属预期) |
| **HPA-FT fix** | hp_aft.py 声称继承"epoch-begin 质心预计算" | **build_hpa_trainer 未注册 callback, run_hp_aft.py 未调用 _precompute_centroids**; 且该预计算对 ragged input_ids 用 torch.tensor 会崩溃 | **BUG (潜伏, 新发现)** | grep 全项目仅 feasibility_hp_aft.py L155 显式调用; hp_aft.py L243-293 |
| energy-anchor fix | 参考质心在 __init__ 同步预计算 (无需 epoch hook); 冻结 + stop-grad; MMD 位级验证 | percell MMD 独立复算 sigma=1.326052 / mmd=0.575198 与已锁完全一致 | **OK** | 独立 MMD 复算 |
| energy-anchor 设计注意 | 质心**未 L2 归一化** (范数 ≈ 27, 原始 CLS 尺度) vs CAFT/ordinal 归一化 | MMD 对标度敏感 → 存在"缩小范数而非靠近质心"的失效模式风险 | **风险(方法论)** | feasibility 日志 norms=[6.5(子集)]; 代码 L218-221 无 normalize |
| energy drift A5 layerwise | 声称 13 层漂移曲线 | CPU 链 A5 **FAILED**: `_collect_extra_arm` 硬编码 num_labels=31 加载 ordinal (3 类) checkpoint → size mismatch; 现存 drift 产物全部 source=percell (仅 final layer) | **BUG(交付缺失)** | cpu_chain.log L87-90; summary.json source=percell |
| **6. TAC 敏感性 / L12 bootstrap CI / DPT** | | | | |
| TAC 敏感性数字 | 稿件: naive 3 类 0.8599–0.8763 (+0.21–0.22), CAFT 0.8479–0.8786, frozen 0.5921 (+0.15) | 独立复算完全一致 (全部命中稿件区间) | **OK** | 独立 F1 复算 |
| L12 bootstrap CI | 独立重实现 (同 seed 42, n_boot=1000, cell-level) | frozen CI=(0.791751, 0.816422) 与 CSV **位级一致** (boot_se 0.006338 一致) | **OK** | /tmp/opencode/boot_ci_ind.log vs figure7 CSV |
| 轨迹推断 DPT | 参数已记录于 CSV (pca-30, k=30, n_dcs=10, root=frozen_ref_min); frozen 0.850 > naive 0.335, CAFT 0.398 居中 → 假设成立 | bootstrap_spearman_ci 模式与已验证 L12 相同; DPT 本体 (scanpy, 14775×768) 未 CPU 重跑 | **OK (代码审查) / 重跑需 GPU** | table_trajectory_continuity.csv |
| 因果头消融 A/B/C | A==B 位级一致 (身体致坍缩, 头无关); C==frozen | sanity gate 3/3 PASS (1e-6) | **OK** | causal_head_ablation_summary.json + run.log |

---

## 2. 判定汇总

- **OK**: frozen macro-F1/ρ/LAA、donor bootstrap、分层 L12 7/7 gate、percell npz 三方一致、TAC 敏感性、MMD、因果头消融、ordinal 修复结构、energy-anchor 结构
- **BUG**: ECE 分箱 (新, P0); HPA-FT callback 缺失 (新, P0 潜伏); CAFT 死钩子 (已知); energy-drift A5 交付失败 (P1)
- **风险**: 2-donor bootstrap CI 统计意义; L1–L11 无独立交叉验证; energy-anchor 质心未归一化; 稿件"CAFT 部分恢复"归因
- **需 GPU 后续验证**: frozen 探针全量重跑、分层 L1–L11、DPT 重跑

## 3. ECE 分箱 bug 详情 (新发现, P0)

`corneal_fm/evaluation.py` L527-534:
```python
edges = np.linspace(0.0, 1.0, n_bins + 1)
for b in range(n_bins):
    lo, hi = edges[b], edges[b + 1]
    mask = conf > lo if b == 0 else (conf > lo) & (conf <= hi)   # ← b=0 缺 (conf <= hi)
```
b=0 时 lo=0.0 → `conf > 0.0` 选中**全部**细胞 → bin 0 覆盖 (0, 1.0] 而非 (0, 0.1]。权重和 ≈ 1 + P(conf>0.1) ≈ 2, 且每细胞被重复计入。ECE 被高估 ≈ |mean_acc − mean_conf| (全局项)。

同一 bug 复制到 `measure_percell_probs._reliability_bins` (L626-630, 图 S1 bin 0 同样全量)。

影响面:
- 全部 13 个 results/epoch2 JSON 的 ECE (frozen 0.5967→0.3010; naive 0.0763→0.0460, 0.1248→0.0690, 0.1013→0.0574; caft 0.1282→0.0695, 0.0922→0.0526, 0.1047→0.0580; hvg_pca 0.0117 未能量化—无 percell 数组)
- paper1.tex L106 "ECE 0.5967"/"ECE 0.0117"、L124 表、图 S1/S2/S3、Table S1
- 方向性结论保留 (frozen 校准差, 微调较好; naive vs CAFT 相对关系不变), 但**所有绝对值错误**, 审稿人用 released percell 数组复算即会暴露。

## 4. HPA-FT 潜伏 bug 详情 (新发现, P0)

- 证据: `grep -rn "add_callback\|_precompute_centroids" scripts/ corneal_fm/` → 仅 feasibility_hp_aft.py L155 显式调用; hp_aft.py 无 callback 注册; run_hp_aft.py L278 build 后直接 train()。
- 机制: HPATrainer 继承 CAFTTrainer 的死 `on_epoch_begin`; transformers 4.57 Trainer 类本身无该方法 (源码证实); `cont_layer_centroids` 恒 None → compute_loss 中 need_hidden=False → loss = loss_cls。
- 后果: 若以 `--mode lwc/both --lambda 0.5` 启动真实 run, LWC 连续性静默不生效 (与 CAFT bug 完全相同), 且若有人"修复"为直接调用 `_precompute_centroids`, 还会因 ragged `torch.tensor(batch["input_ids"])` 崩溃 (L268)。
- 修复方向 (参照 ordinal.py): 注册真实 `TrainerCallback` + ragged-safe padding 预计算。
- 现状: save/ 下仅有 hp_aft_feasibility (报告, 显式调用预计算所以"21/21 通过"), 无真实网格 run → 尚未污染任何已锁结果, 但**任何后续 HPA run 都会静默无连续性**。

## 5. 稿件"CAFT 部分恢复 ρ"归因问题 (P1)

证据链:
1. λ=0 / λ=0.5 的 caft_gf.py 归档脚本 md5 相同, 与当前 scripts/caft_gf.py 相同;
2. 两 run 的 prepared train 数据集 (79684 cells) 全序 hash 一致;
3. 配置 (lr/bs/epochs/seed/manifest) 逐项相同;
4. 连续性项从未执行 (框架级死代码 + 无 L_cont 日志) → 训练目标 = 纯 CE;
5. 但 L12 ρ: naive −0.042/−0.036/+0.054 (均值 −0.008) vs "CAFT" +0.191/+0.230/+0.210 (均值 +0.210)。

→ λ=0.5 组 3/3 为正且与 naive 组不重叠 (3v3 完全分离, 双侧 Mann-Whitney p≈0.05), 但该差异只能归因于运行间 bf16 非确定性/环境差异 (chaotic 累积 7473 步), **不能归因于连续性目标**。paper1.tex L48 (摘要 "CAFT partially restores the gradient (ρ → +0.210; positive in all seeds)")、L82、L90 ("significantly distinct from both naive and frozen")、L114 (promotion criterion 1 "partial")、L176 ("the continuity objective acted on a readout already overwritten by the task" — 该句事实错误: 目标从未作用) 均需重写。诚实阴性结论 (分类无改善, 不 promote) 不受影响。

## 6. 其他发现

| 发现 | 位置 | 判定 | 影响 |
|------|------|------|------|
| `_collect_extra_arm` 硬编码 num_labels=31, 遇 ordinal (3 类) checkpoint 崩溃 | measure_energy_drift.py L466-484 | BUG | A5 layerwise 漂移曲线缺失 (交付不完整); 需按 checkpoint 实际类数加载或跳过 |
| energy-anchor 质心未 L2 归一化 | energy_anchor.py L218-221 | 风险 | 与 CAFT/ordinal 不一致; MMD 对标度敏感, 存在范数收缩失效模式 |
| 2-donor bootstrap CI 退化 | evaluation.py | 风险 | CI=[min,max] of 2 值, 论文中 CI 不宜作为不确定性区间解读; 建议在稿件中显式说明 |
| layerwise JSON 写非严格 JSON (NaN 字面量) | measure_layerwise_rho.py | 风险(低) | RFC8259 不合法; 严格解析器会失败 (Python 可读) |
| 摘要 "17,587 cells" = raw_shape (config L34) vs 15,827 post-QC / 14,775 上皮 | paper1.tex L70 | 文档 | 建议写明 "raw 17,587 → 15,827 post-QC → 14,775 epithelial evaluated" |
| 稿件内部数字核对 | paper1.tex | OK | naive 0.6492 / CAFT 0.6451 / frozen 0.4421 / hvg 0.6192 / TAC 全部与复算一致 |

## 7. 建议修复优先级

| 优先级 | 事项 | 工作量 |
|--------|------|--------|
| **P0** | 修复 `expected_calibration_error` bin 0 上界; 重算全部 ECE (CPU 可做, 用 percell npz); 更新 paper1.tex L106/L124 + Table S1 + 图 S1/S2/S3 数据 | 小 (函数 1 行 + 重算脚本) |
| **P0** | 在 `build_hpa_trainer`/`run_hp_aft.py` 注册真实 epoch-begin callback (镜像 ordinal.py) + ragged-safe padding; 启动任何真实 HPA run 前必须验证 L_cont 进入日志 | 中 |
| **P1** | 重写稿件 CAFT "partial restoration" 归因段落 (L48/82/90/114/176): 表述为"λ=0.5 组检查点 ρ 偏高, 但因连续性目标未执行, 不能归因于 CAFT"; 诚实阴性框架保留 | 小 (文字) |
| **P1** | 修复 `_collect_extra_arm` 类数处理, 重跑 A5 layerwise 能量漂移 (GPU) | 中 |
| **P2** | energy-anchor: 决定质心归一化策略并文档化 (MMD 标度敏感性); 2-donor CI 稿件显式说明; layerwise JSON 写严格 JSON; 摘要细胞数措辞 | 小 |

## 8. 需 GPU 后续验证 (本审计未覆盖)

1. frozen 探针端到端重跑 (train 99,606 + ext 14,775 嵌入) — 一致性链已闭环, 但独立重算需 GPU。
2. measure_layerwise_rho L1–L11 中间层 (hidden_states 1..11) — 需 GPU 前向; L12 已位级验证。
3. DPT 轨迹推断重跑 (14775×768 PCA + kNN, CPU 极慢 / GPU 更快)。
4. ordinal λ=0.5 训练 (GPU 队列中) — 完成后应验证 L_ord/L_cont 日志进入 trainer_state, 且质心 callback 每 epoch 触发。
5. energy-anchor λ=0.5 训练 (GPU 队列中) — 完成后验证 L_anchor_mmd2 日志 + stage_centroids 冻结 (requires_grad=False)。

## 9. 审计产物

- artifacts/paper1_pipeline_audit/verify_metrics_independent.py — 独立指标复算脚本 (7 模型 × 6 指标, diff=0.0)
- artifacts/paper1_pipeline_audit/metric_recompute.json — 复算明细 (含 ece 标准分箱对照)
- 本报告
