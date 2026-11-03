# GX10 部署清单 — Corneal-FM CAFT Training (v2, 2026-07-23 重写)

> **硬件**: ASUS Ascent GX10 ＝ NVIDIA DGX Spark (GB10 Grace Blackwell, 20 核 ARM **aarch64**, 128GB 统一内存, sm_121)
> **系统**: DGX OS 7 (Ubuntu 24.04.4 LTS), CUDA **13.0**, 驱动 580.159.03 — 已确认 (`Build Skill/GX10-STATUS.md`)
> **目标**: RAPIDS 重处理数据 → CAFT-GF + CAFT-scGPT fine-tune + E1-E5 benchmark + 消融 A1-A4 + 表皮 E6
> **关键约束**: Blackwell sm_121 只能用 CUDA 13 wheel；conda/cu12-x86 路线在 GX10 上不可用
> **代码状态**: 24 处硬编码 `D:\Codex\...` 已迁移为 `CORNEAL_FM_DATA` / `__file__` 派生路径（本次完成，跨平台），代码零改动即可在 GX10 运行

---

## ⚠️ 与旧版清单的关键差异（旧版已废弃）

| 项 | 旧版（错误） | 本版（正确） |
|----|------------|------------|
| PyTorch | conda `pytorch-cuda=12.1` / 默认 `pip install torch` | **NGC 容器内置** 或 **torch 2.13.0+cu130**（默认 pip wheel 是 CUDA12/x86，跑不了 GB10） |
| 环境 | conda env | NGC Docker `pytorch:25.11-py3` 主 / 原生 `venv-ds` 备 |
| 注意力 | 未提 | **禁装 flash-attn**，用 PyTorch SDPA（Blackwell 上更快） |
| 内存 | 仅"减 batch" | 统一内存 OOM 会**整机僵尸**，需 swapoff + cgroup + vllm-stop |
| 数据迁移 | Windows→GX10 传输 | ✅ 已迁移完毕，直接验证 |
| 预处理 | 沿用 Windows 数据 | **RAPIDS-singlecell 在 GX10 重处理** + CS_before 重定基线 |

---

## Phase 1: 环境验证 (~30min)

```bash
nvidia-smi                      # GB10, Memory-Usage 显示 "Not Supported" 是正常(统一内存无独立显存)
uname -m                        # → aarch64
nvcc --version                  # CUDA 13.0
source ~/venv-ds/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
#   期望: 2.13.0+cu130 True (12, 1)     ← (12,1)=sm_121, is_available 必须 True
```

- ✅ 已就绪: `HF_ENDPOINT=https://hf-mirror.com`（GX10 已全局配置）、Docker GPU 直通、pip 清华源、apt USTC(ubuntu-ports)
- 若用 Docker 基座替代 venv-ds:
  ```bash
  docker pull nvcr.io/nvidia/pytorch:25.11-py3    # GX10 在用, 含 CUDA13 + torch + FlashInfer
  ```

## Phase 2: 科学栈 + 模型安装 (容器内或 venv-ds)

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -r config/requirements-gx10.txt        # 不含 torch(已就绪); 不装 flash-attn

# geneformer (jkobject fork; 以它自报的 transformers<5 pin 为准 → 全栈单 env)
pip install git+https://github.com/jkobject/geneformer.git
python -c "import transformers, geneformer; print('transformers', transformers.__version__)"
#   ⚠️ 部署时确认 transformers 版本: Geneformer 需 <5 (清单基线 ~4.45.2);
#     AutoModel (scLong) 与 scGPT TransformerModel 均兼容 4.x → 单 env 可行

# scGPT (flash-attn 可选, aarch64 跳过 → 自动 SDPA fallback)
pip install git+https://github.com/bowang-lab/scGPT.git

# Geneformer 4 个 pkl (hf-mirror 下载到 geneformer 包目录)
python - <<'PY'
import geneformer, os
print("geneformer pkg dir:", geneformer.__path__[0])
PY
#   从 https://hf-mirror.com/ctheodoris/Geneformer 下载:
#     gene_median_dictionary.pkl / token_dictionary_gc104M.pkl / 等 4 个
#     放入上面打印的 geneformer 包目录
#   scGPT whole-human 权重: https://hf-mirror.com/bowang-lab/scGPT
```

## Phase 3: 数据完整性验证 (go/no-go 门槛 1)

```bash
# 数据已在 ~/Codex/Ai/data (✅ 已迁移, 405GB Codex 树之一)
cd ~/Codex/Ai
source ~/venv-ds/bin/activate
python tests/verify_preprocessing.py     # shape: 100k×35477 / 20k×35477 + orthologs + vocab
python tests/verify_h5ad.py              # 源 h5ad 结构完整性 (backed, 不全量载入)
```

## Phase 4: RAPIDS-singlecell 重新预处理 (GX10 新能力)

```bash
pip install rapids-singlecell            # GPU 加速 scanpy, aarch64/cu130 wheel
# 需求: ~40GB 空闲统一内存 (docker/加速数据处理)
```

- 用 RAPIDS 在 GPU 上重跑 QC → HVG → normalize → PCA → UMAP → 聚类
- ⚠️ **复现性**: RAPIDS 用**精确 kNN 图 + GPU 浮点**, 与 CPU scanpy 有**微小数值差异**
  - `CS_before` 在 GX10 重处理数据上**重算 → 成为新基线**
  - Windows CS_before (B=0.5477 / A=0.6338) **降级为 sanity range, 非精确目标**
  - GPR = CS_after / CS_before 在**同一份 GX10 数据**上计算 → 内部自洽
  - 复现靠: seed=42 + 锁定环境 + STAR Methods 记录 RAPIDS 版本与全部参数
- 后续 tokenize/finetune/benchmark **全部基于这份 RAPIDS 重处理数据**

## Phase 5: 微调

```bash
vllm-stop                                # ⚠️ 训练与 vLLM 推理共抢 128GB 统一内存, 训练前必须停

# 5.1 冒烟测试 (100 cells, 测显存与温度) — 决定内存防护档位 + batch 上限
python scripts/protocol_06_finetune_gf.py --max_epochs 1 --max_cells 100
python scripts/protocol_07_finetune_scgpt.py --max_epochs 1 --max_cells 100
#   监控: free -h (统一内存) + ~/tools/bin/dgxtop (GPU/温度)
#   ⚠️ 持续训练 >4h 温度可达 85-95°C 超阈值自动关机; 四周留 50mm, 室温 <25°C

# 5.2 内存防护 (统一内存 OOM=整机僵尸/SSH 挂死, 硬重启)
sudo swapoff -a                          # 【最低档, 必做】防 swap 死亡螺旋 → 任务死但系统活
# 升级档(冒烟后按显存占用决定):
# ⚠️ systemd-run --scope 拒绝 OOMScoreAdjust (ExecProperties= 仅限 transient service)
# → 两种模式:
#    Transient service (支持 OOMScoreAdjust):
bash scripts/gx10_cgroup_run.sh --mode service --memory-max 100G --memory-swap-max 0 \
     --oom-score-adjust 500 --runtime-max 259200 --log-dir /home/dq/Codex/Ai/logs \
     --unit gx10-caft-gf -- \
     bash -lc 'cd ~/Codex/Ai && source ~/venv-ds/bin/activate && python scripts/protocol_06_finetune_gf.py --seed 42'
#    Scope (仅 cgroup 资源属性, 无 OOMScoreAdjust):
bash scripts/gx10_cgroup_run.sh --mode scope --memory-max 100G --memory-swap-max 0 \
     --unit gx10-caft-scope -- \
     bash -lc 'cd ~/Codex/Ai && source ~/venv-ds/bin/activate && python scripts/protocol_06_finetune_gf.py --seed 42'
#    预检 (dry-run 验证命令, 不执行):
bash scripts/gx10_cgroup_run.sh --mode service --dry-run --memory-max 100G -- -- python train.py
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'   # 每次实验间清缓存

# 5.3 注意力实现: 确认 SDPA (不装 flash-attn 即默认)
#   Geneformer (BertModel via HF) 与 scGPT (use_fast_transformer=False fallback) 均走 SDPA
#   若日志报 "flash_attention_2 is not installed", 设 attn/sdpa 即可, 勿装 flash-attn

# 5.4 正式训练
python scripts/protocol_06_finetune_gf.py --caft_lambda 0.1 --seed 42          # CAFT-GF 2-3d
python scripts/protocol_07_finetune_scgpt.py --caft_lambda 0.1 --seed 42       # CAFT-scGPT 3-5d
python scripts/protocol_08_inference_sclong.py                                  # scLong zero-shot (BaiDing1234/scLong, hf-mirror) 0.5d
python scripts/protocol_09_benchmark.py                                         # E1-E5 1-2d
python scripts/run_all.py --config config/ablation_configs.yml                  # A1-A4 2-3d
```

## Phase 6: 回传 (GX10 → Windows)

```bash
# 从 Windows 拉取 (网线直连 2.5Gbps / Tailscale):
#   scp -r dq@<GX10-IP>:/home/dq/Codex/Ai/save/ D:\Codex\Ai\save\
# 回传: save/*.log, save/caft_gf_model/, save/caft_scgpt_model/,
#       save/benchmark_results.json, save/ablation_results.json, figures/
```

---

## 故障排查

| 症状 | 原因 | 修复 |
|------|------|------|
| `torch.cuda.is_available()=False` | torch 非 cu130 (x86/CUDA12 wheel) | 用 NGC 容器 或 `pip install torch --index-url https://download.pytorch.org/whl/cu130` |
| `libcudart.so.12: cannot open` | 装了 CUDA12 编译的包 (flash-attn 等) | `pip uninstall flash-attn ...`; 只用 cu130 源 |
| sm_121 warning "(8.0)-(12.0)" | 提示性 | **可忽略** (sm_120/121 二进制兼容, PyTorch 官方确认) |
| OOM → SSH 挂死/整机僵尸 | 统一内存 swap 死亡螺旋 | `sudo swapoff -a` + cgroup MemoryMax + 减 batch |
| `systemd-run --scope -p OOMScoreAdjust=N` → Unknown assignment | scope 不支持 ExecProperties | **改用 `--mode service`** (transient service, `scripts/gx10_cgroup_run.sh`) |
| GPR 恢复失败 "CS_before legacy PCA" | 旧版 `cs_before_results.json` 含 PCA 嵌入 | 用 `gx10_gpr_recovery.sh` 显式路径 paired 模式, 不用 `--cs_before` |
| GPR 恢复失败 "stale 020334" | 使用了过时的失败运行 | 改用有效的 CAFT 输出 (如 `caft_gf_lam0.0_seed42_20260724-085100`)
| `nvidia-smi` 显存 Not Supported | 统一内存架构 | 正常; 用 `free -h` 看内存 |
| Geneformer import 报错 | 老 fork 的 circular import / tokenizer 语法 | 用 jkobject fork 最新版; 缺 pkl 从 hf-mirror 补 |
| Docker 更新后 GPU 失效 | apt upgrade 破坏 runtime | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| HF 下载超时 | 墙 | 确认 `HF_ENDPOINT=https://hf-mirror.com` |
| 温度过高自动关机 | 散热不足 | 四周留 50mm, 室温<25°C, `dgxtop` 监控 |
| **GPU ~96% util 但 ~5-15W / SM 时钟钉死 513-721MHz**（throttle 全 "Not Active"、温度正常、power-limit 显示 N/A） | **USB-C PD 电源楔死**（crash/OOM/休眠唤醒触发；NVIDIA 已承认 under investigation） | **拔电源适配器出墙 1-2 分钟再插回**（仅拔 USB-C 无效，SSH 重启无效）；预防 `sudo fwupdmgr refresh && sudo fwupdmgr update`（新固件明确修 PD）；持续 5W/0% 且拔电无效 → RMA |

---

## Phase 5.5: GPR 恢复 (CAFT 完成后)

```bash
# 干运行 — 仅预检，不执行计算（12 门检查）
bash scripts/gx10_gpr_recovery.sh --dry-run \
  --before-checkpoint /home/dq/.cache/huggingface/hub/.../Geneformer-V2-104M \
  --after-checkpoint  /home/dq/Codex/Ai/save/caft_gf_lam0.0_seed42_20260724-085100/model/checkpoint-1500 \
  --tokenized         /home/dq/Codex/Ai/data/gf_tokenized_train/corneal_100k.dataset \
  --out               /home/dq/Codex/Ai/save/gpr_recovery_result.json

# 完整运行（预检 + GPR 计算，通过 cgroup launcher 执行，需要 GPU）
# 恢复脚本调用 gx10_cgroup_run.sh --mode service，配置如下：
#   MemoryMax=100G  MemorySwapMax=0  RuntimeMaxSec=86400
#   OOMScoreAdjust=500  OOMPolicy=kill
#   Python: /home/dq/venv-ds/bin/python
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
bash scripts/gx10_gpr_recovery.sh \
  --before-checkpoint /home/dq/.cache/huggingface/hub/... \
  --after-checkpoint  /home/dq/Codex/Ai/save/caft_gf_lam0.0_seed42_20260724-085100/model/checkpoint-1500 \
  --tokenized         /home/dq/Codex/Ai/data/gf_tokenized_train/corneal_100k.dataset \
  --out               /home/dq/Codex/Ai/save/gpr_recovery_result.json
```

12 个预检门（v2）：
1. 拒绝过时运行 `020334`
2. 要求 after-checkpoint 父目录中存在 `CAFT_DONE`
3. 拒绝 `cs_before_results.json` 中含旧版 PCA 嵌入
4. 验证 `--tokenized` 目录（包含 HF 数据集文件）
5. 验证 before-checkpoint 存在（model.safetensors + config.json）
6. 验证 after-checkpoint 存在
7. 固定 Python 解释器 `/home/dq/venv-ds/bin/python`，验证存在且可执行
8. 检查无活跃 vLLM/compute 进程（`pgrep`）
9. 检查无活跃 GPU 计算进程（`nvidia-smi --query-compute-apps`）
10. 验证 CUDA 可用（`nvidia-smi`）
11. 要求 MemAvailable ≥ 20 GiB（统一内存中加载模型）
12. 验证 cgroup v2（统一层级）

所有路径必须为绝对路径。不使用 `ls -dt`。不使用 `--cs_before`。
**GPR 计算通过 cgroup launcher 委托执行**（非直接 `python -m corneal_fm.gpr`）。

---

## 可靠性要点 (2026-07-30 外部检索确认 · NVIDIA 论坛 + 官方文档)

> **核心因果链**: OOM/crash → USB-C PD 电源楔死 → GPU 卡 ~5W 低功耗态（只能拔电恢复）。
> 所以**避免 OOM 比恢复更重要**——它是电源楔死的首要触发因素。

- **防 OOM = 防楔死**: 训练规模留余量适配 ~119GB 可用（128GB 减系统/GPU 预留）；实验间 `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`；`sudo swapoff -a`（统一内存 swap 死亡螺旋）。
- **vLLM 是可靠性控制，非仅卫生**: 其默认 `--gpu-memory-utilization 0.90` ≈ 占 115GB；须 `systemctl disable --now` 自启单元 + `docker update --restart=no`（否则被 systemd 重新拉起）+ 调低利用率。训练前 `vllm-stop`（见 Phase 5）。
- **统一内存 mmap 陷阱**: HF `mmap` 载入大模型会在 ~66% 处 OOM（page cache 与 CUDA 张量争同一物理池）。对策: direct-to-CUDA 分片载入 + `posix_fadvise(DONTNEED)` + 及时关 shard；`cudaMemGetInfo` 在 UMA 上**低报**可分配量（忽略可回收 swap，官方 known-issue）。
- **固件**: `sudo fwupdmgr refresh && sudo fwupdmgr update`（新版明确修 PD 电源问题；注意个别版本有 ConnectX-7 回归）。
- **容器/精度（再次确认）**: NGC `pytorch:25.11-py3`（官方 DGX Spark playbook 在用）或 `26.03-py3`（社区验证）；**≤24.12 不可用**（无 sm_12x codegen）。训练用 **bf16**（Blackwell 上比 fp16 安全、无 loss scaling）；**禁 FP4/FP8/NVFP4**（sm_121 无后端，NVIDIA 确认无 roadmap）；CUDA 13.2 上 `adamw_8bit` 会坏 → 用 `adamw_torch`。`TORCH_CUDA_ARCH_LIST=12.1+PTX` + `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`（源码构建扩展时需要）。

## 复现性策略 (发表级)

1. CS_before & CS_after 均在 **同一份 GX10 RAPIDS-reprocessed** 数据上计算 → GPR 有效
2. seed=42 固定; 锁定 pip freeze / NGC 镜像 tag / RAPIDS 参数
3. STAR Methods 记录: RAPIDS-singlecell 版本 + HVG/normalize/PCA/UMAP 全参数, scGPT/Geneformer/transformers 版本, cu130 容器 tag
4. 脚本已 archive 到 save/ (protocol_11_release 打包)
