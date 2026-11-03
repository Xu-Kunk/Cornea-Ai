# 生产环境溯源 Environment Provenance

> 2026-09-22：移除内部主机名与绝对路径，以便随公开仓库发布；技术规格与版本信息未改动。
> C1 环境冻结 · 生成 2026-08-13。完整锁定见 `config/requirements-cellplm.lock.txt`
> （`pip freeze`，243 包）。

## 运行时

- Python 3.12.3，独立虚拟环境（`.venv-cellplm`）
- 硬件：NVIDIA GB10（Grace Blackwell，aarch64，CUDA 13.0/sm_121）
- 关键依赖实测版本（`importlib.metadata`）：

| 包 | 版本 | 安装源 |
|----|------|--------|
| torch | 2.13.0+cu130 | PyPI（cu130 build） |
| transformers | 4.57.6 | PyPI |
| geneformer | 0.1.0 | HF snapshot `04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5`（本地 HF cache，`file://` 安装） |
| scGPT | 0.2.5 | git `bowang-lab/scGPT` @ `cebd6fae655b9c585a4807daa3ac31bb764f06b4` |
| anndata | 0.13.2 | PyPI |
| numpy | 2.4.6 | PyPI |
| scipy | 1.18.0 | PyPI |
| datasets | 5.0.0 | PyPI |
| tokenizers | 0.22.2 | PyPI |

## 注意

- geneformer 与 scGPT 均通过 `direct_url.json` 记录安装源，**已 pin**（非 git
  HEAD，非浮动）：geneformer 钉在本地 HF snapshot，scGPT 钉在具体 commit。
- Geneformer V2-104M checkpoint/tokenizer 语料：Genecorpus-104M，snapshot
  `04c2b2e8…`（见 P1-O1 污染审计报告）。
- 历史 `requirements-core.txt` 曾 pin `torch==2.12.1` / `transformers==5.13.0`，
  与生产环境（2.13.0+cu130 / 4.57.6）不符；本文件为生产实测版本，投稿以
  本文件 + `requirements-cellplm.lock.txt` 为准。
