#!/usr/bin/env bash
# ============================================================================
# CAFT-GF 一键流水线 (λ-Pareto + CS_before/CS_after + GPR + Markdown 汇总)
#
# 用法 (在你的 GX10 终端, 干净会话, 后台挂机):
#   cd /home/dq/Codex/Ai && source ~/venv-ds/bin/activate
#   chmod +x scripts/run_caft_all.sh
#   nohup bash scripts/run_caft_all.sh > caft_all.log 2>&1 &
#   echo $! > caft_all.pid
#   # 监控: tail -f caft_all.log ; tail -f save/caft_sweep_*/summary.log
#
# 预计 3-4h。跑完看:  save/caft_sweep_<时间戳>/CAFT_RESULTS.md
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
export CORNEAL_FM_DATA="${CORNEAL_FM_DATA:-$ROOT/data}"
TS="$(date +%Y%m%d-%H%M%S)"
SUMMARY_DIR="$ROOT/save/caft_sweep_$TS"
mkdir -p "$SUMMARY_DIR"
LOG="$SUMMARY_DIR/summary.log"

LAMBDAS="${LAMBDAS:-0 0.1 0.5 1.0}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-16}"
SEED="${SEED:-42}"
DATA_FILE="${DATA_FILE:-corneal_20k_donor_split.h5ad}"
PY="${PY:-$HOME/venv-ds/bin/python}"

log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
die()  { log "❌ FATAL: $*"; exit 1; }

log "=========================================="
log "CAFT-GF 全流程启动"
log "λ sweep = [$LAMBDAS] | epochs=$EPOCHS | bs=$BS | seed=$SEED"
log "数据(CS) = $DATA_FILE | 输出 = $SUMMARY_DIR"
log "=========================================="

# --- 前置: 环境 ---
command -v "$PY" >/dev/null 2>&1 || die "找不到 python: $PY"
python - <<'PYEOF' 2>/dev/null
import torch
ok = torch.cuda.is_available()
print("CUDA:", ok, torch.cuda.get_device_name(0) if ok else "")
assert ok, "no GPU"
PYEOF
[ $? -ne 0 ] && die "GPU 不可用 (torch.cuda.is_available=False)"
[ -f "$CORNEAL_FM_DATA/corneal_100k_subsample.h5ad" ] || die "缺 corneal_100k_subsample.h5ad"
[ -f "$CORNEAL_FM_DATA/$DATA_FILE" ] || die "缺 CS 数据 $DATA_FILE"
log "✓ python / GPU / 数据 就绪"

# --- 释放统一内存: 停 vLLM (关键) ---
log "停止 vLLM 以释放统一内存..."
systemctl --user stop vllm 2>/dev/null || true
pkill -9 -f "vllm.entrypoints|vllm serve|vllm/vllm" 2>/dev/null || true
sleep 6
if pgrep -f "vllm.entrypoints|vllm serve" >/dev/null 2>&1; then
  log "⚠️ vLLM 似乎仍在运行 (会占 ~74GB, 可能 OOM), 请手动: systemctl --user stop vllm"
fi
log "GPU mem: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a') | temp: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo 'n/a')°C"

# ============================================================================
# 1) λ-Pareto CAFT 微调 (顺序, 每个写 RUN_STATUS/CAFT_DONE/caft_args.json)
# ============================================================================
declare -a RUN_DIRS
for lam in $LAMBDAS; do
  log "──── λ=$lam 训练开始 ────"
  before=$(ls -dt "$ROOT"/save/caft_gf_lam${lam}_seed${SEED}_* 2>/dev/null | head -1)
  "$PY" scripts/caft_gf.py --caft_lambda "$lam" --epochs "$EPOCHS" --batch_size "$BS" --seed "$SEED" >> "$LOG" 2>&1 &
  PID=$!
  log "  λ=$lam PID=$PID, 等待 CAFT_DONE..."
  NEW=""; waited=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 30; waited=$((waited+30))
    cur=$(ls -dt "$ROOT"/save/caft_gf_lam${lam}_seed${SEED}_* 2>/dev/null | head -1)
    if [ -n "$cur" ] && [ "$cur" != "$before" ]; then NEW="$cur"; fi
    # 超时保护: 60 min
    if [ "$waited" -ge 3600 ]; then log "  ⚠️ λ=$lam 超时(60min), 跳过"; kill -9 "$PID" 2>/dev/null; break; fi
  done
  if [ -z "$NEW" ]; then NEW=$(ls -dt "$ROOT"/save/caft_gf_lam${lam}_seed${SEED}_* 2>/dev/null | head -1); fi
  RUN_DIRS[$lam]="$NEW"
  if [ -n "$NEW" ] && [ -f "$NEW/CAFT_DONE" ]; then
    el=$(python -c "import json;print(json.load(open('$NEW/caft_args.json'))['elapsed_min'])" 2>/dev/null || echo '?')
    log "  ✓ λ=$lam 完成 (${el} min): $NEW"
  else
    log "  ⚠️ λ=$lam 未完成 (无 CAFT_DONE): ${NEW:-无目录}; 检查 $NEW/ERROR_TRACEBACK.txt"
  fi
done

# ============================================================================
# 2) CS_before (base 预训练模型)
# ============================================================================
log "──── 计算 CS_before (base model) ────"
CSB="$SUMMARY_DIR/cs_before.json"
"$PY" scripts/cs_after.py --data_file "$DATA_FILE" --out "$CSB" >> "$LOG" 2>&1
CS_BEFORE="?"
if [ -f "$CSB" ]; then
  CS_BEFORE=$(python -c "import json;print(json.load(open('$CSB'))['continuity_score'])" 2>/dev/null || echo '?')
  log "✓ CS_before = $CS_BEFORE"
else
  log "⚠️ CS_before 计算失败 (见 $LOG)"
fi

# ============================================================================
# 3) CS_after + GPR (每个已完成 λ 的 checkpoint)
# ============================================================================
log "──── 计算 CS_after / GPR ────"
for lam in $LAMBDAS; do
  d="${RUN_DIRS[$lam]:-}"
  if [ -z "$d" ]; then log "  λ=$lam: 无 checkpoint (训练未完成), 跳过测量"; continue; fi
  ck=$(ls -d "$d"2607*_geneformer_cellClassifier_CAFT-GF_lam${lam} 2>/dev/null | head -1)
  if [ -z "$ck" ]; then log "  λ=$lam: 找不到模型目录 $d, 跳过"; continue; fi
  out="$SUMMARY_DIR/cs_after_lam${lam}.json"
  if [ "$CS_BEFORE" != "?" ] && [ -n "$CS_BEFORE" ]; then
    "$PY" scripts/cs_after.py --checkpoint "$ck" --cs_before "$CS_BEFORE" --out "$out" >> "$LOG" 2>&1
  else
    "$PY" scripts/cs_after.py --checkpoint "$ck" --out "$out" >> "$LOG" 2>&1
  fi
  [ -f "$out" ] && log "  ✓ λ=$lam CS_after 完成: $out" || log "  ⚠️ λ=$lam CS_after 失败"
done

# ============================================================================
# 4) 汇总 Markdown
# ============================================================================
MD="$SUMMARY_DIR/CAFT_RESULTS.md"
log "汇总 -> $MD"
{
  echo "# CAFT-GF λ-Pareto 实验结果"
  echo
  echo "- 生成时间: $(date)"
  echo "- 模型: Geneformer V2-104M | 训练: epochs=$EPOCHS, bs=$BS, seed=$SEED"
  echo "- CS 测量数据: $DATA_FILE (embedding = [CLS])"
  echo "- **CS_before (base model) = $CS_BEFORE**"
  echo
  echo "| λ | CS_after | GPR | pair相似 | LAA | 训练耗时(min) | checkpoint |"
  echo "|---|----------|-----|--------|-----|---------------|-----------|"
  for lam in $LAMBDAS; do
    out="$SUMMARY_DIR/cs_after_lam${lam}.json"
    d="${RUN_DIRS[$lam]:-}"
    el="-"
    if [ -n "$d" ] && [ -f "$d/caft_args.json" ]; then
      el=$(python -c "import json;print(json.load(open('$d/caft_args.json'))['elapsed_min'])" 2>/dev/null || echo '-')
    fi
    ck=$(ls -d "$d"2607*_geneformer_cellClassifier_CAFT-GF_lam${lam} 2>/dev/null | head -1)
    ck=$(basename "${ck:-无}")
    if [ -f "$out" ]; then
      row=$("$PY" -c "import json;d=json.load(open('$out'));print(str(d['continuity_score'])+' | '+str(d.get('gpr',{}).get('gpr','-'))+' | '+str(d.get('pair_similarity',''))+' | '+str(d.get('layer_adjacency_accuracy','-')))" 2>/dev/null || echo "? | ? | ? | ?")
      echo "| $lam | $row | $el | $ck |"
    else
      echo "| $lam | (无CS) | - | - | - | $el | $ck |"
    fi
  done
  echo
  echo "> GPR = CS_after / CS_before；>1 表示微调后分化连续性保持/提升（CAFT 目标）。"
  echo "> 明细 JSON: $SUMMARY_DIR/cs_*.json"
} > "$MD"

log "================================================"
log "✅ 全流程结束！结果: $MD"
log "================================================"
cat "$MD" | tee -a "$LOG"
