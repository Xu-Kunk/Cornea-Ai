#!/usr/bin/env bash
# ============================================================================
# Corneal-FM CAFT — Day-3 autonomous pipeline (runs detached in tmux overnight).
# Idempotent: each step skips if its output already exists. Re-applies the GX10
# guardrails between long GPU steps.
#
#   STEP 1  E0      frozen scFM vs HVG/PCA on the FULL donor-disjoint split (gate).
#   STEP 2  subsample   representative 30k layer-stratified subset (design "短 naive").
#   STEP 3  naive   lambda=0 fine-tune, 1 epoch, on the subsample -> checkpoint + layer ds.
#   STEP 4  GPR     CS_before (pretrained) + CS_after (naive) + bootstrap 95% CI.
#   STEP 5  report  consolidated summary + A/B/C scenario -> Day3_STOP_REPORT.md.
#
# GX10 forum guardrails applied throughout (unified-memory OOM zombie = P1,
# silent 85-95C thermal shutdown = P2): swap off, swappiness=1, GPU clock cap
# 300-2200 MHz, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False. vLLM must be
# stopped before launch (it reserves ~97GB unified memory and freezes the box).
# ============================================================================
set -uo pipefail
# Point CORNEAL_FM_DATA at the repository root (the directory containing scripts/).
cd "${CORNEAL_FM_DATA:?set CORNEAL_FM_DATA to the repo root}"
# Activate your Python environment first (e.g. `source .venv/bin/activate`).
# source <your-venv>/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=python
PRETRAINED="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ctheodoris--Geneformer/snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5/Geneformer-V2-104M"
OUT=artifacts/day3-20260723
PIPELOG=$OUT/day3_pipeline.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PIPELOG"; }
hb()  { echo "[$(date '+%F %T')] HB step=$1" > "$OUT/day3_heartbeat.txt"; }

guardrails() {
  sudo swapoff -a 2>/dev/null
  sudo sysctl -w vm.swappiness=1 >/dev/null 2>&1
  sudo nvidia-smi -pm 1 >/dev/null 2>&1
  sudo nvidia-smi -lgc 300,2200 >/dev/null 2>&1
}

# Make OUT absolute: caft_gf.py prepends its own data/ dir to --data_file, and
# Path(data_dir)/absolute_path == absolute_path, so an absolute SUB avoids the
# data/artifacts/... mismatch that crashed STEP3 on the first run.
OUT="$PWD/artifacts/day3-20260723"
mkdir -p "$OUT/S5" "$OUT/S6"
guardrails
log "=== Day-3 pipeline START (E0 full -> naive/GPR on 30k subsample) ==="

# ---------------- STEP 1: E0 (full donor-disjoint split) ----------------
E0RES=$OUT/S5/E0_full_results.json
if [ ! -f "$E0RES" ]; then
  log "STEP1 E0 (full 100k train / 20k test) START"; hb "E0"
  timeout 14400 "$PY" scripts/e0_axis1_baseline.py \
    --train data/corneal_100k_donor_split.h5ad --test data/corneal_20k_donor_split.h5ad \
    --pretrained "$PRETRAINED" --label cell_type --seed 42 --n-hvg 2000 --n-pc 50 \
    --out "$E0RES" >>"$OUT/S5/e0_full_pipeline.log" 2>&1
  log "STEP1 E0 END (exit $?)"
  guardrails
else
  log "STEP1 E0 already done"
fi

# ---------------- STEP 2: 30k layer-stratified subsample ----------------
SUB=$OUT/S6/corneal_30k_layer_subsample.h5ad
if [ ! -f "$SUB" ]; then
  log "STEP2 create 30k layer-stratified subsample START"; hb "subsample"
  SUB_OUT="$SUB" "$PY" - <<'PYSUB' >>"$OUT/S6/subsample_pipeline.log" 2>&1
import os
import anndata as ad
import numpy as np
from corneal_fm.caft import author_cell_type_to_layer

out = os.environ["SUB_OUT"]
a = ad.read_h5ad("data/corneal_100k_subsample.h5ad")
layer = a.obs["author_cell_type"].map(author_cell_type_to_layer)
rng = np.random.default_rng(42)
# Scale epithelial layers to keep thousands of cells per differentiation layer
# (stable Continuity Score) while capping total runtime.
targets = {0: 1700, 1: 2800, 2: 2700, 3: 700, 4: 4100}
idx = []
for lyr, tg in targets.items():
    pool = a.obs.index[layer == lyr].tolist()
    idx += list(rng.choice(pool, size=min(tg, len(pool)), replace=False))
rest = 30000 - len(idx)
if rest > 0:
    pool = a.obs.index[layer == -1].tolist()
    idx += list(rng.choice(pool, size=min(rest, len(pool)), replace=False))
sub = a[a.obs_names.isin(idx)].copy()
print("subsample shape:", sub.shape)
sub.write_h5ad(out, compression="gzip")
print("written:", out)
PYSUB
  log "STEP2 subsample END (exit $?)"
else
  log "STEP2 subsample already done"
fi

# ---------------- STEP 3: naive fine-tune (lambda=0, 1 epoch) ----------------
NAIVE_RUN=$(ls -dt save/caft_gf_lam0.0_seed42_* 2>/dev/null | head -1)
if [ -z "$NAIVE_RUN" ] || [ ! -f "$NAIVE_RUN/CAFT_DONE" ]; then
  log "STEP3 naive fine-tune (lambda=0, 1 epoch) START"; hb "naive"
  timeout 14400 "$PY" scripts/caft_gf.py \
    --data_file "$SUB" --caft_lambda 0 --epochs 1 --batch_size 16 --seed 42 \
    >>"$OUT/naive_pipeline.log" 2>&1
  log "STEP3 naive END (exit $?)"
  NAIVE_RUN=$(ls -dt save/caft_gf_lam0.0_seed42_* 2>/dev/null | head -1)
  guardrails
else
  log "STEP3 naive already done ($NAIVE_RUN)"
fi
LAYER_DS="$NAIVE_RUN/corneal_layer_subset_tk/cell.dataset"
NAIVE_CKPT="$NAIVE_RUN/model"
log "naive run=$NAIVE_RUN dataset=$LAYER_DS ckpt=$NAIVE_CKPT"

# ---------------- STEP 4: GPR = CS_after / CS_before ----------------
CSB=$OUT/S6/cs_before_results.json
if [ ! -f "$CSB" ]; then
  log "STEP4a CS_before (pretrained on layer dataset) START"; hb "cs_before"
  timeout 7200 "$PY" corneal_fm/gpr.py --checkpoint "$PRETRAINED" --tokenized "$LAYER_DS" \
    --num_classes 31 --n_boot 1000 --out "$CSB" >>"$OUT/S6/cs_before_pipeline.log" 2>&1
  log "STEP4a CS_before END (exit $?)"
  guardrails
else
  log "STEP4a CS_before already done"
fi

CSG=$OUT/S6/cs_gpr_result.json
if [ ! -f "$CSG" ]; then
  CS_BEFORE=$("$PY" -c "import json;print(json.load(open('$CSB'))['continuity_score'])" 2>/dev/null)
  log "STEP4b CS_after (naive) + GPR START (CS_before=$CS_BEFORE)"; hb "cs_after"
  timeout 7200 "$PY" corneal_fm/gpr.py --checkpoint "$NAIVE_CKPT" --tokenized "$LAYER_DS" \
    --num_classes 31 --cs_before "$CS_BEFORE" --n_boot 1000 --out "$CSG" >>"$OUT/S6/cs_gpr_pipeline.log" 2>&1
  log "STEP4b CS_after+GPR END (exit $?)"
else
  log "STEP4b CS_after+GPR already done"
fi

# ---------------- STEP 5: consolidated report ----------------
log "STEP5 generate Day-3 report"; hb "report"
"$PY" - <<'PYREPORT' 2>>"$OUT/S6/report_pipeline.log"
import json
from pathlib import Path

OUT = Path("artifacts/day3-20260723")
e0 = json.load(open(OUT / "S5/E0_full_results.json"))
cb = json.load(open(OUT / "S6/cs_before_results.json"))
cg = json.load(open(OUT / "S6/cs_gpr_result.json"))

b, s = e0["baseline_hvg_pca"], e0["scfm_frozen"]
gpr = cg.get("gpr", {}) or {}
g = gpr.get("gpr")
if g is None:
    scenario = "UNKNOWN (GPR missing)"
elif g < 0.80:
    scenario = "A — strong degradation; add Direction C (5-strategy remedy comparison)"
elif g <= 0.95:
    scenario = "B — moderate degradation; add Direction A (mechanism / manifold flattening)"
else:
    scenario = "C — minimal/no degradation; consider tissue pivot or 'when to worry' framing"

lines = [
    "# Corneal-FM CAFT — Day-3 Decision Report",
    "",
    "> 自主通宵管线产出 · GX10 (GB10 Blackwell, CUDA13) · seed 42 · 映射 v2026-07-23 (PI ratified)",
    "",
    "## E0 — Axis-1 cell discrimination gate (full donor-disjoint split)",
    "",
    f"- **Verdict: {e0['go_no_go']}** (GO iff frozen scFM macro-F1 > HVG/PCA)",
    f"- HVG/PCA macro-F1 = **{b['macro_f1']:.4f}** (n_hvg={b['n_hvg']}, n_pcs={b['n_pcs']})",
    f"- frozen scFM macro-F1 = **{s['macro_f1']:.4f}",
    f"- 差值 (scFM - HVG) = {s['macro_f1']-b['macro_f1']:+.4f}",
    f"- n_classes={e0['n_classes']} · train_donors={e0['split']['train_donors']} · test_donors={e0['split']['test_donors']} · disjoint={e0['split']['donor_disjoint']}",
    "",
    "> 解读: frozen scFM 在细胞判别上与 HVG/PCA 相当/略低 (符合 scFM 文献张力;",
    "> Ahlmann-Eltze/Richter 等)。CAFT 的贡献在分化连续性轴 (Axis-2)，非 Axis-1 判别。",
    "",
    "## GPR — Gradient Preservation Ratio (differentiation continuity)",
    "",
    f"- CS_before (pretrained) = **{cb['continuity_score']:.4f}**  CI95 {cb['bootstrap_cs_95ci']['ci_low']:.4f}-{cb['bootstrap_cs_95ci']['ci_high']:.4f}",
    f"- CS_after (naive λ=0)   = **{cg['continuity_score']:.4f}**  CI95 {cg['bootstrap_cs_95ci']['ci_low']:.4f}-{cg['bootstrap_cs_95ci']['ci_high']:.4f}",
    f"- **GPR = {g} ({gpr.get('interpretation','?')})**",
    f"- 相邻层相似度 (naive): {cg.get('pair_similarity')}",
    "",
    f"## A/B/C scenario → **{scenario}**",
    "",
    "决策树 (design §11): GPR<0.80→A/+C · 0.80-0.95→B/+A · >0.95→C/pivot.",
    "",
    "## Out-of-scope (本 Day-3 不做，待此判决后用户确认)",
    "",
    "- CAFT (λ>0) 全量微调 + A1-A4 消融",
    "- E1-E5 benchmark + E6 表皮验证",
    "- 全量 100k naive (本管线用 30k 分层子样本的'短 naive')",
    "- Results/Discussion/Figures 撰写与 Patterns 投稿",
]
report = "\n".join(lines) + "\n"
(OUT / "Day3_STOP_REPORT.md").write_text(report)
(OUT / "day3_summary.json").write_text(json.dumps({
    "e0": {"verdict": e0["go_no_go"], "hvg_f1": b["macro_f1"], "scfm_f1": s["macro_f1"],
           "diff": s["macro_f1"] - b["macro_f1"], "n_classes": e0["n_classes"]},
    "cs_before": cb["continuity_score"], "cs_before_ci": cb["bootstrap_cs_95ci"],
    "cs_after": cg["continuity_score"], "cs_after_ci": cg["bootstrap_cs_95ci"],
    "gpr": g, "gpr_interpretation": gpr.get("interpretation"),
    "pair_similarity_naive": cg.get("pair_similarity"), "scenario": scenario,
}, indent=2))
print(report)
print("WROTE Day3_STOP_REPORT.md + day3_summary.json")
PYREPORT
log "STEP5 report END"
hb "done"
log "=== Day-3 pipeline COMPLETE ==="
