#!/bin/bash
# Sensitivity + UMAP extraction driver (p5070 CPU, resumable, lane.log).
# Phases: 1-2 scGPT naive/corn x {42,123,456} (fast); 3-4 GF naive/corn seed42
# (long, ~13-25 h/seed under load); then reduce + figures.
ROOT=/home/summer/corneal_scgpt
PY_VENV=$ROOT/venv/bin/python
PY_SCGPT=$ROOT/repo/.venv/bin/python
SCRIPTS=$ROOT/scripts
CACHE=$ROOT/artifacts/paper1_vanzyl/sensitivity/_caches
SENS=$ROOT/artifacts/paper1_vanzyl/sensitivity
UMAP=$ROOT/artifacts/paper1_vanzyl/umap_collapse
LOG=$ROOT/artifacts/paper1_vanzyl/sensitivity/lane.log
mkdir -p "$CACHE" "$SENS" "$UMAP"
log() { echo "{\"ts\": \"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)\", \"event\": \"$1\", \"msg\": \"$2\"}" >> "$LOG"; }
log LANE_START "sens+umap extraction (scGPT naive/corn x3 + GF naive/corn s42) on p5070 CPU"

run_one() {
  local py=$1 pypath=$2 tag=$3 arch=$4 arm=$5 seed=$6
  if [ -f "$CACHE/$tag.DONE" ]; then
    log SKIP "$tag (DONE)"
    return 0
  fi
  log SEED_START "$tag"
  PYTHONPATH=$pypath $py $SCRIPTS/sens_umap_extract.py --arch $arch --arm $arm \
    ${seed:+--seed $seed} --threads 12 --out "$CACHE" >> "$CACHE/$tag.run.log" 2>&1
  rc=$?
  log SEED_DONE "$tag rc=$rc"
  [ $rc -ne 0 ] && return $rc
  return 0
}

# ---- Phase 1/2: scGPT (fast; essential bistable trio first) ----
# priority order: naive42, naive123 (bistable pair), corn42 (rescue),
# then the remaining robustness seeds
for spec in "naive 42" "naive 123" "corn 42" "naive 456" "corn 123" "corn 456"; do
  set -- $spec
  run_one "$PY_SCGPT" "$ROOT/repo/scGPT:$ROOT:$ROOT/scripts" "scgpt_$1_seed$2" scgpt $1 $2 || exit $?
done
log REDUCE "scGPT caches complete -> sens_umap_reduce"
PYTHONPATH=$ROOT $PY_VENV $SCRIPTS/sens_umap_reduce.py --data-dir "$CACHE" --out-dir "$SENS" --umap-out-dir "$UMAP" >> "$SENS/reduce1.log" 2>&1
log REDUCE_DONE "reduce after scGPT rc=$?"

# ---- Phase 3/4: GF (long) ----
run_one "$PY_VENV" "$ROOT" "gf_naive_seed42" gf naive 42 || exit $?
run_one "$PY_VENV" "$ROOT" "gf_corn_seed42" gf corn 42 || exit $?
log REDUCE "GF caches complete -> final reduce + figures"
PYTHONPATH=$ROOT $PY_VENV $SCRIPTS/sens_umap_reduce.py --data-dir "$CACHE" --out-dir "$SENS" --umap-out-dir "$UMAP" >> "$SENS/reduce2.log" 2>&1
rc=$?
log REDUCE_DONE "final reduce rc=$rc"
PYTHONPATH=$ROOT $PY_VENV $SCRIPTS/sens_umap_fig.py --sensitivity-dir "$SENS" --umap-dir "$UMAP" >> "$SENS/fig.log" 2>&1
log FIG_DONE "figures rc=$?"
log LANE_DONE "all phases complete"
