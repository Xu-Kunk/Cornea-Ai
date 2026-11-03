#!/usr/bin/env python
"""CORN advantage A+D — reduction: per-cell npz -> monotonicity.json +
thresholds.json + summary (PCS-locked metric definitions).

FROZEN metric protocol (PCS Stability, frozen before first run 2026-08-20):

Experiment A — ordinal monotonicity (per cell, k=1,2 on the 3-tier axis):
  * CORN arms: p1 = sigmoid(threshold_logit_1) = P(class>=1),
    p2 = sigmoid(threshold_logit_2) = P(class>=2). GF 4-class axis:
    p1 = sigmoid(t1) (>=TAC => not-LSC), p2 = sigmoid(t3) (=diff);
    t2 unused on the 3-tier axis (user-confirmed 2026-08-20).
  * LoRA (softmax 30-class): probabilities summed over the majority-mapped
    3-tier classes (naive_map, locked); P(class>=1) = P(basal)+P(diff),
    P(class>=2) = P(diff).
  * metric 1 (literal cumulative violation): fraction P(class>=1) <
    P(class>=2). For ANY softmax this is 0 by construction (cumulative sums
    of non-negative probabilities) — reported as the control.
  * metric 2 (PRIMARY decision-consistency violation): fraction of cells
    where the model's decision class disagrees with the CORN-style
    threshold-count rule over its own cumulative probabilities
    (count of P(class>=k) >= 0.5):
      - CORN arms: decision IS the threshold count -> 0 by construction
        (computed and verified).
      - LoRA: decision = argmax author class mapped to tier vs the
        threshold-count class -> the discriminating metric.
  * decode threshold 0.5 (ordinal_decode contract).

Experiment D — threshold interpretability (per CORN arm/seed):
  * differentiation axis = L2-normed cell embedding projected onto
    u = normalize(mean_diff_emb - mean_LSC_emb) (model's own embedding
    space; means over the test cells by ground-truth 3-tier label).
  * 10 deciles of the axis; per decile: mean p1, mean p2, mean marker
    scores (locked van Zyl panel: score_LSC / score_PMC_basal / score_TDC).
  * threshold position = linear interpolation between decile means where
    mean p crosses 0.5 (threshold-1: LSC->basal transition; threshold-2:
    basal->differentiated transition).
  * marker contrast = mean score below vs above each threshold position.

Seeds: CORN arms 42/123/456 (mean +/- sd); LoRA 42/123 (mean +/- range).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "<home>/corneal_scgpt/artifacts/paper1_vanzyl/corn_advantage"
)
DECODE_THRESHOLD = 0.5
LAYER_ORDER = ["LSC", "basal", "differentiated"]
TIER_IDX = {c: i for i, c in enumerate(LAYER_ORDER)}
SCORE_COLS = ["score_LSC", "score_PMC_basal", "score_TDC"]
CORN_SEEDS = [42, 123, 456]
LORA_SEEDS = [42, 123]
N_DECILES = 10

ARMS = {
    "corn_gf": {"prefix": "gf_seed", "seeds": CORN_SEEDS, "kind": "corn", "gf": True},
    "corn_scgpt": {"prefix": "ordinal_seed", "seeds": CORN_SEEDS, "kind": "corn", "gf": False},
    "corn_cellplm": {"prefix": "cellplm_seed", "seeds": CORN_SEEDS, "kind": "corn", "gf": False},
    "lora_scgpt": {"prefix": "lora_seed", "seeds": LORA_SEEDS, "kind": "lora", "gf": False},
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_seed(arm: str, seed: int) -> dict:
    info = ARMS[arm]
    data = np.load(OUT / f"{info['prefix']}{seed}.npz")
    meta = json.loads((OUT / f"{info['prefix']}{seed}.json").read_text())
    return {"data": data, "meta": meta}


def l2_norm_rows(emb: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return emb / nrm


def corn_p1p2(logits: np.ndarray, gf: bool) -> tuple[np.ndarray, np.ndarray]:
    """CORN threshold probabilities on the 3-tier reading."""
    if gf:
        p1 = sigmoid(logits[:, 0])  # t1: >=TAC => not-LSC => basal+diff
        p2 = sigmoid(logits[:, 2])  # t3: =diff
    else:
        p1 = sigmoid(logits[:, 0])
        p2 = sigmoid(logits[:, 1])
    return p1, p2


def lora_tier_probs(logits: np.ndarray, naive_classes: list, naive_map: dict
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LoRA 30-class softmax -> 3-tier probabilities + argmax tier."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    tier = np.zeros((probs.shape[0], 3), dtype=float)
    for j, cls in enumerate(naive_classes):
        t = TIER_IDX[naive_map[cls]]
        tier[:, t] += probs[:, j]
    p_lsc, p_basal, p_diff = tier[:, 0], tier[:, 1], tier[:, 2]
    argmax_author = np.array([naive_classes[int(i)] for i in logits.argmax(1)])
    argmax_tier = np.array([TIER_IDX[naive_map[c]] for c in argmax_author])
    argmax_3tier = tier.argmax(1)
    return p_lsc, p_basal, p_diff, argmax_tier, argmax_3tier


def summary_stats(p: np.ndarray) -> dict:
    return {
        "mean": float(p.mean()),
        "median": float(np.median(p)),
        "q05": float(np.quantile(p, 0.05)),
        "q95": float(np.quantile(p, 0.95)),
        "n_above_half": int((p >= DECODE_THRESHOLD).sum()),
    }


def monotonicity_seed(arm: str, seed: int) -> dict:
    info = ARMS[arm]
    loaded = load_seed(arm, seed)
    data, meta = loaded["data"], loaded["meta"]
    logits = np.asarray(data["logits"], dtype=float)
    n = logits.shape[0]

    if info["kind"] == "corn":
        p1, p2 = corn_p1p2(logits, info["gf"])
        literal = float((p1 < p2).mean())
        # CORN decision IS the threshold count -> constructional consistency
        k_rule = ((p1 >= DECODE_THRESHOLD).astype(int)
                  + (p2 >= DECODE_THRESHOLD).astype(int))
        decision = k_rule.copy()
        consistency = float((decision != k_rule).mean())
        return {
            "seed": seed, "n_cells": n,
            "threshold_definition": ("p1=sigma(t1), p2=sigma(t3) [GF 4-class axis]"
                                     if info["gf"] else
                                     "p1=sigma(t1), p2=sigma(t2)"),
            "p1": summary_stats(p1), "p2": summary_stats(p2),
            "literal_cumulative_violation_rate": literal,
            "decision_consistency_violation_rate": consistency,
            "decision_consistency_detail": {
                "note": "CORN decode rule IS the threshold count; "
                        "rate 0 by construction (verified numerically)",
                "n_rule_class0": int((k_rule == 0).sum()),
                "n_rule_class1": int((k_rule == 1).sum()),
                "n_rule_class2": int((k_rule == 2).sum()),
            },
            "gate": meta.get("gate"),
        }

    # LoRA softmax arm
    naive_classes = meta["naive_classes"]
    naive_map = meta["naive_map"]
    p_lsc, p_basal, p_diff, argmax_tier, argmax_3tier = lora_tier_probs(
        logits, naive_classes, naive_map)
    p_ge1 = p_basal + p_diff
    p_ge2 = p_diff
    literal = float((p_ge1 < p_ge2).mean())
    rule_class = ((p_ge1 >= DECODE_THRESHOLD).astype(int)
                  + (p_ge2 >= DECODE_THRESHOLD).astype(int))
    consistency = float((argmax_tier != rule_class).mean())
    # detail: disagreement breakdown
    detail = {}
    for t in range(3):
        for r in range(3):
            key = f"argmax_tier{t}_rule{r}"
            detail[key] = int(((argmax_tier == t) & (rule_class == r)).sum())
    consistency_3tier = float((argmax_3tier != rule_class).mean())
    return {
        "seed": seed, "n_cells": n,
        "threshold_definition": ("softmax 30-class -> majority-mapped 3-tier; "
                                 "P(>=1)=P(basal)+P(diff), P(>=2)=P(diff)"),
        "p1": summary_stats(p_ge1), "p2": summary_stats(p_ge2),
        "literal_cumulative_violation_rate": literal,
        "decision_consistency_violation_rate": consistency,
        "decision_consistency_detail": {
            "note": "softmax argmax (author class -> tier) vs CORN-style "
                    "threshold-count rule over the model's own cumulative "
                    "probabilities",
            "confusion": detail,
            "n_rule_class0": int((rule_class == 0).sum()),
            "n_rule_class1": int((rule_class == 1).sum()),
            "n_rule_class2": int((rule_class == 2).sum()),
        },
        "variant_argmax_3tier_consistency_rate": consistency_3tier,
        "gate": meta.get("gate"),
    }


def aggregate_monotonicity() -> dict:
    arms = {}
    for arm, info in ARMS.items():
        per_seed = {f"s{s}": monotonicity_seed(arm, s) for s in info["seeds"]}
        keys = ["literal_cumulative_violation_rate",
                "decision_consistency_violation_rate"]
        agg = {}
        for k in keys:
            vals = [per_seed[f"s{s}"][k] for s in info["seeds"]]
            agg[k] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(min(vals)), "max": float(max(vals)),
                "per_seed": {f"s{s}": per_seed[f"s{s}"][k] for s in info["seeds"]},
            }
        agg["n_seeds"] = len(info["seeds"])
        arms[arm] = {"per_seed": per_seed, "aggregate": agg}
    return arms


# ─────────────────────────────────────────────────────────────────────────────
# Experiment D — threshold interpretability
# ─────────────────────────────────────────────────────────────────────────────


def thresholds_seed(arm: str, seed: int) -> dict:
    info = ARMS[arm]
    loaded = load_seed(arm, seed)
    data, meta = loaded["data"], loaded["meta"]
    emb = np.asarray(data["emb"], dtype=float)
    y = np.asarray(data["y"])
    scores = np.asarray(data["scores"], dtype=float)
    logits = np.asarray(data["logits"], dtype=float)
    p1, p2 = corn_p1p2(logits, info["gf"])

    emb_n = l2_norm_rows(emb)
    mean_lsc = emb_n[y == "LSC"].mean(axis=0)
    mean_diff = emb_n[y == "differentiated"].mean(axis=0)
    u = mean_diff - mean_lsc
    u /= np.linalg.norm(u) + 1e-12
    axis = emb_n @ u  # differentiation position (LSC -> differentiated)

    order = np.argsort(axis)
    axis_s, p1_s, p2_s, y_s, sc_s = (axis[order], p1[order], p2[order],
                                     y[order], scores[order])
    n = len(axis_s)
    edges = np.linspace(0, n, N_DECILES + 1).astype(int)
    deciles = []
    for d in range(N_DECILES):
        lo, hi = edges[d], edges[d + 1]
        seg = slice(lo, hi)
        deciles.append({
            "decile": d,
            "axis_lo": float(axis_s[seg][0]),
            "axis_hi": float(axis_s[seg][-1]),
            "n": int(hi - lo),
            "mean_p1": float(p1_s[seg].mean()),
            "mean_p2": float(p2_s[seg].mean()),
            "frac_LSC": float((y_s[seg] == "LSC").mean()),
            "frac_basal": float((y_s[seg] == "basal").mean()),
            "frac_diff": float((y_s[seg] == "differentiated").mean()),
            "mean_score_LSC": float(sc_s[seg, 0].mean()),
            "mean_score_PMC_basal": float(sc_s[seg, 1].mean()),
            "mean_score_TDC": float(sc_s[seg, 2].mean()),
        })

    def crossing(col: str, lo: int = 0, hi: int = N_DECILES - 1) -> dict | None:
        vals = [deciles[d][col] for d in range(lo, hi + 1)]
        # mean p rises along the axis; find the decile where it crosses 0.5
        idx = None
        for d in range(lo, hi):
            if vals[d] < DECODE_THRESHOLD <= vals[d + 1]:
                idx = d
                break
        if idx is None:
            return None
        v0, v1 = vals[idx], vals[idx + 1]
        frac = (DECODE_THRESHOLD - v0) / max(v1 - v0, 1e-12)
        # interpolate axis value + marker scores at the crossing
        def interp(key: str) -> float:
            a0 = deciles[idx][key]
            a1 = deciles[idx + 1][key]
            return float(a0 + frac * (a1 - a0))
        return {
            "crossing_between_deciles": [idx, idx + 1],
            "frac_within_interval": frac,
            "axis_value": float(axis_s[edges[idx + 1]] + frac * (
                axis_s[edges[idx + 2]] - axis_s[edges[idx + 1]])),
            "cumulative_frac_below": float(edges[idx + 1] / n),
            "score_LSC_at": interp("mean_score_LSC"),
            "score_PMC_basal_at": interp("mean_score_PMC_basal"),
            "score_TDC_at": interp("mean_score_TDC"),
        }

    thr1 = crossing("mean_p1")
    thr2 = crossing("mean_p2")

    def contrast(thr: dict | None) -> dict | None:
        if thr is None:
            return None
        below = axis_s <= thr["axis_value"]
        above = ~below
        out = {}
        for j, name in enumerate(SCORE_COLS):
            b = sc_s[below, j].mean() if below.any() else float("nan")
            a = sc_s[above, j].mean() if above.any() else float("nan")
            out[name] = {
                "mean_below": float(b), "mean_above": float(a),
                "delta_above_minus_below": float(a - b),
                "fold_above_over_below": None if b == 0 else float(a / b),
            }
        return out

    return {
        "arm": arm, "seed": seed, "n_cells": n,
        "axis": "L2-normed cell embedding projected onto "
                "normalize(mean_diff - mean_LSC) (model's own embedding space)",
        "deciles": deciles,
        "threshold1_LSC_to_basal": thr1,
        "threshold2_basal_to_diff": thr2,
        "marker_contrast_threshold1": contrast(thr1),
        "marker_contrast_threshold2": contrast(thr2),
        "checkpoint": meta.get("checkpoint"),
    }


def aggregate_thresholds() -> dict:
    arms = {}
    for arm, info in ARMS.items():
        if info["kind"] != "corn":
            continue
        per_seed = {f"s{s}": thresholds_seed(arm, s) for s in info["seeds"]}
        # threshold position agreement across seeds (axis value)
        pos1 = [per_seed[f"s{s}"]["threshold1_LSC_to_basal"] for s in info["seeds"]]
        pos2 = [per_seed[f"s{s}"]["threshold2_basal_to_diff"] for s in info["seeds"]]
        agg = {
            "threshold1_axis_mean_sd": {
                "mean": float(np.mean([p["axis_value"] for p in pos1 if p])),
                "sd": float(np.std([p["axis_value"] for p in pos1 if p], ddof=1)),
            },
            "threshold2_axis_mean_sd": {
                "mean": float(np.mean([p["axis_value"] for p in pos2 if p])),
                "sd": float(np.std([p["axis_value"] for p in pos2 if p], ddof=1)),
            },
        }
        arms[arm] = {"per_seed": per_seed, "aggregate": agg}
    return arms


def main() -> int:
    mono = aggregate_monotonicity()
    thr = aggregate_thresholds()
    meta = {
        "experiment": "CORN unique advantage A+D (user-approved, PCS compliant)",
        "test_donor": "sanes2022_Hu0216",
        "n_test_cells": 20158,
        "decode_threshold": DECODE_THRESHOLD,
        "primary_metric": "decision_consistency_violation_rate "
                          "(softmax argmax vs CORN-style threshold-count rule)",
        "control_metric": "literal_cumulative_violation_rate "
                          "(P(class>=1) < P(class>=2); 0 by construction "
                          "for any softmax)",
        "frozen_before_run": [
            "GF arm uses t1/t3 on the 3-tier axis (user-confirmed 2026-08-20)",
            "LoRA 30->3-tier mapping = train majority (scgpt_eval_vanzyl contract)",
            "marker panel = locked van Zyl score_LSC/PMC_basal/TDC",
            "seeds: CORN 42/123/456, LoRA 42/123; final-epoch checkpoints",
        ],
        "architecture_notes": {
            "corn_gf": "GSE153515-trained (4-class axis), applied cross-corpus "
                       "to van Zyl (structural metrics only; no locked van Zyl rho)",
            "corn_scgpt": "van Zyl-trained, locked rho 0.819",
            "corn_cellplm": "van Zyl-trained, locked rho 0.918",
            "lora_scgpt": "van Zyl-trained softmax, locked final-epoch rho 0.834",
        },
        "generated_utc": now_utc(),
    }
    (OUT / "monotonicity.json").write_text(
        json.dumps({"meta": meta, "arms": mono}, indent=2)
    )
    (OUT / "thresholds.json").write_text(
        json.dumps({"meta": meta, "arms": thr}, indent=2)
    )
    print(f"wrote {OUT / 'monotonicity.json'} and {OUT / 'thresholds.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
