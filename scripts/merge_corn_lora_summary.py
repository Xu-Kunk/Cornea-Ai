#!/usr/bin/env python
"""E5 merge: CORN+LoRA 3-seed aggregation + cross-arm comparison.

Reads per-seed jsons (corn_lora/seed{S}_args.json, written by the runner)
and the archived reference jsons (CORN-only 6-run eval, LoRA-only lora_v2,
naive convergence eval, frozen baseline), then writes corn_lora/summary.json:

  * combo aggregate: rho_normed / null_z / macro_f1 mean+sd over seeds
    {42,123,456}, separately for best_val and final_epoch checkpoints
  * comparison table vs CORN-only / LoRA-only / naive / frozen with
    per-mode verdict: does the combination beat every single arm?

Combo = 15 epochs, best-val (macro-F1) primary + final-epoch sensitivity,
exactly the E4-LoRA protocol; CORN-only is the §8 3-epoch final checkpoint
(different epoch budget — disclosed, not asserted as a controlled contrast).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("<home>/corneal_scgpt")
OUT = ROOT / "artifacts" / "paper1_vanzyl" / "corn_lora"
SEEDS = [42, 123, 456]
CKPT_TYPES = ["best_val", "final_epoch"]
LAYER_ORDER = ["LSC", "basal", "differentiated"]

# archived references
CORNAL_EVAL = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "eval"
LORA_CKPT = ROOT / "artifacts" / "paper1_vanzyl" / "run6" / "lora_v2" / "checkpoints"
NAIVE_EVAL = ROOT / "artifacts" / "paper1_vanzyl" / "convergence" / "final_test_eval.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(vals: list[float]) -> dict:
    import numpy as np
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "per_seed": {f"s{s}": float(v) for s, v in zip(SEEDS, vals)}}


def main() -> int:
    combo = {s: load(OUT / f"seed{s}_args.json") for s in SEEDS}
    agg: dict = {}
    for ck in CKPT_TYPES:
        agg[ck] = {
            "rho_normed": mean_sd([combo[s]["test_donor_eval"][ck]["rho_normed"]
                                   for s in SEEDS]),
            "rho_raw": mean_sd([combo[s]["test_donor_eval"][ck]["rho_raw"]
                                for s in SEEDS]),
            "null_z": mean_sd([combo[s]["test_donor_eval"][ck]["null_z"]
                               for s in SEEDS]),
            "macro_f1": mean_sd([combo[s]["test_donor_eval"][ck]["macro_f1"]["macro_f1"]
                                 for s in SEEDS]),
        }

    # ── reference arms ──────────────────────────────────────────────────
    # CORN-only: §8 3-epoch final checkpoints (6-run eval jsons)
    corn = {s: load(CORNAL_EVAL / f"ordinal_seed{s}.json") for s in (42, 123, 456)}
    corn_final = {
        "rho_normed": [corn[s]["rho_normed"] for s in (42, 123, 456)],
        "null_z": [corn[s]["null_z"] for s in (42, 123, 456)],
        "macro_f1": [corn[s]["macro_f1"]["macro_f1"] for s in (42, 123, 456)],
    }
    import numpy as np
    corn_seeds_42_123 = {"rho_normed": float(np.mean([corn[42]["rho_normed"],
                                                      corn[123]["rho_normed"]])),
                         "null_z": float(np.mean([corn[42]["null_z"], corn[123]["null_z"]])),
                         "macro_f1": float(np.mean([corn[42]["macro_f1"]["macro_f1"],
                                                    corn[123]["macro_f1"]["macro_f1"]]))}

    # LoRA-only: E4-LoRA v2 (same protocol: 15 ep, best-val + final-epoch)
    lora = {s: load(LORA_CKPT / f"lora_seed{s}" / "args.json") for s in SEEDS}
    lora_arms = {}
    for ck in CKPT_TYPES:
        lora_arms[ck] = {
            "rho_normed": [lora[s]["test_donor_eval"][ck]["rho_normed"] for s in SEEDS],
            "null_z": [lora[s]["test_donor_eval"][ck]["null_z"] for s in SEEDS],
            "macro_f1": [lora[s]["test_donor_eval"][ck]["macro_f1"]["macro_f1"] for s in SEEDS],
        }

    # naive: 15-epoch convergence arm (3 seeds; bistable per-seed)
    naive = load(NAIVE_EVAL)
    naive_arms = {}
    for ck, key in [("best_val", "best_val"), ("final_epoch", "final_epoch")]:
        n = naive["aggregate"][key]
        naive_arms[ck] = {"rho_normed": n["rho_normed"]["mean"],
                          "null_z": n["null_z"]["mean"],
                          "macro_f1": n["macro_f1"]["mean"]}

    # frozen baseline (G1 npz, test-donor masked)
    frozen = load(CORNAL_EVAL / "frozen_baseline.json")
    frozen_rho = frozen["rho"]["rho_normed"]

    # ── comparison + verdict (combo > every single arm, per ckpt mode) ──
    comparison: dict = {}
    verdicts: dict = {}
    for ck in CKPT_TYPES:
        c_rho = agg[ck]["rho_normed"]["mean"]
        l_rho = float(np.mean(lora_arms[ck]["rho_normed"]))
        n_rho = naive_arms[ck]["rho_normed"]
        corn_mean = float(np.mean(corn_final["rho_normed"]))
        ck_comparison = {
            "combo_mean": c_rho,
            "combo_per_seed": agg[ck]["rho_normed"]["per_seed"],
            "corn_only_3ep_final_mean_3seeds": corn_mean,
            "corn_only_3ep_final_mean_seeds_42_123": corn_seeds_42_123["rho_normed"],
            "corn_only_per_seed_final": {f"s{s}": corn[s]["rho_normed"] for s in (42, 123, 456)},
            "lora_only_mean": l_rho,
            "lora_only_per_seed": {f"s{s}": lora_arms[ck]["rho_normed"][i]
                                   for i, s in enumerate(SEEDS)},
            "naive_mean": n_rho,
            "frozen_baseline": frozen_rho,
            "aligned_seed_42_123_lora": {f"s{s}": {
                "combo": agg[ck]["rho_normed"]["per_seed"][f"s{s}"],
                "lora": lora[s]["test_donor_eval"][ck]["rho_normed"],
                "corn_3ep_final": corn[s]["rho_normed"]} for s in SEEDS},
        }
        comparison[ck] = ck_comparison
        beat = {
            "combo_gt_corn_3ep_final": bool(c_rho > corn_mean),
            "combo_gt_lora": bool(c_rho > l_rho),
            "combo_gt_naive": bool(c_rho > n_rho),
            "combo_gt_frozen": bool(c_rho > frozen_rho),
        }
        beat["combo_gt_every_single_arm"] = bool(all(beat.values()))
        beat["note"] = ("combo is 15-epoch best-val/final; CORN-only is the §8 "
                        "3-epoch final checkpoint (epoch-budget mismatch disclosed); "
                        "LoRA-only is the aligned 15-epoch E4-LoRA v2 protocol")
        verdicts[ck] = beat

    summary = {
        "arm": "CORN ordinal head + QKV LoRA (r=8 alpha=16)",
        "seeds": SEEDS,
        "epochs": 15,
        "aggregate": agg,
        "reference_arms": {
            "corn_only_3ep_final_3seeds": {k: float(np.mean(v)) for k, v in corn_final.items()},
            "corn_only_per_seed": {f"s{s}": {"rho_normed": corn[s]["rho_normed"],
                                             "null_z": corn[s]["null_z"],
                                             "macro_f1": corn[s]["macro_f1"]["macro_f1"]}
                                   for s in (42, 123, 456)},
            "lora_only": {ck: {k: list(v) for k, v in lora_arms[ck].items()}
                          for ck in CKPT_TYPES},
            "naive_15ep": naive_arms,
            "frozen_baseline_test_donor": {"rho_normed": frozen_rho,
                                           "null_z": frozen["rho"]["null_z"],
                                           "probe_f1": frozen["probe_f1"]["f1_macro"]},
        },
        "comparison_rho_normed": comparison,
        "verdicts": verdicts,
        "meta": {
            "eval_set": "test donor sanes2022_Hu0216 (20,158 cells)",
            "bootstrap": {"n": 1000, "seed": 42, "level": "cell-level"},
            "null": {"n": 100, "seed": 42, "primary": "L2-normed"},
            "layer_order": LAYER_ORDER,
            "source_jsons": {
                "combo_per_seed": [f"corn_lora/seed{s}_args.json" for s in SEEDS],
                "corn_only": "run6/eval/ordinal_seed{S}.json (6-run eval suite)",
                "lora_only": "run6/lora_v2/checkpoints/lora_seed{S}/args.json",
                "naive": "convergence/final_test_eval.json",
                "frozen": "run6/eval/frozen_baseline.json",
            },
        },
    }
    out = OUT / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved " + str(out))
    print(json.dumps(verdicts, indent=1))
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
