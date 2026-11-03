# CellPLM LoRA-only arm (van Zyl, third architecture)

Recipe (unchanged across seeds): LoRA on the `Flow_Attention`
`query/key/value` projections, **r = 8, alpha = 16**, base trunk frozen,
adapters + classification head + latent trained through the official
`CellTypeAnnotationPipeline.fit()`; test donor `sanes2022_Hu0216`
(20,158 eval cells); 38,233 train / 4,249 valid cells.

* script: `scripts/cellplm_lora_vanzyl.py`
  (sha256 `fa979f425c5d4db77dcb51c273c2ed6ae1e89e0c3dd8808b3f940caab807b950`)
* launch (per seed, on `<compute-node>`):
  `<home>/corneal_scgpt/venv/bin/python scripts/cellplm_lora_vanzyl.py --seed {42,123,456} --out-root <home>/corneal_scgpt/artifacts/paper1_vanzyl/cellplm_lora`
* raw records: `seed{42,123,456}/seed{seed}.json` in this directory
  (pulled from the path above); the fitted checkpoint and per-cell embeddings
  stay on `<compute-node>` (`model_best_valid.pt`, `cell_emb.npz`, `frozen_emb.npz`).

## Per-seed results

| seed | ρ (normed) | ρ (raw) | macro-F1 | acc | **epochs actually run** | elapsed (min) | record sha256 |
|---|---|---|---|---|---|---|---|
| 42 | 0.8735 | 0.1984 | 0.9578 | 0.9644 | **184** | 9.82 | `adfe5413…` |
| 123 | 0.1542 | 0.1172 | 0.9595 | 0.9665 | **366** | 6.85 | `e712b744…` |
| 456 | 0.6262 | 0.2537 | 0.9516 | 0.9590 | **80** | 3.40 | `adac1835…` |
| **mean ± SD** | **0.5513 ± 0.3654** | 0.1898 ± 0.0702 | **0.9563 ± 0.0041** | 0.9633 ± 0.0039 | **210.0 ± 144.8** | 6.69 ± 3.24 | |

(`SD` = sample SD, ddof = 1. Frozen reference on the same test donor:
ρ_normed 0.8652, `seed42.json → frozen_baseline`.)

## Seed sensitivity and the early-stopping spread

ρ (normed) is seed-unstable (0.1542 – 0.8735, SD 0.3654) while macro-F1 is
stable (0.9516 – 0.9595, SD 0.0041). **Early stopping fired at 80 / 184 / 366
epochs** for seeds 456 / 42 / 123 — a 4.6× spread in the number of epochs
actually run (`training.epochs_actual`, `best_valid_f1` 0.9444 / 0.9471 /
0.9521). The epoch counts are recorded per seed in `summary.json`
(`aggregate.epochs_actual`, `per_seed.*.epochs_actual`) so the spread is
visible alongside the mean; the mean alone (0.5513) is reported in the main
text with an explicit seed-sensitivity note.
