# epi4-CE — 4-class CE baselines on corneal epithelial cells (Paper 1 referee control)

Purpose: separate **CE supervision** from **label-set difference** in the
naive-vs-ordinal comparison. The naive arm trains cross-entropy over 31
`author_cell_type` classes; the ordinal (CORN) arm trains over the 4 corneal
stages on epithelial cells only. These arms train **CE over the same 4 stages**,
so the ordinal head can be compared at matched supervision.

## Layout and which comparison is primary

```
results/paper1_vanzyl/epi4_ce/
├── README.md                          this file
├── summary.json                       combined record (variants + reference arms)
├── paired_drho_reference_arms.json     frozen−naive, frozen−ordinal, naive−ordinal
├── paired_drho_drop_nonepi.json        frozen−drop_nonepi, drop_nonepi−ordinal
├── paired_drho_mask_nonepi.json         frozen−mask_nonepi, mask_nonepi−ordinal
├── gpu_contention_probe.json           card facts, contention test, ETA basis
├── drop_nonepi/                       VARIANT A — COMPLETE (27,878 rows, 2,616 steps)
│   ├── summary.json, provenance_enrichment.json
│   └── seed{42,123,456}/{eval_gse153515.json, percell_epi4_{seed}.npz}
└── mask_nonepi/                       VARIANT B — COMPLETE (79,684 rows, 7,473 steps)
    ├── summary.json, provenance_enrichment.json
    └── seed{42,123,456}/{eval_gse153515.json, percell_epi4_{seed}.npz}
```

**Primary comparison: `mask_nonepi`** (complete, three seeds). It supervises the
same epithelial cells as the ordinal arm **and** takes the same number of
optimizer steps (7,473), so label set, supervising cells, loss form and step
budget are matched. `drop_nonepi` is the secondary control: it additionally removes the
51,806 non-epithelial rows, which changes row composition and step count
(2,616), confounding the label-space question with those two.

| arm | train rows | rows contributing loss | steps/epoch | steps (3 epochs) | implementation |
|---|---|---|---|---|---|
| `drop_nonepi` | 27,878 | 27,878 | 872 | **2,616** | `--arm ce --label-space epi4` |
| `mask_nonepi` (primary) | 79,684 | 27,878 (epithelial only) | 2,491 | **7,473** | `--arm axis_ft --lambda-arm 0.0` |
| ordinal-only CORN (reference) | 79,684 | 27,878 | 2,491 | 7,473 | `scripts/ordinal_gf.py` |
| naive 31-class CE (reference) | 79,684 | 79,684 | 2,491 | 7,473 | `scripts/caft_gf.py` |

`mask_nonepi` = `AxisFTTrainer` at `axis_lambda=0`: `compute_loss` returns
`ordinal_ce_loss(...)` (4-class CE on `class_to_ordinal`, epithelial-masked,
non-epithelial rows contribute nothing), with no anchor term, no h0 capture and
`output_hidden_states=False`; head width 4. Verified in isolation:
`loss == F.cross_entropy(logits[valid], ordinals[valid])`, relabelling a
non-epithelial row to another non-epithelial class leaves the loss bit-identical,
an all-non-epithelial batch yields a grad-carrying 0.0.

## `drop_nonepi` — three seeds (GSE153515, 14,775 epithelial cells, 2 donors)

| seed | ρ | macro-F1 [95 % CI] | LAA | ECE | steps | train wall |
|---|---|---|---|---|---|---|
| 42 | 0.2142 | 0.6632 [0.6370, 0.6894] | 0.9794 | 0.0718 | 2,616 | 4,467 s |
| 123 | 0.1413 | 0.6597 [0.6346, 0.6848] | 0.9827 | 0.0774 | 2,616 | 3,894 s |
| 456 | 0.1191 | 0.6583 [0.6290, 0.6876] | 0.9796 | 0.0817 | 2,616 | 3,848 s |
| **mean ± SD** | **0.1582 ± 0.0497** | **0.6604 ± 0.0025** | 0.9806 ± 0.0019 | 0.0770 ± 0.0050 | 2,616 each | |

Per-class F1 (mean ± SD over the three seeds):

| class | mean ± SD | per seed (42 / 123 / 456) |
|---|---|---|
| LSC | 0.8671 ± 0.0033 | 0.8685 / 0.8695 / 0.8633 |
| TAC | **0.0000 ± 0.0000** | 0.0000 / 0.0000 / 0.0000 |
| basal | 0.9024 ± 0.0031 | 0.9059 / 0.9016 / 0.8997 |
| differentiated | 0.8495 ± 0.0060 | 0.8563 / 0.8472 / 0.8450 |

TAC (support 171 / 14,775 cells) is predicted zero times by all three seeds —
visible in `per_class.TAC.recall = 0` and in row 2 of each `confusion` matrix
(e.g. seed 42: `[17, 0, 149, 5]`).

## `mask_nonepi` — three seeds (GSE153515, 14,775 epithelial cells, 2 donors)

| seed | ρ | macro-F1 [95 % CI] | LAA | ECE | steps | train wall |
|---|---|---|---|---|---|---|
| 42 | 0.2663 | 0.6639 [0.6395, 0.6883] | 0.9779 | 0.0965 | 7,473 | 9,131 s |
| 123 | 0.2293 | 0.6544 [0.6243, 0.6844] | 0.9809 | 0.1091 | 7,473 | 9,126 s |
| 456 | 0.2667 | 0.6545 [0.6221, 0.6870] | 0.9796 | 0.1087 | 7,473 | 9,133 s |
| **mean ± SD** | **0.2541 ± 0.0215** | **0.6576 ± 0.0054** | 0.9794 ± 0.0015 | 0.1048 ± 0.0072 | 7,473 each | |

Per-class F1 (mean ± SD over the three seeds):

| class | mean ± SD | per seed (42 / 123 / 456) |
|---|---|---|
| LSC | 0.8580 ± 0.0096 | 0.8688 / 0.8547 / 0.8504 |
| TAC | **0.0000 ± 0.0000** | 0.0000 / 0.0000 / 0.0000 |
| basal | 0.8988 ± 0.0065 | 0.9063 / 0.8945 / 0.8956 |
| differentiated | 0.8479 ± 0.0092 | 0.8585 / 0.8420 / 0.8434 |

TAC (support 171 / 14,775 cells) is again predicted zero times by all three
seeds. Step-matched against the ordinal arm, the four-class CE head retains
0.2541 of a possible 0.8330 — the label set explains about a quarter of the
naive-to-ordinal gap, and the step budget (2,616 → 7,473) moves it the wrong
way for a "simply undertrained" account.

## Paired Δρ contrasts (donor-stratified bootstrap, axis refit per replicate, 2,000 reps, seed 42)

Cells resampled with replacement **within each donor** (donor_1 n=5,339;
donor_2 n=9,436); one index vector per replicate applied to both arms; each
arm's per-layer-centroid axis re-fitted inside the replicate via
`corneal_fm.evaluation.marker_gradient_correlation`. Sign:
`Δρ = ρ(arm_A) − ρ(arm_B)`.

| contrast | Δρ mean over seeds | per-seed Δρ [95 % CI] |
|---|---|---|
| frozen − naive | +0.8145 | s42 +0.8488 [+0.7721,+0.9152]; s123 +0.8424 [+0.7629,+0.9133]; s456 +0.7524 [+0.6756,+0.8542] |
| frozen − ordinal | −0.0266 | s42 −0.0277 [−0.0424,−0.0172]; s123 −0.0235 [−0.0378,−0.0130]; s456 −0.0287 [−0.0430,−0.0182] |
| naive − ordinal (ordinal − naive = +0.8412) | −0.8412 | s42 −0.8765 [−0.9382,−0.8055]; s123 −0.8659 [−0.9329,−0.7922]; s456 −0.7811 [−0.8795,−0.7095] |
| **frozen − drop_nonepi** | **+0.6482** (sd 0.0497) | s42 +0.5922 [+0.5340,+0.6835]; s123 +0.6650 [+0.6044,+0.7694]; s456 +0.6873 [+0.6227,+0.8055] |
| **drop_nonepi − ordinal** | **−0.6748** (sd 0.0495) | s42 −0.6199 [−0.7149,−0.5579]; s123 −0.6886 [−0.7888,−0.6303]; s456 −0.7160 [−0.8308,−0.6542] |
| **frozen − mask_nonepi** | **+0.5523** (sd 0.0215) | s42 +0.5400 [+0.5109,+0.5697]; s123 +0.5771 [+0.5429,+0.6161]; s456 +0.5396 [+0.5106,+0.5697] |
| **mask_nonepi − ordinal** | **−0.5789** (sd 0.0188) | s42 −0.5678 [−0.5992,−0.5380]; s123 −0.6006 [−0.6425,−0.5652]; s456 −0.5684 [−0.5996,−0.5388] |
| frozen − mask_nonepi / mask_nonepi − ordinal | complete | `paired_drho_mask_nonepi.json` |

### Interval conventions in `paired_drho_*.json`

Every pair records four intervals, so any number quoted in the manuscript can be
recomputed by hand:

| field | what it is |
|---|---|
| `per_seed.<seed>.ci_lower/ci_upper` | that seed's own 2.5th/97.5th percentiles over its 2,000 replicates (`ci_method` states this) |
| `averaged_ci95.ci95` | arithmetic mean of the three per-seed bounds — the convention the manuscript uses for the frozen−naive contrast (mean Δρ 0.8145, averaged bounds 0.7369/0.8942, printed 0.737–0.894) |
| `pooled.ci95` | percentile over the *concatenated* per-seed replicates (3 × 2,000) |
| `across_seed.ci95` | two-sided 95 % t-interval on the three seed means (`df` = 2, `t` = 4.303) |

The averaged interval is the one the four-class-CE contrast quotes in Results
§6; `pooled` and `across_seed` are stored as alternatives, not as a correction.

Pairing pre-check (all arms): labels **and** donor arrays element-wise identical
across the 14,775 cells — ordinal-embedding `labels` ≡ Tier-B `y_true_ext` ≡
`artifacts/paper1_corn_boundary` `labels`, donors `donor_1`/`donor_2` identical;
recomputing ρ from the stored embeddings reproduces the recorded values exactly
(frozen 0.8063606041550953, naive s42 −0.04239536391926124, ordinal s42
0.8340699807044084, drop_nonepi s42 0.21415669551060146).

## Launch — host, GPU, exact commands

* host `p5070` (`summer@192.168.50.2`, `summer-NUC13RNGi7`), NVIDIA RTX PRO 5000
  72 GB Blackwell (73,415 MiB, driver 595.91.07, CUDA 13.2), one job at a time
* workspace `/home/summer/epi4_ce`; data symlinked to sha-verified copies
  (train `bc357c7d…`, external `2e4ef308…`), manifest `f4393ae1…`

```
# drop_nonepi (3 seeds, 2,616 steps each)
python3 scripts/method_arm_gf.py --arm ce --label-space epi4 \
    --data_file corneal_100k_donor_split.h5ad --epochs 3 --seed {42,123,456} \
    --strict-fair --split_manifest artifacts/paper1_fair_split/manifest.json
bash scripts/eval_with_diagnostics.sh {SEED} --skip-tokenize \
    --out-dir results/paper1_vanzyl/epi4_ce/drop_nonepi

# mask_nonepi (primary; 3 seeds, 7,473 steps each) — started 19:00:41
python3 scripts/method_arm_gf.py --arm axis_ft --lambda-arm 0.0 \
    --data_file corneal_100k_donor_split.h5ad --epochs 3 --seed {42,123,456} \
    --strict-fair --split_manifest artifacts/paper1_fair_split/manifest.json
bash scripts/eval_with_diagnostics.sh {SEED} --skip-tokenize \
    --out-dir results/paper1_vanzyl/epi4_ce/mask_nonepi \
    --arm-tag method_axis_ft --variant mask_nonepi
```

Hyperparameters (identical to the locked naive/ordinal arms):
`epochs=3`, per-device batch 8 × grad-accum 4 (effective 32), `lr=5e-5`,
`lr_scheduler=linear`, `warmup_steps=500`, `weight_decay=0.001`,
`max_grad_norm=1.0`, bf16 weights, gradient checkpointing, `save_strategy=epoch`,
final-epoch checkpoint (step 2,616 for `drop_nonepi`), `group_by_length=True`.
Epithelial split of the fair-split train partition:
`kept=27878 dropped=51806`, stages LSC 3,634 · TAC 6,215 · basal 5,853 ·
differentiated 12,176.

## Code change (`scripts/method_arm_gf.py`, additive only)

Diff baseline: `save/method_soft_spearman_lam0.5_seed456_20260814-183748/method_arm_gf.py`.

* new `--label-space {author31,epi4}`, **default `author31`** — every existing
  invocation unchanged (`apply_label_space` returns its inputs untouched);
* new `--arm ce` (plain CE; `lambda` forced to 0) — same `CAFTTrainer` at
  `caft_lambda=0.0` as the naive arm;
* `epi4`: drop non-epithelial rows, rewrite `label` to the 4 stage indices,
  4-wide head, frozen axis skipped;
* guards: `--label-space epi4` requires `--arm ce`; `--arm ce` requires `lambda=0`;
* provenance additions: `label_space`, `label_space_info`, `class_names`,
  `n_ce_classes`.
* `mask_nonepi` needs **no** code change: `--arm axis_ft --lambda-arm 0.0` is
  the already-existing masked 4-class CE path.

## Evaluation script: both config layouts + diagnostics

`scripts/epi4_ce_eval.py` resolves the head width without assuming
`config.json["num_labels"]`, which this export path never writes: order
`num_labels` → `len(id2label)` → `classifier.weight.shape[0]`. Verified on
`caft_gf` (31), `ordinal_gf` (3) and `axis_ft` (4) checkpoints — the same
exporter used by `mask_nonepi`, so both layouts are handled.

The first seed-42 evaluation (16:24:23, rc=1 after 32 s) died on
`KeyError: 'num_labels'` — a Python exception, not a signal and not a GPU event
(no Xid / OOM in the window; the only host Xid is from 2026-10-20 22:39). Fixed
by the fallback above; the retry succeeded at 16:31:35. All later evaluations run
through `scripts/eval_with_diagnostics.sh` (`python3 -X faulthandler -u`,
separate stdout/stderr per attempt, append-only `logs/eval_seed{S}.attempts.log`
with explicit rc/signal, no retry loop).

## Scheduling note

The card is compute/power-bound (SM 100 %, 295–303 W of a 300 W cap, VRAM ~4.3 of
73.4 GB). A measured contention test found two concurrent training jobs each
degrade ~2× (epi4 1.40 → 2.88–3.11 s/step) for ~3.4 % aggregate throughput, so
all arms run strictly serialised (`gpu_contention_probe.json`).

## Artifact hashes (sha256)

| file | sha256 |
|---|---|
| `drop_nonepi/seed42/eval_gse153515.json` | `ff7b914707833b88ca2dcf5d79acbc5ba8cff6f0814a8c7fd196a76a8fd4fa52` |
| `drop_nonepi/seed123/eval_gse153515.json` | `4d96f490842213bb3dbd25e7eca374ded601cdb723ded4abedb975bd370d3196` |
| `drop_nonepi/seed456/eval_gse153515.json` | `63e0f22222a629af5cabcaa534f52743981575290073c6feac22b07ce3b6b022` |
| `drop_nonepi/seed42/percell_epi4_42.npz` | `fd1ce9ee58e12e4878a6ec514e075713f0f45bac245f264537d21c3c4b49e885` |
| `drop_nonepi/seed123/percell_epi4_123.npz` | `483dfb950cfc1147b8cbb1caa530eaecf292ca6492b974328582f699593cc9fa` |
| `drop_nonepi/seed456/percell_epi4_456.npz` | `bb2e22026945b8cea35c1ca96c4363ee9154ce88967a658f5794c9437b208d42` |
| `drop_nonepi/summary.json` | `55f0f85d5f9a25fdda80ad1c0ebcbfcdf6e53045b8c4e14b4419ff6c9a295f2c` |
| `mask_nonepi/seed42/eval_gse153515.json` | `f807fe137ab7f460ab736afc832b91366e22e3957eea5c8125bc0313c041366c` |
| `mask_nonepi/seed123/eval_gse153515.json` | `1ba51281cfcf46af29a887e47bcb21bcd3f8052397bc795fc876e230023d2f26` |
| `mask_nonepi/seed456/eval_gse153515.json` | `a96e7186f04a74189a25a19a5a431bd0afaeb41f601d1d2c5954752aced69722` |
| `mask_nonepi/seed42/percell_epi4_42.npz` | `ee3d87331364927bca97850bffeaf5c68e300e54e5561a76613a6633721fc7b0` |
| `mask_nonepi/seed123/percell_epi4_123.npz` | `d944cce39d4512d893f1ad85b8986f41c2e41dd1810d12859fc32e4da7793850` |
| `mask_nonepi/seed456/percell_epi4_456.npz` | `072d4199b1cc647139705157de7b28325172d081989ca2fd882b4ea92003a94c` |
| `mask_nonepi/summary.json` | `4e72189fe305471ae4710659e3e3bc12db55d2e969b2c9d780d52c8d597576ce` |
| `mask_nonepi/provenance_enrichment.json` | `3a24f5dd8632d187dd08c69ae066e374e9ed89d9f31f8b34942ecb91fa3ed8bb` |
| `paired_drho_mask_nonepi.json` | `07695c3f117bf158441070e237201b56897ec95331ebb99f2a85dc91a9297422` |
| `summary.json` | `235cd4e76c6254d12b8bda4fd384998b00f92c6151cbc4d6ded36956abe172a0` |
| `scripts/method_arm_gf.py` (local = remote) | `7bf09d2cedac7fe9b73ed9f16ee65dcae506a78dca60e53a1094f7ab359c522b` |
| `scripts/epi4_ce_eval.py` (local = remote) | `267d2ba8d7b4028813f4c3f6a719ebe3bf8b3077223120a630c144c73b8cc7d8` |
| `scripts/eval_with_diagnostics.sh` (local = remote) | `f34f3038d099042c3f6a59f0862bf0bb03bf96dbe2104e745a0040d671c1e941` |

Checkpoints stay on `p5070` under
`~/epi4_ce/save/method_ce_epi4_lam0.0_seed{S}_<ts>/model/checkpoint-2616`
(final epoch); each record's `archived_inputs.checkpoint` carries the
`model.safetensors` / `config.json` hashes. Nothing under `docs/manuscript/**`
was touched.
