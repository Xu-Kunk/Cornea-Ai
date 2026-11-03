# FDR table consistency note — 2026-10-09

**Scope:** `results/paper1_vanzyl/fdr/bh_fdr_table.json` and
`artifacts/paper1_vanzyl/fdr/bh_fdr_table.json`.

## Discrepancy

The row `GF epidermis naive` (architecture GF, dataset GSE147424,
tissue epidermis) records `rho = 0.4551`, computed from the **single
seed-42** Geneformer naive checkpoint (`z_source`: "computed locally
(n=100 seed 42, rho gate PASS)"; `z = 53.69`, `q = 0.0`).

The manuscript reports the **three-seed mean** for the same cell
(GSE147424, CE-naive): `0.298` (per-seed range `0.19--0.35`), in
Table `tab:domains` and the associated text.

The two numbers describe the same arm/domain at different seed
aggregation (single seed vs three-seed mean); they are not a data
error. The single-seed value is higher than the three-seed mean.

## Impact

- The row remains significant (`q = 0.0`), so it does **not** change the
  BH-FDR family result (20 of 21 significant) nor the
  `21 comparisons` / `20 significant` count reported in the manuscript.
- Per the v24 revision decision (item B17), the JSON numbers are
  **not** modified; this note documents the discrepancy only.
- Recommended follow-up (out of scope for batches A–C): recompute this
  row from the three-seed mean if strict per-artifact alignment is
  required for a later submission package refresh.

## Convention addendum (2026-10-09, v24 batch D)

`tab:gfbudget` reports **raw** per-seed differentiation-gradient ρ, and the official
1-epoch arm is summarized over all six pre-registered seeds (six-seed raw mean
**0.058**; the three-seed subsets are 0.096 for seeds 42/123/456 and 0.020 for the
E3 seeds 789/2024/31415). The BH-FDR grid row for the same arm uses the
**L2-normed** convention (`rho = 0.0568`, i.e. 0.057 to three decimals). The two
numbers therefore differ only by the raw-vs-normed convention; both are computed
from the same `gf_official_1epoch/eval_armb_seed*.json` evaluation JSONs and are
not contradictory.
