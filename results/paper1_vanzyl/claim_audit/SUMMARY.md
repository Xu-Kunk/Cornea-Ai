# Claim audit — Δρ and ΔLAA for the frozen → fine-tuned differentiation gradient

**Question.** The manuscript's headline is the *change* in differentiation-order
Spearman ρ between the frozen Geneformer-V2-104M encoder and its fine-tuned
derivatives, but the archived statistics only test positive association. This
audit adds the paired test for the change, the paired test for layer-adjacency
accuracy (LAA), and a permutation null for LAA.

- **Data (read-only):** `artifacts/paper1_tier_b_eval/percell_*.npz`, 14,775
  external cells, 2 donors (`donor_1` n = 5,339; `donor_2` n = 9,436), class
  counts LSC 3,423 / TAC 171 / basal 6,698 / differentiated 4,483.
- **Metrics:** imported unmodified from `corneal_fm/evaluation.py`
  (`marker_gradient_correlation`, `layer_adjacency_accuracy`). The bootstrap
  loops use numerically identical re-implementations; equality was asserted at
  run time (max |Δ| = 0 for ρ and LAA on the full sample, ≤ 1.1 × 10⁻¹⁶ for
  ρ and 0 for LAA on random resamples).
- **Resampling unit:** **cell, stratified by donor** — cells are drawn with
  replacement within each donor and donor sizes are preserved. This is a
  stratified **cell** bootstrap, **not** a donor-level bootstrap (only two
  donors exist).
- **Seed:** bootstrap RNG `20261021`; permutation RNG `20261022`.
- **Replicates:** 2,000 bootstrap / 2,000 permutations.
- **Wall-clock runtime:** **357.4 s** total (281.5 s for the 2,000-replicate
  bootstrap loop), Python 3.12.3, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0,
  CPU only (the GPU was occupied by an unrelated vLLM worker).
- **Output:** `results/paper1_vanzyl/claim_audit/delta_rho_laa.json`.

---

## 1. Reproduction gate (must pass before any downstream statistic)

All values recomputed from the per-cell `emb_ext` / `y_true_ext` arrays and
compared with the archived `results/paper1_tier_b_eval/results_*.json`
(tolerance ±0.002). **Passed.**

| Quantity | Reproduced | Expected | \|diff\| | Pass |
|---|---:|---:|---:|:--:|
| Frozen ρ | 0.806361 | 0.8064 | 3.9e-05 | ✅ |
| Frozen LAA | 0.966497 | 0.9665 | 2.5e-06 | ✅ |
| Naive ρ, seed 42 | −0.042395 | −0.0424 | 4.6e-06 | ✅ |
| Naive ρ, seed 123 | −0.036025 | −0.0360 | 2.5e-05 | ✅ |
| Naive ρ, seed 456 | +0.053929 | +0.0539 | 2.9e-05 | ✅ |
| Naive ρ, three-seed mean | −0.008164 | −0.0082 | 3.6e-05 | ✅ |
| Naive LAA, three-seed mean | 0.976808 | 0.9768 | 7.7e-06 | ✅ |

No divergence was observed in axis orientation, stage-rank coding, L2
normalisation, or cell ordering: `y_true_ext` and `donor_ext` are element-wise
identical across all seven arms, the frozen embeddings are L2-normalised
(mean row norm 1.0) and the fine-tuned embeddings are raw (mean row norm
≈ 27.9), matching the arm protocol.

---

## 2. Point estimates

| Arm | ρ | LAA |
|---|---:|---:|
| Frozen Geneformer-V2-104M | **+0.8064** | **0.9665** |
| Naive, seed 42 | −0.0424 | 0.9760 |
| Naive, seed 123 | −0.0360 | 0.9777 |
| Naive, seed 456 | +0.0539 | 0.9767 |
| Naive, three-seed mean | **−0.0082** | **0.9768** |
| CAFT, seed 42 (secondary) | +0.1913 | 0.9786 |
| CAFT, seed 123 (secondary) | +0.2298 | 0.9764 |
| CAFT, seed 456 (secondary) | +0.2095 | 0.9769 |
| CAFT, three-seed mean (secondary) | +0.2102 | 0.9773 |

---

## 3. Paired Δρ = ρ_frozen − ρ_adapted

Same resampled cells for both arms in every replicate. **Primary analysis:
stage-centroid axis refit on the resampled cells inside each replicate**, per
arm. Secondary: axis estimated once on the full sample and held fixed.

### 3a. Primary — naive fine-tuning, axis refit in replicate

| Contrast | Δρ | 95% percentile CI | Replicates ≤ 0 |
|---|---:|---:|:--:|
| Frozen − naive s42 | +0.8488 | +0.7708 … +0.9209 | 0 / 2,000 |
| Frozen − naive s123 | +0.8424 | +0.7632 … +0.9174 | 0 / 2,000 |
| Frozen − naive s456 | +0.7524 | +0.6760 … +0.8598 | 0 / 2,000 |
| **Frozen − naive (three-seed mean)** | **+0.8145** | **+0.7375 … +0.8980** | **0 / 2,000** |

### 3b. Secondary — fixed full-sample axis

| Contrast | Δρ | 95% percentile CI | Replicates ≤ 0 |
|---|---:|---:|:--:|
| Frozen − naive s42 | +0.8488 | +0.8294 … +0.8677 | 0 / 2,000 |
| Frozen − naive s123 | +0.8424 | +0.8229 … +0.8615 | 0 / 2,000 |
| Frozen − naive s456 | +0.7524 | +0.7339 … +0.7708 | 0 / 2,000 |
| **Frozen − naive (three-seed mean)** | **+0.8145** | **+0.7954 … +0.8334** | **0 / 2,000** |

**The fixed-axis interval is 4.2× narrower than the refit-axis interval**
(width 0.0380 vs 0.1606 for the pooled contrast; 2.3× for CAFT). Reusing the
full-sample axis therefore understates the uncertainty of the change; the
refit-in-replicate interval is the one to report.

### 3c. Secondary contrast — CAFT fine-tuning, axis refit in replicate

| Contrast | Δρ | 95% percentile CI | Replicates ≤ 0 |
|---|---:|---:|:--:|
| Frozen − CAFT s42 | +0.6151 | +0.5861 … +0.6453 | 0 / 2,000 |
| Frozen − CAFT s123 | +0.5766 | +0.5308 … +0.6328 | 0 / 2,000 |
| Frozen − CAFT s456 | +0.5969 | +0.5648 … +0.6331 | 0 / 2,000 |
| **Frozen − CAFT (three-seed mean)** | **+0.5962** | **+0.5757 … +0.6204** | **0 / 2,000** |

Fixed-axis pooled CAFT interval: +0.5865 … +0.6060 (width 0.0195).

### 3d. Leave-one-donor-out Δρ (descriptive, no CI)

| Retained donor | n | Frozen ρ | Δρ, naive pooled | Δρ, CAFT pooled |
|---|---:|---:|---:|---:|
| donor_1 | 5,339 | +0.6078 | +0.8728 | +0.3647 |
| donor_2 | 9,436 | +0.8247 | +0.5371 | +0.3360 |

Per-seed naive Δρ on donor_1: s42 +0.8266, s123 +0.8861, s456 +0.9055; on
donor_2: s42 +0.5866, s123 +0.5349, s456 +0.4898. The direction is the same in
both donors; the magnitude is larger on the smaller donor.

---

## 4. Paired ΔLAA = LAA_frozen − LAA_adapted

Point estimate: **−0.0103** (frozen 0.9665, naive three-seed mean 0.9768), i.e.
LAA is higher after fine-tuning.

k-NN LAA is not Fréchet-differentiable under a with-replacement cell bootstrap:
resampled duplicates are distance-0 neighbours and are trivially adjacent,
which inflates LAA in every replicate. As a result the requested
duplicates-retained interval **does not contain its own point estimate**
(reported below and flagged in the JSON). The duplicate-free variant — same
resamples, collapsed to their unique cells before the metric is evaluated —
does contain the point estimate and is the interval suitable for the paper.

### 4a. Requested estimator (duplicates retained, as briefed) — flagged

| Contrast | ΔLAA | 95% percentile CI | Point in CI | Replicates ≤ 0 |
|---|---:|---:|:--:|:--:|
| Frozen − naive s42 | −0.0095 | −0.0051 … −0.0017 | no | 2,000 / 2,000 |
| Frozen − naive s123 | −0.0112 | −0.0057 … −0.0022 | no | 2,000 / 2,000 |
| Frozen − naive s456 | −0.0102 | −0.0051 … −0.0019 | no | 2,000 / 2,000 |
| **Frozen − naive (three-seed mean)** | **−0.0103** | **−0.0052 … −0.0021** | **no** | **2,000 / 2,000** |

### 4b. Duplicate-free estimator (unique cells; recommended interval)

| Contrast | ΔLAA | 95% percentile CI | Point in CI | Replicates ≤ 0 |
|---|---:|---:|:--:|:--:|
| Frozen − naive s42 | −0.0095 | −0.0127 … −0.0061 | yes | 2,000 / 2,000 |
| Frozen − naive s123 | −0.0112 | −0.0138 … −0.0072 | yes | 2,000 / 2,000 |
| Frozen − naive s456 | −0.0102 | −0.0126 … −0.0064 | yes | 2,000 / 2,000 |
| **Frozen − naive (three-seed mean)** | **−0.0103** | **−0.0127 … −0.0070** | **yes** | **2,000 / 2,000** |
| Frozen − CAFT (three-seed mean, secondary) | −0.0108 | −0.0133 … −0.0076 | yes | 2,000 / 2,000 |

In both estimators the sign is the same in every one of the 2,000 replicates.

---

## 5. LAA permutation null

Stage labels permuted across cells with class counts preserved; the
nearest-neighbour graph is held fixed, so each permutation re-labels the same
edges. 2,000 permutations per arm.

| Arm | Observed LAA | Null mean | Null 95th pct | Obs − null mean | Obs − null p95 | Upper-tail p |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 0.9665 | 0.6423 | 0.6508 | +0.3242 | +0.3157 | 0.0005 |
| Naive s42 | 0.9760 | 0.6423 | 0.6487 | +0.3338 | +0.3273 | 0.0005 |
| Naive s123 | 0.9777 | 0.6423 | 0.6491 | +0.3354 | +0.3286 | 0.0005 |
| Naive s456 | 0.9767 | 0.6423 | 0.6489 | +0.3344 | +0.3278 | 0.0005 |

Analytic cross-check for the observed class proportions:
Σᵢ pᵢ(pᵢ₋₁ + pᵢ + pᵢ₊₁) = **0.64234** (i.i.d. form; finite-population form
0.64231), matching the permutation null mean (0.6423). The observed LAA is
therefore well above the chance level, not near the null ceiling.

**k-NN variants (full data, descriptive):**

| Arm | k-th neighbour, k=1 | k=5 | k=10 | Fraction of k NN adjacent, k=5 | k=10 |
|---|---:|---:|---:|---:|---:|
| Frozen | 0.9665 | 0.9597 | 0.9579 | 0.9632 | 0.9613 |
| Naive s42 | 0.9760 | 0.9686 | 0.9678 | 0.9722 | 0.9706 |
| Naive s123 | 0.9777 | 0.9712 | 0.9683 | 0.9740 | 0.9712 |
| Naive s456 | 0.9767 | 0.9702 | 0.9679 | 0.9722 | 0.9705 |

The frozen < adapted ordering is unchanged at k = 5 and k = 10, so it is not an
artefact of a single neighbour.

---

## 6. Not computable from the archived arrays

**Δρ(ordinal − naive)** cannot be computed. The ordinal (CORN) arms archive
per-cell `logits`, `probs`, `preds`, `labels`, `layer`, and `donor_id`
(`artifacts/paper1_corn_boundary/percell_lam0.0_*.npz`) but **no CLS
embeddings**, so ρ cannot be recomputed without re-running inference.

---

## 7. Ready-to-paste sentences

1. **Δρ (primary).** Freezing rather than naively fine-tuning the encoder
   changed the differentiation-order Spearman ρ by Δρ = 0.815 (95% percentile
   CI 0.738–0.898; 0 of 2,000 replicates ≤ 0), estimated with a paired
   stratified cell bootstrap in which cells were resampled with replacement
   within each of the two external donors (donor sizes preserved) and the same
   resampled cells were used for both arms, with the stage-centroid axis refit
   on the resampled cells in every replicate.

2. **Δρ (per-seed and CAFT).** Per-seed Δρ ranged from 0.752 to 0.849 for the
   three naive runs, and the three-seed mean for the CAFT runs was 0.596 (95%
   CI 0.576–0.620), with the same axis-refit cell bootstrap.

3. **Axis handling.** Holding the full-sample stage-centroid axis fixed instead
   of refitting it within each replicate narrowed the pooled naive interval to
   0.795–0.833 (4.2-fold narrower), so the refit-in-replicate interval is the
   one reported.

4. **ΔLAA.** Layer-adjacency accuracy was 0.0103 higher after naive
   fine-tuning (ΔLAA = −0.0103, 95% CI −0.0127 to −0.0070; 2,000 of 2,000
   replicates ≤ 0), evaluated on the same donor-stratified cell resamples with
   duplicate draws collapsed to unique cells before the nearest-neighbour
   metric was computed.

5. **LAA null.** Observed LAA (0.9665) exceeded the label-permutation null
   (mean 0.6423, 95th percentile 0.6508) by 0.316, and the analytic i.i.d.
   expectation for the observed class proportions was 0.6423, so the observed
   value is far above chance rather than near a chance ceiling.
