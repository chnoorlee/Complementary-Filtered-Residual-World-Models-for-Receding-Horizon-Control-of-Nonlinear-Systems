# Stage 15 Research Decision — v1

## Decision: PIVOT

The complete v1 benchmark contains 90 primary training runs, 60 diagnostic
ablation runs, and 15 system–regime cells. All 150 saved checkpoints were
uniformly re-evaluated with the audited fixed-training-scale metric before the
statistics below were computed. The original trajectory-local-scale outputs
remain immutable in `archive/v1_metric_bug_raw/` and
`archive/v1_ablation_metric_bug_raw/`.

## Corrected primary result

- PGNN was the lowest-error primary comparator, selected descriptively after
  inspecting the v1 test results; therefore the contrast is not treated as a
  preselected confirmatory comparison.
- RGP-WM minus PGNN NRMSE-25: **−0.001723** (lower favors RGP-WM).
- Signed relative improvement: **+1.555%**.
- 95% paired hierarchical-bootstrap CI: **[−0.006344, +0.005955]**.
- Two-sided bootstrap p-value: **0.4784**.
- Paired trajectory records: **4,320**.
- Preregistered H1 threshold: at least 10% relative improvement with the CI
  wholly below zero.
- H1 decision: **not supported**.

H2 is not supported because corrupt-regime NLL/calibration did not jointly
improve, although one-step non-inferiority passed (+2.54%, within the 5%
margin). Gate–oracle Spearman correlation was near zero (mean 0.034), and the
catastrophic-rate diagnostic did not favor RGP-WM. H3 is therefore not
supported.

## Corrected ablation diagnosis

- Versus `RGP-fixed`, full RGP-WM was 1.48% worse at horizon 25 (95% CI of the
  difference: −0.000728 to +0.006306) and better on only one of five systems.
- Versus `RGP-no-phys`, full RGP-WM was 1.46% worse (95% CI: −0.006631 to
  +0.012107) and better on only one system.
- Versus `RGP-1step`, full RGP-WM was 53.71% better (95% CI: −0.174672 to
  −0.079820), supporting multi-step rather than teacher-forced training.
- Removing reliability supervision did not produce a consistent gate-alignment
  loss: the full-minus-ablation Spearman difference was only 0.01285 and full
  RGP-WM was better on two of five systems.
- The all-components ablation hypothesis H4 is **not supported**.

These results identify multi-step training as useful but falsify the proposed
reliability-gating mechanism as the source of robust improvement. They do not
support a positive RGP-WM paper claim.

## Integrity action

The v1 method will not be described as outperforming recent baselines. No
threshold, comparator, or cell set was changed after seeing the corrected
result. The corrected machine-readable results are in
`results/analysis/primary_inference.json` and
`results/analysis/ablation_inference.json`.

Any v2 must:

1. be motivated by v1 diagnostics rather than test-cell cherry-picking;
2. freeze a materially different, falsifiable mechanism;
3. use newly generated independent confirmatory trajectories/seeds for its main
   claim; and
4. preserve v1 artifacts as negative evidence.
