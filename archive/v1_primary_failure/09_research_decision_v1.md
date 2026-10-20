# Stage 15 Research Decision — v1

## Decision: PIVOT

The complete primary grid contains 90 trained conditions and 270 evaluation cells. All 90 checkpoints were uniformly re-evaluated with the audited metric code before inference.

## Frozen primary result

- Best comparator on the trajectory/seed/cell-balanced primary estimand: PGNN.
- RGP-WM minus PGNN NRMSE-25: **+0.010219** (lower would favor RGP-WM).
- Signed relative improvement: **-1.3976%**.
- 95% paired hierarchical-bootstrap CI: **[-0.044119, +0.113432]**.
- Two-sided bootstrap p-value: **0.998**.
- Paired trajectory records: **4,320**.
- H1 threshold: at least +10% relative improvement and CI wholly below zero.
- H1 decision: **not supported**.

H2 is not supported because corrupt-regime NLL/calibration did not jointly improve, although one-step non-inferiority passed (+2.54%, below the 5% margin). Gate--oracle Spearman was positive but small (mean 0.034), and the catastrophic-rate diagnostic did not favor RGP-WM; the preliminary H3 decision is not supported pending the fixed/no-physics comparisons.

## Integrity action

The v1 method will not be described as outperforming recent baselines. No threshold, comparator, or cell set is changed. The frozen four-ablation grid is run only to diagnose whether reliability supervision, dynamic gating, physics, or multi-step training behaved as intended.

Any v2 must:

1. be motivated by v1 diagnostics rather than test-cell cherry-picking;
2. freeze a materially different, falsifiable mechanism;
3. use newly generated independent confirmatory trajectories/seeds for its main claim; and
4. preserve v1 artifacts as negative evidence.
