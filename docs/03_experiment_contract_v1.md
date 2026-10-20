# Experiment Contract v1 (Provisional Pending Literature Gate)

## Dataset generation

- Reference dynamics: explicit controlled ODEs integrated with fourth-order Runge–Kutta at a small internal step.
- Systems: damped oscillator, forced Duffing, controlled Van der Pol, pendulum, cart-pole.
- Trajectory sampling: independent initial conditions, piecewise-smooth bounded controls, and physical parameters.
- Split: disjoint trajectories for train, validation, in-distribution test, and shifted-parameter test. No window from one trajectory may cross a split.
- Label: all data are synthetic-but-mechanistic simulator trajectories, never described as real sensor data.

## Forecast protocol

- Context length: 8 steps.
- Forecast horizons reported: 1, 5, 10, 25.
- Free rollout: after the context, predictions rather than ground truth are fed back.
- Observation corruption: per-feature Gaussian noise plus missing-at-random masks; mask and time-since-observation are inputs.
- Parameter shift: test parameters outside the training interval but within a prespecified physically stable range.

## Methods

1. RSSM-25 transition-backbone adaptation.
2. SimNorm-WM-24 transition-backbone adaptation.
3. TransWM-23 transition-backbone adaptation.
4. PGNN-22 matched physics-guided residual comparator.
5. Proposed RGP-WM.

## RGP-WM components

- differentiable nominal-physics expert;
- learned recurrent data expert;
- branch-error predictors supervised by realized expert errors;
- reliability gate derived from predicted relative errors;
- multi-step Gaussian rollout likelihood.

## Ablations

- `RGP-no-rel`: remove branch-error supervision; gate learned only through forecast loss.
- `RGP-fixed`: replace reliability gate with a fixed validation-selected weight.
- `RGP-no-phys`: remove nominal physics expert.
- `RGP-1step`: replace rollout loss with teacher-forced one-step loss.

## Metrics

- Primary: normalized RMSE at horizon 25 (NRMSE-25), with each state
  channel divided by the standard deviation estimated from the training split
  only. Because stored targets are already in those normalized coordinates,
  the evaluation-time denominator is one for every channel.
- Secondary accuracy: NRMSE-1/5/10, MAE-25.
- Probabilistic: Gaussian NLL, empirical 90% coverage, mean interval width, calibration error across nominal coverages.
- Robustness: catastrophic rollout rate using a prespecified normalized-error threshold.
- Physics: normalized nominal-dynamics defect, interpreted as consistency with the imperfect prior rather than ground-truth law satisfaction.
- Efficiency: parameter count, training seconds, peak GPU memory if available, and batch-1 rollout latency.

## Statistics

- Seeds: 11, 23, 37, paired across methods.
- Primary aggregate: hierarchical bootstrap over seed, system-regime cell, and trajectory.
- Cell contrasts: paired seed/trajectory bootstrap; Holm-adjusted family-wise inference.
- Effect sizes and confidence intervals reported alongside p-values.
- No statement of significance unless implemented in the analysis code and present in machine-readable results.

## Pilot and time guard

Before the full grid, run one system, one seed, and a reduced epoch count. The runner must print `TIME_ESTIMATE: <seconds>` for the projected full training grid. Every training loop checks a wall-clock budget and saves partial results before 80% of the assigned budget.

## Decision after v1

- PROCEED only if all models run without NaN/Inf, the data splits are leakage-free, and a single-condition pilot completes within the allocated budget.
- REFINE if calibration collapses, parameter counts are badly unmatched, or the proposed gate degenerates to one branch.
- PIVOT if the literature gate identifies the same method or if the full primary endpoint fails.

## Audited metric amendment (2026-08-05, before v2)

The first v1 evaluation accidentally re-estimated the NRMSE denominator from
each test trajectory. This is undefined at horizon one and overweights
low-dynamic trajectories. The error was discovered by an independent mechanism
audit after all v1 primary runs and before any v2 design or confirmatory run.
The original primary and ablation outputs are preserved in
`archive/v1_metric_bug_raw/` and `archive/v1_ablation_metric_bug_raw/`.
All 150 v1 checkpoints were uniformly re-evaluated with the training-only scale
above before the corrected inference was generated.
The amendment cannot rescue the preregistered v1 hypothesis: its 10% success
threshold was already missed by a wide margin. No v2 endpoint will use the
uncorrected statistic.
