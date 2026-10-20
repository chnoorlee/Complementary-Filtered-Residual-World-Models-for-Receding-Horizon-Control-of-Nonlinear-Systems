# v1-to-v2 pivot audit

Date: 2026-08-05. This document is a development record, not a confirmatory
result. All v1 weights and test outputs remain frozen separately.

## Why v1 was abandoned

Independent mechanism review found that v1's data branch was defined as the
physics branch plus a bounded residual, while the final prediction depended on
the product of that residual and its learned gate. The two quantities could
therefore compensate for one another and were not separately identifiable.
The same audit found a mean gate--oracle Spearman correlation of 0.034 and
nearly unchanged gate weights across clean, corrupt, and OOD-corrupt regimes.
This is a structural failure, not a hyperparameter shortfall.

## Rejected pivot: cross-fitted pairwise-advantage gate

The proposed repair first froze an independently trained PGNN expert and
considered a classifier for whether its forecast would beat nominal physics.
On seed-11 validation windows, a truth-informed hard selector had 9.88%--18.17%
NRMSE-25 headroom over the best fixed blend across the five systems. The
switching problem therefore existed in hindsight. However, group-held-out
logistic and gradient-boosted probes using only deployable context, expert
disagreement, horizon, action, missingness, and context-innovation features had
AUROCs of approximately 0.40--0.61. Their soft blends did not beat the best
fixed blend in a reproducible way. The required information was not predictable
from the eight-step context, so this direction failed its development kill test
and will not be taken to confirmatory data.

## Candidate pivot: Physics-Filtered Residual World Model (PFR-WM)

The next candidate targets the directly observed v1 failure: both experts start
from the same noisy, last-value-imputed final context state. PFR-WM instead
applies, within the observed context only, a differentiable sequence of:

1. nominal-physics prediction;
2. mask-exact measurement update with a learned gain in [0, 1]; and
3. a probabilistic physics-residual free rollout from the filtered state.

Missing channels always have zero update. A gain fixed to 0.5 before training,
with the same hidden width and gain-head architecture, is retained as the
mechanism ablation. The pendulum state is projected back to its physical
unit-circle manifold after both measurement and residual updates. The method is
not claimed to invent learned filtering: KalmanNet and modern data-assimilation
work establish that prior art. The narrower question is whether integrating a
mask-exact physics filter into the forecast origin of a residual world model
improves long-horizon controlled dynamics under missing/noisy context.

## Development kill rule (frozen before PFR-WM training)

Seed 11 from the already-consumed v1 experiment is development-only. Across its
15 system-regime cells, PFR-WM must satisfy all of the following before any new
seed is generated:

- at least 5% cell-balanced NRMSE-25 improvement over the retrained PGNN;
- at least 2% improvement over the prespecified gain-0.5 fixed-filter ablation;
- no more than two cells with degradation greater than 10%;
- finite metrics and no more than one system containing a regime in which over
  95% of observed gains are at most 0.05 or at least 0.95.

If this gate fails, no confirmatory PFR-WM grid will be run and the paper will
remain a negative/diagnostic study. If it passes, the architecture, endpoints,
and new seeds 41, 53, and 67 will be frozen before confirmatory execution.

## Frozen development outcome

The complete 15-run development grid finished without non-finite metrics. Its
cell-balanced NRMSE-25 values were 0.110504 for retrained PGNN, 0.097068 for
learned-gain PFR-WM, and 0.097285 for the gain-0.5 ablation. Thus PFR-WM
improved over PGNN by 12.159% but over the fixed-filter ablation by only 0.223%,
below the frozen 2% mechanism threshold. Exactly two cells degraded by more
than 10% relative to PGNN, no system showed saturated gains, and all metrics
were finite. The machine-readable decision is
`results/analysis/pfr_development_kill.json`; `kill_rule_passed` is `false`.

Consequently, no confirmatory claim will be made for learned adaptive gains.
The failure is informative: forecast-origin filtering helped, while learning
the gain did not add reproducible development value. A versioned v3 may test
the simpler prespecified equal-trust update on untouched seeds, but it must not
reuse the failed PFR-WM mechanism claim or weaken this kill rule.

## Exact-search novelty boundary

Exact-keyword searches covered physics-guided neural filters, differentiable
physics filtering, mask-aware state estimation, neural data assimilation, and
learned Kalman filters (2021--2026). KalmanNet (2022) is the closest generic
learned-filter precedent; PgDMM (2022) is the closest probabilistic
physics-residual transition; the 2023 data-assimilation review maps their wider
intersection. No broad claim to learned filtering, data assimilation, or
physics-residual novelty is permissible.
