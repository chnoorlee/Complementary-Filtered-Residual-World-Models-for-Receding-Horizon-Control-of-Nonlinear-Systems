# Literature Synthesis, Novelty Audit, and Hypotheses v1

## Evidence base

The reproducible search returned 389 records and 315 deduplicated candidates. Deterministic title-family screening plus manual relevance inspection retained 40 sources: 21 world-model papers/reviews, 14 hybrid-physics/dynamical-modeling papers, and 5 uncertainty/calibration papers. Duplicate preprint and archival records were collapsed in favor of the stronger venue record. All retained items are dated 2021–2026 and carry a DOI or official academic URL.

## Cluster 1 — recurrent and implicit world models

DreamerV3 demonstrates the breadth of recurrent latent world models, while TD-MPC2 shows that decoder-free normalized latent dynamics scale across continuous-control tasks. Their published evidence concerns complete agents and task return, not controlled state-forecast calibration. They nevertheless motivate two strong transition families: a recurrent stochastic state-space baseline and a SimNorm deterministic latent baseline.

## Cluster 2 — transformer and selective-memory world models

IRIS uses autoregressive Transformers for data-efficient world-model imagination; STORM and later GLAM explore stochastic and selective-state-space alternatives. Recent methods such as UniZero and DINO-WM pursue long-history planning and feature-space dynamics. These works address representation and memory capacity, but the retrieved abstracts do not frame prediction as selection between a misspecified physical prior and a learned correction whose local reliabilities are supervised.

## Cluster 3 — hybrid physics and probabilistic dynamics

The strongest novelty constraint comes from PGNN and PgDMM. PGNN shows that injecting partial physical knowledge can improve nonlinear-dynamics learning. PgDMM already combines a physics-based state-space model with probabilistic deep latent dynamics. Therefore, “physics-informed,” “hybrid,” “probabilistic,” or “uncertainty-aware” alone are not contributions. The defensible gap is narrower: these retrieved works do not explicitly train two branch-error predictors and use their relative predictions as an interpretable gate during free rollout under both missing observations and parameter shift.

## Cluster 4 — newest structured and trustworthy-horizon work

PRISM-WM (2025 preprint) gates specialized experts by latent physical mode, not by physics-versus-data reliability. HaM-World (2026 preprint) adds Hamiltonian residual structure and selective memory; its assumptions are less natural for dissipative systems. A 2026 equivariant-world-model preprint studies certified predictable horizons from symmetry and Lyapunov spectra. These works make a broad “structured long-horizon world model” novelty claim untenable. RGP-WM is positioned instead as an empirically calibrated hybrid-transition mechanism for partially observed dissipative and controlled systems.

## Gap statement

Within the screened 2021–2026 evidence, no retained archival source states the following combined mechanism and protocol: (i) an imperfect differentiable nominal dynamics expert, (ii) a learned residual expert, (iii) supervised prediction of each expert’s realized local error, (iv) relative-error gating during free rollout, and (v) joint evaluation of long-horizon accuracy, interval calibration, missing observations, and parameter shift under a common state-space benchmark.

This is a literature-bounded absence statement, not proof that no such paper exists. Citation verification and a final exact-title/keyword refresh are required before submission.

## Prespecified hypotheses

### H1 — aggregate long-horizon prediction

Across five systems and three regimes, RGP-WM will reduce aggregate NRMSE at horizon 25 by at least 10% relative to the best protocol-matched comparator. Failure occurs if the paired hierarchical-bootstrap 95% confidence interval includes zero improvement or the point improvement is below 10%.

### H2 — calibrated uncertainty under shift

RGP-WM will achieve lower aggregate Gaussian NLL and 90%-interval calibration error than the best comparator in the corruption and parameter-shift regimes, while its one-step NRMSE is no more than 5% worse. Failure occurs if either uncertainty metric is not improved or the one-step non-inferiority margin is violated.

### H3 — interpretable gate behavior

The weight assigned to the data-residual expert will increase when the nominal-physics branch has larger realized error relative to the data branch. This is measured by positive rank correlation between predicted and oracle relative-error weights and by lower catastrophic-rollout rate than both a fixed-fusion model and a data-only ablation. Failure occurs if the correlation confidence interval includes zero or catastrophic rollouts do not decrease.

### H4 — component necessity

Removing reliability supervision, nominal physics, or multi-step training will worsen its prespecified target metric in at least four of five systems. This hypothesis fails component-wise when that ablation does not degrade the target metric in four systems.

Clarification frozen before any ablation run: the target for `RGP-no-rel` is gate--oracle Spearman correlation (higher is better); the targets for `RGP-fixed`, `RGP-no-phys`, and `RGP-1step` are NRMSE-25 (lower is better). The fixed-gate condition tests the necessity of dynamic rather than constant fusion.

## Research decision

**PROCEED to the pilot, with a narrowed claim.** The literature invalidates broad novelty claims but leaves a testable reliability-gating gap. PgDMM-22 is promoted to a required core comparator. PRISM-WM and HaM-World are mandatory discussion references but remain non-archival and protocol-incompatible in the current metadata snapshot.
