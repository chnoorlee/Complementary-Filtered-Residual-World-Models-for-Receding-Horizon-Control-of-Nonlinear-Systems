# Stage 12 — Experiment Run

## Objective

Execute 180 equal-budget tuning trials, 90 primary trainings, 270 primary evaluation cells, and 60 ablation trainings without test-driven changes.

## Checks

- All losses and predictions finite.
- Completed JSON and checkpoint counts agree.
- Resume skips completed conditions.
- Recompute evaluation from checkpoints after metric-code audit.
- Preserve stdout/stderr and wall-clock evidence.
