# Stage 14 — Result Analysis

## Objective

Analyze all conditions from trajectory-level artifacts before drafting result claims.

## Procedure

- Select the best comparator on the same seed/cell/trajectory-balanced NRMSE-25 estimand.
- Run 5,000-replicate paired hierarchical bootstrap.
- Run 15 paired cell contrasts with Holm correction.
- Evaluate accuracy, calibration, catastrophic failures, gate alignment, nominal dynamics defect, parameters, and latency.
- Use independent analysis/reviewer contexts at the prompt-mandated review stages; do not expose them to desired conclusions.
