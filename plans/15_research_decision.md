# Stage 15 — Research Decision

## Decision tree

- `PROCEED`: H1 reaches >=10% and its 95% CI excludes zero; complete ablations and manuscript.
- `REFINE`: implementation/evaluation defect exists; repair and uniformly recompute without changing trained weights where possible.
- `PIVOT`: mechanism-level H1 failure; preserve v1, diagnose ablations, and either report bounded negative evidence or design a new version with independent confirmation.

The decision document must quote machine-readable effect sizes and cannot be based on a favorable cell.
