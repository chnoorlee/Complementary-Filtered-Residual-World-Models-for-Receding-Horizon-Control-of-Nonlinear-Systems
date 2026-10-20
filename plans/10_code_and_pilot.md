# Stage 10-11 — Code and Pilot

## Frozen implementation boundary

- Ground-truth trajectories use sampled parameters; every model sees only the nominal parameterization.
- Future observations and targets are never encoder inputs.
- The learned rollouts are free-running for all reported horizons.
- All methods share trajectory splits, corruption draws, optimizer family, stopping rule, and evaluation code.
- Hidden widths are chosen to keep the six primary models near 62k parameters: RSSM/SimNorm 80, Transformer 48, PGNN/PgDMM 84, RGP-WM 64.

## Test-first evidence

- RED: test collection failed because `rgpwm` did not exist.
- GREEN: 27 tests passed on 2026-08-05.
- Covered contracts: deterministic RK4, dissipative oscillator, pendulum manifold, disjoint OOD ranges, reproducible splits/corruptions, window shapes, all model outputs, differentiable nominal physics, metric identities, and finite loss/backpropagation.

## Pilot protocol

- System: forced Duffing.
- Seed: 11.
- Methods: all six primary methods.
- Epochs: 3, full data counts and horizons.
- Outcome gate: finite execution, no leakage, no collapsed intervals, and projected primary grid within eight hours.
