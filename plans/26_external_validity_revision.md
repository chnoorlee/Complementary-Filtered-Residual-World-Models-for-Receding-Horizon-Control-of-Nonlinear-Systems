# External validity and concise writing revision

Date: 2026-11-04. Authorized by the user's request to improve external validity
and remove defensive manuscript statements. Original freeze: Git `4929325`.

## Experimental decision before external test evaluation

Use the publicly released NanoDrone real-flight benchmark (Busetto et al.,
Control Engineering Practice 172, 106871, 2026; DOI
10.1016/j.conengprac.2026.106871). This adds measured physical records, a
larger state space, a held-out trajectory family, and a structurally incomplete
nominal model. It is a new external replication, separate from frozen H1/H2.

- Split by complete flight: Square/Random/Chirp runs 1--3 for training; run 4
  of each for validation; Melon runs 1--3 for final testing.
- Represent orientation by a three-component rotation vector; predict position,
  velocity, orientation, and angular velocity (12 coordinates). All scaling and
  the nominal thrust coefficient are fitted on training flights alone.
- The prior uses rigid-body thrust direction, gravity, translation and constant
  body angular velocity. Rotational acceleration and drag are unmodeled.
- Compare CFR gain 0.5, unfiltered probabilistic physics-residual, and a
  recurrent data-only transition with exactly shared learned architecture;
  also report the nominal prior. These are protocol implementations, not scores
  attributed to the original NanoDrone reference models.
- Context 8; training rollout 25; evaluation rollouts 25 and 50 at 100 Hz.
- Four test conditions: as-recorded; independent Gaussian measurement noise
  (0.1 training-standard-deviation per coordinate); 30% independent missing
  coordinates; both perturbations. Targets remain the recorded measurements.
- Training uses a fixed equal mixture of as-recorded and modestly perturbed
  context (noise 0.05, missingness 0.1); targets and controls are unmodified.
- Training/validation stride 10; testing stride 50. No window crosses a flight.
- Hidden width 64; AdamW; at most 40 epochs, patience 8; batch 256. Two learning
  rates (0.001, 0.003) per learned method, development seed 17; choose by validation
  NLL. Final seeds 131, 149, 167, 181, 199. Training and test perturbations are
  paired across methods. Test set remains unused until all models are selected.
- Primary descriptive score: equal-flight mean normalized RMSE over all forecast
  times and state coordinates (training scales). Report each condition/horizon,
  per-flight scores, seed variability, and physical-unit position/velocity error.
  Three test flights from one platform are not independent platforms; training
  seeds measure optimizer variability. No new confirmatory p-values.
- Performance-only pilot: three training epochs, no test scoring. Report timing;
  any budget adjustment must be recorded before test evaluation. Preserve failures.

## Parallel tasks and completion conditions

1. Verify primary literature and data provenance, download pinned public records.
2. Implement/test separate external adapter and runner; preserve frozen modules.
3. Execute development selection and five-seed final grid; save checkpoints,
   training histories, predictions, per-window errors and provenance hashes.
4. Integrate all outcomes, including unfavorable ones, into paper and figures.
5. Remove redundant apologetic framing; retain essential conditions and H2.
6. Independent code/evidence review, regression tests, both LaTeX builds, PDF QA.

No manuscript claim of improved performance on external data is authorized by
this plan alone: its content must follow actual final results.
