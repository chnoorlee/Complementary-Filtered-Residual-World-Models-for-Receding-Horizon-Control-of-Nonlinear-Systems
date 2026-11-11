# T-ASE closed-loop revision protocol

Status: fixed before execution on 2026-11-08. This additional study does not
change the confirmatory H1/H2 analysis or overwrite its artifacts.

## Question

Does the forecast-origin correction improve receding-horizon regulation when
the same controller and objective query each prediction model online?

## Frozen design

- Systems: all five controlled nonlinear systems in `rgpwm.dynamics.SYSTEMS`.
- Training/data seeds: the eight frozen confirmatory seeds.
- Plant conditions: ID clean, ID corrupt, and OOD-parameter corrupt.
- Replicates: four independently generated plant episodes per
  seed--system--condition cell.
- Context: eight states. Seven zero-input burn-in transitions precede control.
- Control interval: 20 receding-horizon steps.
- Controller: deterministic-mean cross-entropy MPC with horizon 8, 64 action
  sequences, two iterations, and eight elites. The initial action standard
  deviation is 0.6 of the system limit and the floor is 0.05 of the limit.
- Objective: summed squared state error in training-standard-deviation units,
  plus 0.01 times squared action/limit, with an additional factor of two on
  terminal state error. Pendulum target is `(sin(theta), cos(theta), omega) =
  (0, 1, 0)`; all other targets are zero.
- Controllers: nominal dynamics from the last causally imputed state;
  gain-0.5 complementary context filter plus nominal dynamics; PGNN-style;
  and CFR-WM. Learned controllers use the corresponding frozen checkpoint.
- Observation corruption: ID-clean and ID-corrupt share each plant initial
  state and parameter draw, matching the open-loop benchmark's paired ID
  design. The clean condition has exact complete state
  observations. Corrupt conditions use Gaussian noise with scalar standard
  deviation equal to 8% of the mean training-state standard deviation and
  independent 25% channel missingness. Missing values are causally carried
  forward. Exogenous standard-normal draws, masks, initial states, plant
  parameters, and planner draws are paired across controllers.
- Primary closed-loop metric: mean realized state regulation loss, i.e. mean
  squared state error in training-standard-deviation units across the 20
  controlled transitions. Action effort and the combined realized objective
  are reported separately.
- Aggregation: episodes are averaged within each seed with equal weight across
  the five systems and three conditions. CFR-WM is paired with PGNN-style at
  the seed level. A 5,000-replicate seed bootstrap gives a 95% interval for
  relative improvement, and an exact one-sided sign-flip test assesses an
  improvement above zero. Condition-specific values remain descriptive.

## Integrity rules

The test outcomes will not tune the controller, objective, horizon, model,
corruption, target, or reporting rule. All episode rows, protocol constants,
checkpoint hashes, source hashes, and aggregate calculations are retained.
Negative or mixed results remain reportable. The study is an additional
closed-loop evaluation, not a new confirmation of the original hypotheses.
