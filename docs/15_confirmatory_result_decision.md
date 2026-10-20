# v3 confirmatory result and research decision

Date: 2026-08-06 (Asia/Singapore)  
Decision: **PROCEED with bounded claims; reject the broad robustness claim**

## Evidence lock

- Frozen contract: `docs/12_cfr_confirmatory_contract.json`, SHA-256
  `9af39c8d2858f391006e7dc9ce4a86fdfb60ec980d7b8e58294e8fef6127a240`.
- Completed run manifest: `results/confirmatory_run_manifest.json`; 440
  method--seed--system trainings, 16,563.3 seconds, CUDA, `complete=true`.
- Locked inference: `results/confirmatory_analysis/confirmatory_inference.json`;
  1,320 aggregate cells, 126,720 trajectory rows, 40 distinct dataset hashes,
  all metrics finite.
- The primary estimator balances the 15 prespecified system--regime cells
  within each of eight untouched seeds and then averages seed-specific relative
  NRMSE-25 improvements. Confidence intervals use 5,000 nested paired
  bootstrap replicates. Confirmatory p values use all `2^8` seed sign flips.

## Analysis-only amendment

The first call to the original frozen analyzer failed before bootstrap,
statistics, or output creation. Its generic loader left the descriptive
`parameter_count` at the payload top level, while its summarizer expected the
field in each aggregate row. The stderr and empty stdout are retained under
`logs/`.

No frozen file was silently modified. The versioned amendment
`docs/14_cfr_analysis_amendment.json` authenticates the original failure,
contract, run manifest, and source hashes. The entry point
`code/rgpwm/confirm_analyze_amended.py` validates the exact 440 JSON/PT pairs,
re-load equality, paths, and absence of temporary files; joins one positive
integer parameter count by method--seed--system; and invokes the unchanged
frozen statistics. It changes no performance value, estimator, hypothesis,
margin, random seed, replicate count, multiplicity family, or decision rule.

## Primary hypothesis H1

Comparator: prespecified PGNN. Metric: cell-balanced NRMSE-25. Required margin:
5% relative improvement.

- PGNN mean: 0.1096146.
- CFR-WM mean: 0.0997669.
- Absolute difference (CFR-WM minus PGNN): -0.0098476, 95% CI
  [-0.0121114, -0.0075151].
- Relative improvement: **8.973%**, 95% CI **[6.883%, 10.952%]**.
- Exact one-sided sign-flip p value at the 5% margin: **0.00390625**.
- Seed-specific improvements ranged from 5.290% to 12.762%.

**Decision: H1 supported.** Both the bootstrap lower bound and the exact
margin test pass the frozen gate.

## Hierarchical robustness hypothesis H2

H2 was evaluated because H1 passed. It required both a 5% benefit in the ten
corrupt/shift cells and clean-regime noninferiority at a maximum 5%
degradation.

- Corrupt/shift: 12.045% improvement, 95% CI [10.513%, 13.494%], exact
  one-sided 5%-margin p=0.00390625.
- Clean: -13.080% relative improvement (13.080% degradation), 95% relative
  improvement CI [-20.601%, -5.669%], exact one-sided -5% noninferiority
  p=0.95703125.

**Decision: H2 not supported.** The corrupted-context component passes; the
clean noninferiority component fails. This result cannot be summarized as
general robustness.

## Secondary and descriptive evidence

All five broad-comparator contrasts favor CFR-WM: RSSM 19.077%, SimNorm
27.344%, Transformer 24.094%, PGNN 8.973%, and PgDMM 12.441%. Their two-sided
exact sign-flip p values are 0.0078125 and their separate five-test Holm values
are 0.0390625.

All five sensitivity contrasts favor CFR-WM: gain 0 (30.886%), gain 0.25
(2.973%), gain 0.75 (6.973%), gain 1 (12.132%), and one-step training
(59.381%); the separate five-test Holm values are 0.0390625. These contrasts
support the use of a nonzero complementary update, the prespecified equal-trust
gain within this benchmark, and multi-step training. They do not prove that
gain 0.5 is universally optimal.

Three descriptive cells are more than 10% worse than PGNN, all clean:
cartpole (47.33%), controlled Van der Pol (19.36%), and Duffing (24.37%). Every
ID-corrupt cell improves. These cell contrasts are exploratory descriptions,
not a third confirmatory family.

## Publication claim boundary

The manuscript may claim:

1. an 8.973% mean NRMSE-25 advantage over the prespecified PGNN comparator on
   the frozen 15-cell synthetic-mechanistic benchmark;
2. a strong advantage specifically under corrupted or shifted physical
   context; and
3. evidence that equal-trust complementary filtering and multi-step training
   contribute within this benchmark.

It may not claim universal filter optimality, clean-regime noninferiority,
visual world-model performance, policy return, real-robot transfer, causal
mechanism, SOTA status beyond the protocol-matched comparators, or guaranteed
SCI-Q2 acceptance.
