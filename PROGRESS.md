# Research Progress

## Project identity

- Current working title: **Complementary-Filtered Residual World Models: Controlled Benchmarks and Real-Flight Evaluation**
- Direction: world models
- Working method name: **CFR-WM** (v3 confirmatory candidate; RGP-WM and learned-gain PFR-WM are rejected development versions)
- Started: 2026-08-05 (Asia/Singapore)
- Original confirmation cutoff: 2026-08-06; external source supplement: 2026-11-04
- Submission target: not selected; the earlier JIRS profile is historical venue screening.

## Nonlinear workflow

`Definition -> Literature -> Hypotheses -> Design -> Pilot -> Full runs -> Analysis`

- If the pilot exposes infeasible runtime or unstable optimization: **REFINE -> Design**.
- If full results fail the prespecified primary hypothesis: **PIVOT -> Hypotheses/Method**, version the previous artifacts, and rerun.
- If peer review finds a method-evidence mismatch: **REFINE -> Experiments**, never patch prose around missing evidence.
- If the quality gate is below 7/10: **REFINE -> Draft or Experiments** according to the failure source.

## Stage ledger

| Stage | Version | Status | Evidence/artifact | Decision |
|---|---:|---|---|---|
| 1 TOPIC_INIT | v1 | completed | `docs/01_research_definition.md` | World-model direction selected |
| 2 PROBLEM_DECOMPOSE | v1 | completed | `docs/01_research_definition.md` | Four falsifiable subproblems frozen |
| A+ HARDWARE_CHECK | v1 | completed | RTX 5070 Laptop 8,151 MiB; Ryzen 9 9955HX; 16 GB RAM; MiKTeX present | Small state-space models, not image-scale RL |
| 3 SEARCH_STRATEGY | v1 | completed | `docs/02_search_and_baseline_contract.md` | OpenAlex, DBLP, Crossref/arXiv verification |
| 4 LITERATURE_COLLECT | v1 | completed | 389 raw, 315 deduplicated records in `literature/` | Retrieval provenance saved |
| 5 LITERATURE_SCREEN | v1 | completed | 40 retained sources plus one targeted v2 supplement; 41/41 current metadata checks in `literature/citation_audit.json` | Gate passed; all cited sources 2021–2026 |
| 6 KNOWLEDGE_EXTRACT | v1 | completed | `literature/knowledge_cards.json` | Core claims limited to inspected metadata/abstracts |
| 7 SYNTHESIS | v1 | completed | `docs/04_literature_synthesis_and_hypotheses.md` | PROCEED with narrowed reliability-gate novelty |
| 8 HYPOTHESIS_GEN | v1 | completed | four prespecified hypotheses in `docs/04_literature_synthesis_and_hypotheses.md` | H1 is primary |
| 8.5 THEORETICAL_BOUNDS | v1 | completed | `docs/05_method_and_theory.md` | Conditional oracle-gap and Lipschitz bounds only |
| 9 EXPERIMENT_DESIGN | v1 | completed | `docs/06_experiment_design.yaml` | Gate passed; pilot required before full grid |
| 10 CODE_GENERATION | v1 | completed | 27 unit tests; `code/rgpwm/` | GREEN; all tests pass |
| 11 PILOT_RUN | v1 | completed | `results/pilot/`; six methods, three epochs | PROCEED; projected primary grid 5,084.6 s |
| 11.5 HYPERPARAMETER_TUNING | v1 | completed | `results/tuning/`; 180/180 unique trials | Equal six-trial budget; test data untouched |
| 12 EXPERIMENT_RUN | v1 | completed | 90/90 trainings; 270/270 evaluations; 3,927.1 s | All finite; no time guard |
| 13 ITERATIVE_REFINE | v1 | completed | metric v2 plus immutable raw archives | all 150 checkpoints uniformly re-evaluated with training-only scale |
| 14 RESULT_ANALYSIS | v1 | completed | `results/analysis/`; 5,000 paired bootstrap replicates | corrected effect +1.56%, CI crosses zero; H1–H4 not supported |
| 15 RESEARCH_DECISION | v1 | completed | `docs/09_research_decision_v1.md` | PIVOT; no positive primary claim |
| 12b ABLATION_DIAGNOSIS | v1 | completed | 60/60 trainings; corrected inference in `results/analysis/ablation_inference.json` | only multi-step training helped; gate/physics components not supported |
| 9b-13b CONFIRMATORY_FREEZE | v3 | completed | `docs/12_cfr_confirmatory_contract.json`; independent protocol and statistical GO reviews | 11 methods, 8 untouched seeds, exact provenance-bound runner |
| 12c CONFIRMATORY_RUN | v3 | completed | 440 JSON/checkpoint pairs; `results/confirmatory_run_manifest.json`; 16,563.3 s on CUDA | Complete exact grid; no intermediate performance metrics inspected |
| 14b RESULT_ANALYSIS | v3 | completed | `results/confirmatory_analysis/`; 1,320 aggregate rows; 126,720 trajectory rows; 5,000 paired bootstrap replicates | H1 supported; H2 rejected because clean noninferiority failed |
| 15b RESEARCH_DECISION | v3 | completed | `docs/15_confirmatory_result_decision.md` | PROCEED with a bounded primary claim and explicit negative robustness result |
| 16-17 PAPER_DRAFT | v3 | completed | all required sections; result-bound tables and figures; 236-word rendered abstract | No unsupported policy, robot-transfer, or universal-filter claims |
| 18-19 PEER_REVIEW/REVISION | v3 | completed | `docs/17_hostile_manuscript_review_and_revision.md`; `docs/19_third_party_review_and_rebuttal.md` | Title, scope, comparator labels, mechanism language, and paper-alone detail revised |
| 20-25 QUALITY/EXPORT/REVIEW | v3 | completed | 91/91 tests; 18- and 27-page clean PDFs; `docs/18_quality_gate.md`; `docs/20_completion_audit.md` | Scientific package complete; author-controlled submission fields remain open |

### v2 development decision — 2026-08-05

- Fifteen seed-11 development trainings compared retrained PGNN, learned-gain
  PFR-WM, and its prespecified gain-0.5 matched ablation on all 15 cells.
- PFR-WM improved cell-balanced NRMSE-25 over PGNN by 12.159%, but improved
  over the fixed ablation by only 0.223%; the frozen mechanism gate required
  2%. The exact decision is `kill_rule_passed=false` in
  `results/analysis/pfr_development_kill.json`.
- Learned adaptive filtering is rejected for confirmation. The simpler
  mask-exact equal-trust physics filter becomes a versioned v3 candidate and
  must be evaluated on untouched seeds with its own frozen contract.

### v3 confirmatory freeze — 2026-08-05

- CFR-WM fixes the context measurement gain at 0.5 and retains the
  probabilistic physics-residual 25-step free rollout. Learned-gain PFR-WM is
  explicitly excluded from the confirmatory method grid.
- The fail-closed contract freezes 11 methods, eight independent seeds
  (41, 53, 67, 79, 83, 97, 109, 113), five systems, three regimes, 440 training
  runs, 1,320 aggregate cells, and 126,720 paired trajectory rows.
- The primary estimand is the mean of eight seed-specific relative NRMSE-25
  improvements over the PGNN-style comparator. H1 directly tests a 5% improvement margin using a
  bundled seed/trajectory bootstrap and an exact one-sided seed sign-flip
  test. H2 is hierarchically gated and uses a -5% clean noninferiority margin.
- The dedicated runner pins training-code, analysis-code, tuning-file, and
  contract hashes. The generic experiment CLI rejects all frozen seeds and the
  reserved confirmatory output directory before data generation.
- Eight clusters make significance attainable for both five-test Holm families;
  the fifteen cell contrasts are explicitly descriptive. OOD trajectory IDs
  must be disjoint from the repeated ID-clean/ID-corrupt identifiers.
- At this freeze checkpoint the full suite contained 64 passing tests, and the
  confirmatory directory, analysis directory, and manifest were still absent.
  Later execution and amendment tests are recorded in the outcome below.

### v3 confirmatory outcome — 2026-08-06

- All 440 prespecified trainings completed on CUDA in 16,563.3 seconds. The
  completed manifest authenticates the frozen contract, training code,
  analysis code, and tuning archive. The locked analysis contains 1,320
  aggregate rows, 126,720 paired trajectory rows, 40 seed--system dataset
  hashes, and finite metrics throughout.
- The original frozen analyzer stopped before bootstrap or inference because
  the generic loader did not propagate the top-level descriptive
  `parameter_count` field. The zero-output failure is retained in `logs/`.
  `docs/14_cfr_analysis_amendment.json` and
  `code/rgpwm/confirm_analyze_amended.py` form a separately hashed,
  analysis-only amendment: it authenticates all JSON/PT pairs, performs a
  many-to-one schema join, and calls the unchanged frozen statistics. It does
  not alter an estimand, metric, seed, resampling scheme, replicate count,
  family, threshold, or decision rule.
- H1 passed. Against the preselected PGNN-style comparator, CFR-WM improved
  cell-balanced NRMSE-25 by 8.973% (95% paired bootstrap CI
  6.883--10.952%); the exact one-sided sign-flip p value at the 5% margin was
  0.00390625. Every seed-level effect exceeded 5%.
- H2 failed. Corrupt and shifted cells improved by 12.045% (95% CI
  10.513--13.494%; margin p=0.00390625), but clean cells degraded by 13.080%
  (95% CI 5.669--20.601% degradation) and failed the -5% noninferiority test
  (p=0.95703125). Cartpole, controlled Van der Pol, and Duffing clean cells
  were descriptively more than 10% worse than PGNN; every ID-corrupt cell
  improved.
- The bounded publication claim is therefore an overall 15-cell advantage and
  robust performance under corrupted physical context, not universal
  improvement or clean-regime noninferiority. The manuscript, abstract,
  discussion, limitations, and conclusion state the negative H2 result.
- The rendered Springer abstract is 236 words. The final 18-page working PDF
  and 27-page Springer PDF passed the full
  PDFLaTeX--BibTeX--PDFLaTeX--PDFLaTeX chain with clean final logs. All 45
  rendered pages passed automated and manual visual QA.
- Final delivery validation passed 91/91 tests and Python byte-compilation.
  Scoped source, bibliography, result-asset, submission-source, and build
  manifests validate; `archive/manifest_sha256.json` freezes the completed
  package after the documentation closeout.

## External validity and concise-writing revision — 2026-11-04

- Completed the separate, pre-recorded NanoDrone extension: 9 training flights,
  3 validation flights, and 3 held-out Melon flights. Normalization and thrust
  fitting use only training records. The three learned models each have
  47,000 parameters; 6 learning-rate development runs and 15 final runs completed.
- Evaluated 387 forecast windows, 4 context conditions and 2 horizons; saved
  64 prediction arrays and independently recomputed all 384 flight-level
  rows, 32 summaries and 16 paired contrasts. The external asset manifest
  validates the raw-record, code, checkpoint, prediction and asset chain.
- Full-state mean NRMSE is higher for CFR-WM than the unfiltered residual
  model in all 8 comparisons. The mean paired-seed relative error increase
  spans 1.15--3.45%. CFR-WM has lower 50-step position error than the other
  learned models across 4 conditions; velocity effects are mixed, and the
  nominal prior wins position error on recorded and missing-only contexts.
- Rewrote the abstract, external results, discussion and conclusion around
  this component trade-off. Removed repetitive defensive statements and
  consolidated procedural details. Original H1/H2, theory conditions and
  the single-platform/offline-acquisition scope remain explicit.
- Independent code review and quick field/EIC agent review completed; the
  multi-step sensitivity interpretation and two final wording issues were
  corrected against actual results. No journal-specific criteria were bound.
- Validation: 121 tests and byte-compilation passed; 20-page generic and
  28-page Springer PDFs built with clean final logs; all 48 rendered pages
  checked. Prior PDFs/source/build manifests are preserved in
  `archive/pre_external_revision_20261104/`.
- Completion record: `docs/23_external_validity_revision.md`. The new external
  evidence strengthens the setting coverage while limiting performance claims
  to the observed full-state/component pattern.

## Version history

### v1 — 2026-08-05

- Froze the world-model scope and excluded unsupported full-agent RL claims.
- Selected a low-dimensional, action-conditioned physical-dynamics benchmark that is feasible on the verified hardware.
- Defined fifteen evaluation cells (five systems by three deployment regimes), three paired seeds, recent transition-backbone comparators, and evidence gates.
- Next loop point: literature screening may rename or alter RGP-WM if a materially equivalent recent method is found.
- Literature loop outcome: broad physics-plus-uncertainty novelty was rejected because PgDMM-22 already covers it. The method was narrowed to supervised relative-reliability gating; PgDMM-22 was added as a required comparator.
- Code loop outcome: the RED suite initially failed on the absent package, then 27 tests passed after implementing five RK4 systems, leakage-safe windowing, nine model/ablation interfaces, probabilistic metrics, and finite-loss backpropagation.
- Pilot outcome: all six primary models completed on Duffing; parameter counts were 59,188--61,963, RGP-WM's mean OOD data-gate weight was 0.444, and the projected primary grid was 5,084.6 seconds.
- Tuning outcome: 180/180 unique equal-budget validation trials completed. Several winners reached the learning-rate search boundary (0.003); this is retained as a disclosed limitation rather than followed by unequal post hoc search.
- Corrected primary v1 outcome: after uniform fixed-training-scale
  re-evaluation, RGP-WM minus the descriptively selected PGNN comparator was
  -0.001723 NRMSE-25 (95% CI -0.006344 to +0.005955), a signed relative
  improvement of +1.555%, p=0.4784. This misses the preregistered 10% threshold;
  H1--H3 remain unsupported.
- Independent audit outcome: v1's per-trajectory NRMSE denominator was
  re-estimated from each test trajectory, causing an undefined one-step scale
  and unstable weighting. The implementation now uses the fixed training-split
  state scale. The original primary and ablation JSONs are retained for
  provenance, and all 150 checkpoints were re-evaluated before corrected
  inference. The audit also found
  that the v1 residual and gate are non-identifiable because only their product
  determines the prediction, motivating a structurally distinct v2 rather than
  post-hoc tuning. The corrected ablations support multi-step training but not
  the reliability gate or physics branch; H4 is unsupported.
