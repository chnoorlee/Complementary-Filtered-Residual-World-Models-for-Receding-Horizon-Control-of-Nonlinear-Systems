# CFR-WM paper and reproducibility package

This repository studies the **Complementary-Filtered Residual World Model
(CFR-WM)** for forecasting controlled physical dynamics from noisy and incomplete
context. It combines a frozen five-system synthetic confirmation with a separate
NanoDrone measured-flight extension. RGP-WM (v1) and learned-gain PFR-WM (v2)
remain archived development candidates.

Public repository:
https://github.com/chnoorlee/Complementary-Filtered-Residual-World-Models-for-Receding-Horizon-Control-of-Nonlinear-Systems

Licensed under the MIT License; see `LICENSE`. Citation metadata is in
`CITATION.cff`.

## Primary artifacts

- Manuscript source: `paper/mypaper/main.tex`
- Compiled review manuscript: `paper/mypaper/main.pdf`
- Springer single-file submission source/PDF:
  `paper/mypaper/main_springer.tex` and `paper/mypaper/main_springer.pdf`
- Immutable v3 contract: `docs/12_cfr_confirmatory_contract.json`
- Method and theory: `sections/method.tex`
- Reproducibility commands: `code/README.md`
- Literature corpus and audit: `literature/shortlist.json` and
  `literature/citation_audit.json`
- Audited bibliography and binding manifest: `paper/references.bib` and
  `literature/bibliography_manifest.json`
- Confirmatory checkpoints: `results/confirmatory/`
- Locked inference outputs: `results/confirmatory_analysis/`
- Workflow and pivot ledger: `PROGRESS.md`
- Scope and evidence constraints: `RESTRICTS.yaml`
- Flattened-source provenance: `paper/submission_source_manifest.json`
- Springer build provenance: `paper/springer_build_manifest.json`
- Completion audit: `docs/20_completion_audit.md`
- Original package checksum inventory: `archive/manifest_sha256.json`
- External protocol and source ledger: `plans/26_external_validity_revision.md`
  and `docs/21_external_sources_and_protocol.md`
- External experiment and reproducibility commands: `code/external_validity/`
  and `code/README.md`
- Prose revision record: `docs/22_anti_defensive_revision.md`
- Current revision results and verification: `docs/23_external_validity_revision.md`

## Measured-flight results (2026-11-04)

The completed external study adds 21 development/final trainings, three
held-out Melon flights, five final training seeds, four context conditions,
and two forecast horizons. All three learned methods have exactly 47,000
parameters. Full-state mean NRMSE favors the unfiltered residual model over
CFR-WM in all eight comparisons; the mean paired-seed relative effects
correspond to 1.15--3.45% higher error for CFR-WM. Its 50-step position error
is lower than the other learned models across all four conditions, while
velocity effects are mixed. Under combined noise and missingness, CFR-WM
and unfiltered position errors are 0.094 and 0.125 m. The nominal prior has
the lowest position error on recorded and missing-only contexts.

The evidence now includes measured flights and a held-out trajectory family
on one platform, with the authors' offline-preprocessed state estimates.
These descriptive results are integrated into the abstract, results,
discussion, and conclusion; the original confirmation remains frozen.

The revised suite passes 121 tests and byte-compilation. The current PDFs
have 20 pages (generic) and 28 pages (Springer), with clean final build logs
and rendered layout checks across all 48 pages. Saved predictions, result
tables, figures, macros, manuscript sources, and the Springer PDF have
separate validated provenance manifests. See `docs/23_external_validity_revision.md`
for the results and revision record.

## Frozen confirmation

The grid contains 11 methods, eight untouched seeds, five controlled nonlinear
systems, and three regimes. This gives 440 trained models, 1,320 aggregate
evaluation cells, and 126,720 trajectory-level metric rows. Every method within
a seed--system group shares the same deterministic dataset fingerprint.
ID-clean and ID-corrupt evaluations are repeated measurements on the same
trajectory identifiers; OOD identifiers are disjoint.

H1 is the only confirmation gate: the mean of eight seed-specific CFR-WM
improvements over the preselected PGNN-style comparator must exceed a prespecified
5% NRMSE-25 margin under both a fixed-cell nested bootstrap and an exact
one-sided seed sign-flip test. H2 is a hierarchically gated robustness claim.
Five broad-comparator contrasts and five CFR sensitivity contrasts are separate
Holm-adjusted families. System--regime contrasts are descriptive only.

No intermediate confirmatory performance metrics were inspected. The dedicated
runner is fail-closed on contract, tuning, code, grid, and payload provenance.

The completed confirmation supports H1: CFR-WM improves cell-balanced
NRMSE-25 over the preselected PGNN-style comparator by 8.973% (95% paired bootstrap
CI 6.883--10.952%; exact one-sided 5%-margin sign-flip p=0.00390625). H2 is
not supported. Corrupt/shifted conditions improve by 12.045%, but clean
conditions degrade by 13.080%, failing the prespecified -5% noninferiority
margin (p=0.95703125). The manuscript reports this negative robustness result
and the three clean cells with more than 10% degradation.

The original frozen analyzer stopped before inference because its generic
loader omitted a top-level descriptive `parameter_count` field. The failure
logs are retained. A separately hashed analysis-only amendment validates all
440 JSON/PT pairs and joins that field by method--seed--system before calling
the unchanged frozen statistics. The amendment is documented in
`docs/14_cfr_analysis_amendment.json`; it changes no estimand, metric, random
seed, replicate count, multiplicity family, or decision rule.

## Scope

The study predicts explicit state coordinates under known future controls.
The controlled benchmark spans five nonlinear systems. The external protocol
holds out complete Melon flights from the NanoDrone authors' offline-preprocessed
records, with training and validation drawn from Square, Random, and Chirp flights.
Four context conditions separate added noise and missingness; 25- and 50-step
forecasts test a trajectory family absent from training on one drone platform.
Online acquisition, closed-loop control, and transfer between platforms require
further evaluation.

## Original verification (2026-08-06)

The original suite passed 91/91 tests and Python byte-compilation. The 18-page
working PDF and 27-page Springer PDF complete the full bibliography build
chain with clean final logs. All 45 rendered pages passed programmatic and
visual blank-page, clipping, overlap, and boundary checks. The rendered
Springer abstract was 236 words. Detailed evidence and the remaining
author-controlled submission fields are recorded in
`docs/20_completion_audit.md` and `docs/13_pre_submission_checklist.md`.
