# CFR-WM paper and reproducibility package

This repository contains the completed confirmatory, evidence-bounded study of the
**Complementary-Filtered Residual World Model (CFR-WM)** for 25-step prediction
of controlled physical dynamics from noisy and incomplete context. The data are
analytically simulated trajectories, not real sensor recordings. RGP-WM (v1)
and learned-gain PFR-WM (v2) are archived rejected development candidates and
must not be used as confirmatory evidence.

The package is written to the methodological standard expected of an SCI-Q2
submission. It does not guarantee acceptance, and the target journal's current
JCR quartile must be checked institutionally on the actual submission date.

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
- Package-wide checksum inventory: `archive/manifest_sha256.json`

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

The evidence concerns transition prediction on a fixed, low-dimensional,
five-system synthetic-mechanistic benchmark with known future controls. It does
not establish visual representation learning, policy quality, universal filter
optimality, real-robot transfer, or hardware safety.

## Final verification

The final suite passes 91/91 tests and Python byte-compilation. The 18-page
working PDF and 27-page Springer PDF complete the full bibliography build
chain with clean final logs. All 45 rendered pages passed programmatic and
visual blank-page, clipping, overlap, and boundary checks. The rendered
Springer abstract is 236 words. Detailed evidence and the remaining
author-controlled submission fields are recorded in
`docs/20_completion_audit.md` and `docs/13_pre_submission_checklist.md`.
