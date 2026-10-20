# CFR-WM completion audit

Date: 2026-10-20

## Scientific outcome

The frozen confirmatory grid contains 440 trained models, 1,320 aggregate
system--regime cells, 126,720 trajectory-level rows, and 40 authenticated
seed--system dataset hashes.

- H1 passed: CFR-WM improved cell-balanced NRMSE-25 over the prespecified
  PGNN-style comparator by 8.9729577%, with a 95% paired bootstrap interval of
  `[6.8825106, 10.9515351]%` and an exact one-sided 5%-margin sign-flip
  `p=0.00390625`.
- H2 failed: corrupt/shifted cells improved by 12.04514%, but clean cells had a
  signed improvement of -13.07955% (13.08% degradation) and failed the frozen
  -5% noninferiority test (`p=0.95703125`).

The paper reports both outcomes. It does not claim clean-regime
noninferiority, causal identification of a robustness mechanism, real-robot
transfer, policy improvement, or general world-model superiority.

## Final artifacts

- Working manuscript: `paper/mypaper/main.pdf` (18 pages), SHA-256
  `df6a64d729b360314b2536053e040fa108e468c9cc69d56bf90c4e34a2c4b2bb`.
- Springer manuscript: `paper/mypaper/main_springer.pdf` (27 pages), SHA-256
  `d9531cf5e3c70bd0d442cd67748b12dd0f75a0211711bf13c958d846e35e1339`.
- Locked inference: `results/confirmatory_analysis/confirmatory_inference.json`.
- Build provenance: `paper/submission_source_manifest.json` and
  `paper/springer_build_manifest.json`.
- Package inventory: `archive/manifest_sha256.json`, generated as the final
  content freeze after this audit record.

## Verification evidence

- 91/91 tests pass; Python byte-compilation succeeds.
- Both final build logs are free of LaTeX/package warnings, undefined
  references/citations, overfull boxes, fatal errors, and compilation errors.
- Every PDF page was rasterized and inspected programmatically; contact sheets
  and dense appendix pages were inspected visually. No blank, clipped,
  overlapping, or boundary-touching content was found. The final rebuilt PDFs
  were rasterized again, and all 45 pages were pixel-identical to those
  inspected versions.
- The Springer abstract contains 236 rendered words.
- The source, bibliography, result assets, flattened submission source, and
  final build each have validated scoped provenance manifests.

## Submission boundary

The scientific and reproducibility package is complete. Before submission, the
authors must replace identity and affiliation placeholders, resolve funding
and contribution declarations, approve competing-interest and AI-assistance
statements, deposit the package to obtain a persistent identifier, select and
verify a current journal/quartile route, and complete the portal submission.
No local test, PDF build, or review can establish acceptance or current SCI-Q2
classification.
