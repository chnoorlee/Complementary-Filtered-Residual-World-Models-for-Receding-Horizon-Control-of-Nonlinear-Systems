# Final statistical and code review

Date: 2026-08-06  
Mode: independent, read-only agents with separate task contexts

## Statistical review

Verdict: **GO; no statistical blocker**.

The independent reviewer authenticated the frozen contract
(`9af39c8d...7a240`), completed-run manifest, 440 run payloads, 1,320 aggregate
rows, 126,720 trajectory rows, 40 seed--system fingerprints, and the frozen
analysis code (`56fd06a3...f9a3`). The analysis-only amendment was judged
isolated: its only operation before the frozen inference call is a validated
many-to-one `parameter_count` join on method--seed--system. Its recorded
`inference_change` is `none`.

The reviewer independently reproduced the decision interpretation:

- H1: PGNN 0.10961458 versus CFR-WM 0.09976695; difference -0.00984764;
  relative improvement 8.97296%, 95% CI [6.88251%, 10.95154%]; exact
  one-sided 5%-margin p=0.00390625. H1 passes.
- H2: corrupt/shift improvement 12.04514%, 95% CI
  [10.51334%, 13.49379%], p=0.00390625; clean improvement -13.07955%, 95% CI
  [-20.60060%, -5.66868%], -5% noninferiority p=0.95703125. H2 fails.
- The two five-test Holm families are separate; raw two-sided p=0.0078125 and
  Holm p=0.0390625. All 15 cell contrasts remain descriptive. The three cells
  over the 10% degradation guard are the three clean cells reported in the
  manuscript.

The reviewer found no direction error, pseudo-replication claim, gate error,
selective omission, manuscript-result discrepancy, or unsupported general
robustness claim.

## Code review and repair loop

The first final code review issued two high-severity provenance blockers:

1. bibliography generation did not require a complete citation audit; and
2. the flattened Springer source was not bound to the manuscript inputs.

The repair made bibliography generation fail closed on an exact 41-row
verified audit and introduced `literature/bibliography_manifest.json`.
Result-dependent fixed-cell values were moved from prose literals to generated
macros. The submission builder now records section, bibliography, result
manifest, builder, output, and graphic hashes in
`paper/submission_source_manifest.json`.

A second review then identified one remaining high-severity transitive gap:
static Figure 1 and nested result assets could change after the top-level
manifest was written. The repair added `includegraphics` discovery, direct
graphic hashes, and recursive revalidation of the bibliography and result
asset manifests. Regression tests mutate prose and graphics and require
validation failure.

Final code-review verdict: **GO; no remaining blocker**. The independent
reviewer ran `validate_submission_manifest()` and the full suite. The final
suite has 82 passing tests. The sole warning is PyTorch's informational nested
tensor warning for a pre-norm Transformer; it is not a numerical failure and
does not affect the tested rollout interface.
