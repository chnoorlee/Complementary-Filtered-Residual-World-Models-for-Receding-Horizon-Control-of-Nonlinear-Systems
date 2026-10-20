# Final quality-gate record

Date: 2026-10-20

## Initial gate

An independent delivery audit scored the package 8.9/10. It judged the
scientific manuscript ready for closure but the repository not yet ready for
delivery because current verification, review disposition, and a package-wide
checksum snapshot had not been durably recorded.

## Closed technical gates

- The final suite passes 91/91 tests, and `compileall` succeeds.
- The only emitted runtime warning is the documented PyTorch nested-tensor
  performance advisory associated with a `norm_first` Transformer; it is not
  a correctness or numerical-finiteness failure.
- The working manuscript is 18 pages and the Springer manuscript is 27 pages.
  Both complete the PDFLaTeX--BibTeX--PDFLaTeX--PDFLaTeX sequence without a
  logged LaTeX/package warning, undefined citation/reference, overfull box,
  fatal error, or compilation error.
- All 45 rendered pages were checked for blank output, boundary contact,
  clipping, and visible overlap. The contact sheets and the densest appendix
  pages were also inspected manually.
- After the final provenance-code rebuild, all 45 pages were rasterized again
  and matched the manually inspected versions pixel for pixel. The changed PDF
  hashes therefore reflect build metadata rather than visible content drift.
- The rendered Springer abstract contains 236 words, within the stated
  150--250-word range, and six keywords are present.
- Bibliography checks bind every emitted field to its source-snapshot value;
  title and DOI remain the independently verified subset. The Springer source
  and separate build manifests bind the template, bibliography style,
  flattened source, PDF, log, and toolchain evidence.

## Confidence-interval clarification

The primary H1 interval uses the locked bootstrap stream with seed 20260805:
`[6.8825, 10.9515]%`. The broad-comparator table uses its separately locked
family bootstrap stream with seed 20260808: `[6.8924, 10.9570]%`. Both use
5,000 paired bootstrap replicates and the same H1 point estimate and estimand.
The endpoints differ by no more than 0.01 percentage point after two-decimal
rounding. Only the primary stream determines the H1 decision.

## Gate decision

The technical scientific package is **GO for author completion and journal
selection**. It is not yet a submission: author identity and affiliations,
funding, contributions, author approvals, archival DOI or persistent URL, the
required interpretation of “SCI 2区,” and the journal's current policies and
quartile must be supplied or verified by the authors at submission time.
