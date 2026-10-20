# JIRS pre-submission checklist

This checklist separates completed scientific work from author-controlled or
subscription-controlled metadata. A checked scientific item must have a local
artifact; it is not satisfied by prose alone.

## Scientific package

- [x] `results/confirmatory_run_manifest.json` validates exactly 440 paired JSON/PT
  runs and no unexpected files.
- [x] `results/confirmatory_analysis/confirmatory_inference.json` was produced
  by the versioned no-override amendment, which authenticates the original
  analyzer and calls its unchanged frozen statistics; it matches all
  contract/code/tuning/amendment hashes.
- [x] The analysis contains 1,320 aggregate rows, 126,720 trajectory rows, 40
  distinct seed--system data hashes, and finite metrics.
- [x] H1 and H2 are reported exactly as frozen, including negative or
  inconclusive outcomes.
- [x] Five broad-comparator and five sensitivity families retain their exact
  two-sided seed sign-flip and Holm values; cell contrasts remain descriptive.
- [x] Manuscript macros, figures, and tables were regenerated after locked
  analysis, not copied from v1/v2 artifacts.
- [x] Full tests pass 91/91 and `compileall` succeeds; the sole PyTorch
  nested-tensor performance warning is classified and disclosed.
- [x] Citation audit reports 41/41 DOI/title verifications, binds every emitted
  source-snapshot field, confirms that all cited keys exist in the
  bibliography, and finds no unused bibliography entry.
- [x] Both working and Springer PDFs compile through the full
  PDFLaTeX--BibTeX--PDFLaTeX--PDFLaTeX chain without undefined references or
  citations.
- [x] A final statistical reviewer, code reviewer, evidence-bound manuscript
  reviewer, and hostile external-style reviewer have issued no unresolved
  blocking findings.

## Journal and formatting

- [ ] The submitting author has specified whether “SCI 2区” means JCR Q2, CAS
  2区, or both.
- [ ] JIRS is verified in the required current institutional quartile database
  on the actual submission date; otherwise select a scope-matched verified Q2
  journal and reformat.
- [x] Official JIRS scope and author instructions were checked on 2026-08-05.
- [x] The rendered Springer abstract is 236 words and six keywords are present.
- [x] JIRS Category (2) and MSC codes are placed on the Springer title page.
- [x] Official December 2024 Springer Nature template assets are archived with
  their download checksum.
- [ ] Recheck the official template and policies on the actual submission date.

## Author-supplied fields

- [ ] Replace author, email, affiliation, city, and country placeholders.
- [ ] Supply funding and grant identifiers, or state “Not applicable.”
- [ ] Supply an author-contribution statement agreed by all authors.
- [ ] Confirm the competing-interests declaration with every author.
- [ ] Deposit code and results in an archival repository and add its DOI or
  persistent URL to the data/code availability statement.
- [ ] Confirm that all authors approve the submitted manuscript and that it is
  not under consideration elsewhere.
- [ ] Review and, if needed, adapt the generative-AI disclosure to the Springer
  policy in force at submission.
- [ ] Perform a final human review of equations, claims, reference metadata,
  figures, accessibility, and supplementary files.
