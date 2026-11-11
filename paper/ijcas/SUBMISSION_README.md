# IJCAS submission package

Prepared for the *International Journal of Control, Automation, and Systems*
on 2026-11-10. The journal's official submission guidance was checked on that
date: https://link.springer.com/journal/12555/submission-guidelines

## Files to upload

1. `IJCAS_Manuscript.pdf` — the complete single-blind manuscript. It includes
   the author's name, affiliation, corresponding-author email, ORCID, abstract,
   keywords, figures, tables, references, and Statements and Declarations.
2. `IJCAS_Source_Files.zip` — the complete editable LaTeX source set required
   by the journal at initial submission and revision. It contains `main.tex`,
   the bibliography, Springer class and bibliography-style files, all five
   figure PDFs, the generated bibliography, and the compiled manuscript PDF.
3. `IJCAS_Cover_Letter.pdf` — a factual cover letter based on the manuscript's
   reported study design and mixed results.
4. `IJCAS_Title_Page.pdf` — a separate author-information and declarations
   file. The journal uses single-blind review, so the same author information
   is also correctly retained in the manuscript.

The Editorial Manager interface may combine or relabel some categories. Use
the manuscript PDF as the main manuscript and the source ZIP as the editable
source file; upload the cover letter and separate title page under their named
categories when those categories are offered.

## Files for the author, not journal uploads

- `IJCAS_Cover_Letter.tex` and `IJCAS_Title_Page.tex` are editable sources.
- `SHA256SUMS.txt` and `submission_package_manifest.json` verify package
  integrity and record the build status.
- `REPOSITORY_DOI_GUIDE.md` explains the GitHub/Zenodo and direct-Zenodo
  options.

## Author actions in the submission portal

- Confirm that the work has not been published and is not under consideration
  elsewhere. The official policy treats submission as this confirmation; the
  package does not infer it on the author's behalf.
- Enter the Author Contributions and Competing Interests statements in the
  portal as well as retaining them in the manuscript. IJCAS states that the
  portal-entered versions are used for publication.
- Confirm the article type, suggested or opposed reviewers if requested, and
  any institutional approval or permissions requested by the portal.
- After acceptance, choose the subscription publishing route if a non-open-
  access publication with no APC is desired. IJCAS is a hybrid journal and
  offers both subscription and open-access routes after acceptance.

## Repository status

The manuscript contains the required Data and Code Availability statement,
which cites the public GitHub repository:

https://github.com/chnoorlee/Complementary-Filtered-Residual-World-Models-for-Receding-Horizon-Control-of-Nonlinear-Systems

The repository holds the executable code, frozen configurations, seeds, split
identifiers, saved predictions, metrics, analysis scripts, and permitted data
provenance metadata. Third-party NanoDrone flight files are not redistributed.
A persistent archival identifier (Zenodo DOI) is not yet issued; the statement
describes the planned archival step. Once the DOI is issued, replace the
Zenodo sentence with the DOI and rebuild the package; do not insert a
provisional or invented DOI.
