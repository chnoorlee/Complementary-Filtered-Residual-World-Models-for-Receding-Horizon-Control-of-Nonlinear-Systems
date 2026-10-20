# Third-party code review and rebuttal

Date: 2026-10-20

## Review findings

The independent code review found no critical or high-severity defect. It
identified two medium-severity provenance gaps and one low-severity package
closure gap:

1. the flattened Springer source manifest did not bind the class/style assets
   or compiled build products;
2. the citation audit checked title and DOI while the bibliography generator
   also emitted author, year, venue, and URL metadata; and
3. no final whole-package checksum snapshot had yet been generated.

During final regression, an indexed-row labeling defect was also found in the
secondary manuscript-table generator. The code indexed rows by method and then
attempted to read the removed `method` column.

## Implemented response

- The flattened-source manifest now includes `sn-jnl.cls`,
  `sn-mathphys-num.bst`, and the reproduction instructions.
- A separate Springer build manifest authenticates the final PDF, log,
  template/style assets, flattened source, source manifest, bibliography, and
  detected build-tool versions. Its Windows `pdfinfo` probe uses the correct
  `-v` argument and is regression tested.
- All bibliography fields emitted from the source records are recorded and
  compared against the citation-audit source snapshot. Drift tests cover these
  fields. The audit language distinguishes independently verified title/DOI
  from source-snapshot author/year/venue/URL provenance.
- The table generator now uses the preserved index value for method lookup,
  with a regression assertion for every displayed style label.
- The manuscript-asset manifest now binds its generator script, so a changed
  generator cannot silently reuse authenticated outputs.
- Build-manifest validation now recursively authenticates the flattened
  submission-source chain and checks recorded tool versions against the
  detected local toolchain instead of trusting self-reported metadata.
- The package inventory now covers the root-level `paper.txt` and both final
  `.bbl`/`.log` pairs. Narrow Git ignore exceptions retain those four build
  records in the recovery snapshot while auxiliary LaTeX files stay excluded.
  Repository attributes disable line-ending normalization so a checkout
  preserves the exact bytes authenticated by the package and build manifests.
- The whole-package checksum snapshot is generated only after the manuscript,
  documentation, tests, and build records are frozen.

## Disposition

The final post-fix code review returned `APPROVE`: no critical, high, medium,
or low actionable finding remained. It independently confirmed the generator
binding, recursive source-chain authentication, current-toolchain comparison,
14/14 focused tests, and 91/91 full tests. A second focused review also
approved the final package-inventory scope and recovery-file exceptions with
no actionable finding. This clean local review does not convert verification
into external journal acceptance.
