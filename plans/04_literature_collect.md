# Stage 4 Plan — LITERATURE_COLLECT

## Goal

Retrieve real paper metadata from official academic APIs, deduplicate by DOI/title, and retain the provenance of every record.

## Procedure

1. Query every search family in `docs/02_search_and_baseline_contract.md`.
2. Save raw API responses separately from the curated shortlist.
3. Resolve exact-title records for all core baselines.
4. Flag preprint-only records rather than silently presenting them as peer reviewed.

## Exit criteria

- >=50 deduplicated candidates before screening.
- Exact records for DreamerV3, TD-MPC2, IRIS, and the 2022 physics-guided dynamics comparator.
- Retrieval date, query, source, DOI/URL, year, and venue retained.

