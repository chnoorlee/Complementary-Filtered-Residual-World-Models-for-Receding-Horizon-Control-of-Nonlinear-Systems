"""Shared validation for DOI/title checks and frozen source-metadata fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SOURCE_SNAPSHOT_FIELDS = (
    ("source_snapshot_authors", "authors"),
    ("source_snapshot_year", "year"),
    ("source_snapshot_venue", "venue"),
    ("source_snapshot_url", "url"),
)


def audit_row_matches_record(
    row: Mapping[str, Any], record: Mapping[str, Any],
) -> bool:
    """Return whether every emitted BibTeX field matches the frozen audit row."""
    if (
        row.get("status") != "verified"
        or row.get("expected_title") != record.get("title")
        or (row.get("doi") or None) != (record.get("doi") or None)
    ):
        return False
    return all(row.get(audit_name) == record.get(source_name)
               for audit_name, source_name in SOURCE_SNAPSHOT_FIELDS)
