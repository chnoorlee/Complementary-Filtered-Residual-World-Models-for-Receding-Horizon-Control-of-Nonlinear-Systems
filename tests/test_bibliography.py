from __future__ import annotations

import json

import pytest

import make_bibliography as bibliography


def _record(key: str, title: str, doi: str | None = None) -> dict[str, object]:
    return {
        "cite_key": key,
        "title": title,
        "authors": ["A. Author"],
        "year": 2025,
        "venue": "Journal",
        "doi": doi,
        "url": f"https://example.org/{key}",
    }


def _audit_record(record: dict[str, object]) -> dict[str, object]:
    return {
        "cite_key": record["cite_key"],
        "doi": record["doi"],
        "expected_title": record["title"],
        "status": "verified",
        "source_snapshot_authors": record["authors"],
        "source_snapshot_year": record["year"],
        "source_snapshot_venue": record["venue"],
        "source_snapshot_url": record["url"],
    }


def test_bibliography_requires_exact_verified_audit(tmp_path, monkeypatch) -> None:
    primary = tmp_path / "shortlist.json"
    supplement = tmp_path / "supplement.json"
    audit = tmp_path / "audit.json"
    first = _record("first2025", "First title", "10.1/first")
    second = _record("second2025", "Second title")
    primary.write_text(json.dumps({"shortlist": [first]}), encoding="utf-8")
    supplement.write_text(json.dumps({"shortlist": [second]}), encoding="utf-8")
    audit.write_text(json.dumps({
        "verified": 2,
        "records": [_audit_record(first), _audit_record(second)],
    }), encoding="utf-8")
    monkeypatch.setattr(bibliography, "SHORTLIST_PATH", primary)
    monkeypatch.setattr(bibliography, "SUPPLEMENT_PATH", supplement)
    monkeypatch.setattr(bibliography, "AUDIT_PATH", audit)

    assert [row["cite_key"] for row in bibliography._load_verified_records()] == [
        "first2025", "second2025",
    ]

    first["title"] = "Edited after audit"
    primary.write_text(json.dumps({"shortlist": [first]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or unverified"):
        bibliography._load_verified_records()


@pytest.mark.parametrize(
    ("field", "edited"),
    [
        ("authors", ["Changed Author"]),
        ("year", 2024),
        ("venue", "Changed Journal"),
        ("url", "https://example.org/changed"),
    ],
)
def test_bibliography_rejects_emitted_metadata_drift(
    tmp_path, monkeypatch, field, edited,
) -> None:
    primary = tmp_path / "shortlist.json"
    audit = tmp_path / "audit.json"
    record = _record("first2025", "First title", "10.1/first")
    primary.write_text(json.dumps({"shortlist": [record]}), encoding="utf-8")
    audit.write_text(json.dumps({
        "verified": 1, "records": [_audit_record(record)],
    }), encoding="utf-8")
    monkeypatch.setattr(bibliography, "SHORTLIST_PATH", primary)
    monkeypatch.setattr(bibliography, "SUPPLEMENT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(bibliography, "AUDIT_PATH", audit)

    record[field] = edited
    primary.write_text(json.dumps({"shortlist": [record]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or unverified"):
        bibliography._load_verified_records()


def test_bibliography_rejects_missing_audit_key(tmp_path, monkeypatch) -> None:
    primary = tmp_path / "shortlist.json"
    audit = tmp_path / "audit.json"
    primary.write_text(
        json.dumps({"shortlist": [_record("first2025", "First title")]}),
        encoding="utf-8",
    )
    audit.write_text(json.dumps({"verified": 0, "records": []}), encoding="utf-8")
    monkeypatch.setattr(bibliography, "SHORTLIST_PATH", primary)
    monkeypatch.setattr(bibliography, "SUPPLEMENT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(bibliography, "AUDIT_PATH", audit)

    with pytest.raises(RuntimeError, match="keys do not exactly match"):
        bibliography._load_verified_records()
