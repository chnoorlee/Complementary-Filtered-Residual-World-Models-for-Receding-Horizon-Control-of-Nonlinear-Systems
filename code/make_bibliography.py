"""Create a metadata-bound BibTeX file from the screened literature set."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from bibliography_provenance import audit_row_matches_record


SHORTLIST_PATH = Path("literature/shortlist.json")
SUPPLEMENT_PATH = Path("literature/v2_supplement.json")
AUDIT_PATH = Path("literature/citation_audit.json")
DESTINATION = Path("paper/references.bib")
MANIFEST_PATH = Path("literature/bibliography_manifest.json")
GENERATOR_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).resolve().with_name("bibliography_provenance.py"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verified_records() -> list[dict[str, object]]:
    """Load exactly the records covered by a complete citation audit."""
    if not SHORTLIST_PATH.is_file() or not AUDIT_PATH.is_file():
        raise RuntimeError("shortlist and citation audit are both required")
    records = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))["shortlist"]
    if SUPPLEMENT_PATH.is_file():
        records.extend(
            json.loads(SUPPLEMENT_PATH.read_text(encoding="utf-8"))["shortlist"],
        )
    keys = [str(record["cite_key"]) for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("literature source contains duplicate cite keys")

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    audit_rows = audit.get("records")
    if not isinstance(audit_rows, list):
        raise RuntimeError("citation audit lacks a records list")
    audited = {str(row.get("cite_key")): row for row in audit_rows}
    if len(audited) != len(audit_rows) or set(audited) != set(keys):
        raise RuntimeError("citation audit keys do not exactly match literature sources")
    if audit.get("verified") != len(records):
        raise RuntimeError("citation audit verified count is incomplete")
    for record in records:
        key = str(record["cite_key"])
        row = audited[key]
        if not audit_row_matches_record(row, record):
            raise RuntimeError(f"citation audit is stale or unverified for {key}")
    return records


def escape(value: str) -> str:
    value = value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    source = _load_verified_records()
    entries = []
    for record in source:
        fields = {
            "author": " and ".join(record["authors"]),
            "title": "{{" + record["title"] + "}}",
            "year": str(record["year"]),
            "journal": record.get("venue") or "Preprint",
            "doi": record.get("doi") or "",
            "url": record.get("url") or "",
        }
        body = ",\n".join(
            f"  {name} = {{{escape(value)}}}" for name, value in fields.items() if value
        )
        entries.append(f"@article{{{record['cite_key']},\n{body}\n}}")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".bib.tmp")
    temporary.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    temporary.replace(DESTINATION)
    sources = [SHORTLIST_PATH, AUDIT_PATH]
    if SUPPLEMENT_PATH.is_file():
        sources.insert(1, SUPPLEMENT_PATH)
    manifest = {
        "complete": True,
        "entry_count": len(entries),
        "sources": {path.as_posix(): _sha256(path) for path in sources},
        "generators": {
            path.relative_to(Path.cwd().resolve()).as_posix(): _sha256(path)
            for path in GENERATOR_PATHS
        },
        "bibliography": {DESTINATION.as_posix(): _sha256(DESTINATION)},
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_temporary.replace(MANIFEST_PATH)
    print(f"wrote {len(entries)} verified-metadata entries to {DESTINATION}")
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
