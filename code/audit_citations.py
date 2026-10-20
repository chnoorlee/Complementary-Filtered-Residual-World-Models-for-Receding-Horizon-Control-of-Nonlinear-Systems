"""Verify screened DOI metadata against the current OpenAlex academic graph."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def fetch(record: dict) -> tuple[dict, str]:
    if record.get("doi"):
        identifier = urllib.parse.quote(f"https://doi.org/{record['doi']}", safe="")
        route = "doi"
    else:
        identifier = record["paper_id"].rsplit("/", 1)[-1]
        route = "openalex_id_for_preprint"
    request = urllib.request.Request(
        f"https://api.openalex.org/works/{identifier}",
        headers={"User-Agent": "CFR-WM-citation-audit/1.0 (mailto:anonymous@example.com)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response), route


def main() -> None:
    shortlist_path = Path("literature/shortlist.json")
    records = json.loads(shortlist_path.read_text(encoding="utf-8"))["shortlist"]
    supplement_path = Path("literature/v2_supplement.json")
    if supplement_path.exists():
        records.extend(json.loads(supplement_path.read_text(encoding="utf-8"))["shortlist"])
    output = []
    for index, record in enumerate(records, 1):
        audit = {
            "cite_key": record["cite_key"], "doi": record.get("doi"),
            "expected_title": record["title"], "status": "unverified",
        }
        try:
            resolved, route = fetch(record)
            resolved_title = resolved.get("title") or ""
            similarity = SequenceMatcher(None, normalize(record["title"]), normalize(resolved_title)).ratio()
            audit.update({
                "status": "verified" if similarity >= 0.92 else "title_mismatch",
                "resolved_title": resolved_title,
                "title_similarity": similarity,
                "openalex_id": resolved.get("id"),
                "publication_date": resolved.get("publication_date"),
                "verification_route": route,
            })
        except (urllib.error.URLError, TimeoutError, KeyError, TypeError) as error:
            audit["error"] = str(error)
        output.append(audit)
        print(json.dumps({"index": index, "total": len(records), **audit}, ensure_ascii=False))
        time.sleep(0.08)
    destination = Path("literature/citation_audit.json")
    destination.write_text(json.dumps({
        "audited_on": date.today().isoformat(),
        "source": "OpenAlex works-by-DOI endpoint",
        "verified": sum(item["status"] == "verified" for item in output),
        "records": output,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    failures = [item for item in output if item["status"] != "verified"]
    print(json.dumps({"verified": len(output) - len(failures), "failed": len(failures)}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
