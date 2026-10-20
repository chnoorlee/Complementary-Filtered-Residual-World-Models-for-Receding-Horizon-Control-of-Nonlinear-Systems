"""Reproducible academic-metadata retrieval for the RGP-WM paper.

This script queries official OpenAlex and DBLP APIs.  It stores metadata and
provenance only; it does not download copyrighted full text.  The output is a
candidate pool for human screening, not an automatically accepted bibliography.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEARCH_QUERIES = [
    "world model reinforcement learning robust prediction",
    "DreamerV3 world model uncertainty",
    "TD-MPC2 latent dynamics SimNorm",
    "IRIS transformer sample efficient world model",
    "stochastic transformer world model",
    "physics guided neural nonlinear dynamics",
    "hybrid physics data driven dynamical systems",
    "differentiable physics learned correction",
    "uncertainty calibration dynamical systems",
    "missing observations latent state model",
    "long horizon rollout error world model",
    "model based reinforcement learning survey",
]

CORE_TITLES = [
    "Mastering diverse control tasks through world models",
    "TD-MPC2: Scalable, Robust World Models for Continuous Control",
    "Transformers are Sample-Efficient World Models",
    "Physics guided neural networks for modelling of non-linear dynamics",
]

RELEVANCE_TERMS = {
    "world model",
    "model-based reinforcement",
    "dynamics",
    "dynamical",
    "physics-informed",
    "physics guided",
    "hybrid model",
    "uncertainty",
    "calibration",
    "state space",
    "missing observation",
    "rollout",
    "forecast",
}


@dataclass(frozen=True)
class RetrievalConfig:
    per_query: int = 25
    timeout_seconds: int = 40
    retries: int = 3
    pause_seconds: float = 0.35


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_json(url: str, config: RetrievalConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "RGP-WM-literature-audit/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(config.retries):
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < config.retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed after {config.retries} attempts: {url}") from last_error


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    length = 1 + max(position for positions in index.values() for position in positions)
    words = [""] * length
    for word, positions in index.items():
        for position in positions:
            words[position] = word
    return " ".join(words)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value


def normalize_title(title: str) -> str:
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in title).split())


def source_name(work: dict[str, Any]) -> str:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    return source.get("display_name") or ""


def author_names(work: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for authorship in work.get("authorships") or []:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            names.append(name)
    return names


def openalex_url(query: str, per_query: int) -> str:
    params = {
        "search": query,
        "filter": "from_publication_date:2021-01-01,to_publication_date:2026-08-05",
        "sort": "relevance_score:desc",
        "per-page": str(per_query),
        "select": (
            "id,doi,title,display_name,publication_year,publication_date,type,"
            "primary_location,authorships,abstract_inverted_index,cited_by_count,ids"
        ),
    }
    return "https://api.openalex.org/works?" + urllib.parse.urlencode(params)


def convert_work(work: dict[str, Any], query: str, retrieved_at: str) -> dict[str, Any]:
    title = work.get("title") or work.get("display_name") or ""
    doi = normalize_doi(work.get("doi"))
    location = work.get("primary_location") or {}
    landing_page = location.get("landing_page_url")
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    searchable = f"{title} {abstract}".lower()
    relevance_hits = sorted(term for term in RELEVANCE_TERMS if term in searchable)
    return {
        "paper_id": work.get("id"),
        "title": title,
        "normalized_title": normalize_title(title),
        "authors": author_names(work),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "venue": source_name(work),
        "work_type": work.get("type"),
        "doi": doi,
        "url": landing_page or (f"https://doi.org/{doi}" if doi else work.get("id")),
        "abstract": abstract,
        "cited_by_count_snapshot": work.get("cited_by_count"),
        "relevance_terms": relevance_hits,
        "discovery_query": query,
        "metadata_source": "OpenAlex",
        "retrieved_at": retrieved_at,
    }


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"doi:{record['doi']}" if record.get("doi") else f"title:{record['normalized_title']}"
        if key not in retained:
            retained[key] = record
            retained[key]["discovery_queries"] = [record["discovery_query"]]
        else:
            queries = retained[key]["discovery_queries"]
            if record["discovery_query"] not in queries:
                queries.append(record["discovery_query"])
            if len(record.get("abstract", "")) > len(retained[key].get("abstract", "")):
                preserved_queries = retained[key]["discovery_queries"]
                retained[key] = record
                retained[key]["discovery_queries"] = preserved_queries
    return sorted(
        retained.values(),
        key=lambda item: (
            -len(item.get("relevance_terms", [])),
            -(item.get("cited_by_count_snapshot") or 0),
            item.get("title", ""),
        ),
    )


def query_dblp(title: str, config: RetrievalConfig) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"q": title, "h": "5", "format": "json"})
    payload = get_json(f"https://dblp.org/search/publ/api?{params}", config)
    raw_hits = ((payload.get("result") or {}).get("hits") or {}).get("hit") or []
    if isinstance(raw_hits, dict):
        raw_hits = [raw_hits]
    results: list[dict[str, Any]] = []
    for hit in raw_hits:
        info = hit.get("info") or {}
        results.append(
            {
                "title": info.get("title"),
                "year": info.get("year"),
                "venue": info.get("venue"),
                "doi": normalize_doi(info.get("doi")),
                "url": info.get("url"),
                "metadata_source": "DBLP",
                "query": title,
                "retrieved_at": utc_now(),
            }
        )
    return results


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("literature"))
    parser.add_argument("--per-query", type=int, default=25)
    args = parser.parse_args()
    config = RetrievalConfig(per_query=args.per_query)
    retrieved_at = utc_now()

    all_records: list[dict[str, Any]] = []
    source_log: list[dict[str, Any]] = []
    for query in SEARCH_QUERIES + CORE_TITLES:
        url = openalex_url(query, config.per_query)
        payload = get_json(url, config)
        results = payload.get("results") or []
        all_records.extend(convert_work(work, query, retrieved_at) for work in results)
        source_log.append(
            {
                "source": "OpenAlex",
                "query": query,
                "url": url,
                "records": len(results),
                "retrieved_at": retrieved_at,
                "status": "ok",
            }
        )
        time.sleep(config.pause_seconds)

    deduplicated = deduplicate(all_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in deduplicated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    dblp_core: dict[str, list[dict[str, Any]]] = {}
    for title in CORE_TITLES[:3]:
        try:
            dblp_core[title] = query_dblp(title, config)
            status = "ok"
        except RuntimeError as exc:
            dblp_core[title] = [{"error": str(exc)}]
            status = "failed"
        source_log.append(
            {
                "source": "DBLP",
                "query": title,
                "url": "https://dblp.org/search/publ/api",
                "records": len(dblp_core[title]),
                "retrieved_at": utc_now(),
                "status": status,
            }
        )
        time.sleep(config.pause_seconds)

    write_json(args.output_dir / "dblp_core_verification.json", dblp_core)
    write_json(args.output_dir / "retrieval_sources.json", source_log)
    write_json(
        args.output_dir / "retrieval_summary.json",
        {
            "generated_at": utc_now(),
            "date_window": ["2021-01-01", "2026-08-05"],
            "queries": len(SEARCH_QUERIES) + len(CORE_TITLES),
            "raw_records": len(all_records),
            "deduplicated_records": len(deduplicated),
            "records_with_doi": sum(bool(record.get("doi")) for record in deduplicated),
            "records_with_abstract": sum(bool(record.get("abstract")) for record in deduplicated),
        },
    )
    print(f"raw_records: {len(all_records)}")
    print(f"deduplicated_records: {len(deduplicated)}")
    print(f"output_dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

