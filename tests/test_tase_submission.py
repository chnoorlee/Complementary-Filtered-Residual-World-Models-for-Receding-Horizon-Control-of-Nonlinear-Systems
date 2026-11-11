from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import build_tase_submission as tase


PAPER = ROOT / "paper" / "tase"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_front_matter_order_and_word_limits() -> None:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    abstract = main.index(r"\begin{abstract}")
    practitioners = main.index(r"\section*{Note to Practitioners}")
    keywords = main.index(r"\begin{IEEEkeywords}")
    assert abstract < practitioners < keywords
    assert 1 <= tase.tex_words(PAPER / "sections" / "abstract.tex") <= 200
    assert 100 <= tase.tex_words(PAPER / "sections" / "note_to_practitioners.tex") <= 300


def test_citations_are_defined_and_literature_is_not_sparse() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PAPER / "sections").glob("*.tex"))
    )
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]*)\}", source)
        for key in group.split(",")
    }
    bibliography = (PAPER / "references.bib").read_text(encoding="utf-8")
    defined = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert cited <= defined
    assert len(cited) >= 20


def test_negative_and_external_evidence_remain_visible() -> None:
    abstract = (PAPER / "sections" / "abstract.tex").read_text(encoding="utf-8")
    results = (PAPER / "sections" / "results.tex").read_text(encoding="utf-8")
    conclusion = (PAPER / "sections" / "discussion_conclusion.tex").read_text(encoding="utf-8")
    combined = " ".join(" ".join((abstract, results, conclusion)).lower().split())
    assert "noninferiority criterion is not met" in combined
    assert "filter plus nominal" in combined
    assert "undercoverage" in combined or "undercovered" in combined
    assert "learned residual does not improve point accuracy" in combined


def test_anonymous_source_and_required_ai_disclosure() -> None:
    tase.validate_anonymity()
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert r"\author{Anonymous Author(s)}" in main
    assert "OpenAI ChatGPT/Codex was used" in main
    assert "take responsibility for its content" in main


def test_build_and_package_receipts_match_current_artifacts() -> None:
    build = json.loads((PAPER / "build_manifest.json").read_text(encoding="utf-8"))
    package = json.loads((PAPER / "package_manifest.json").read_text(encoding="utf-8"))
    assert build["complete"] is True and build["anonymous"] is True
    assert build["main_pages"] <= tase.MAX_MAIN_PAGES
    assert build["abstract_words"] <= 200
    assert 100 <= build["note_to_practitioners_words"] <= 300
    assert package["complete"] is True
    archive = PAPER / package["archive"]
    assert tase.sha256(archive) == package["archive_sha256"]
    with zipfile.ZipFile(archive) as source_zip:
        assert set(source_zip.namelist()) == set(package["files"])
        for name, digest in package["files"].items():
            assert sha256_bytes(source_zip.read(name)) == digest
        assert "figures/Fig2.pdf" not in source_zip.namelist()
        assert "figures/Fig3.pdf" not in source_zip.namelist()
        assert "table_structured_outage.tex" not in source_zip.namelist()
