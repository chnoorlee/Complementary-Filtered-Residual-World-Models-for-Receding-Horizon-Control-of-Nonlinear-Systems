from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

import build_springer_submission as springer


def test_expand_inputs_flattens_nested_sources(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    main = root / "paper" / "mypaper"
    inputs = root / "results"
    main.mkdir(parents=True)
    inputs.mkdir()
    (inputs / "outer.tex").write_text(
        "before \\input{../../results/inner} after", encoding="utf-8",
    )
    (inputs / "inner.tex").write_text("locked table", encoding="utf-8")
    monkeypatch.setattr(springer, "ROOT", root)

    expanded = springer._expand_inputs(
        r"start \input {../../results/outer} end", main,
    )

    assert expanded == "start before locked table after end"
    assert r"\input{" not in expanded


def test_expand_inputs_rejects_project_escape(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    main = root / "paper" / "mypaper"
    main.mkdir(parents=True)
    outside = tmp_path / "outside.tex"
    outside.write_text("not allowed", encoding="utf-8")
    monkeypatch.setattr(springer, "ROOT", root)

    with pytest.raises(RuntimeError, match="outside the project"):
        springer._expand_inputs(r"\input{../../../outside}", main)


def test_expand_inputs_accepts_spacing_and_final_guard_covers_include() -> None:
    assert springer._expand_inputs.__name__ == "_expand_inputs"
    assert springer.re.search(r"\\(?:input|include)\b", r"\include{chapter}")
    assert springer.re.search(r"\\(?:input|include)\b", r"\input file")


def test_asset_manifest_rejects_stale_result_asset(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    asset = root / "paper" / "result.tex"
    sources = (
        root / "docs" / "contract.json",
        root / "results" / "run.json",
    )
    asset.parent.mkdir(parents=True)
    for path in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    asset.write_text("authenticated", encoding="utf-8")
    generator = root / "code" / "make_confirmatory_manuscript_assets.py"
    generator.parent.mkdir(parents=True)
    generator.write_text("# generator\n", encoding="utf-8")
    manifest_path = root / "paper" / "manifest.json"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps({
        "complete": True,
        "generator": {"code/make_confirmatory_manuscript_assets.py": digest(generator)},
        "assets": {"paper/result.tex": digest(asset)},
        "authenticated_sources": {
            path.relative_to(root).as_posix(): digest(path) for path in sources
        },
    }), encoding="utf-8")
    monkeypatch.setattr(springer, "ROOT", root)
    monkeypatch.setattr(springer, "ASSET_MANIFEST", manifest_path)
    monkeypatch.setattr(springer, "ASSET_GENERATOR", generator)
    monkeypatch.setattr(springer, "_analysis_sources", lambda: sources)
    monkeypatch.setattr(springer, "_validate_analysis_code_provenance", lambda manifest: None)

    springer._validate_asset_manifest([asset])
    asset.write_text("stale or edited", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        springer._validate_asset_manifest([asset])
    asset.write_text("authenticated", encoding="utf-8")
    generator.write_text("# changed generator\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        springer._validate_asset_manifest([asset])


def test_analysis_code_provenance_rejects_code_drift(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    package = root / "code" / "rgpwm"
    package.mkdir(parents=True)
    for name in ("confirm_protocol.py", "confirm_analyze.py", "statistics.py"):
        (package / name).write_text(f"# {name}\n", encoding="utf-8")
    monkeypatch.setattr(springer, "ROOT", root)
    digest = springer._analysis_code_sha256()
    contract = root / "docs" / "12_cfr_confirmatory_contract.json"
    contract.parent.mkdir()
    contract.write_text(json.dumps({
        "hashes": {"analysis_code_sha256": digest},
    }), encoding="utf-8")
    amendment_code = package / "confirm_analyze_amended.py"
    amendment_code.write_text("# amendment\n", encoding="utf-8")
    amendment_code_digest = hashlib.sha256(amendment_code.read_bytes()).hexdigest()
    amendment_contract = root / "docs" / "14_cfr_analysis_amendment.json"
    amendment_contract.write_text(json.dumps({
        "amendment_code_sha256": amendment_code_digest,
    }), encoding="utf-8")
    amendment_contract_digest = hashlib.sha256(amendment_contract.read_bytes()).hexdigest()
    manifest = {
        "analysis_code_sha256": digest,
        "analysis_amendment_code_sha256": amendment_code_digest,
        "analysis_amendment_contract_sha256": amendment_contract_digest,
    }

    springer._validate_analysis_code_provenance(manifest)
    (package / "statistics.py").write_text("# drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="analysis-code provenance mismatch"):
        springer._validate_analysis_code_provenance(manifest)


def test_bibliography_provenance_rejects_stale_audit(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    literature = root / "literature"
    paper = root / "paper"
    literature.mkdir(parents=True)
    paper.mkdir()
    shortlist = literature / "shortlist.json"
    supplement = literature / "v2_supplement.json"
    audit = literature / "citation_audit.json"
    references = paper / "references.bib"
    bibliography_manifest = literature / "bibliography_manifest.json"
    generators = (root / "code" / "make_bibliography.py", root / "code" / "bibliography_provenance.py")
    for path in generators:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    record = {
        "cite_key": "verified2025", "title": "Verified title", "doi": "10.1/x",
        "authors": ["A. Author"], "year": 2025, "venue": "Journal",
        "url": "https://example.org/verified2025",
    }
    shortlist.write_text(json.dumps({"shortlist": [record]}), encoding="utf-8")
    supplement.write_text(json.dumps({"shortlist": []}), encoding="utf-8")
    audit.write_text(json.dumps({
        "verified": 1,
        "records": [{
            "cite_key": "verified2025", "expected_title": "Verified title",
            "doi": "10.1/x", "status": "verified",
            "source_snapshot_authors": ["A. Author"],
            "source_snapshot_year": 2025,
            "source_snapshot_venue": "Journal",
            "source_snapshot_url": "https://example.org/verified2025",
        }],
    }), encoding="utf-8")
    references.write_text("@article{verified2025}\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    bibliography_manifest.write_text(json.dumps({
        "complete": True,
        "entry_count": 1,
        "sources": {
            path.relative_to(root).as_posix(): digest(path)
            for path in (shortlist, supplement, audit)
        },
        "generators": {
            path.relative_to(root).as_posix(): digest(path)
            for path in generators
        },
        "bibliography": {"paper/references.bib": digest(references)},
    }), encoding="utf-8")
    monkeypatch.setattr(springer, "ROOT", root)
    monkeypatch.setattr(springer, "BIBLIOGRAPHY_GENERATORS", generators)
    monkeypatch.setattr(springer, "REFERENCES", references)
    monkeypatch.setattr(springer, "BIBLIOGRAPHY_MANIFEST", bibliography_manifest)

    springer._validate_bibliography_provenance()
    audit.write_text(json.dumps({"verified": 0, "records": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bibliography-source manifest mismatch"):
        springer._validate_bibliography_provenance()


def test_submission_manifest_detects_prose_drift(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    sections = root / "sections"
    paper = root / "paper"
    main = paper / "mypaper"
    sections.mkdir(parents=True)
    main.mkdir(parents=True)
    names = ("abstract", *springer.BODY_SECTIONS, "appendix")
    for name in names:
        (sections / f"{name}.tex").write_text(name, encoding="utf-8")
    figures = main / "figures"
    figures.mkdir()
    graphic = figures / "result.png"
    graphic.write_bytes(b"original graphic")
    (sections / "results.tex").write_text(
        r"results \includegraphics{figures/result.png}", encoding="utf-8",
    )
    references = paper / "references.bib"
    asset_manifest = paper / "asset.json"
    bibliography_manifest = root / "literature" / "bibliography.json"
    bibliography_manifest.parent.mkdir()
    for path in (references, asset_manifest, bibliography_manifest):
        path.write_text(path.name, encoding="utf-8")
    output = main / "main_springer.tex"
    output.write_text("flattened", encoding="utf-8")
    submission_manifest = paper / "submission.json"
    builder = root / "code" / "builder.py"
    builder.parent.mkdir(exist_ok=True)
    builder.write_text("# builder", encoding="utf-8")
    template_sources = (
        root / "paper" / "springer" / "sn-jnl.cls",
        root / "paper" / "springer" / "sn-mathphys-num.bst",
        root / "code" / "README.md",
    )
    for path in template_sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    monkeypatch.setattr(springer, "ROOT", root)
    monkeypatch.setattr(springer, "SECTIONS", sections)
    monkeypatch.setattr(springer, "MAIN_DIR", main)
    monkeypatch.setattr(springer, "REFERENCES", references)
    monkeypatch.setattr(springer, "ASSET_MANIFEST", asset_manifest)
    monkeypatch.setattr(springer, "BIBLIOGRAPHY_MANIFEST", bibliography_manifest)
    monkeypatch.setattr(springer, "OUTPUT", output)
    monkeypatch.setattr(springer, "SUBMISSION_MANIFEST", submission_manifest)
    monkeypatch.setattr(springer, "__file__", str(builder))
    monkeypatch.setattr(springer, "_template_sources", lambda: template_sources)
    monkeypatch.setattr(springer, "_validate_bibliography_provenance", lambda: None)
    monkeypatch.setattr(springer, "_validate_asset_manifest", lambda required: None)

    springer._write_submission_manifest()
    springer.validate_submission_manifest()
    (sections / "results.tex").write_text("edited result claim", encoding="utf-8")
    with pytest.raises(RuntimeError, match="submission source manifest mismatch"):
        springer.validate_submission_manifest()
    (sections / "results.tex").write_text(
        r"results \includegraphics{figures/result.png}", encoding="utf-8",
    )
    springer._write_submission_manifest()
    graphic.write_bytes(b"edited graphic")
    with pytest.raises(RuntimeError, match="submission source manifest mismatch"):
        springer.validate_submission_manifest()
    graphic.write_bytes(b"original graphic")
    springer._write_submission_manifest()
    template_sources[0].write_text("edited class", encoding="utf-8")
    with pytest.raises(RuntimeError, match="submission source manifest mismatch"):
        springer.validate_submission_manifest()
