from __future__ import annotations

import json
import zipfile

import pytest

import build_ijdc_submission as ijdc


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    directory = root / "paper" / "ijdc"
    sections = directory / "sections"
    sections.mkdir(parents=True)
    monkeypatch.setattr(ijdc, "ROOT", root)
    monkeypatch.setattr(ijdc, "DIRECTORY", directory)
    monkeypatch.setattr(ijdc, "SECTIONS", sections)
    monkeypatch.setattr(ijdc, "MANIFEST", directory / "source_manifest.json")
    monkeypatch.setattr(ijdc, "BUILD_MANIFEST", directory / "build_manifest.json")
    monkeypatch.setattr(ijdc, "scientific_provenance", lambda: None)
    monkeypatch.setattr(ijdc, "key", lambda path: path.resolve().as_posix())
    bib = root / "references.bib"
    bib.write_text("@article{source, title={Verified source}, year={2026}}", encoding="utf-8")
    audit = root / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ijdc, "BIBLIOGRAPHY_SOURCES", (bib,))
    monkeypatch.setattr(ijdc, "CITATION_AUDIT", audit)
    reviewer_macros = root / "reviewer_additional_macros.tex"
    reviewer_macros.write_text(r"\newcommand{\AuditValue}{1}", encoding="utf-8")
    reviewer_manifest = root / "reviewer_manifest.json"
    reviewer_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ijdc, "REVIEWER_MACROS", reviewer_macros)
    monkeypatch.setattr(ijdc, "REVIEWER_MANIFEST", reviewer_manifest)
    cls = root / "sn-jnl.cls"
    bst = root / "sn-basic.bst"
    cls.write_text("template", encoding="utf-8")
    bst.write_text("style", encoding="utf-8")
    monkeypatch.setattr(ijdc, "COPY_SOURCES", {
        "references.bib": bib, "sn-jnl.cls": cls, "sn-basic.bst": bst,
    })
    for name in ("abstract", *ijdc.BODY_SECTIONS, "appendix"):
        text = "\\begin{abstract}Evidence.\\end{abstract}" if name == "abstract" else name
        (sections / f"{name}.tex").write_text(text, encoding="utf-8")
    figures = "\n".join(r"\begin{figure}\includegraphics{" + old + r"}\end{figure}"
                        for old in ijdc.FIGURE_MAP)
    (sections / "method.tex").write_text(figures + r"\cite{source}", encoding="utf-8")
    for relative in ijdc.FIGURE_MAP.values():
        path = directory / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"vector artwork")
    return directory


def test_portable_anonymous_build_and_manifest_detect_edit(project):
    ijdc.build()
    source = (project / "main.tex").read_text(encoding="utf-8")
    assert r"\documentclass[iicol,pdflatex,sn-basic,Numbered]" in source
    assert r"\author{Anonymous authors}" in source
    assert r"\bibliography{references}" in source
    assert r"\input{" not in source
    assert "author@example.com" not in source
    assert "Author Placeholder" not in source
    assert r"\begin{figure}" in source
    assert r"\begin{figure*}" not in source
    ijdc.validate_source_manifest()
    (project / "sections" / "conclusion.tex").write_text("Revised conclusion", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        ijdc.validate_source_manifest()


def test_missing_bibliography_fails_before_build(project):
    ijdc.BIBLIOGRAPHY_SOURCES[0].unlink()
    with pytest.raises(RuntimeError, match="required IJDC input missing"):
        ijdc.build()


def test_missing_citation_is_not_silently_packaged(project):
    with (project / "sections" / "discussion.tex").open("a", encoding="utf-8") as handle:
        handle.write(r"\cite{invented}")
    with pytest.raises(RuntimeError, match="lacks cited keys"):
        ijdc.build()


def test_only_anonymous_sources_are_packaged_and_stale_pdf_rejected(project):
    ijdc.build()
    for suffix in ("pdf", "bbl", "log"):
        (project / f"main.{suffix}").write_text("compiled", encoding="utf-8")
    (project / "title_page.tex").write_text("PRIVATE AUTHOR DETAILS", encoding="utf-8")
    receipt = {
        "complete": True, "source_manifest_sha256": ijdc.sha256(ijdc.MANIFEST),
        "outputs": ijdc.hash_files([project / f"main.{ext}" for ext in ("pdf", "bbl", "log")]),
    }
    ijdc.BUILD_MANIFEST.write_text(json.dumps(receipt), encoding="utf-8")
    archive_path = ijdc.package_sources()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "title_page.tex" not in names
        assert "main.tex" in names and "main.bbl" in names
        assert "figures/Fig4.pdf" in names
        assert all(not name.startswith(("/", "..")) for name in names)
    (project / "main.pdf").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="compiled output is stale"):
        ijdc.package_sources()


def test_figure_or_audit_drift_is_detected(project):
    ijdc.build()
    ijdc.CITATION_AUDIT.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        ijdc.validate_source_manifest()


def test_author_metadata_in_sections_is_rejected(project):
    with (project / "sections" / "method.tex").open("a", encoding="utf-8") as handle:
        handle.write(r"\email{author@example.com}")
    with pytest.raises(RuntimeError, match="identifying author metadata"):
        ijdc.build()


def test_layout_adaptation_preserves_explicit_float_widths():
    source = (r"\begin{figure}\includegraphics{figures/figure1_method.png}"
              r"\end{figure}\begin{table*}wide\end{table*}")
    adapted = ijdc.adapt_layout(source)
    assert r"\begin{figure}" in adapted
    assert r"\begin{figure*}" not in adapted
    assert r"\begin{table*}wide\end{table*}" in adapted
    assert r"\includegraphics{figures/Fig1.pdf}" in adapted


def test_reviewer_analysis_manifest_detects_output_drift(project):
    analysis_input = ijdc.ROOT / "analysis.py"
    analysis_output = ijdc.ROOT / "reviewer_table.tex"
    analysis_input.write_text("verified analysis", encoding="utf-8")
    analysis_output.write_text("verified table", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "complete": True,
        "analysis_scope": "post_review_secondary_no_confirmatory_decision_change",
        "inputs": {"analysis.py": ijdc.sha256(analysis_input)},
        "outputs": {"reviewer_table.tex": ijdc.sha256(analysis_output)},
    }
    ijdc.REVIEWER_MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    ijdc.validate_reviewer_analysis()
    analysis_output.write_text("changed after analysis", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reviewer-analysis outputs drift"):
        ijdc.validate_reviewer_analysis()
