from __future__ import annotations

import json
import re
from types import SimpleNamespace
from pathlib import Path

import pytest

import create_springer_build_manifest as build_manifest


def test_pdf_metadata_accepts_current_manuscript_title_and_rejects_old_title(monkeypatch):
    manuscript = (build_manifest.MAIN_DIR / "main.tex").read_text(encoding="utf-8")
    title = re.search(r"pdftitle=\{([^}]+)\}", manuscript).group(1)
    assert title == (
        "Complementary-Filtered Residual World Models: "
        "Controlled Benchmarks and Real-Flight Evaluation"
    )
    calls = []

    def pdfinfo_output(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(stdout=f"Title: {title}\nPages: 31\nAuthor: Placeholder\n")

    monkeypatch.setattr(build_manifest.subprocess, "run", pdfinfo_output)
    assert build_manifest._pdf_metadata() == {"title": title, "pages": 31, "author": "Placeholder"}
    assert calls == [["pdfinfo", str(build_manifest.MAIN_DIR / "main_springer.pdf")]]
    title = ("A Confirmatory Study of Complementary-Filtered Residual World Models "
             "under Synthetic Context Corruption")
    with pytest.raises(RuntimeError, match="unexpected PDF metadata"):
        build_manifest._pdf_metadata()


def test_run_version_uses_tool_specific_arguments(monkeypatch) -> None:
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(stdout="pdfinfo version 1.2.3\n", stderr="")

    monkeypatch.setattr(build_manifest.subprocess, "run", fake_run)
    assert build_manifest._run_version("pdfinfo") == "pdfinfo version 1.2.3"
    assert calls[0][0] == ["pdfinfo", "-v"]


def test_critical_log_findings_detects_build_failures() -> None:
    assert build_manifest._critical_log_findings("Output written on main.pdf") == []
    assert build_manifest._critical_log_findings("Overfull \\hbox (1.0pt too wide)")
    assert build_manifest._critical_log_findings("LaTeX Warning: Citation `x' undefined")


def test_validate_build_manifest_detects_hashed_output_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "main.tex"
    log = root / "main.log"
    pdf = root / "main.pdf"
    for path, value in ((source, "source"), (log, "clean log"), (pdf, "pdf")):
        path.write_text(value, encoding="utf-8")
    manifest = root / "manifest.json"
    monkeypatch.setattr(build_manifest, "ROOT", root)
    monkeypatch.setattr(build_manifest, "MAIN_DIR", root)
    monkeypatch.setattr(build_manifest, "MANIFEST", manifest)
    monkeypatch.setattr(build_manifest, "INPUTS", (source,))
    monkeypatch.setattr(build_manifest, "OUTPUTS", (log, pdf))
    monkeypatch.setattr(
        build_manifest.springer_submission, "validate_submission_manifest", lambda: None,
    )
    toolchain = {
        "pdflatex": "pdfTeX test",
        "bibtex": "BibTeX test",
        "pdfinfo": "pdfinfo test",
    }
    monkeypatch.setattr(build_manifest, "_current_toolchain", lambda: toolchain)
    monkeypatch.setattr(
        build_manifest, "_pdf_metadata",
        lambda: {"title": "test", "pages": 1, "author": ""},
    )
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "complete": True,
        "inputs": {"main.tex": build_manifest._sha256(source)},
        "outputs": {
            "main.log": build_manifest._sha256(log),
            "main.pdf": build_manifest._sha256(pdf),
        },
        "pdf": {"title": "test", "pages": 1, "author": ""},
        "toolchain": toolchain,
    }), encoding="utf-8")

    build_manifest.validate_build_manifest()
    pdf.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        build_manifest.validate_build_manifest()


def test_validate_build_manifest_rejects_stale_submission_chain(
    monkeypatch,
) -> None:
    def reject_stale_source() -> None:
        raise RuntimeError("submission source manifest mismatch")

    monkeypatch.setattr(
        build_manifest.springer_submission,
        "validate_submission_manifest",
        reject_stale_source,
    )
    with pytest.raises(RuntimeError, match="submission source manifest mismatch"):
        build_manifest.validate_build_manifest()


def test_validate_build_manifest_rejects_fabricated_toolchain(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "main.tex"
    log = root / "main.log"
    pdf = root / "main.pdf"
    for path, value in ((source, "source"), (log, "clean log"), (pdf, "pdf")):
        path.write_text(value, encoding="utf-8")
    manifest = root / "manifest.json"
    actual_toolchain = {
        "pdflatex": "pdfTeX actual",
        "bibtex": "BibTeX actual",
        "pdfinfo": "pdfinfo actual",
    }
    monkeypatch.setattr(build_manifest, "ROOT", root)
    monkeypatch.setattr(build_manifest, "MAIN_DIR", root)
    monkeypatch.setattr(build_manifest, "MANIFEST", manifest)
    monkeypatch.setattr(build_manifest, "INPUTS", (source,))
    monkeypatch.setattr(build_manifest, "OUTPUTS", (log, pdf))
    monkeypatch.setattr(
        build_manifest.springer_submission, "validate_submission_manifest", lambda: None,
    )
    monkeypatch.setattr(build_manifest, "_current_toolchain", lambda: actual_toolchain)
    monkeypatch.setattr(
        build_manifest, "_pdf_metadata",
        lambda: {"title": "test", "pages": 1, "author": ""},
    )
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "complete": True,
        "inputs": {"main.tex": build_manifest._sha256(source)},
        "outputs": {
            "main.log": build_manifest._sha256(log),
            "main.pdf": build_manifest._sha256(pdf),
        },
        "pdf": {"title": "test", "pages": 1, "author": ""},
        "toolchain": {**actual_toolchain, "pdfinfo": "fabricated"},
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="toolchain differs"):
        build_manifest.validate_build_manifest()
