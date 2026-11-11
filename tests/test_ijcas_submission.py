from __future__ import annotations

import json
import re
import zipfile

import build_ijcas_submission as ijcas


def test_ijcas_render_is_portable_and_uses_author_year_style():
    document, _, abstract_words = ijcas.render()
    assert 150 <= abstract_words <= 250
    assert r"\documentclass[pdflatex,sn-basic]{sn-jnl}" in document
    assert "Numbered" not in document.splitlines()[2]
    assert r"\cite{" not in document
    assert r"\citep{" in document
    assert r"\input{" not in document
    assert r"\bibliography{references}" in document


def test_ijcas_render_retains_all_evidence_boundaries():
    document, _, _ = ijcas.render()
    normalized = " ".join(document.split())
    required = (
        "clean-regime noninferiority is not established",
        r"\ClosedLoopEpisodes{} paired simulated",
        "does not show an additional benefit from the learned residual",
        "increases full-state error in all eight",
        "predictive intervals also under-cover",
        "No stability certificate follows",
    )
    for phrase in required:
        assert phrase in normalized


def test_ijcas_figures_follow_manuscript_order():
    document, _, _ = ijcas.render()
    figures = re.findall(
        r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}", document
    )
    assert figures == [f"figures/Fig{index}.pdf" for index in range(1, 6)]


def test_ijcas_bibliography_deduplicates_doi_resolver_urls_only():
    bibliography = ijcas.bibliography_text()
    assert "doi = {10.1073/pnas.2311808121}" in bibliography
    assert "url = {https://doi.org/10.1073/pnas.2311808121}" not in bibliography
    assert "url = {https://arxiv.org/abs/2310.09615}" in bibliography


def test_ijcas_source_manifest_records_completed_metadata_and_repository_gate():
    ijcas.validate_source_manifest()
    manifest = json.loads(ijcas.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["reference_style"] == "Springer Basic author-year"
    assert manifest["abstract_words"] == 168
    assert manifest["author_metadata_pending"] is False
    assert manifest["declarations_pending"] is False
    assert manifest["repository_identifier_pending"] is True


def test_ijcas_render_includes_confirmed_author_metadata():
    document, _, _ = ijcas.render()
    assert r"\author*[1]{\fnm{Yongzhi} \sur{Li}}" in document
    assert "1337958@wku.edu.cn" in document
    assert "0009-0008-8642-7385" in document
    assert "Computer Science and Technology" in document
    assert "Wenzhou-Kean University" in document
    assert "Anonymous authors" not in document
    assert "The author verified the sources" in document
    assert "The authors verified the sources" not in document


def test_ijcas_render_includes_confirmed_declarations_without_placeholders():
    document, _, _ = ijcas.render()
    required = (
        r"\section*{Statements and Declarations}",
        "no funds, grants, or other support were received",
        "The author declares no competing interests.",
        "Yongzhi Li made the principal contribution",
        "The author has no acknowledgements to declare.",
        "involved neither human participants nor animals",
        r"\subsection*{Consent to participate}",
        r"\subsection*{Consent for publication}",
    )
    for phrase in required:
        assert phrase in document
    assert "Author confirmation required" not in document
    assert document.index(r"\section*{Statements and Declarations}") < document.index(
        r"\begin{appendices}"
    )
    assert document.count(r"\subsection*{Data and code availability}") == 1


def test_ijcas_source_package_excludes_separate_submission_documents():
    assert ijcas.PACKAGE.is_file()
    with zipfile.ZipFile(ijcas.PACKAGE) as archive:
        names = archive.namelist()
    assert "main.tex" in names
    assert "main.pdf" in names
    assert "title_page.tex" not in names
    assert "cover_letter.tex" not in names
    assert "author_information_required.md" not in names
    assert names.count("figures/Fig4.pdf") == 1
    assert all(not name.startswith(("/", "..")) for name in names)


def test_ijcas_source_package_is_not_blocked_by_pending_repository_identifier():
    manifest = json.loads(ijcas.PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["submission_source_archive"] is True
    assert manifest["repository_identifier_pending"] is True
    assert manifest["repository_identifier_required_for_initial_submission"] is False
    assert manifest["submission_blocker"] is None


def test_ijcas_submission_bundle_has_upload_files_and_integrity_receipts():
    assert ijcas.SUBMISSION_PACKAGE.is_file()
    with zipfile.ZipFile(ijcas.SUBMISSION_PACKAGE) as archive:
        names = archive.namelist()
        assert archive.testzip() is None
    assert names[:4] == [
        "IJCAS_Manuscript.pdf",
        "IJCAS_Source_Files.zip",
        "IJCAS_Title_Page.pdf",
        "IJCAS_Cover_Letter.pdf",
    ]
    assert "SUBMISSION_README.md" in names
    assert "SHA256SUMS.txt" in names
    assert "submission_package_manifest.json" in names

    manifest = json.loads(ijcas.SUBMISSION_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["peer_review"] == "single-blind"
    assert manifest["technically_ready_for_portal_upload"] is True
    assert manifest["author_portal_confirmations_pending"] is True
    assert manifest["repository_identifier_required_for_initial_submission"] is False
    assert manifest["manuscript_pages"] > 0
    assert manifest["title_page_pages"] == 1
    assert manifest["cover_letter_pages"] == 1


def test_ijcas_cover_letter_preserves_mixed_results_without_unconfirmed_claims():
    text = ijcas.COVER_LETTER_SOURCE.read_text(encoding="utf-8")
    required = (
        "1,920-episode",
        "clean-regime noninferiority is not established",
        "slightly outperforms the complete model",
        "higher full-state error and undercoverage",
    )
    for phrase in required:
        assert phrase in " ".join(text.split())
    assert "not under consideration" not in text
    assert "not been published" not in text
