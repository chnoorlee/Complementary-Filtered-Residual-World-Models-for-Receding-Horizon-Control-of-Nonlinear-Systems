"""Build and package the IJCAS submission edition without changing frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import build_ijdc_submission as ijdc


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "paper" / "ijcas"
SECTIONS = DIRECTORY / "sections"
BASE_SECTIONS = ROOT / "paper" / "ijdc" / "sections"
TEMPLATE = ROOT / "paper" / "springer-official" / "sn-article-template"
BIBLIOGRAPHY_SOURCES = ijdc.BIBLIOGRAPHY_SOURCES
RESULT_MACROS = ROOT / "paper" / "results_macros.tex"
EXTERNAL_MACROS = ROOT / "paper" / "external_results_macros.tex"
REVIEWER_MACROS = ROOT / "paper" / "reviewer_additional_macros.tex"
CLOSED_LOOP_MACROS = ROOT / "paper" / "tase" / "closed_loop_macros.tex"
CLOSED_LOOP_TABLE = ROOT / "paper" / "tase" / "table_closed_loop.tex"
CLOSED_LOOP_ASSET_MANIFEST = ROOT / "paper" / "tase" / "closed_loop_assets_manifest.json"
CLOSED_LOOP_RESULTS_MANIFEST = ROOT / "results" / "tase_closed_loop" / "manifest.json"
MANIFEST = DIRECTORY / "source_manifest.json"
BUILD_MANIFEST = DIRECTORY / "build_manifest.json"
PACKAGE_MANIFEST = DIRECTORY / "package_manifest.json"
SUBMISSION_MANIFEST = DIRECTORY / "submission_package_manifest.json"
PACKAGE = DIRECTORY / "IJCAS_Source_Files.zip"
SUBMISSION_DIRECTORY = DIRECTORY / "submission_package_20261110"
SUBMISSION_PACKAGE = DIRECTORY / "IJCAS_submission_package_20261110.zip"
TITLE_PAGE_SOURCE = DIRECTORY / "title_page.tex"
TITLE_PAGE_PDF = DIRECTORY / "title_page.pdf"
COVER_LETTER_SOURCE = DIRECTORY / "cover_letter.tex"
COVER_LETTER_PDF = DIRECTORY / "cover_letter.pdf"
SUBMISSION_README = DIRECTORY / "SUBMISSION_README.md"
TITLE = (
    "Complementary-Filtered Residual World Models for Receding-Horizon "
    "Control of Nonlinear Systems"
)
AUTHOR_GIVEN = "Yongzhi"
AUTHOR_FAMILY = "Li"
AUTHOR_EMAIL = "1337958@wku.edu.cn"
AUTHOR_ORCID = "0009-0008-8642-7385"
AUTHOR_DIVISION = "Computer Science and Technology"
AUTHOR_INSTITUTION = "Wenzhou-Kean University"
AUTHOR_CITY = "Wenzhou"
AUTHOR_STATE = "Zhejiang"
AUTHOR_COUNTRY = "China"
KEYWORDS = (
    "complementary filtering",
    "model predictive control",
    "nonlinear systems",
    "physics-guided learning",
    "state estimation",
    "world models",
)
SECTION_SOURCES = (
    ("introduction", BASE_SECTIONS / "introduction.tex"),
    ("related_work", BASE_SECTIONS / "related_work.tex"),
    ("method", BASE_SECTIONS / "method.tex"),
    ("experiments", BASE_SECTIONS / "experiments.tex"),
    ("results", BASE_SECTIONS / "results.tex"),
    ("closed_loop", SECTIONS / "closed_loop.tex"),
    ("external_validation", BASE_SECTIONS / "external_validation.tex"),
    ("discussion", SECTIONS / "discussion.tex"),
    ("limitations", SECTIONS / "limitations.tex"),
    ("conclusion", SECTIONS / "conclusion.tex"),
)
FIGURE_SOURCES = {
    "figures/Fig1.pdf": ROOT / "paper" / "ijdc" / "figures" / "Fig1.pdf",
    "figures/Fig2.pdf": ROOT / "paper" / "ijdc" / "figures" / "Fig2.pdf",
    "figures/Fig3.pdf": ROOT / "paper" / "ijdc" / "figures" / "Fig3.pdf",
    "figures/Fig4.pdf": ROOT / "paper" / "tase" / "figures" / "FigClosedLoop.pdf",
    "figures/Fig5.pdf": ROOT / "paper" / "ijdc" / "figures" / "Fig4.pdf",
}
FIGURE_REPLACEMENTS = {
    "figures/figure1_method.png": "figures/Fig1.pdf",
    "figures/confirmatory_overview.png": "figures/Fig2.pdf",
    "figures/cell_difference_heatmap.png": "figures/Fig3.pdf",
    "figures/external_flight.png": "figures/Fig5.pdf",
    "figures/FigClosedLoop.pdf": "figures/Fig4.pdf",
}
COPY_SOURCES = {
    "sn-jnl.cls": TEMPLATE / "sn-jnl.cls",
    "sn-basic.bst": TEMPLATE / "bst" / "sn-basic.bst",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"input outside project: {resolved}") from error


def bibliography_text() -> str:
    text = "\n\n".join(
        path.read_text(encoding="utf-8").strip() for path in BIBLIOGRAPHY_SOURCES
    ) + "\n"
    return re.sub(
        r"(?im)^(\s*doi\s*=\s*\{([^}]+)\},?)\s*\n"
        r"\s*url\s*=\s*\{https?://(?:dx\.)?doi\.org/\2\},?\s*$",
        r"\1",
        text,
    )


def validate_closed_loop_evidence() -> None:
    for path in (CLOSED_LOOP_ASSET_MANIFEST, CLOSED_LOOP_RESULTS_MANIFEST):
        if not path.is_file():
            raise RuntimeError(f"closed-loop manifest missing: {path}")
    assets = json.loads(CLOSED_LOOP_ASSET_MANIFEST.read_text(encoding="utf-8"))
    results = json.loads(CLOSED_LOOP_RESULTS_MANIFEST.read_text(encoding="utf-8"))
    if assets.get("complete") is not True or results.get("complete") is not True:
        raise RuntimeError("closed-loop evidence is incomplete")
    if assets.get("closed_loop_manifest_sha256") != sha256(CLOSED_LOOP_RESULTS_MANIFEST):
        raise RuntimeError("closed-loop results manifest drift")
    if results.get("row_count") != 1920:
        raise RuntimeError("closed-loop episode inventory is not the frozen 1,920-row study")
    for relative, expected in assets.get("inputs", {}).items():
        path = CLOSED_LOOP_RESULTS_MANIFEST.parent / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"closed-loop input drift: {relative}")
    for relative, expected in assets.get("outputs", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"closed-loop output drift: {relative}")


def scientific_provenance() -> None:
    ijdc.scientific_provenance()
    validate_closed_loop_evidence()


def adapt_layout(text: str) -> str:
    for old, new in FIGURE_REPLACEMENTS.items():
        text = text.replace("{" + old + "}", "{" + new + "}")
    return text


def adapt_venue_content(name: str, text: str) -> str:
    """Apply venue-specific framing while preserving the reported evidence."""
    text = text.replace("recent IJDC work demonstrates", "recent work demonstrates")
    text = re.sub(
        r"The authors verified the sources,\s+equations, code, results, and scientific "
        r"claims, approved the final manuscript,\s+and take responsibility for its content\.",
        "The author verified the sources, equations, code, results, and scientific "
        "claims, approved the final manuscript, and takes responsibility for its content.",
        text,
    )
    if name != "introduction":
        return text
    text = text.replace(
        "The evaluation has two layers. First, five controlled nonlinear systems:",
        "The evaluation has three layers. First, five controlled nonlinear systems:",
    )
    external_marker = (
        "Second, an external experiment uses measured flight states from the NanoDrone"
    )
    closed_loop_paragraph = (
        "Second, a frozen receding-horizon regulation study holds the controller and "
        "prediction interface fixed across 1,920 paired episodes. It reuses all five "
        "systems, three plant/observation conditions, and eight seeds to test whether "
        "open-loop differences persist under a common cross-entropy model predictive "
        "controller.\n\n"
        "Third, an external experiment uses measured flight states from the NanoDrone"
    )
    if text.count(external_marker) != 1:
        raise RuntimeError("IJCAS introduction external-evaluation marker drift")
    text = text.replace(external_marker, closed_loop_paragraph)
    contribution = (
        " \\item secondary classical-filter, residual-scale, structured-outage,\n"
        " projection, latency, and calibration audits, together with an external\n"
        " measured-flight evaluation."
    )
    revised_contribution = (
        " \\item a frozen 1,920-episode receding-horizon regulation study using a\n"
        " common cross-entropy model predictive controller; and\n"
        " \\item secondary classical-filter, residual-scale, structured-outage,\n"
        " projection, latency, and calibration audits, together with an external\n"
        " measured-flight evaluation."
    )
    if text.count(contribution) != 1:
        raise RuntimeError("IJCAS introduction contribution marker drift")
    text = text.replace(contribution, revised_contribution)
    source_summary = (
        "The controlled benchmark supports the aggregate primary hypothesis against the\n"
        "preselected PGNN-style comparator but not the gated clean-regime noninferiority\n"
        "requirement. Classical component analysis and residual-scale sensitivity show\n"
        "that the gain-0.5 filter followed by nominal rollout performs better than the\n"
        "learned residual transition. The flight\n"
        "evaluation shows that lower long-horizon position error can coexist with higher\n"
        "full-state error and undercovered intervals."
    )
    revised_summary = (
        "The controlled benchmark supports the aggregate primary hypothesis against the\n"
        "preselected PGNN-style comparator but not the gated clean-regime noninferiority\n"
        "requirement. The closed-loop study shows a small state-loss improvement over\n"
        "PGNN-style, while classical component analysis shows that gain-0.5 filtering\n"
        "followed by nominal rollout performs slightly better than the complete model.\n"
        "The flight evaluation shows that lower long-horizon position error can coexist\n"
        "with higher full-state error and undercovered intervals."
    )
    if text.count(source_summary) != 1:
        raise RuntimeError("IJCAS introduction result-summary marker drift")
    text = text.replace(source_summary, revised_summary)
    return text.replace(
        "controlled and flight experiments; and discusses implications",
        "controlled open- and closed-loop and measured-flight experiments; and discusses implications",
    )


def expand_section(path: Path) -> tuple[str, list[Path]]:
    text = path.read_text(encoding="utf-8")
    if path == SECTIONS / "closed_loop.tex":
        marker = r"\input{table_closed_loop}"
        if text.count(marker) != 1:
            raise RuntimeError("closed-loop section must contain one table input")
        text = text.replace(marker, CLOSED_LOOP_TABLE.read_text(encoding="utf-8").strip())
        return text, [CLOSED_LOOP_TABLE]
    return ijdc.expand_inputs(text)


def abstract_words(text: str) -> int:
    clean = re.sub(r"%.*", " ", text)
    clean = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", clean)
    clean = re.sub(r"[{}$~]", " ", clean)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", clean))


def declarations_text(data_availability: str) -> str:
    return rf"""\section*{{Statements and Declarations}}

\subsection*{{Funding}}
The author declares that no funds, grants, or other support were received
during the preparation of this manuscript.

\subsection*{{Competing interests}}
The author declares no competing interests.

\subsection*{{Ethics approval}}
Not applicable. This study used numerical simulations and publicly released
non-human flight records and involved neither human participants nor animals.

\subsection*{{Consent to participate}}
Not applicable.

\subsection*{{Consent for publication}}
Not applicable.

\subsection*{{Data and code availability}}
{data_availability.strip()}

\subsection*{{Author contributions}}
Yongzhi Li made the principal contribution to this work and read and approved
the final manuscript.

\subsection*{{Acknowledgements}}
The author has no acknowledgements to declare."""


def output_sources() -> list[Path]:
    names = ("main.tex", "references.bib", *COPY_SOURCES, *FIGURE_SOURCES)
    return [DIRECTORY / name for name in dict.fromkeys(names)]


def render() -> tuple[str, list[Path], int]:
    scientific_provenance()
    abstract_path = SECTIONS / "abstract.tex"
    appendix_path = BASE_SECTIONS / "appendix.tex"
    data_path = BASE_SECTIONS / "data_availability.tex"
    required = [
        abstract_path,
        appendix_path,
        data_path,
        *(path for _, path in SECTION_SOURCES),
        *COPY_SOURCES.values(),
        *BIBLIOGRAPHY_SOURCES,
        RESULT_MACROS,
        EXTERNAL_MACROS,
        REVIEWER_MACROS,
        CLOSED_LOOP_MACROS,
        CLOSED_LOOP_TABLE,
        CLOSED_LOOP_ASSET_MANIFEST,
        CLOSED_LOOP_RESULTS_MANIFEST,
        *FIGURE_SOURCES.values(),
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required IJCAS input missing: {path}")

    abstract_source = abstract_path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(
        r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}",
        abstract_source,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("abstract must contain exactly one abstract environment")
    word_count = abstract_words(match.group(1))
    if not 150 <= word_count <= 250:
        raise RuntimeError(f"IJCAS abstract has {word_count} words; required range is 150-250")

    dependencies = list(required)
    documents: dict[str, str] = {}
    for name, path in SECTION_SOURCES:
        expanded, nested = expand_section(path)
        documents[name] = adapt_layout(adapt_venue_content(name, expanded)).strip()
        dependencies.extend(nested)
    appendix, nested = ijdc.expand_inputs(appendix_path.read_text(encoding="utf-8"))
    appendix = adapt_layout(re.sub(r"^\\appendix\s*", "", appendix, count=1)).strip()
    dependencies.extend(nested)
    data_availability, nested = ijdc.expand_inputs(data_path.read_text(encoding="utf-8"))
    dependencies.extend(nested)
    declarations = declarations_text(data_availability)
    macros = "\n".join(
        path.read_text(encoding="utf-8").strip()
        for path in (RESULT_MACROS, EXTERNAL_MACROS, REVIEWER_MACROS, CLOSED_LOOP_MACROS)
    )
    body = "\n\n".join(documents[name] for name, _ in SECTION_SOURCES)
    document = rf"""% IJCAS submission edition generated from hash-validated evidence.
% Author identity, affiliation, and declarations supplied or confirmed by the author.
\documentclass[pdflatex,sn-basic]{{sn-jnl}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{booktabs,siunitx,xcolor}}
\usepackage[title]{{appendix}}
\hypersetup{{pdftitle={{{TITLE}}},pdfauthor={{{AUTHOR_GIVEN} {AUTHOR_FAMILY}}},pdfsubject={{}},pdfkeywords={{}}}}
\newcommand{{\method}}{{CFR-WM}}
\newcommand{{\vect}}[1]{{\mathbf{{#1}}}}
{macros}
\begin{{document}}
\title[Complementary-Filtered Residual World Models]{{{TITLE}}}
\author*[1]{{\fnm{{{AUTHOR_GIVEN}}} \sur{{{AUTHOR_FAMILY}}}}}
\email{{\href{{mailto:{AUTHOR_EMAIL}}}{{{AUTHOR_EMAIL}}};
\href{{https://orcid.org/{AUTHOR_ORCID}}}{{ORCID: {AUTHOR_ORCID}}}}}
\affil*[1]{{\orgdiv{{{AUTHOR_DIVISION}}},
\orgname{{{AUTHOR_INSTITUTION}}},
\orgaddress{{\city{{{AUTHOR_CITY}}}, \state{{{AUTHOR_STATE}}},
\country{{{AUTHOR_COUNTRY}}}}}}}
\abstract{{{match.group(1).strip()}}}
\keywords{{{', '.join(KEYWORDS)}}}
\maketitle

{body}

\backmatter
{declarations}

\begin{{appendices}}
{appendix}
\end{{appendices}}

\bibliography{{references}}
\end{{document}}
"""
    document = re.sub(r"\\cite\{", r"\\citep{", document)
    if re.search(r"\\(?:input|include)\b", document):
        raise RuntimeError("portable source still contains input/include dependencies")
    graphics = re.findall(
        r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}", document
    )
    if set(graphics) != set(FIGURE_SOURCES):
        raise RuntimeError(f"unexpected IJCAS figure inventory: {graphics}")
    if "Anonymous authors" in document or re.search(
        r"Author Placeholder|author@example\.com", document
    ):
        raise RuntimeError("author placeholders remain in the IJCAS manuscript")
    required_author_values = (
        AUTHOR_GIVEN,
        AUTHOR_FAMILY,
        AUTHOR_EMAIL,
        AUTHOR_ORCID,
        AUTHOR_DIVISION,
        AUTHOR_INSTITUTION,
        AUTHOR_CITY,
        AUTHOR_STATE,
        AUTHOR_COUNTRY,
    )
    if any(value not in document for value in required_author_values):
        raise RuntimeError("confirmed IJCAS author metadata is incomplete")
    bibliography_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography_text())
    cited_keys = {
        item.strip()
        for group in re.findall(r"\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}", document)
        for item in group.split(",")
    }
    if len(bibliography_keys) != len(set(bibliography_keys)):
        raise RuntimeError("duplicate IJCAS bibliography keys")
    missing = cited_keys - set(bibliography_keys)
    if missing:
        raise RuntimeError(f"IJCAS bibliography lacks cited keys: {sorted(missing)}")
    dependencies.extend((Path(__file__), Path(ijdc.__file__)))
    return document, list(dict.fromkeys(dependencies)), word_count


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {project_key(path): sha256(path) for path in paths}


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build() -> None:
    document, inputs, word_count = render()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY / "figures").mkdir(exist_ok=True)
    (DIRECTORY / "main.tex").write_text(document, encoding="utf-8")
    (DIRECTORY / "references.bib").write_text(bibliography_text(), encoding="utf-8")
    for name, source in COPY_SOURCES.items():
        shutil.copyfile(source, DIRECTORY / name)
    for name, source in FIGURE_SOURCES.items():
        shutil.copyfile(source, DIRECTORY / name)
    write_json(
        MANIFEST,
        {
            "schema_version": 1,
            "complete": True,
            "venue": "International Journal of Control, Automation and Systems",
            "reference_style": "Springer Basic author-year",
            "abstract_words": word_count,
            "author_metadata_pending": False,
            "declarations_pending": False,
            "repository_identifier_pending": True,
            "inputs": hash_files(inputs),
            "outputs": hash_files(output_sources()),
        },
    )
    validate_source_manifest()


def validate_source_manifest() -> None:
    document, inputs, word_count = render()
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "complete": True,
        "venue": "International Journal of Control, Automation and Systems",
        "reference_style": "Springer Basic author-year",
        "abstract_words": word_count,
        "author_metadata_pending": False,
        "declarations_pending": False,
        "repository_identifier_pending": True,
        "inputs": hash_files(inputs),
        "outputs": hash_files(output_sources()),
    }
    if recorded != expected or (DIRECTORY / "main.tex").read_text(encoding="utf-8") != document:
        raise RuntimeError("IJCAS source manifest mismatch; rebuild before compiling")


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=DIRECTORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} failed:\n{result.stdout[-6000:]}\n{result.stderr[-2000:]}"
        )
    return result.stdout + result.stderr


def compile_document() -> None:
    validate_source_manifest()
    for suffix in ("aux", "bbl", "blg", "log", "out", "pdf"):
        path = DIRECTORY / f"main.{suffix}"
        if path.exists():
            path.unlink()
    for command in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ):
        run(command)
    log = (DIRECTORY / "main.log").read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "LaTeX Warning: There were undefined references",
        "LaTeX Warning: Citation",
        "multiply defined",
        "! LaTeX Error",
        "Emergency stop",
        "Overfull \\hbox",
    )
    present = [item for item in forbidden if item in log]
    if present:
        raise RuntimeError(f"IJCAS build log failed validation: {present}")
    info = run(["pdfinfo", "main.pdf"])
    page_match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    if not page_match:
        raise RuntimeError("cannot determine IJCAS PDF page count")
    write_json(
        BUILD_MANIFEST,
        {
            "schema_version": 1,
            "complete": True,
            "source_manifest_sha256": sha256(MANIFEST),
            "pages": int(page_match.group(1)),
            "author_metadata_pending": False,
            "declarations_pending": False,
            "repository_identifier_pending": True,
            "outputs": hash_files(
                [DIRECTORY / f"main.{ext}" for ext in ("pdf", "bbl", "log")]
            ),
        },
    )


def package_sources() -> Path:
    validate_source_manifest()
    receipt = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    compiled = [DIRECTORY / f"main.{ext}" for ext in ("pdf", "bbl", "log")]
    if (
        receipt.get("complete") is not True
        or receipt.get("source_manifest_sha256") != sha256(MANIFEST)
        or receipt.get("outputs") != hash_files(compiled)
    ):
        raise RuntimeError("IJCAS compiled output is stale; compile before packaging")
    paths = [*output_sources(), DIRECTORY / "main.bbl", DIRECTORY / "main.pdf"]
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(DIRECTORY).as_posix())
    payload = {
        "complete": True,
        "submission_source_archive": True,
        "repository_identifier_pending": True,
        "repository_identifier_required_for_initial_submission": False,
        "submission_blocker": None,
        "archive_sha256": sha256(PACKAGE),
        "files": {
            path.relative_to(DIRECTORY).as_posix(): sha256(path) for path in paths
        },
    }
    write_json(PACKAGE_MANIFEST, payload)
    return PACKAGE


def pdf_pages(path: Path) -> int:
    info = run(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    if not match:
        raise RuntimeError(f"cannot determine page count: {path}")
    return int(match.group(1))


def package_submission() -> Path:
    source_archive = package_sources()
    required = (
        DIRECTORY / "main.pdf",
        source_archive,
        TITLE_PAGE_SOURCE,
        TITLE_PAGE_PDF,
        COVER_LETTER_SOURCE,
        COVER_LETTER_PDF,
        SUBMISSION_README,
        DIRECTORY / "REPOSITORY_DOI_GUIDE.md",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"IJCAS submission artifacts missing: {missing}")

    for source in (TITLE_PAGE_SOURCE, COVER_LETTER_SOURCE, SUBMISSION_README):
        text = source.read_text(encoding="utf-8")
        if re.search(r"\b(?:TODO|TBD)\b|\[[^]]*(?:confirm|required|pending)[^]]*\]", text, re.I):
            raise RuntimeError(f"submission-facing placeholder remains: {source.name}")

    upload_files = {
        "IJCAS_Manuscript.pdf": DIRECTORY / "main.pdf",
        "IJCAS_Source_Files.zip": source_archive,
        "IJCAS_Title_Page.pdf": TITLE_PAGE_PDF,
        "IJCAS_Cover_Letter.pdf": COVER_LETTER_PDF,
    }
    support_files = {
        "IJCAS_Title_Page.tex": TITLE_PAGE_SOURCE,
        "IJCAS_Cover_Letter.tex": COVER_LETTER_SOURCE,
        "SUBMISSION_README.md": SUBMISSION_README,
        "REPOSITORY_DOI_GUIDE.md": DIRECTORY / "REPOSITORY_DOI_GUIDE.md",
    }
    SUBMISSION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, source in {**upload_files, **support_files}.items():
        shutil.copyfile(source, SUBMISSION_DIRECTORY / name)

    checksums = {
        name: sha256(SUBMISSION_DIRECTORY / name)
        for name in (*upload_files, *support_files)
    }
    checksum_path = SUBMISSION_DIRECTORY / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "venue": "International Journal of Control, Automation, and Systems",
        "peer_review": "single-blind",
        "technically_ready_for_portal_upload": True,
        "author_metadata_pending": False,
        "declarations_pending": False,
        "repository_identifier_pending": True,
        "repository_identifier_required_for_initial_submission": False,
        "author_portal_confirmations_pending": True,
        "manuscript_pages": pdf_pages(DIRECTORY / "main.pdf"),
        "title_page_pages": pdf_pages(TITLE_PAGE_PDF),
        "cover_letter_pages": pdf_pages(COVER_LETTER_PDF),
        "upload_files": {
            name: {"sha256": checksums[name], "bytes": (SUBMISSION_DIRECTORY / name).stat().st_size}
            for name in upload_files
        },
        "support_files": {
            name: {"sha256": checksums[name], "bytes": (SUBMISSION_DIRECTORY / name).stat().st_size}
            for name in support_files
        },
        "checksum_file_sha256": sha256(checksum_path),
    }
    write_json(SUBMISSION_MANIFEST, manifest)
    shutil.copyfile(SUBMISSION_MANIFEST, SUBMISSION_DIRECTORY / SUBMISSION_MANIFEST.name)

    members = [
        *upload_files,
        *support_files,
        "SHA256SUMS.txt",
        SUBMISSION_MANIFEST.name,
    ]
    with zipfile.ZipFile(SUBMISSION_PACKAGE, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(SUBMISSION_DIRECTORY / name, name)
    with zipfile.ZipFile(SUBMISSION_PACKAGE) as archive:
        if archive.testzip() is not None or archive.namelist() != members:
            raise RuntimeError("IJCAS submission bundle failed ZIP integrity validation")
    return SUBMISSION_PACKAGE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--submission-package", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate_source_manifest()
    else:
        build()
    if args.compile:
        compile_document()
    if args.package:
        print(package_sources())
    if args.submission_package:
        print(package_submission())
    print(MANIFEST)


if __name__ == "__main__":
    main()
