"""Build and package the anonymous IJDC edition without changing frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import build_springer_submission as frozen


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "paper" / "ijdc"
SECTIONS = DIRECTORY / "sections"
TEMPLATE = ROOT / "paper" / "springer-official" / "sn-article-template"
BIBLIOGRAPHY_SOURCES = (
    ROOT / "paper" / "references.bib",
    ROOT / "paper" / "external_references.bib",
    ROOT / "paper" / "ijdc_additional_references.bib",
)
CITATION_AUDIT = ROOT / "literature" / "ijdc_citation_audit.json"
REVIEWER_MANIFEST = ROOT / "results" / "reviewer_additional_analysis" / "manifest.json"
REVIEWER_MACROS = ROOT / "paper" / "reviewer_additional_macros.tex"
MANIFEST = DIRECTORY / "source_manifest.json"
BUILD_MANIFEST = DIRECTORY / "build_manifest.json"
TITLE = (
    "Complementary-Filtered Residual World Models for Multistep Prediction "
    "of Nonlinear Controlled Dynamics"
)
KEYWORDS = ("nonlinear dynamics", "state estimation", "complementary filtering",
            "physics-guided learning", "multistep prediction", "model predictive control")
BODY_SECTIONS = frozen.BODY_SECTIONS
FIGURE_MAP = {
    "figures/figure1_method.png": "figures/Fig1.pdf",
    "figures/confirmatory_overview.png": "figures/Fig2.pdf",
    "figures/cell_difference_heatmap.png": "figures/Fig3.pdf",
    "figures/external_flight.png": "figures/Fig4.pdf",
}
COPY_SOURCES = {
    "sn-jnl.cls": TEMPLATE / "sn-jnl.cls",
    "sn-basic.bst": TEMPLATE / "bst" / "sn-basic.bst",
}


def bibliography_text() -> str:
    return "\n\n".join(path.read_text(encoding="utf-8").strip()
                         for path in BIBLIOGRAPHY_SOURCES) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"input outside project: {resolved}") from error


def validate_reviewer_analysis() -> None:
    if not REVIEWER_MANIFEST.is_file():
        raise RuntimeError(f"reviewer-analysis manifest missing: {REVIEWER_MANIFEST}")
    recorded = json.loads(REVIEWER_MANIFEST.read_text(encoding="utf-8"))
    if (recorded.get("schema_version") != 1
            or recorded.get("complete") is not True
            or recorded.get("analysis_scope")
            != "post_review_secondary_no_confirmatory_decision_change"):
        raise RuntimeError("reviewer-analysis manifest is incomplete or has the wrong scope")
    for group in ("inputs", "outputs"):
        inventory = recorded.get(group)
        if not isinstance(inventory, dict) or not inventory:
            raise RuntimeError(f"reviewer-analysis {group} inventory is empty")
        for relative, expected_hash in inventory.items():
            path = (ROOT / relative).resolve()
            try:
                path.relative_to(ROOT.resolve())
            except ValueError as error:
                raise RuntimeError(f"reviewer-analysis path outside project: {path}") from error
            if not path.is_file() or sha256(path) != expected_hash:
                raise RuntimeError(f"reviewer-analysis {group} drift: {relative}")


def scientific_provenance() -> None:
    frozen._validate_bibliography_provenance()
    frozen._validate_asset_manifest(frozen._required_result_assets())
    frozen._validate_external_citation()
    frozen._validate_external_assets()
    validate_reviewer_analysis()


def section_sources() -> list[Path]:
    return [SECTIONS / f"{name}.tex"
            for name in ("abstract", *BODY_SECTIONS, "appendix")]


def expand_inputs(text: str, seen: tuple[Path, ...] = ()) -> tuple[str, list[Path]]:
    """Preserve the original main-directory resolution of frozen table inputs."""
    dependencies: list[Path] = []

    def replace(match: re.Match[str]) -> str:
        path = frozen.MAIN_DIR / match.group(1)
        if not path.suffix:
            path = path.with_suffix(".tex")
        path = path.resolve()
        key(path)
        if path in seen:
            raise RuntimeError(f"cyclic LaTeX input: {path}")
        if not path.is_file():
            raise RuntimeError(f"missing LaTeX input: {path}")
        nested, children = expand_inputs(path.read_text(encoding="utf-8"), (*seen, path))
        dependencies.extend((path, *children))
        return nested.strip()

    expanded = re.sub(r"\\input\s*\{([^}]+)\}", replace, text)
    if re.search(r"\\(?:input|include)\b", expanded):
        raise RuntimeError("unresolved input/include dependency")
    return expanded, dependencies


def adapt_layout(text: str) -> str:
    for old, new in FIGURE_MAP.items():
        text = text.replace("{" + old + "}", "{" + new + "}")
    return text


def output_sources() -> list[Path]:
    names = ("main.tex", "references.bib", *COPY_SOURCES, *FIGURE_MAP.values())
    return [DIRECTORY / name for name in dict.fromkeys(names)]


def render() -> tuple[str, list[Path]]:
    scientific_provenance()
    sources = section_sources()
    required = [*sources, *COPY_SOURCES.values(), *BIBLIOGRAPHY_SOURCES, CITATION_AUDIT,
                REVIEWER_MANIFEST, REVIEWER_MACROS,
                *(DIRECTORY / path for path in FIGURE_MAP.values())]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required IJDC input missing: {path}")
    documents: dict[str, str] = {}
    dependencies = list(required)
    for path in sources:
        expanded, nested = expand_inputs(path.read_text(encoding="utf-8"))
        documents[path.stem] = adapt_layout(expanded).strip()
        dependencies.extend(nested)
    match = re.fullmatch(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}",
                         documents["abstract"], re.DOTALL)
    if not match:
        raise RuntimeError("abstract must contain exactly one abstract environment")
    macros = []
    for path in (frozen.RESULT_MACROS, frozen.EXTERNAL_MACROS, REVIEWER_MACROS):
        macros.append(path.read_text(encoding="utf-8").strip())
        dependencies.append(path)
    body = "\n\n".join(documents[name] for name in BODY_SECTIONS)
    appendix = re.sub(r"^\\appendix\s*", "", documents["appendix"], count=1)
    document = rf"""% Anonymous IJDC edition, generated from the journal revision sources.
\documentclass[iicol,pdflatex,sn-basic,Numbered]{{sn-jnl}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{booktabs,siunitx,xcolor}}
\usepackage[title]{{appendix}}
\hypersetup{{pdftitle={{{TITLE}}},pdfauthor={{}},pdfsubject={{}},pdfkeywords={{}}}}
\newcommand{{\method}}{{CFR-WM}}
\newcommand{{\vect}}[1]{{\mathbf{{#1}}}}
{chr(10).join(macros)}
\begin{{document}}
\title[Complementary-Filtered Residual World Models]{{{TITLE}}}
\author{{Anonymous authors}}
\abstract{{{match.group(1).strip()}}}
\keywords{{{', '.join(KEYWORDS)}}}
\maketitle

{body}

\backmatter
\begin{{appendices}}
{appendix}
\end{{appendices}}
\bibliography{{references}}
\end{{document}}
"""
    if re.search(r"\\(?:input|include)\b", document):
        raise RuntimeError("portable source still contains input/include dependencies")
    graphics = re.findall(r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}", document)
    if set(graphics) != set(FIGURE_MAP.values()):
        raise RuntimeError(f"unexpected figure inventory: {graphics}")
    if re.search(r"Author Placeholder|author@example\.com|\\(?:affil|email)\b", document):
        raise RuntimeError("identifying author metadata or placeholders in anonymous manuscript")
    bibliography_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography_text())
    cited_keys = {item.strip() for group in re.findall(
        r"\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}", document)
        for item in group.split(",")}
    if len(bibliography_keys) != len(set(bibliography_keys)):
        raise RuntimeError("duplicate IJDC bibliography keys")
    missing_keys = cited_keys - set(bibliography_keys)
    if missing_keys:
        raise RuntimeError(f"IJDC bibliography lacks cited keys: {sorted(missing_keys)}")
    dependencies.extend((Path(__file__), Path(frozen.__file__),
                         frozen.ASSET_MANIFEST, frozen.BIBLIOGRAPHY_MANIFEST,
                         frozen.EXTERNAL_ASSET_MANIFEST, frozen.EXTERNAL_CITATION_AUDIT))
    return document, list(dict.fromkeys(dependencies))


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {key(path): sha256(path) for path in paths}


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build() -> None:
    document, inputs = render()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY / "main.tex").write_text(document, encoding="utf-8")
    (DIRECTORY / "references.bib").write_text(bibliography_text(), encoding="utf-8")
    for name, source in COPY_SOURCES.items():
        shutil.copyfile(source, DIRECTORY / name)
    write_json(MANIFEST, {"schema_version": 1, "complete": True,
                         "inputs": hash_files(inputs),
                         "outputs": hash_files(output_sources())})
    validate_source_manifest()


def validate_source_manifest() -> None:
    document, inputs = render()
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "complete": True,
                "inputs": hash_files(inputs), "outputs": hash_files(output_sources())}
    if recorded != expected or (DIRECTORY / "main.tex").read_text(encoding="utf-8") != document:
        raise RuntimeError("IJDC source manifest mismatch; rebuild before compiling")


def compile_document() -> None:
    validate_source_manifest()
    commands = (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    for command in commands:
        result = subprocess.run(command, cwd=DIRECTORY, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=180)
        if result.returncode:
            raise RuntimeError(f"{command[0]} failed:\n{result.stdout[-5000:]}\n{result.stderr}")
    log = (DIRECTORY / "main.log").read_text(encoding="utf-8", errors="replace")
    critical = re.findall(r"^.*(?:undefined|multiply defined|Overfull|^!).*$", log, re.MULTILINE)
    if critical:
        raise RuntimeError("IJDC build needs attention:\n" + "\n".join(critical))
    write_json(BUILD_MANIFEST, {"schema_version": 1, "complete": True,
                              "source_manifest_sha256": sha256(MANIFEST),
                              "outputs": hash_files([DIRECTORY / f"main.{ext}"
                                                     for ext in ("pdf", "bbl", "log")])})


def package_sources() -> Path:
    validate_source_manifest()
    receipt = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    expected = hash_files([DIRECTORY / f"main.{ext}" for ext in ("pdf", "bbl", "log")])
    if (receipt.get("complete") is not True
            or receipt.get("source_manifest_sha256") != sha256(MANIFEST)
            or receipt.get("outputs") != expected):
        raise RuntimeError("IJDC compiled output is stale; compile before packaging")
    paths = [*output_sources(), DIRECTORY / "main.bbl", DIRECTORY / "main.pdf"]
    destination = DIRECTORY / "IJDC_anonymous_source.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(DIRECTORY).as_posix())
    write_json(DIRECTORY / "package_manifest.json", {
        "complete": True, "archive_sha256": sha256(destination),
        "files": {path.relative_to(DIRECTORY).as_posix(): sha256(path) for path in paths},
    })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="compile the freshly built source")
    parser.add_argument("--package", action="store_true", help="package a verified compiled edition")
    parser.add_argument("--validate", action="store_true", help="validate without rewriting sources")
    args = parser.parse_args()
    if args.validate:
        validate_source_manifest()
    else:
        build()
    if args.compile:
        compile_document()
    if args.package:
        print(package_sources())
    print(MANIFEST)


if __name__ == "__main__":
    main()
