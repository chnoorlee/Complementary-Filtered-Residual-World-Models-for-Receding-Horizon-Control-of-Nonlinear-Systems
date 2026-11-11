"""Build, validate, and package the anonymous IEEE T-ASE edition."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import make_tase_closed_loop_assets as closed_loop_assets


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "tase"
BIBLIOGRAPHY_SOURCES = (
    ROOT / "paper" / "references.bib",
    ROOT / "paper" / "external_references.bib",
    ROOT / "paper" / "ijdc_additional_references.bib",
)
SOURCE_COPY = {
    "results_macros.tex": ROOT / "paper" / "results_macros.tex",
    "reviewer_additional_macros.tex": ROOT / "paper" / "reviewer_additional_macros.tex",
    "external_results_macros.tex": ROOT / "paper" / "external_results_macros.tex",
    "table_confirmatory.tex": ROOT / "results" / "confirmatory_analysis" / "table_confirmatory.tex",
    "table_reviewer_posthoc.tex": ROOT / "paper" / "table_reviewer_posthoc.tex",
    "table_external_results.tex": ROOT / "paper" / "table_external_results.tex",
    "table_external_physical.tex": ROOT / "paper" / "table_external_physical.tex",
    "table_external_uncertainty.tex": ROOT / "paper" / "table_external_uncertainty.tex",
    "table_cell_contrasts.tex": ROOT / "paper" / "table_cell_contrasts.tex",
}
FIGURE_COPY = {
    "Fig1.pdf": ROOT / "paper" / "ijdc" / "figures" / "Fig1.pdf",
    "Fig4.pdf": ROOT / "paper" / "ijdc" / "figures" / "Fig4.pdf",
}
PACKAGE = PAPER / "TASE_anonymous_source.zip"
BUILD_MANIFEST = PAPER / "build_manifest.json"
PACKAGE_MANIFEST = PAPER / "package_manifest.json"
MAX_MAIN_PAGES = 12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path = PAPER) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{output[-8000:]}")
    return output


def locate_tex_file(name: str) -> Path:
    result = subprocess.run(
        ["kpsewhich", name], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).stdout.strip()
    path = Path(result)
    if not path.is_file():
        raise FileNotFoundError(f"TeX dependency not found: {name}")
    return path


def prepare_sources() -> None:
    closed_loop_assets.main()
    for destination, source in SOURCE_COPY.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, PAPER / destination)
    figures = PAPER / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for destination, source in FIGURE_COPY.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, figures / destination)
    bibliography = "\n\n".join(
        source.read_text(encoding="utf-8").strip()
        for source in BIBLIOGRAPHY_SOURCES
    ) + "\n"
    (PAPER / "references.bib").write_text(bibliography, encoding="utf-8")
    shutil.copy2(locate_tex_file("IEEEtran.cls"), PAPER / "IEEEtran.cls")
    shutil.copy2(locate_tex_file("IEEEtran.bst"), PAPER / "IEEEtran.bst")


def tex_words(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\cite\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}$~]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def compile_document(stem: str) -> None:
    for suffix in ("aux", "bbl", "blg", "log", "out", "pdf"):
        path = PAPER / f"{stem}.{suffix}"
        if path.exists():
            path.unlink()
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"])
    auxiliary = (PAPER / f"{stem}.aux").read_text(encoding="utf-8", errors="replace")
    if "\\citation" in auxiliary:
        run(["bibtex", stem])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"])


def page_count(path: Path) -> int:
    output = run(["pdfinfo", str(path)], cwd=ROOT)
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"cannot read page count from {path}")
    return int(match.group(1))


def pdf_metadata(path: Path) -> dict[str, str]:
    output = run(["pdfinfo", str(path)], cwd=ROOT)
    metadata = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def validate_log(stem: str) -> None:
    text = (PAPER / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "LaTeX Warning: There were undefined references",
        "LaTeX Warning: Citation",
        "multiply defined",
        "! LaTeX Error",
        "Emergency stop",
    )
    present = [item for item in forbidden if item in text]
    if present:
        raise RuntimeError(f"{stem}.log failed validation: {present}")
    overfull = re.findall(r"Overfull \\hbox \(([^)]*)\)", text)
    if overfull:
        raise RuntimeError(f"{stem}.log contains overfull boxes: {overfull[:8]}")


def validate_anonymity() -> None:
    text_files = [
        PAPER / "main.tex", PAPER / "supplement.tex",
        *sorted((PAPER / "sections").glob("*.tex")),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
    forbidden = ("Chnoor", "Li Yongzhi", "Yongzhi Li", "University of", "@gmail", "@outlook")
    hits = [term for term in forbidden if term.lower() in combined.lower()]
    if hits:
        raise RuntimeError(f"anonymous source contains identifying terms: {hits}")
    for pdf in (PAPER / "main.pdf", PAPER / "supplement.pdf"):
        metadata = pdf_metadata(pdf)
        if metadata.get("Author", "").strip():
            raise RuntimeError(f"PDF author metadata is not blank: {pdf.name}")


def package_sources() -> dict[str, str]:
    figure_files = [PAPER / "figures" / name for name in (*FIGURE_COPY, "FigClosedLoop.pdf")]
    table_files = [PAPER / name for name in (
        "table_confirmatory.tex", "table_reviewer_posthoc.tex",
        "table_closed_loop.tex", "table_external_results.tex",
        "table_external_physical.tex", "table_external_uncertainty.tex",
        "table_cell_contrasts.tex",
    )]
    macro_files = [PAPER / name for name in (
        "results_macros.tex", "reviewer_additional_macros.tex",
        "external_results_macros.tex", "closed_loop_macros.tex",
    )]
    include = [
        PAPER / "main.tex", PAPER / "supplement.tex", PAPER / "references.bib",
        PAPER / "IEEEtran.cls", PAPER / "IEEEtran.bst",
        *sorted((PAPER / "sections").glob("*.tex")),
        *figure_files, *macro_files, *table_files,
    ]
    unique = sorted(set(include), key=lambda path: path.relative_to(PAPER).as_posix())
    hashes = {path.relative_to(PAPER).as_posix(): sha256(path) for path in unique}
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique:
            archive.write(path, path.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(PACKAGE) as archive:
        if set(archive.namelist()) != set(hashes):
            raise RuntimeError("source package inventory differs from the manifest")
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise RuntimeError(f"source package hash mismatch: {name}")
    return hashes


def main() -> None:
    prepare_sources()
    abstract_words = tex_words(PAPER / "sections" / "abstract.tex")
    practitioner_words = tex_words(PAPER / "sections" / "note_to_practitioners.tex")
    if not 1 <= abstract_words <= 200:
        raise RuntimeError(f"abstract has {abstract_words} words; T-ASE maximum is 200")
    if not 100 <= practitioner_words <= 300:
        raise RuntimeError(f"Note to Practitioners has {practitioner_words} words; required range is 100-300")
    compile_document("main")
    compile_document("supplement")
    validate_log("main")
    validate_log("supplement")
    validate_anonymity()
    main_pages = page_count(PAPER / "main.pdf")
    supplement_pages = page_count(PAPER / "supplement.pdf")
    if main_pages > MAX_MAIN_PAGES:
        raise RuntimeError(f"main paper is {main_pages} pages; target is at most {MAX_MAIN_PAGES}")
    hashes = package_sources()
    package_payload = {
        "complete": True,
        "archive": PACKAGE.name,
        "archive_sha256": sha256(PACKAGE),
        "files": hashes,
    }
    PACKAGE_MANIFEST.write_text(json.dumps(package_payload, indent=2) + "\n", encoding="utf-8")
    build_payload = {
        "complete": True,
        "venue": "IEEE Transactions on Automation Science and Engineering",
        "anonymous": True,
        "abstract_words": abstract_words,
        "note_to_practitioners_words": practitioner_words,
        "main_pages": main_pages,
        "supplement_pages": supplement_pages,
        "main_pdf_sha256": sha256(PAPER / "main.pdf"),
        "supplement_pdf_sha256": sha256(PAPER / "supplement.pdf"),
        "closed_loop_assets_manifest_sha256": sha256(PAPER / "closed_loop_assets_manifest.json"),
        "package_manifest_sha256": sha256(PACKAGE_MANIFEST),
    }
    BUILD_MANIFEST.write_text(json.dumps(build_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(build_payload, indent=2))


if __name__ == "__main__":
    main()
