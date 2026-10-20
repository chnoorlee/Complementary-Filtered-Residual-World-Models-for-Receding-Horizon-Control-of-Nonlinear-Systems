"""Validate and hash the final Springer PDF, log, toolchain, and build inputs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import build_springer_submission as springer_submission


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "paper" / "mypaper"
MANIFEST = ROOT / "paper" / "springer_build_manifest.json"
EXPECTED_TITLE = (
    "A Confirmatory Study of Complementary-Filtered Residual World Models "
    "under Synthetic Context Corruption"
)
INPUTS = (
    MAIN_DIR / "main_springer.tex",
    ROOT / "paper" / "references.bib",
    ROOT / "paper" / "submission_source_manifest.json",
    ROOT / "paper" / "springer-official" / "sn-article-template" / "sn-jnl.cls",
    ROOT / "paper" / "springer-official" / "sn-article-template" / "bst" / "sn-mathphys-num.bst",
    ROOT / "code" / "build_springer_submission.py",
    ROOT / "code" / "README.md",
    Path(__file__).resolve(),
)
OUTPUTS = (
    MAIN_DIR / "main_springer.bbl",
    MAIN_DIR / "main_springer.log",
    MAIN_DIR / "main_springer.pdf",
)
CRITICAL_LOG_PATTERNS = (
    r"^! ",
    r"LaTeX Error",
    r"Undefined control sequence",
    r"There were undefined (?:references|citations)",
    r"(?:Citation|Reference) .+ undefined",
    r"destination with the same identifier",
    r"Overfull \\[hv]box",
)
VERSION_ARGUMENTS = {
    "pdflatex": ("--version",),
    "bibtex": ("--version",),
    "pdfinfo": ("-v",),
}


def _key(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _critical_log_findings(text: str) -> list[str]:
    findings = []
    for pattern in CRITICAL_LOG_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            findings.append(pattern)
    return findings


def _run_version(command: str) -> str:
    try:
        arguments = VERSION_ARGUMENTS[command]
    except KeyError as error:
        raise RuntimeError(f"no version arguments configured for {command}") from error
    result = subprocess.run(
        [command, *arguments], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    lines = (result.stdout + result.stderr).splitlines()
    if not lines:
        raise RuntimeError(f"{command} produced no version output")
    return lines[0].strip()


def _current_toolchain() -> dict[str, str]:
    return {
        command: _run_version(command)
        for command in ("pdflatex", "bibtex", "pdfinfo")
    }


def _pdf_metadata() -> dict[str, object]:
    result = subprocess.run(
        ["pdfinfo", str(MAIN_DIR / "main_springer.pdf")],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    fields = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            fields[name.strip()] = value.strip()
    title = fields.get("Title")
    try:
        pages = int(fields.get("Pages", "0"))
    except ValueError as error:
        raise RuntimeError("pdfinfo returned an invalid page count") from error
    if title != EXPECTED_TITLE or pages <= 0:
        raise RuntimeError(f"unexpected PDF metadata: title={title!r}, pages={pages}")
    return {"title": title, "pages": pages, "author": fields.get("Author", "")}


def _hash_group(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [_key(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Springer build artifacts are missing: {missing}")
    return {_key(path): _sha256(path) for path in paths}


def _single_output_with_suffix(suffix: str) -> Path:
    matches = [path for path in OUTPUTS if path.suffix == suffix]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {suffix} output, found {len(matches)}"
        )
    return matches[0]


def validate_build_manifest() -> None:
    springer_submission.validate_submission_manifest()
    if not MANIFEST.is_file():
        raise RuntimeError("Springer build manifest is missing")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("complete") is not True:
        raise RuntimeError("Springer build manifest is incomplete")
    for group, paths in (("inputs", INPUTS), ("outputs", OUTPUTS)):
        recorded = payload.get(group)
        if not isinstance(recorded, dict):
            raise RuntimeError(f"Springer build manifest lacks {group}")
        for path in paths:
            if not path.is_file() or recorded.get(_key(path)) != _sha256(path):
                raise RuntimeError(f"Springer build manifest mismatch: {_key(path)}")
    log_findings = _critical_log_findings(_single_output_with_suffix(".log").read_text(
        encoding="utf-8", errors="replace",
    ))
    if log_findings:
        raise RuntimeError(f"critical Springer log findings: {log_findings}")
    if payload.get("pdf") != _pdf_metadata():
        raise RuntimeError("Springer PDF metadata differs from the build manifest")
    if payload.get("toolchain") != _current_toolchain():
        raise RuntimeError("Springer build toolchain differs from the build manifest")


def main() -> None:
    springer_submission.validate_submission_manifest()
    log = _single_output_with_suffix(".log")
    if log.is_file():
        findings = _critical_log_findings(log.read_text(encoding="utf-8", errors="replace"))
        if findings:
            raise RuntimeError(f"critical Springer log findings: {findings}")
    payload = {
        "schema_version": 1,
        "complete": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": _hash_group(INPUTS),
        "outputs": _hash_group(OUTPUTS),
        "pdf": _pdf_metadata(),
        "toolchain": _current_toolchain(),
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)
    validate_build_manifest()
    print(MANIFEST)


if __name__ == "__main__":
    main()
