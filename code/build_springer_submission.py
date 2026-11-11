"""Build a single-file Springer Nature submission from the audited paper sources."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from bibliography_provenance import audit_row_matches_record


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN_DIR = PAPER / "mypaper"
SECTIONS = ROOT / "sections"
OUTPUT = MAIN_DIR / "main_springer.tex"
RESULT_MACROS = PAPER / "results_macros.tex"
ASSET_MANIFEST = PAPER / "manuscript_assets_manifest.json"
ASSET_GENERATOR = ROOT / "code" / "make_confirmatory_manuscript_assets.py"
BIBLIOGRAPHY_MANIFEST = ROOT / "literature" / "bibliography_manifest.json"
SUBMISSION_MANIFEST = PAPER / "submission_source_manifest.json"
REFERENCES = PAPER / "references.bib"
EXTERNAL_REFERENCES = PAPER / "external_references.bib"
EXTERNAL_CITATION_AUDIT = ROOT / "literature" / "external_citation_audit.json"
EXTERNAL_ASSET_MANIFEST = PAPER / "external_assets_manifest.json"
EXTERNAL_ASSET_GENERATOR = ROOT / "code" / "make_external_assets.py"
EXTERNAL_MACROS = PAPER / "external_results_macros.tex"
TITLE = (
    "Complementary-Filtered Residual World Models: "
    "Controlled Benchmarks and Real-Flight Evaluation"
)
BIBLIOGRAPHY_GENERATORS = (
    ROOT / "code" / "make_bibliography.py",
    ROOT / "code" / "bibliography_provenance.py",
)
BODY_SECTIONS = (
    "introduction", "related_work", "method", "experiments", "results",
    "external_validation", "discussion", "limitations", "conclusion",
)


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"refusing to inline a file outside the project: {resolved}") from error
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_key(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _analysis_sources() -> tuple[Path, ...]:
    analysis = ROOT / "results" / "confirmatory_analysis"
    return (
        ROOT / "docs" / "12_cfr_confirmatory_contract.json",
        ROOT / "docs" / "14_cfr_analysis_amendment.json",
        ROOT / "code" / "rgpwm" / "confirm_analyze_amended.py",
        ROOT / "results" / "confirmatory_run_manifest.json",
        analysis / "confirmatory_inference.json",
        analysis / "confirmatory_summary.csv",
        analysis / "aggregate_cells.csv",
        analysis / "trajectory_metrics.csv",
        analysis / "cell_contrasts.csv",
    )


def _bibliography_sources() -> tuple[Path, ...]:
    return (
        ROOT / "literature" / "shortlist.json",
        ROOT / "literature" / "v2_supplement.json",
        ROOT / "literature" / "citation_audit.json",
        REFERENCES,
    )


def _template_sources() -> tuple[Path, ...]:
    template = PAPER / "springer-official" / "sn-article-template"
    return (
        template / "sn-jnl.cls",
        template / "bst" / "sn-mathphys-num.bst",
        ROOT / "code" / "README.md",
    )


def _manuscript_sources() -> tuple[Path, ...]:
    names = ("abstract", *BODY_SECTIONS, "appendix")
    return tuple(SECTIONS / f"{name}.tex" for name in names)


def _graphics_sources() -> tuple[Path, ...]:
    """Resolve every graphic referenced by a manuscript section."""
    pattern = re.compile(r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}")
    graphics: list[Path] = []
    for source in _manuscript_sources():
        for match in pattern.finditer(source.read_text(encoding="utf-8")):
            candidate = MAIN_DIR / match.group(1)
            if not candidate.suffix:
                matches = [
                    candidate.with_suffix(suffix)
                    for suffix in (".pdf", ".png", ".jpg", ".jpeg", ".eps")
                    if candidate.with_suffix(suffix).is_file()
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"graphic path is missing or ambiguous: {candidate}")
                candidate = matches[0]
            graphic = _inside_root(candidate)
            if not graphic.is_file():
                raise RuntimeError(f"required manuscript graphic is missing: {graphic}")
            graphics.append(graphic)
    return tuple(dict.fromkeys(graphics))


def _required_result_assets() -> list[Path]:
    return [
        RESULT_MACROS,
        ROOT / "results/confirmatory_analysis/table_confirmatory.tex",
        MAIN_DIR / "figures/confirmatory_overview.png",
        MAIN_DIR / "figures/cell_difference_heatmap.png",
        PAPER / "table_comparator_contrasts.tex",
        PAPER / "table_sensitivity_contrasts.tex",
        PAPER / "table_secondary_metrics.tex",
        PAPER / "table_cell_contrasts.tex",
        PAPER / "table_seed_effects.tex",
    ]


def _external_assets() -> tuple[Path, ...]:
    return (
        *(ROOT / "results/external_nanodrone" / f"{name}.csv"
          for name in ("summary", "contrasts", "seed_scores")),
        PAPER / "table_external_results.tex", PAPER / "table_external_physical.tex",
        EXTERNAL_MACROS,
        MAIN_DIR / "figures/external_flight.png", MAIN_DIR / "figures/external_flight.pdf",
    )


def _validate_hashes(recorded: object, label: str) -> None:
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError(f"{label} lacks file hashes")
    for key, expected in recorded.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise RuntimeError(f"{label} has invalid file hashes")
        path = _inside_root(ROOT / key)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"{label} mismatch: {key}")


def _validate_external_citation() -> None:
    if not EXTERNAL_CITATION_AUDIT.is_file() or not EXTERNAL_REFERENCES.is_file():
        raise RuntimeError("external citation audit or bibliography is missing")
    audit = json.loads(EXTERNAL_CITATION_AUDIT.read_text(encoding="utf-8"))
    fields = audit.get("fields")
    required = {"author", "title", "journal", "volume", "pages", "year", "doi", "url"}
    if (audit.get("status") != "verified" or audit.get("cite_key") != "busetto2026_nanodrone"
            or not isinstance(fields, dict) or set(fields) != required
            or any(not isinstance(value, str) for value in fields.values())):
        raise RuntimeError("external citation audit is incomplete")
    expected = "@article{busetto2026_nanodrone,\n" + "".join(
        f"  {key} = {{{fields[key]}}},\n" for key in sorted(fields)
    ) + "}\n"
    if (EXTERNAL_REFERENCES.read_text(encoding="utf-8") != expected
            or _sha256(EXTERNAL_REFERENCES) != audit.get("bibliography_sha256")):
        raise RuntimeError("external bibliography differs from verified metadata")
    _validate_hashes(audit.get("local_evidence"), "external citation evidence")


def _validate_external_assets() -> None:
    """Require the new empirical chain without modifying the frozen synthetic chain."""
    if not EXTERNAL_ASSET_MANIFEST.is_file():
        raise RuntimeError("external result-asset manifest is missing")
    manifest = json.loads(EXTERNAL_ASSET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError("external result-asset manifest is incomplete")
    expected_assets = {_manifest_key(path) for path in _external_assets()}
    if set(manifest.get("assets", {})) != expected_assets:
        raise RuntimeError("external result-asset inventory mismatch")
    if set(manifest.get("generator", {})) != {_manifest_key(EXTERNAL_ASSET_GENERATOR)}:
        raise RuntimeError("external asset-generator inventory mismatch")
    for group in ("generator", "assets", "authenticated_sources"):
        _validate_hashes(manifest.get(group), f"external {group}")
    directory = ROOT / "results/external_nanodrone"
    train_path, eval_path = directory / "training_manifest.json", directory / "evaluation_manifest.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    if (train.get("complete") is not True or evaluation.get("complete") is not True
            or evaluation.get("training_manifest_sha256") != _sha256(train_path)
            or train.get("sources") != evaluation.get("sources")):
        raise RuntimeError("external training/evaluation provenance mismatch")
    for source, label in ((train.get("sources"), "source"),
                          (train.get("training_outputs"), "training output"),
                          (evaluation.get("files"), "evaluation output")):
        _validate_hashes(source, f"external {label}")
    expected_sources = {**train["sources"], **train["training_outputs"], **evaluation["files"],
                        _manifest_key(train_path): _sha256(train_path),
                        _manifest_key(eval_path): _sha256(eval_path)}
    if manifest["authenticated_sources"] != expected_sources:
        raise RuntimeError("external asset-source inventory mismatch")
    if manifest.get("counts") != {"flight_rows": 384, "summary_rows": 32,
                                   "contrast_rows": 16, "prediction_files": 64}:
        raise RuntimeError("external result inventory is incomplete")


def _external_submission_inputs() -> tuple[Path, ...]:
    return (EXTERNAL_REFERENCES, EXTERNAL_CITATION_AUDIT, EXTERNAL_ASSET_MANIFEST,
            EXTERNAL_ASSET_GENERATOR, *_external_assets())


def _analysis_code_sha256() -> str:
    digest = hashlib.sha256()
    package = ROOT / "code" / "rgpwm"
    for name in ("confirm_protocol.py", "confirm_analyze.py", "statistics.py"):
        path = _inside_root(package / name)
        if not path.is_file():
            raise RuntimeError(f"frozen analysis source is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_analysis_code_provenance(manifest: dict[str, object]) -> None:
    contract_path = _inside_root(ROOT / "docs" / "12_cfr_confirmatory_contract.json")
    if not contract_path.is_file():
        raise RuntimeError(f"frozen confirmatory contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    frozen = contract.get("hashes", {}).get("analysis_code_sha256")
    recorded = manifest.get("analysis_code_sha256")
    current = _analysis_code_sha256()
    if not isinstance(frozen, str) or recorded != frozen or current != frozen:
        raise RuntimeError(
            "analysis-code provenance mismatch between current code, contract, and asset manifest"
        )
    amendment_path = _inside_root(ROOT / "docs" / "14_cfr_analysis_amendment.json")
    amendment_code = _inside_root(ROOT / "code" / "rgpwm" / "confirm_analyze_amended.py")
    if not amendment_path.is_file() or not amendment_code.is_file():
        raise RuntimeError("analysis amendment contract or entry point is missing")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment_contract_hash = _sha256(amendment_path)
    amendment_code_hash = _sha256(amendment_code)
    if (
        amendment.get("amendment_code_sha256") != amendment_code_hash
        or manifest.get("analysis_amendment_code_sha256") != amendment_code_hash
        or manifest.get("analysis_amendment_contract_sha256") != amendment_contract_hash
    ):
        raise RuntimeError("analysis-amendment provenance mismatch")


def _validate_bibliography_provenance() -> None:
    """Require the bibliography to match an exact, fully verified audit."""
    if not BIBLIOGRAPHY_MANIFEST.is_file():
        raise RuntimeError("verified bibliography manifest is missing")
    manifest = json.loads(BIBLIOGRAPHY_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError("verified bibliography manifest is incomplete")
    recorded_sources = manifest.get("sources")
    recorded_generators = manifest.get("generators")
    recorded_bibliography = manifest.get("bibliography")
    if (
        not isinstance(recorded_sources, dict)
        or not isinstance(recorded_generators, dict)
        or not isinstance(recorded_bibliography, dict)
    ):
        raise RuntimeError("verified bibliography manifest has invalid groups")
    sources = _bibliography_sources()
    for path in sources[:-1]:
        source = _inside_root(path)
        if (
            not source.is_file()
            or recorded_sources.get(_manifest_key(source)) != _sha256(source)
        ):
            raise RuntimeError(f"bibliography-source manifest mismatch: {_manifest_key(source)}")
    for path in BIBLIOGRAPHY_GENERATORS:
        generator = _inside_root(path)
        if (
            not generator.is_file()
            or recorded_generators.get(_manifest_key(generator)) != _sha256(generator)
        ):
            raise RuntimeError(
                f"bibliography-generator manifest mismatch: {_manifest_key(generator)}"
            )
    bibliography = _inside_root(sources[-1])
    if (
        not bibliography.is_file()
        or recorded_bibliography.get(_manifest_key(bibliography)) != _sha256(bibliography)
    ):
        raise RuntimeError("bibliography manifest mismatch: paper/references.bib")

    records = json.loads(sources[0].read_text(encoding="utf-8"))["shortlist"]
    records.extend(json.loads(sources[1].read_text(encoding="utf-8"))["shortlist"])
    audit = json.loads(sources[2].read_text(encoding="utf-8"))
    audit_rows = audit.get("records")
    if not isinstance(audit_rows, list):
        raise RuntimeError("citation audit lacks records")
    source_map = {str(row["cite_key"]): row for row in records}
    audit_map = {str(row.get("cite_key")): row for row in audit_rows}
    if (
        len(source_map) != len(records)
        or len(audit_map) != len(audit_rows)
        or set(source_map) != set(audit_map)
        or audit.get("verified") != len(records)
        or manifest.get("entry_count") != len(records)
    ):
        raise RuntimeError("citation audit does not exactly cover bibliography sources")
    for key, record in source_map.items():
        row = audit_map[key]
        if not audit_row_matches_record(row, record):
            raise RuntimeError(f"citation audit is stale or unverified for {key}")


def _validate_asset_manifest(required_assets: list[Path]) -> None:
    if not ASSET_MANIFEST.is_file():
        raise RuntimeError(f"result-asset completion manifest is missing: {ASSET_MANIFEST}")
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError("result-asset completion manifest is not complete")
    _validate_analysis_code_provenance(manifest)
    expected_groups = {
        "generator": (ASSET_GENERATOR,),
        "assets": tuple(required_assets),
        "authenticated_sources": _analysis_sources(),
    }
    for group, paths in expected_groups.items():
        recorded = manifest.get(group)
        if not isinstance(recorded, dict):
            raise RuntimeError(f"result-asset manifest lacks {group}")
        for path in paths:
            source = _inside_root(path)
            key = _manifest_key(source)
            if not source.is_file() or recorded.get(key) != _sha256(source):
                raise RuntimeError(f"result-asset manifest mismatch: {key}")


def _expand_inputs(text: str, base: Path) -> str:
    pattern = re.compile(r"\\input\s*\{([^}]+)\}")

    def replace(match: re.Match[str]) -> str:
        candidate = base / match.group(1)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".tex")
        source = _inside_root(candidate)
        if not source.is_file():
            raise RuntimeError(f"required LaTeX input is missing: {source}")
        return _expand_inputs(source.read_text(encoding="utf-8"), base)

    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(replace, text)
    return text


def _abstract_text() -> str:
    source = (SECTIONS / "abstract.tex").read_text(encoding="utf-8").strip()
    match = re.fullmatch(
        r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}", source, re.DOTALL,
    )
    if match is None:
        raise RuntimeError("abstract.tex must contain exactly one abstract environment")
    return match.group(1).strip()


def _body(name: str) -> str:
    source = (SECTIONS / f"{name}.tex").read_text(encoding="utf-8")
    return _expand_inputs(source, MAIN_DIR).strip()


def build() -> str:
    required = _required_result_assets()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"locked manuscript assets are incomplete: {missing}")
    _validate_bibliography_provenance()
    _validate_asset_manifest(required)
    _validate_external_citation()
    _validate_external_assets()

    macros = "\n".join(path.read_text(encoding="utf-8").strip()
                       for path in (RESULT_MACROS, EXTERNAL_MACROS))
    body = "\n\n".join(_body(name) for name in BODY_SECTIONS)
    appendix = _body("appendix")
    appendix = re.sub(r"^\\appendix\s*", "", appendix, count=1)
    document = rf"""% Generated from audited sources; do not edit this flattened file manually.
% Springer Nature journal article template, December 2024 version.
\documentclass[pdflatex,sn-mathphys-num]{{sn-jnl}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{booktabs}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\usepackage[title]{{appendix}}
\hypersetup{{pdftitle={{{TITLE}}}}}
\newcommand{{\method}}{{CFR-WM}}
\newcommand{{\vect}}[1]{{\mathbf{{#1}}}}
{macros}

\raggedbottom
\begin{{document}}

\title[Complementary-Filtered Residual World Models]{{{TITLE}}}

% Replace all placeholder metadata before submission.
\author*[1]{{\fnm{{Author}} \sur{{Placeholder}}}}\email{{author@example.com}}
\affil*[1]{{\orgdiv{{Department}}, \orgname{{Institution}}, \orgaddress{{\city{{City}}, \country{{Country}}}}}}

\abstract{{{_abstract_text()}}}
\keywords{{world models, physics-guided machine learning, complementary filtering, long-horizon prediction, missing observations, distribution shift}}
\pacs[MSC 2020]{{68T07, 68T40, 93B07}}

\maketitle

{body}

\backmatter

\section*{{Declarations}}

\textbf{{Funding}} [To be completed by the authors before submission.]

\textbf{{Competing interests}} [To be confirmed by the authors before submission.]

\textbf{{Ethics approval and consent to participate}} The study uses simulated dynamics and publicly released quadrotor flight records. [Authors to confirm the applicable ethics statement before submission.]

\textbf{{Consent for publication}} Not applicable.

\textbf{{Data and code availability}} The controlled-system trajectories are generated by the supplied differential-equation code. External evaluation uses the NanoDrone authors' publicly released, offline-preprocessed flight state estimates~\cite{{busetto2026_nanodrone}}. The accompanying research package contains source code, seeds, split identifiers, checkpoints, saved predictions, metrics, analysis scripts, and data-download URLs with SHA-256 hashes. Upstream flight records are obtained from the authors' repository; this package grants no additional license to those records. [A public archival repository and persistent identifier must be added before submission.]

\textbf{{Author contributions}} [To be completed by the authors before submission.]

\textbf{{Generative-AI assistance}} Generative-AI tools assisted drafting and software development. [Human author review and approval of the equations, sources, code outputs, statistical results, and claims remain to be documented. The authors must confirm this disclosure against the selected journal's policy before submission.]

\begin{{appendices}}
{appendix}
\end{{appendices}}

\bibliography{{../references,../external_references}}
\end{{document}}
"""
    if re.search(r"\\(?:input|include)\b", document):
        raise RuntimeError("Springer submission must not retain input/include dependencies")
    return document


def _write_submission_manifest() -> None:
    """Bind the flattened source to every prose, bibliography, and build input."""
    inputs = (
        *_manuscript_sources(),
        *_graphics_sources(),
        REFERENCES,
        ASSET_MANIFEST,
        BIBLIOGRAPHY_MANIFEST,
        *_external_submission_inputs(),
        *_template_sources(),
        Path(__file__),
    )
    missing = [str(path) for path in (*inputs, OUTPUT) if not path.is_file()]
    if missing:
        raise RuntimeError(f"submission source manifest inputs are missing: {missing}")
    payload = {
        "complete": True,
        "inputs": {_manifest_key(path): _sha256(path) for path in inputs},
        "output": {_manifest_key(OUTPUT): _sha256(OUTPUT)},
    }
    temporary = SUBMISSION_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SUBMISSION_MANIFEST)


def validate_submission_manifest() -> None:
    if not SUBMISSION_MANIFEST.is_file():
        raise RuntimeError("submission source manifest is missing")
    manifest = json.loads(SUBMISSION_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError("submission source manifest is incomplete")
    _validate_bibliography_provenance()
    _validate_asset_manifest(_required_result_assets())
    _validate_external_citation()
    _validate_external_assets()
    expected = {
        "inputs": (
            *_manuscript_sources(), *_graphics_sources(), REFERENCES,
            ASSET_MANIFEST, BIBLIOGRAPHY_MANIFEST, *_template_sources(),
            *_external_submission_inputs(),
            Path(__file__),
        ),
        "output": (OUTPUT,),
    }
    for group, paths in expected.items():
        recorded = manifest.get(group)
        if not isinstance(recorded, dict):
            raise RuntimeError(f"submission source manifest lacks {group}")
        for path in paths:
            if not path.is_file() or recorded.get(_manifest_key(path)) != _sha256(path):
                raise RuntimeError(f"submission source manifest mismatch: {_manifest_key(path)}")


def main() -> None:
    document = build()
    temporary = OUTPUT.with_suffix(".tex.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(OUTPUT)
    _write_submission_manifest()
    validate_submission_manifest()
    print(OUTPUT)
    print(SUBMISSION_MANIFEST)


if __name__ == "__main__":
    main()
