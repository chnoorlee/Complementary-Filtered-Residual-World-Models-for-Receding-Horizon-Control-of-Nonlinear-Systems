"""Immutable grid constants and analysis fingerprint for CFR-WM confirmation."""

from __future__ import annotations

import hashlib
from pathlib import Path


CONFIRM_METHODS = (
    "rssm", "simnorm", "transformer", "pgnn", "pgdmm",
    "cfrwm", "cfr_gain0", "cfr_gain025", "cfr_gain075", "cfr_gain1",
    "cfr_1step",
)
CONFIRM_SEEDS = (41, 53, 67, 79, 83, 97, 109, 113)
CONFIRM_REGIMES = ("id_clean", "id_corrupt", "ood_parameter_corrupt")
COMPARATORS = ("rssm", "simnorm", "transformer", "pgnn", "pgdmm")
ABLATIONS = (
    "cfr_gain0", "cfr_gain025", "cfr_gain075", "cfr_gain1", "cfr_1step",
)
METHOD_LABELS = {
    "rssm": "RSSM",
    "simnorm": "SimNorm-WM",
    "transformer": "TransWM",
    "pgnn": "PGNN",
    "pgdmm": "PgDMM",
    "cfrwm": "CFR-WM",
    "cfr_gain0": "CFR-gain-0",
    "cfr_gain025": "CFR-gain-0.25",
    "cfr_gain075": "CFR-gain-0.75",
    "cfr_gain1": "CFR-gain-1",
    "cfr_1step": "CFR-one-step",
}


def analysis_code_sha256() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for name in ("confirm_protocol.py", "confirm_analyze.py", "statistics.py"):
        path = package / name
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
