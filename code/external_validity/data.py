"""Pinned NanoDrone records and training-only normalization.

Source CSVs contain motor angular speeds in rad/s and states in SI units.
Quaternion xyzw is transformed sample by sample to the principal rotation
vector; this deterministic transformation never uses neighboring samples.
Downloaded source files are retained locally for research, not relicensed.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation

from rgpwm.data import Normalizer


SOURCE_REPOSITORY = "idsia-robotics/nanodrone-sysid-benchmark"
SOURCE_COMMIT = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
SOURCE_DOI = "10.1016/j.conengprac.2026.106871"
DT = 0.01
POSITION_COLUMNS = ("x", "y", "z")
VELOCITY_COLUMNS = ("vx", "vy", "vz")
QUATERNION_COLUMNS = ("qx", "qy", "qz", "qw")
ANGULAR_VELOCITY_COLUMNS = ("wx", "wy", "wz")
ACTION_COLUMNS = ("m1_rads", "m2_rads", "m3_rads", "m4_rads")
STATE_COLUMNS = (*POSITION_COLUMNS, *VELOCITY_COLUMNS, "rx", "ry", "rz", *ANGULAR_VELOCITY_COLUMNS)


@dataclass(frozen=True)
class Record:
    record_id: str
    split: str
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    dt: float = DT
    source_path: str = ""
    sha256: str = ""

    @property
    def trajectory_id(self) -> str:
        """Compatibility with the existing, read-only windowing API."""
        return self.record_id


def expected_records() -> list[tuple[str, str, str]]:
    """Return (analysis split, record identifier, pinned source path)."""
    records = []
    for trajectory in ("square", "random", "chirp"):
        for run in (1, 2, 3, 4):
            name = f"{trajectory}_20251017_run{run}"
            records.append(("val" if run == 4 else "train", name, f"data/train/{name}.csv"))
    for run in (1, 2, 3):
        name = f"melon_20251017_run{run}"
        records.append(("test", name, f"data/test/{name}.csv"))
    return records


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "CFR-WM-external-validity-research"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def download_dataset(root: str | Path) -> dict:
    """Acquire and hash-check pinned LFS content; never overwrite different data.

    License discovery records the pinned tree, rather than treating public
    visibility as a redistribution license. No upstream Python code is copied.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    api_base = f"https://api.github.com/repos/{SOURCE_REPOSITORY}"
    tree_bytes = _fetch(f"{api_base}/git/trees/{SOURCE_COMMIT}?recursive=1")
    tree = json.loads(tree_bytes)
    if tree.get("truncated"):
        raise ValueError("Pinned repository tree is truncated")
    available = {item["path"] for item in tree["tree"]}
    license_files = sorted(path for path in available if Path(path).name.lower().startswith(("license", "copying", "notice")))
    metadata = root / "source_metadata"
    metadata.mkdir(exist_ok=True)
    for filename, content in (
        ("repository_tree.json", tree_bytes),
        ("README.md", _fetch(f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/README.md")),
    ):
        target = metadata / filename
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Existing source metadata differs: {target}")
        target.write_bytes(content)

    def acquire(item: tuple[str, str, str]) -> dict:
        split, record_id, source_path = item
        if source_path not in available:
            raise FileNotFoundError(f"Missing pinned source: {source_path}")
        raw_url = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/{source_path}"
        pointer = _fetch(raw_url).decode("utf-8")
        pointer_fields = dict(line.split(" ", 1) for line in pointer.strip().splitlines())
        if pointer_fields.get("version") != "https://git-lfs.github.com/spec/v1":
            raise ValueError(f"Expected LFS pointer: {source_path}")
        oid = pointer_fields["oid"]
        if not oid.startswith("sha256:"):
            raise ValueError(f"Unsupported LFS checksum: {oid}")
        expected_hash = oid.removeprefix("sha256:")
        expected_bytes = int(pointer_fields["size"])
        media_url = f"https://media.githubusercontent.com/media/{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/{source_path}"
        relative_path = Path(source_path).relative_to("data")
        destination = root / relative_path
        content = destination.read_bytes() if destination.exists() else _fetch(media_url)
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash or len(content) != expected_bytes:
            raise ValueError(f"Source content failed LFS integrity check: {destination}")
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            destination.write_bytes(content)
        return {
            "record_id": record_id, "analysis_split": split,
            "source_path": source_path, "local_path": relative_path.as_posix(),
            "url": media_url, "sha256": actual_hash, "bytes": len(content),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(acquire, expected_records()))
    manifest = {
        "source_repository": f"https://github.com/{SOURCE_REPOSITORY}",
        "source_commit": SOURCE_COMMIT, "source_doi": SOURCE_DOI,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_files_in_pinned_tree": license_files,
        "license_status": "not_found" if not license_files else "requires_review",
        "redistribution": "not_granted_by_this_manifest; local research download only",
        "source_metadata_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(metadata.iterdir()) if path.is_file()
        },
        "state_order": list(STATE_COLUMNS), "action_order": list(ACTION_COLUMNS),
        "units": {"position": "m", "velocity": "m/s", "rotation_vector": "rad", "body_angular_velocity": "rad/s", "motor_angular_velocity": "rad/s", "time": "s"},
        "analysis_split": "run1-3 square/random/chirp train; run4 validation; melon run1-3 test",
        "files": files,
    }
    manifest_path = root / "download_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["source_commit"] != SOURCE_COMMIT or previous["files"] != files:
            raise ValueError("Existing download manifest identifies different records")
        return previous
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_records(root: str | Path) -> dict[str, list[Record]]:
    """Load all 15 required records with verified hashes and fixed whole-run split."""
    root = Path(root)
    manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Dataset commit differs from the pinned protocol")
    entries = {item["record_id"]: item for item in manifest["files"]}
    wanted = expected_records()
    if len(entries) != len(manifest["files"]) or set(entries) != {item[1] for item in wanted}:
        raise ValueError("Manifest record inventory is incomplete or duplicated")
    output: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    required = ("t", *POSITION_COLUMNS, *VELOCITY_COLUMNS, *QUATERNION_COLUMNS, *ANGULAR_VELOCITY_COLUMNS, *ACTION_COLUMNS)
    seen_hashes: set[str] = set()
    for split, record_id, source_path in wanted:
        entry = entries[record_id]
        relative_path = Path(source_path).relative_to("data")
        if entry["analysis_split"] != split or entry["local_path"] != relative_path.as_posix():
            raise ValueError(f"Manifest split/path mismatch: {record_id}")
        path = root / relative_path
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != entry["sha256"] or len(content) != entry["bytes"]:
            raise ValueError(f"Dataset changed since download: {record_id}")
        if checksum in seen_hashes:
            raise ValueError("Duplicate physical record across the analysis inventory")
        seen_hashes.add(checksum)
        table = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64, encoding="utf-8")
        if not table.dtype.names or not set(required).issubset(table.dtype.names):
            raise ValueError(f"Required NanoDrone fields missing: {record_id}")
        if table.ndim != 1 or len(table) < 2:
            raise ValueError(f"Record requires at least two samples: {record_id}")
        values = np.column_stack([table[name] for name in required])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite measured values: {record_id}")
        timestamps = table["t"].copy()
        if not np.allclose(np.diff(timestamps), DT, rtol=0.0, atol=1e-7):
            raise ValueError(f"Expected monotonic 100 Hz timestamps: {record_id}")
        quaternion = np.column_stack([table[name] for name in QUATERNION_COLUMNS])
        if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, rtol=0.0, atol=0.02):
            raise ValueError(f"Malformed orientation quaternion: {record_id}")
        rotvec = Rotation.from_quat(quaternion).as_rotvec()
        states = np.column_stack([*[table[name] for name in POSITION_COLUMNS], *[table[name] for name in VELOCITY_COLUMNS], rotvec, *[table[name] for name in ANGULAR_VELOCITY_COLUMNS]])
        actions = np.column_stack([table[name] for name in ACTION_COLUMNS])
        if np.any(actions < 0.0):
            raise ValueError(f"Motor angular speed must be nonnegative: {record_id}")
        output[split].append(Record(record_id, split, states, actions, timestamps, DT, source_path, checksum))
    return output


def require_training_records(records: Iterable[Record]) -> list[Record]:
    items = list(records)
    if not items or any(item.split != "train" for item in items):
        raise ValueError("Fitting requires nonempty training records only")
    if len({item.record_id for item in items}) != len(items):
        raise ValueError("Duplicate training record identifiers")
    for item in items:
        if item.states.ndim != 2 or item.states.shape[1] != 12 or item.actions.shape != (len(item.states), 4):
            raise ValueError(f"Invalid record dimensions: {item.record_id}")
        if len(item.states) < 2 or not np.isfinite(item.states).all() or not np.isfinite(item.actions).all():
            raise ValueError(f"Invalid training values: {item.record_id}")
        if not np.isfinite(item.dt) or item.dt <= 0.0:
            raise ValueError(f"Invalid sample interval: {item.record_id}")
    return items


def fit_normalizer(train_records: Iterable[Record]) -> Normalizer:
    items = require_training_records(train_records)
    states = np.concatenate([item.states for item in items], axis=0)
    actions = np.concatenate([item.actions for item in items], axis=0)
    return Normalizer(states.mean(axis=0), np.maximum(states.std(axis=0), 1e-6), actions.mean(axis=0), np.maximum(actions.std(axis=0), 1e-6))

