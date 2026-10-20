"""Create a SHA-256 snapshot manifest for the final research package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOTS = (
    "archive", "code", "data", "docs", "figures", "literature", "logs",
    "results", "paper", "sections", "plans", "tests",
)
ROOT_FILES = (
    ".gitattributes", ".gitignore", "environment.yml", "MEGA_PROMPT.md",
    "PROGRESS.md", "paper.txt", "pyproject.toml", "README.md",
    "RESTRICTS.yaml",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    entries = []
    destination = Path("archive/manifest_sha256.json")
    candidates = [Path(name) for name in ROOT_FILES if Path(name).is_file()]
    for root_name in ROOTS:
        root = Path(root_name)
        if not root.exists():
            continue
        candidates.extend(item for item in root.rglob("*") if item.is_file())
    for path in sorted(set(candidates)):
        if path == destination:
            continue
        if (
            "__pycache__" in path.parts
            or "render_pages_current" in path.parts
            or path.suffix in {
                ".aux", ".blg", ".out", ".fdb_latexmk", ".fls",
            }
        ):
            continue
        entries.append({
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SHA-256",
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"manifested {len(entries)} files ({payload['total_bytes']} bytes)")


if __name__ == "__main__":
    main()
