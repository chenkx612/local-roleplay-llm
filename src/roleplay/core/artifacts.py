"""Small, dependency-free helpers for inspectable workflow artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4


def repository_root() -> Path:
    """Return the repository root for an editable or source checkout."""
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path, label: str = "JSON") -> dict[str, Any]:
    """Read a JSON object with a concise, path-aware error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} 不是有效 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON 对象: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonblank JSONL rows and require each row to be an object."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 不是对象")
        rows.append(value)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically write UTF-8 JSON in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    refuse_overwrite: bool = False,
) -> None:
    """Atomically write UTF-8 JSONL, optionally refusing replacement."""
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if refuse_overwrite and path.exists():
            raise FileExistsError(f"拒绝覆盖已有文件: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def artifact_metadata(path: Path, root: Path) -> dict[str, Any]:
    """Describe an artifact relative to its run directory."""
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

