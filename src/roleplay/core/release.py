"""Reusable, verified release exchange for training run artifacts."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from roleplay.core.artifacts import (
    artifact_metadata,
    sha256_file,
    write_json_atomic,
)


@dataclass(frozen=True)
class ReleaseSpec:
    """Stage-specific labels and contracts for the shared release workflow."""

    tag_prefix: str
    cli_name: str
    title: str
    notes: str
    expected_files: frozenset[str]
    contract_label: str
    default_repository: str
    tag_pattern: str = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    manifest_extra: Mapping[str, Any] = field(default_factory=dict)


def _error(error_type: type[Exception], message: str) -> Exception:
    return error_type(message)


def _bundle_paths(
    output_dir: Path, run_id: str, spec: ReleaseSpec
) -> tuple[Path, Path, str]:
    tag = f"{spec.tag_prefix}-{run_id}"
    return (
        output_dir / f"{tag}.tar.gz",
        output_dir / f"{tag}.manifest.json",
        tag,
    )


def create_release_bundle(
    run_dir: Path,
    output_dir: Path,
    *,
    spec: ReleaseSpec,
    error_type: type[Exception],
    validate_archive: Callable[[Path], None],
    metadata_builder: Callable[[Path, Path], dict[str, Any]] = artifact_metadata,
    reuse_existing: bool = True,
) -> tuple[Path, Path, str]:
    """Create or safely reuse one verified release bundle."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    validate_archive(run_dir)
    summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    run_id = summary.get("run", {}).get("id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise _error(error_type, "run_summary.json 缺少有效 run.id")
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise _error(error_type, "发布包目录不能位于 run 目录内")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path, manifest_path, tag = _bundle_paths(output_dir, run_id, spec)
    contents = {
        name: metadata_builder(run_dir / name, run_dir)
        for name in sorted(spec.expected_files)
    }
    if bundle_path.exists() or manifest_path.exists():
        if not reuse_existing:
            raise _error(error_type, "拒绝覆盖现有发布包")
        if not bundle_path.is_file() or not manifest_path.is_file():
            raise _error(error_type, "发布包或 manifest 不完整")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("contents") != contents
            or manifest.get("bundle", {}).get("sha256")
            != sha256_file(bundle_path)
        ):
            raise _error(error_type, "现有发布包与 run 不一致")
        return bundle_path, manifest_path, tag

    temporary = output_dir / f".{bundle_path.name}.{uuid4().hex}.tmp"
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name in sorted(spec.expected_files):
                archive.add(
                    run_dir / name,
                    arcname=f"{run_id}/{name}",
                    recursive=False,
                )
        temporary.replace(bundle_path)
        manifest = {
            "schema_version": 1,
            **spec.manifest_extra,
            "run_id": run_id,
            "run_status": summary["status"],
            "source_commit": summary.get("run", {}).get("commit"),
            "github_release_tag": tag,
            "bundle": {
                "file": bundle_path.name,
                "bytes": bundle_path.stat().st_size,
                "sha256": sha256_file(bundle_path),
            },
            "contents": contents,
        }
        write_json_atomic(manifest_path, manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        bundle_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return bundle_path, manifest_path, tag


def publish_release(
    run_dir: Path,
    output_dir: Path,
    repository: str,
    *,
    spec: ReleaseSpec,
    error_type: type[Exception],
    validate_archive: Callable[[Path], None],
    run_command: Callable[..., Any] = subprocess.run,
    metadata_builder: Callable[[Path, Path], dict[str, Any]] = artifact_metadata,
) -> tuple[Path, Path, str]:
    """Bundle and upload one run through the GitHub CLI."""
    bundle_path, manifest_path, tag = create_release_bundle(
        run_dir,
        output_dir,
        spec=spec,
        error_type=error_type,
        validate_archive=validate_archive,
        metadata_builder=metadata_builder,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command = [
        "gh",
        "release",
        "create",
        tag,
        str(bundle_path),
        str(manifest_path),
        "--repo",
        repository,
        "--title",
        spec.title.format(run_id=manifest["run_id"]),
        "--notes",
        spec.notes,
    ]
    source_commit = manifest.get("source_commit")
    if isinstance(source_commit, str) and source_commit:
        command.extend(["--target", source_commit])
    try:
        run_command(command, check=True)
    except FileNotFoundError as exc:
        raise _error(error_type, "未安装 GitHub CLI：gh") from exc
    except subprocess.CalledProcessError as exc:
        raise _error(
            error_type, "GitHub Release 上传失败；本地发布包已保留"
        ) from exc
    return bundle_path, manifest_path, tag


def format_download_command(
    tag: str, repository: str, *, spec: ReleaseSpec
) -> str:
    """Format a copyable legacy-compatible download command."""
    command = [spec.cli_name, "download", "--tag", tag]
    if repository != spec.default_repository:
        command.extend(["--repo", repository])
    return shlex.join(command)


def extract_release_bundle(
    bundle_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    spec: ReleaseSpec,
    error_type: type[Exception],
) -> Path:
    """Verify and atomically extract one release bundle."""
    bundle_path = bundle_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(
        manifest.get(name) != expected
        for name, expected in spec.manifest_extra.items()
    ):
        raise _error(error_type, "manifest stage 不正确")
    if manifest.get("bundle", {}).get("sha256") != sha256_file(bundle_path):
        raise _error(error_type, "下载包 SHA-256 与 manifest 不匹配")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id
    ):
        raise _error(error_type, "manifest 缺少有效 run_id")
    contents = manifest.get("contents")
    if (
        not isinstance(contents, dict)
        or set(contents) != spec.expected_files
        or any(not isinstance(metadata, dict) for metadata in contents.values())
    ):
        raise _error(
            error_type,
            f"manifest 文件清单不满足 {spec.contract_label}归档契约",
        )

    expected_names = {f"{run_id}/{name}" for name in contents}
    output_root = output_root.resolve()
    destination = output_root / run_id
    if destination.exists():
        raise _error(
            error_type,
            f"本地 run 目录已存在，拒绝覆盖: {destination}",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / f".download-{uuid4().hex}"
    staging_root.mkdir()
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != expected_names or not all(
                member.isfile() for member in members
            ):
                raise _error(error_type, "发布包内容与 manifest 不一致")
            for member in members:
                target = (staging_root / member.name).resolve()
                try:
                    target.relative_to(staging_root)
                except ValueError as exc:
                    raise _error(error_type, "发布包包含不安全路径") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise _error(
                        error_type,
                        f"无法读取发布包文件: {member.name}",
                    )
                with source, target.open("xb") as output_file:
                    shutil.copyfileobj(source, output_file)
        staged_run = staging_root / run_id
        for relative, metadata in contents.items():
            path = staged_run / relative
            if (
                not path.is_file()
                or path.stat().st_size != metadata.get("bytes")
                or sha256_file(path) != metadata.get("sha256")
            ):
                raise _error(error_type, f"解包文件校验失败: {relative}")
        staged_run.replace(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination


def download_release(
    tag: str,
    output_root: Path,
    repository: str,
    *,
    spec: ReleaseSpec,
    error_type: type[Exception],
    run_command: Callable[..., Any] = subprocess.run,
) -> Path:
    """Download, verify, and extract one GitHub Release."""
    if not re.fullmatch(spec.tag_pattern, tag):
        raise _error(error_type, "release tag 格式无效")
    with tempfile.TemporaryDirectory() as temporary:
        download_dir = Path(temporary)
        try:
            run_command(
                [
                    "gh",
                    "release",
                    "download",
                    tag,
                    "--repo",
                    repository,
                    "--dir",
                    str(download_dir),
                ],
                check=True,
            )
        except FileNotFoundError as exc:
            raise _error(error_type, "未安装 GitHub CLI：gh") from exc
        except subprocess.CalledProcessError as exc:
            raise _error(error_type, f"GitHub Release 下载失败: {tag}") from exc
        bundle = download_dir / f"{tag}.tar.gz"
        manifest = download_dir / f"{tag}.manifest.json"
        if not bundle.is_file() or not manifest.is_file():
            raise _error(error_type, "Release 缺少发布包或 manifest")
        return extract_release_bundle(
            bundle,
            manifest,
            output_root,
            spec=spec,
            error_type=error_type,
        )
