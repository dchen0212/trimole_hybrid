#!/usr/bin/env python3
"""Build a fail-closed Zenodo-ready archive from tracked lightweight files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REJECTED_SUFFIXES = {
    ".ckpt",
    ".joblib",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".safetensors",
}
REJECTED_PARTS = {".git", "__pycache__", "node_modules", "results_audit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-file-mb", type=float, default=25.0)
    return parser.parse_args()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    status = git(repo, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise SystemExit("Tracked worktree changes are present; commit or revert them before archiving.")
    commit = git(repo, "rev-parse", "HEAD")
    short_commit = commit[:12]
    tracked = [Path(line) for line in git(repo, "ls-files").splitlines() if line]

    records: list[dict[str, object]] = []
    accepted: list[Path] = []
    size_limit = int(args.max_file_mb * 1024 * 1024)
    for relative in tracked:
        source = repo / relative
        if not source.is_file():
            continue
        if any(part in REJECTED_PARTS for part in relative.parts):
            raise SystemExit(f"Rejected path is tracked: {relative}")
        if source.suffix.lower() in REJECTED_SUFFIXES:
            raise SystemExit(f"Rejected binary suffix is tracked: {relative}")
        size = source.stat().st_size
        if size > size_limit:
            raise SystemExit(f"Tracked file exceeds {args.max_file_mb:g} MB: {relative}")
        digest = sha256(source)
        records.append({"path": relative.as_posix(), "size_bytes": size, "sha256": digest})
        accepted.append(relative)

    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "file_count": len(records),
        "exclusions": sorted(REJECTED_SUFFIXES),
        "files": records,
    }
    manifest_path = output_dir / "ARCHIVE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums_path = output_dir / "SHA256SUMS"
    checksums_path.write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in records),
        encoding="utf-8",
    )

    archive_name = f"trimole-hybrid-bioinf-2026-2212-{short_commit}.zip"
    archive_path = output_dir / archive_name
    archive_root = f"trimole-hybrid-{short_commit}"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in accepted:
            archive.write(repo / relative, f"{archive_root}/{relative.as_posix()}")
        archive.write(manifest_path, f"{archive_root}/ARCHIVE_MANIFEST.json")
        archive.write(checksums_path, f"{archive_root}/SHA256SUMS")

    print(json.dumps({"archive": str(archive_path), "git_commit": commit, "files": len(records), "sha256": sha256(archive_path)}, indent=2))


if __name__ == "__main__":
    main()
