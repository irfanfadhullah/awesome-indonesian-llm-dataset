"""Strict JSONL input and atomic export directories with provenance manifests."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def save_json(path: str | Path, value: dict, *, overwrite: bool = False) -> None:
    """Atomically write a JSON report; caller explicitly owns mutable reports."""
    path = Path(path)
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"Already exists: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        if not overwrite and path.exists():
            raise FileExistsError(f"Already exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON number: {value}")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_constant)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield record


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(
    output: str | Path,
    datasets: dict[str, Iterable[dict]],
    metadata: dict | Callable[[], dict],
) -> dict:
    """Publish a complete export directory, refusing to overwrite any existing path."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".indo-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        files = {}
        for filename, records in datasets.items():
            if Path(filename).name != filename or not filename.endswith(".jsonl"):
                raise ValueError("Bundle file names must be simple .jsonl names")
            count = 0
            digest = hashlib.sha256()
            with (staging / filename).open("wb") as handle:
                for record in records:
                    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    encoded = line.encode()
                    handle.write(encoded)
                    digest.update(encoded)
                    count += 1
            files[filename] = {"records": count, "sha256": digest.hexdigest()}
        manifest = {
            "schema_version": 1,
            "package_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata() if callable(metadata) else metadata),
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output already exists: {output}")
        os.rename(staging, output)
    return manifest
