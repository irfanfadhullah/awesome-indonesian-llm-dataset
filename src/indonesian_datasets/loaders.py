"""Lazy loaders: bounded, cached CSV/TSV downloads and optional Hub streaming."""

import csv
import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from urllib.request import Request, urlopen

from .registry import get_dataset

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/indonesian-datasets``, else ``~/.cache/indonesian-datasets``.

    Shared by every backend so loaders, adapters and batches reuse one cache.
    An empty ``XDG_CACHE_HOME`` counts as unset, per the XDG specification.
    """
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "indonesian-datasets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, cache_dir: Path, max_bytes: int) -> Path:
    if max_bytes <= 0:
        raise ValueError("max_download_bytes must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / hashlib.sha256(url.encode()).hexdigest()
    checksum = path.with_suffix(".sha256")
    if path.exists() and checksum.exists():
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Cached source exceeds {max_bytes} bytes")
        if _sha256(path) == checksum.read_text(encoding="ascii").strip():
            return path
        warnings.warn(f"Cache integrity check failed for {url}; downloading again", stacklevel=2)

    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "awesome-indonesian-datasets/0.2"})
        with urlopen(request, timeout=30) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError(f"Expected a data file but received an HTML page: {url}")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > max_bytes:
                raise ValueError(f"Source exceeds {max_bytes} bytes: {url}")
            with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
                temporary = Path(handle.name)
                size = 0
                while block := response.read(65536):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError(f"Source exceeds {max_bytes} bytes: {url}")
                    handle.write(block)
        digest = _sha256(temporary)
        os.replace(temporary, path)
        # The source is pinned; the checksum detects local corruption, not upstream authenticity.
        checksum.write_text(digest + "\n", encoding="ascii")
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _csv_rows(entry: dict, split: str, cache_dir: Path, max_bytes: int) -> Iterator[dict]:
    url = entry["url_template"].format(revision=entry["revision"], file=entry["files"][split])
    path = _download(url, cache_dir, max_bytes)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        options = {"delimiter": entry.get("delimiter", ",")}
        if "columns" in entry:
            # IndoNLU TSV is unquoted: literal quotation marks are part of the text.
            options["quoting"] = csv.QUOTE_NONE
        reader = csv.DictReader(handle, fieldnames=entry.get("columns"), **options)
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed row near line {reader.line_num} in {entry['id']}")
            yield row


def _hub_rows(entry: dict, split: str, cache_dir: Path) -> Iterator[dict]:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise ImportError('Install Hub support: pip install -e ".[hf]"') from exc
    # Never execute remote dataset scripts. HF_TOKEN is handled by the upstream library.
    yield from hf_load_dataset(
        entry["repo_id"],
        name=entry.get("config"),
        revision=entry["revision"],
        split=split,
        streaming=True,
        cache_dir=str(cache_dir / "huggingface"),
    )


def _canonicalize(row: dict, entry: dict, split: str) -> dict:
    mapped = {}
    for target, source in entry["fields"].items():
        if source not in row:
            raise ValueError(f"Schema changed for {entry['id']}: missing column {source!r}")
        value = row[source]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid {source!r} in {entry['id']}: expected non-empty text")
        mapped[target] = value
    raw = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
    identity = f"{entry['id']}\0{entry['revision']}\0{split}\0{raw}"
    return {
        "id": hashlib.sha256(identity.encode()).hexdigest(),
        **mapped,
        "language": entry["language"],
        "source": entry["id"],
        "split": split,
        "revision": entry["revision"],
        "metadata": {
            key: value for key, value in row.items() if key not in entry["fields"].values()
        },
    }


def load_dataset(
    name: str,
    *,
    split: str = "train",
    limit: int | None = None,
    cache_dir: str | Path | None = None,
    max_download_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Iterator[dict]:
    """Yield canonical records without modifying upstream train/validation/test boundaries.

    CSV/TSV sources are cached as complete files (100 MiB cap by default). Hub
    sources stream; the byte cap does not apply to that backend. ``limit`` caps
    rows, not network bytes. No network access occurs until iteration begins.
    """
    entry = get_dataset(name)
    splits = entry.get("splits", list(entry.get("files", {})))
    if split not in splits:
        raise ValueError(f"Invalid split {split!r} for {name}; choose from {', '.join(splits)}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if max_download_bytes <= 0:
        raise ValueError("max_download_bytes must be positive")
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    def records() -> Iterator[dict]:
        if limit == 0:
            return
        if entry["license"] == "unknown":
            warnings.warn(
                f"{name}: license unverified; review {entry['license_url']}", stacklevel=2
            )
        rows = (
            _csv_rows(entry, split, cache, max_download_bytes)
            if entry["backend"] == "csv"
            else _hub_rows(entry, split, cache)
        )
        try:
            for row in islice(rows, limit):
                yield _canonicalize(row, entry, split)
        finally:
            rows.close()

    return records()
