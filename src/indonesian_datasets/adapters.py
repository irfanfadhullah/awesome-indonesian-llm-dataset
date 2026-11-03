"""Lossless source adapters. Unknown task schemas are preserved, never guessed."""

import base64
import bz2
import codecs
import csv
import dataclasses
import gzip
import hashlib
import io
import json
import lzma
import math
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import date, datetime
from itertools import islice
from pathlib import Path

from .catalog import get_source
from .loaders import MAX_DOWNLOAD_BYTES, _download, load_dataset

FORMATS = ("csv", "csv_rows", "tsv_rows", "json", "jsonl", "lines", "conll", "conllu", "parquet")


class BoundedReader(io.RawIOBase):
    """Count decompressed bytes, not just the size of the compressed download."""

    def __init__(self, stream, limit: int):
        self.stream, self.limit, self.count = stream, limit, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.stream.read(min(len(buffer), self.limit - self.count + 1))
        self.count += len(block)
        if self.count > self.limit:
            raise ValueError(f"Decompressed input exceeds {self.limit} bytes")
        buffer[: len(block)] = block
        return len(block)


def json_safe(value):
    """Encode media bytes explicitly and normalize missing floats to JSON null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return {"_encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (date, datetime)):
        return {"_encoding": "iso8601", "data": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Unsupported record value {type(value).__name__}; provide an explicit adapter")


def _conll(handle, *, ud: bool) -> Iterator[dict]:
    rows, comments = [], []
    for number, line in enumerate(handle, 1):
        line = line.rstrip("\r\n")
        if not line:
            if rows:
                yield {
                    "rows": rows,
                    "comments": comments,
                    "tokens": [r[1] if ud else r[0] for r in rows],
                    **({} if ud else {"tags": [r[-1] for r in rows]}),
                }
                rows, comments = [], []
        elif line.startswith("#"):
            comments.append(line)
        else:
            columns = line.split("\t")
            if (ud and len(columns) != 10) or (not ud and len(columns) != 2):
                raise ValueError(f"Invalid {'CoNLL-U' if ud else 'token/tag'} row at line {number}")
            rows.append(columns)
    if rows:
        yield {
            "rows": rows,
            "comments": comments,
            "tokens": [r[1] if ud else r[0] for r in rows],
            **({} if ud else {"tags": [r[-1] for r in rows]}),
        }


def iter_file(
    path: str | Path,
    *,
    format: str,
    source_name: str | None = None,
    max_uncompressed_bytes: int = 256 * 1024 * 1024,
    encoding: str = "utf-8-sig",
) -> Iterator[dict]:
    if format not in FORMATS:
        raise ValueError(f"Unsupported format {format!r}; choose from {FORMATS}")
    if max_uncompressed_bytes <= 0:
        raise ValueError("max_uncompressed_bytes must be positive")
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise ValueError(f"Unknown source encoding {encoding!r}") from exc
    path = Path(path)
    if format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError('Install Parquet support: pip install -e ".[hf]"') from exc
        total = 0
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=64):
                total += batch.nbytes
                if total > max_uncompressed_bytes:
                    raise ValueError("Decoded Parquet exceeds the uncompressed byte limit")
                yield from batch.to_pylist()
        return
    filename = (source_name or str(path)).lower().split("?", 1)[0]
    with ExitStack() as stack:
        stream = stack.enter_context(path.open("rb"))
        if filename.endswith(".gz"):
            stream = stack.enter_context(gzip.GzipFile(fileobj=stream))
        elif filename.endswith(".bz2"):
            stream = stack.enter_context(bz2.BZ2File(stream))
        elif filename.endswith(".xz"):
            stream = stack.enter_context(lzma.LZMAFile(stream))
        elif filename.endswith(".zst"):
            try:
                import zstandard
            except ImportError as exc:
                raise ImportError(
                    'Install Zstandard support: pip install -e ".[compression]"'
                ) from exc
            stream = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(stream))
        bounded = BoundedReader(stream, max_uncompressed_bytes)
        handle = stack.enter_context(
            io.TextIOWrapper(io.BufferedReader(bounded), encoding=encoding, newline="")
        )
        if format in {"csv", "csv_rows", "tsv_rows"}:
            reader = (
                csv.DictReader(handle)
                if format == "csv"
                else csv.reader(handle, delimiter="\t" if format == "tsv_rows" else ",")
            )
            for row in reader:
                if isinstance(row, dict) and (
                    None in row or any(value is None for value in row.values())
                ):
                    raise ValueError(f"Malformed CSV near line {reader.line_num}")
                yield row if isinstance(row, dict) else {"columns": row}
        elif format in {"conll", "conllu"}:
            yield from _conll(handle, ud=format == "conllu")
        elif format == "json":
            value = json.load(handle)
            if isinstance(value, list):
                for row in value:
                    yield row if isinstance(row, dict) else {"value": row}
            elif isinstance(value, dict):
                # Do not flatten SQuAD/nested documents or invent train/test boundaries.
                yield value
            else:
                raise ValueError("JSON source must contain an object or array")
        else:
            for number, line in enumerate(handle, 1):
                if format == "lines":
                    yield {"text": line.rstrip("\r\n")}
                elif line.strip():
                    try:
                        value = json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"Invalid JSONL at line {number}: {exc}") from exc
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL line {number} is not an object")
                    yield value


def _no_decode(feature):
    """Keep nested audio/image fields as paths/bytes instead of decoding or fetching media."""
    if isinstance(feature, dict):
        return {key: _no_decode(value) for key, value in feature.items()}
    if isinstance(feature, list):
        return [_no_decode(value) for value in feature]
    if dataclasses.is_dataclass(feature):
        if hasattr(feature, "decode"):
            return dataclasses.replace(feature, decode=False)
        if hasattr(feature, "feature"):
            return dataclasses.replace(feature, feature=_no_decode(feature.feature))
    return feature


def source_variants(entry: dict) -> list[dict]:
    if entry["backend"] == "registered":
        return [{"config": "default", "split": split} for split in entry["splits"]]
    if entry.get("variants"):
        return entry["variants"]
    if entry.get("download_url"):
        return [
            {
                "config": "default",
                "split": "all",
                "urls": [entry["download_url"]],
                "format": entry.get("format"),
            }
        ]
    return []


def choose_variant(entry: dict, config: str | None, split: str | None) -> dict:
    variants = source_variants(entry)
    configs = sorted({v["config"] for v in variants})
    if config is None:
        if len(configs) != 1:
            raise ValueError(f"Choose an explicit config for {entry['id']}: {configs}")
        config = configs[0]
    choices = [
        v for v in variants if v["config"] == config and (split is None or v["split"] == split)
    ]
    if len(choices) != 1:
        raise ValueError(
            f"Choose an explicit valid split/config for {entry['id']}; "
            f"available: {[(v['config'], v['split']) for v in variants]}"
        )
    return choices[0]


def iter_source(
    entry: dict,
    *,
    config: str | None = None,
    split: str | None = None,
    limit: int | None = 1000,
    cache_dir: str | Path | None = None,
    max_download_bytes: int = MAX_DOWNLOAD_BYTES,
    max_uncompressed_bytes: int = 256 * 1024 * 1024,
) -> Iterator[dict]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if max_download_bytes <= 0 or max_uncompressed_bytes <= 0:
        raise ValueError("Byte limits must be positive")
    if limit == 0:
        return
    if entry.get("backend") not in {"registered", "file", "hf_parquet"}:
        raise ValueError(f"Source {entry['id']} has no row adapter; {entry['instructions']}")
    variant = choose_variant(entry, config, split)
    split, config = variant["split"], variant["config"]
    if entry["backend"] == "registered":
        yield from load_dataset(
            entry["id"],
            split=split,
            limit=limit,
            cache_dir=cache_dir,
            max_download_bytes=max_download_bytes,
        )
        return
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache/indonesian-datasets"
    if entry["backend"] == "hf_parquet":
        try:
            from datasets import Features
            from datasets import load_dataset as hf_load_dataset
        except ImportError as exc:
            raise ImportError('Install Hub support: pip install -e ".[hf]"') from exc
        dataset = hf_load_dataset(
            "parquet",
            data_files={split: variant["urls"]},
            split=split,
            streaming=True,
            cache_dir=str(cache / "huggingface"),
        )
        if dataset.features is None:
            dataset = dataset._resolve_features()
        if dataset.features is None:
            raise ValueError(
                "Feature schema unavailable; refusing possible automatic media decoding"
            )
        dataset = dataset.cast(Features(_no_decode(dict(dataset.features))))
        rows = iter(dataset)
    elif entry["backend"] == "file":

        def files():
            for url in variant["urls"]:
                path = _download(url, cache, max_download_bytes)
                yield from iter_file(
                    path,
                    format=variant["format"],
                    source_name=url,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                    encoding=entry.get("encoding", "utf-8-sig"),
                )

        rows = files()
    else:
        raise ValueError(f"Source {entry['id']} has no row adapter; {entry['instructions']}")
    try:
        for index, row in enumerate(islice(rows, limit)):
            data = json_safe(row)
            encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode()) > 16 * 1024 * 1024:
                raise ValueError(
                    "Record exceeds 16 MiB JSON export cap; use a dedicated media adapter"
                )
            identity = json.dumps(
                [
                    entry["id"],
                    entry.get("revision"),
                    entry.get("conversion_revision"),
                    config,
                    split,
                    index,
                    data,
                ],
                sort_keys=True,
                ensure_ascii=False,
            )
            yield {
                "id": hashlib.sha256(identity.encode()).hexdigest(),
                "source": entry["id"],
                "revision": entry.get("revision"),
                "conversion_revision": entry.get("conversion_revision"),
                "config": config,
                "split": split,
                "language": entry["language"],
                "task": entry["task"],
                "role": entry["role"],
                "data": data,
            }
    finally:
        close = getattr(rows, "close", None)
        if close:
            close()


def load_source(name: str, *, audit_path: str | Path | None = None, **options) -> Iterator[dict]:
    """Load a reviewed source adapter. Raw task schemas/media references stay under ``data``.

    Unlike the batch planner, this low-level API does not grant or check legal
    permission. Inspect source terms before calling it, as with load_dataset().
    """
    return iter_source(get_source(name, audit_path=audit_path), **options)
