"""Read-only source auditing. No data shards or remote Python scripts are downloaded."""

import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .catalog import fingerprint

MAX_METADATA_BYTES = 16 * 1024 * 1024
_GIT_TREES: dict[tuple[str, str], dict] = {}


@lru_cache(maxsize=128)
def _git_snapshot(repo: str, ref: str) -> dict:
    """Optional public-Git fallback: fetch commit/tree objects, never file contents or hooks."""
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", ref)
        or ref.startswith("-")
    ):
        raise ValueError("Unsafe public Git reference")
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    with tempfile.TemporaryDirectory(prefix="indo-audit-git-") as temporary:
        base = ["git", "-C", temporary, "-c", "core.hooksPath=/dev/null"]

        def run(*args):
            result = subprocess.run(
                [*base, *args], env=env, capture_output=True, timeout=90, check=True
            )
            if len(result.stdout) > MAX_METADATA_BYTES:
                raise ValueError("Git metadata exceeds the audit cap")
            return result.stdout

        run("init", "--bare", "--quiet")
        run(
            "fetch",
            "--quiet",
            "--no-tags",
            "--filter=blob:none",
            "--depth=1",
            "https://github.com/" + repo + ".git",
            ref,
        )
        revision = run("rev-parse", "FETCH_HEAD").decode().strip()
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Invalid Git commit identity")
        # Do NOT request sizes (-l): that would lazily fetch every blob.
        listing = run("ls-tree", "-r", "-z", "FETCH_HEAD")
        items = []
        for record in listing.split(b"\0"):
            if not record:
                continue
            metadata, path = record.split(b"\t", 1)
            mode, kind, object_id = metadata.decode().split()
            if kind == "blob" and mode in {"100644", "100755"}:
                items.append({"path": path.decode("utf-8"), "type": "blob", "sha": object_id})
        tree = {"tree": items, "truncated": False, "metadata_method": "public_git_partial_fetch"}
        _GIT_TREES[(repo, revision)] = tree
        return {"revision": revision, "tree": tree}


@lru_cache(maxsize=128)
def _github_tree(repo: str, revision: str) -> dict:
    if (repo, revision) in _GIT_TREES:
        return _GIT_TREES[(repo, revision)]
    return _json(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1")


def _github_files(entry: dict, revision: str) -> dict:
    repo = entry["github_repo"]
    tree = _github_tree(repo, revision)
    if tree.get("truncated"):
        raise ValueError("GitHub tree is truncated; use an explicit directory/file adapter")
    directory = entry.get("github_directory", "").strip("/")
    variants = {}
    for item in tree.get("tree", []):
        name = item["path"]
        path = PurePosixPath(name)
        if item["type"] != "blob" or (directory and not name.startswith(directory + "/")):
            continue
        if path.suffix not in {".csv", ".tsv", ".jsonl", ".json", ".txt", ".conllu"}:
            continue
        if any(
            part.lower()
            in {
                "experiments",
                "results",
                "predictions",
                "models",
                "scripts",
                "tests",
                "examples",
                "configs",
                "boomer",
            }
            for part in path.parts
        ):
            continue
        if path.stem.lower() in {"requirements", "license", "readme", "changelog", "manifest"}:
            continue
        fmt = {
            ".csv": "csv_rows",
            ".tsv": "tsv_rows",
            ".jsonl": "jsonl",
            ".json": "json",
            ".txt": "lines",
            ".conllu": "conllu",
        }[path.suffix]
        config, split = str(path.with_suffix("")), "unspecified"
        evidence = "No official partition inferred: explicit file configuration only."
        if repo.lower() == "indonlp/nusa-writes" and name.startswith("data/"):
            match = re.fullmatch(r"(.+)-(train|valid|test)\.csv", path.name)
            if match:
                config, split = match[1], {"valid": "validation"}.get(match[2], match[2])
                evidence = "NusaWrites data/ naming convention inspected."
                fmt = "csv"
        elif repo.lower() == "ir-nlp-csui/indonli" and name.startswith("data/indonli/"):
            config, split, fmt = "default", {"val": "validation"}.get(path.stem, path.stem), "jsonl"
            evidence = "Original IndoNLI data/indonli partitions."
        elif repo.lower() == "universaldependencies/ud_indonesian-gsd" and path.suffix == ".conllu":
            config, split = (
                "default",
                {"dev": "validation"}.get(
                    path.stem.rsplit("-", 1)[-1], path.stem.rsplit("-", 1)[-1]
                ),
            )
            evidence = "UD CoNLL-U train/dev/test naming convention."
        elif entry.get("task") == "lexicon" and path.suffix in {".csv", ".tsv", ".txt"}:
            split = "all"
        key = (config, split, fmt)
        variant = variants.setdefault(
            key,
            {
                "config": config,
                "split": split,
                "format": fmt,
                "urls": [],
                "shard_sizes": [],
                "split_evidence": evidence,
            },
        )
        variant["urls"].append(
            f"https://raw.githubusercontent.com/{repo}/{revision}/{quote(name, safe='/')}"
        )
        variant["shard_sizes"].append(item.get("size", 0))
    return {
        "backend": "file" if variants else None,
        "revision": revision,
        "access": "public" if variants else "manual",
        "availability": "native_file_index_verified" if variants else "no_indexed_data_files",
        "variants": sorted(variants.values(), key=lambda v: (v["config"], v["split"])),
        "record_mode": "raw",
        "instructions": (
            "Versioned data-file candidates from the original repository; "
            "not an assertion of task-ready examples. Select file configs, review terms "
            "and use explicit task recipes. Unknown splits stay unspecified; "
            "no Python, pickle or archive extraction is executed."
        ),
    }


def _native_hub_files(info: dict, entry: dict) -> dict | None:
    """Use versioned native data files if an older dataset has no conversion branch."""
    repo = info.get("id", entry["repo_id"])
    revision = info["sha"]
    variants = {}
    for item in info.get("siblings", []):
        name = item["rfilename"]
        path = PurePosixPath(name)
        if path.suffix not in {".csv", ".tsv", ".jsonl", ".parquet"}:
            continue
        if any(part in {"tests", "examples"} for part in path.parts):
            continue
        match = re.match(r"(train|validation|test|dev)(?:[-_.]|$)", path.name)
        split = (
            match[1]
            if match
            else "evaluation"
            if entry.get("role") == "evaluation"
            else "unspecified"
        )
        config = str(path.parent) if match else str(path.with_suffix(""))
        if config in {".", "data"}:
            config = "default"
        fmt = {".csv": "csv_rows", ".tsv": "tsv_rows", ".parquet": "parquet", ".jsonl": "jsonl"}[
            path.suffix
        ]
        key = (config, split, fmt)
        variant = variants.setdefault(
            key, {"config": config, "split": split, "format": fmt, "urls": [], "shard_sizes": []}
        )
        variant["urls"].append(
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(name, safe='/')}"
        )
        if item.get("size") is not None:
            variant["shard_sizes"].append(item["size"])
    if not variants:
        # An archive can still be fetched without interpreting or extracting it.
        archives = [
            item["rfilename"]
            for item in info.get("siblings", [])
            if item["rfilename"].endswith((".zip", ".tar.gz", ".tar.bz2"))
        ]
        if len(archives) == 1:
            return {
                "backend": "artifact",
                "access": "public",
                "download_url": (
                    f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
                    f"{quote(archives[0], safe='/')}"
                ),
                "record_mode": "artifact",
                "instructions": (
                    "Pinned archive, downloaded only as an opaque artifact. "
                    "No extraction or task conversion implied."
                ),
            }
        return None
    formats = {variant["format"] for variant in variants.values()}
    return {
        "backend": "hf_parquet" if formats == {"parquet"} else "file",
        "access": "public",
        "availability": "native_file_index_verified",
        "record_mode": "raw",
        "variants": sorted(variants.values(), key=lambda v: (v["config"], v["split"])),
        "instructions": (
            "Pinned native data files, no remote scripts. Unreviewed CSV/TSV headers "
            "are retained as raw rows. Unidentified splits remain 'unspecified' "
            "and are excluded from default training plans."
        ),
    }


def _json(url: str, *, timeout: int = 20) -> dict:
    headers = {"User-Agent": "awesome-indonesian-datasets/0.2"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and urlparse(url).hostname == "api.github.com":
        headers["Authorization"] = "Bearer " + token
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read(MAX_METADATA_BYTES + 1)
    if len(body) > MAX_METADATA_BYTES:
        raise ValueError("Metadata response exceeds the 16 MiB audit cap")
    return json.loads(body)


def _head(url: str) -> dict:
    request = Request(url, method="HEAD", headers={"User-Agent": "awesome-indonesian-datasets/0.2"})
    with urlopen(request, timeout=15) as response:
        return {
            "content_length": int(response.headers["Content-Length"])
            if response.headers.get("Content-Length")
            else None,
            "content_type": response.headers.get("Content-Type", ""),
            "resolved_url": response.url,
        }


def _hub(entry: dict) -> dict:
    repo = entry["repo_id"]
    endpoint = "https://huggingface.co/api/datasets/" + quote(repo, safe="/")
    info = _json(endpoint + "?blobs=true")
    card = info.get("cardData") or {}
    license_value = card.get("license", "unknown")
    if isinstance(license_value, list):
        license_value = ",".join(license_value)
    updates = {
        "revision": info["sha"],
        "resolved_repo_id": info.get("id", repo),
        "license": str(license_value),
        "license_evidence": [endpoint],
        "license_note": (
            "License declared by the Hub card, not independently established legal clearance. "
            "Original content rights still apply."
        ),
        "size_bytes_reported": info.get("usedStorage"),
        "availability": "metadata_available",
        "language": "mul",
    }
    language = card.get("language", [])
    if isinstance(language, str):
        language = [language]
    if len(language) == 1:
        updates["language"] = language[0]
    updates["languages_declared"] = language
    if info.get("gated"):
        updates.update(
            access="gated",
            backend=None,
            instructions=(
                "Accept the dataset's terms with your own account, then use an authorized "
                "local copy. This auditor does not bypass Hub gates."
            ),
        )
        return updates
    if info.get("disabled"):
        updates.update(access="unavailable", backend=None, availability="disabled")
        return updates
    # Converted Parquet is a data-only route even for legacy script-based datasets.
    # Pin the conversion commit separately; never misrepresent it as the original commit.
    try:
        parquet = _json(
            "https://datasets-server.huggingface.co/parquet?dataset=" + quote(repo, safe="")
        )
        converted = _json(endpoint + "/revision/refs%2Fconvert%2Fparquet")
    except (HTTPError, URLError, ValueError, KeyError) as exc:
        native = _native_hub_files(info, entry)
        if native:
            updates.update(native)
            return updates
        updates.update(
            backend=None,
            access="manual",
            instructions=(
                f"No verified data-only Parquet export: {type(exc).__name__}. "
                "Select an explicit native data file; remote scripts will not run."
            ),
        )
        return updates
    conversion_revision = converted["sha"]
    variants: dict[tuple[str, str], dict] = {}
    resolved_repo = info.get("id", repo)
    for shard in parquet.get("parquet_files", []):
        url = shard["url"]
        parsed = urlparse(url)
        if parsed.hostname != "huggingface.co" or "/resolve/" not in parsed.path:
            raise ValueError("Unexpected Parquet URL from dataset server")
        prefix = f"https://huggingface.co/datasets/{resolved_repo}/resolve/"
        # Do not accept external origins or signed URLs containing credentials.
        if not url.startswith(prefix) or parsed.query:
            raise ValueError("Parquet shard origin does not match the audited repository")
        rest = url[len(prefix) :]
        # Conversion branch may be encoded or literal; preserve only the file path.
        if rest.startswith("refs%2Fconvert%2Fparquet/"):
            filename = rest[len("refs%2Fconvert%2Fparquet/") :]
        elif rest.startswith("refs/convert/parquet/"):
            filename = rest[len("refs/convert/parquet/") :]
        else:
            raise ValueError("Unrecognized Parquet conversion reference")
        key = (shard["config"], shard["split"])
        variant = variants.setdefault(
            key,
            {
                "config": key[0],
                "split": key[1],
                "urls": [],
                "format": "parquet",
                "size_bytes": 0,
                "shard_sizes": [],
            },
        )
        variant["urls"].append(prefix + conversion_revision + "/" + filename)
        variant["size_bytes"] += shard.get("size", 0)
        variant["shard_sizes"].append(shard.get("size", 0))
    if not variants:
        updates.update(
            backend=None,
            access="manual",
            instructions="Dataset server returned no readable Parquet shards.",
        )
        return updates
    updates.update(
        backend="hf_parquet",
        access="public",
        availability="parquet_index_verified",
        conversion_revision=conversion_revision,
        variants=sorted(variants.values(), key=lambda v: (v["config"], v["split"])),
        record_mode="raw",
        instructions=(
            "Data-only Parquet streaming; media decoding disabled. "
            "Choose configurations explicitly for multilingual sources."
        ),
    )
    return updates


def _inspect(entry: dict, revisions: dict, *, git_fallback: bool = False) -> dict:
    backend = entry.get("backend")
    if backend == "hf_pending":
        return _hub(entry)
    if backend == "github_pending":
        key = (entry["github_repo"], entry["github_ref"])
        revision = revisions[key]
        if isinstance(revision, Exception):
            raise revision
        try:
            return _github_files(entry, revision)
        except HTTPError as exc:
            if git_fallback and exc.code in {403, 429}:
                _git_snapshot(entry["github_repo"], revision)
                return _github_files(entry, revision)
            raise
    if backend == "registered":
        definition = entry["definition"]
        if definition["backend"] == "huggingface":
            info = _json(
                "https://huggingface.co/api/datasets/"
                + quote(definition["repo_id"], safe="/")
                + "/revision/"
                + definition["revision"]
            )
            return {
                "availability": "pinned_metadata_verified",
                "access": "gated" if info.get("gated") else "public",
            }
        split = next(iter(definition["files"]))
        url = definition["url_template"].format(
            revision=definition["revision"], file=definition["files"][split]
        )
        return {"availability": "reachable", "first_file": _head(url)}
    if backend == "file_pending":
        key = (entry["github_repo"], entry["github_ref"])
        revision = revisions[key]
        if isinstance(revision, Exception):
            raise revision
        url = f"https://raw.githubusercontent.com/{entry['github_repo']}/{revision}/{entry['github_path']}"
        headers = _head(url)
        fmt = entry["format"]
        if fmt == "txt":
            fmt = "lines"
        if fmt == "tsv":
            # Unknown header conventions: preserve ALL rows as column arrays.
            fmt = "tsv_rows"
        if fmt == "csv":
            fmt = "csv_rows"
        return {
            "backend": "file",
            "download_url": url,
            "format": fmt,
            "revision": revision,
            "availability": "reachable",
            "access": "public",
            **headers,
        }
    if backend in {"file", "artifact"}:
        url = entry.get("download_url") or entry["variants"][0]["urls"][0]
        return {"availability": "reachable", **_head(url)}
    if entry.get("homepage"):
        headers = _head(entry["homepage"])
        return {"availability": "landing_page_reachable", "landing_page": headers}
    return {"availability": "no_download_url"}


def audit_sources(entries: list[dict], *, workers: int = 6, git_fallback: bool = False) -> dict:
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    # Resolve each shared GitHub ref once, before parallel HEAD requests.
    refs = sorted(
        {
            (e["github_repo"], e["github_ref"])
            for e in entries
            if e.get("backend") in {"file_pending", "github_pending"}
        }
    )
    revisions = {}

    def resolve(key):
        repo, ref = key
        try:
            return _json(f"https://api.github.com/repos/{repo}/commits/{quote(ref, safe='')}")[
                "sha"
            ]
        except HTTPError as exc:
            if git_fallback and exc.code in {403, 429}:
                return _git_snapshot(repo, ref)["revision"]
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(resolve, key): key for key in refs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                revisions[key] = future.result()
            except Exception as exc:
                revisions[key] = exc
    observations = {}

    def inspect(entry):
        observed = {
            "catalog_fingerprint": fingerprint(entry),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "evidence_url": entry.get("homepage"),
        }
        try:
            observed["updates"] = _inspect(entry, revisions, git_fallback=git_fallback)
        except HTTPError as exc:
            status = (
                "not_found"
                if exc.code in {404, 410}
                else "rate_limited"
                if exc.code == 403
                and exc.headers
                and exc.headers.get("X-RateLimit-Remaining") == "0"
                else "access_denied"
                if exc.code in {401, 403}
                else "rate_limited"
                if exc.code == 429
                else "http_error"
            )
            observed.update(error=f"HTTP {exc.code}", updates={"availability": status})
        except Exception as exc:
            # A timeout/rate limit is not evidence that a dataset does not exist.
            observed.update(
                error=f"{type(exc).__name__}: {exc}", updates={"availability": "check_failed"}
            )
        return entry["id"], observed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for source_id, observation in pool.map(inspect, entries):
            observations[source_id] = observation
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "metadata and HTTP headers only; no license approval, sample validation "
            "or full dataset downloads implied"
        ),
        "sources": observations,
    }
