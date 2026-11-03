"""Explicit policy-aware plans and resumable, failure-isolated source exports."""

import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .adapters import iter_source, source_variants
from .catalog import fingerprint
from .loaders import _download
from .storage import file_sha256, save_json, write_bundle

BAD_AVAILABILITY = {
    "not_found",
    "disabled",
    "access_denied",
    "http_error",
    "check_failed",
    "rate_limited",
}


def plan_sources(
    entries: list[dict],
    *,
    purpose: str = "research",
    allow_unknown_license: bool = False,
    allow_unpinned: bool = False,
    include_evaluation: bool = False,
    all_configs: bool = False,
    config: str | None = None,
    limit: int | None = 1000,
    max_download_bytes: int = 100 * 1024 * 1024,
    max_uncompressed_bytes: int = 256 * 1024 * 1024,
) -> dict:
    if purpose not in {"research", "commercial"}:
        raise ValueError("purpose must be research or commercial")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    if min(max_download_bytes, max_uncompressed_bytes) <= 0:
        raise ValueError("Byte limits must be positive")
    if config is not None and all_configs:
        raise ValueError("config and all_configs are mutually exclusive")
    jobs, decisions = [], []
    for entry in entries:
        reason = None
        license_value = entry.get("license", "unknown").lower()
        if entry["resource_type"] != "dataset_source":
            reason = "not_a_dataset_source"
        elif entry["access"] not in {"public", "public_unverified", "unverified"}:
            reason = entry["access"]
        elif entry["availability"] in BAD_AVAILABILITY:
            reason = entry["availability"]
        elif entry.get("backend") not in {"registered", "file", "hf_parquet", "artifact"}:
            reason = "adapter_or_audit_required"
        elif entry["role"] == "evaluation" and not include_evaluation:
            reason = "evaluation_excluded"
        elif purpose == "commercial" and any(
            term in license_value for term in ["-nc", "non-commercial", "research-only"]
        ):
            reason = "noncommercial_restriction"
        elif (
            license_value in {"unknown", "other", "", "none", "research-only"}
            and not allow_unknown_license
        ):
            reason = "license_review_required"
        elif not entry.get("revision") and not allow_unpinned:
            reason = "unpinned_source"
        variants = source_variants(entry) if entry.get("backend") else []
        configs = sorted({v["config"] for v in variants})
        if reason is None and len(configs) > 1 and config is None and not all_configs:
            reason = "configuration_required"
        if reason is None and config is not None and config not in configs:
            reason = "configuration_not_available"
        decision = {
            "id": entry["id"],
            "name": entry["name"],
            "status": "skipped" if reason else "planned",
            "reason": reason,
            "instructions": entry["instructions"],
            "license": entry["license"],
            "access": entry["access"],
        }
        decisions.append(decision)
        if reason:
            continue
        selected = [v for v in variants if config is None or v["config"] == config]
        if not selected:
            decision.update(status="skipped", reason="no_data_variants")
        for variant in selected:
            if not include_evaluation and variant["split"].lower() not in {
                "train",
                "training",
                "all",
            }:
                decisions.append(
                    {
                        "id": entry["id"],
                        "config": variant["config"],
                        "split": variant["split"],
                        "status": "skipped",
                        "reason": "held_out_split_excluded",
                    }
                )
                continue
            sizes = variant.get("shard_sizes", [])
            known_size = max(sizes) if sizes else entry.get("content_length")
            if known_size and known_size > max_download_bytes:
                decisions.append(
                    {
                        "id": entry["id"],
                        "config": variant["config"],
                        "split": variant["split"],
                        "status": "skipped",
                        "reason": "source_file_exceeds_byte_cap",
                        "size_bytes": known_size,
                    }
                )
                continue
            key = fingerprint(
                {"id": entry["id"], "config": variant["config"], "split": variant["split"]}
            )[:16]
            label = re.sub(
                r"[^a-zA-Z0-9_-]",
                "-",
                entry["id"] + "-" + variant["config"] + "-" + variant["split"],
            )[:100]
            jobs.append(
                {
                    "key": key,
                    "directory": label + "-" + key,
                    "source": entry,
                    "config": variant["config"],
                    "split": variant["split"],
                }
            )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "purpose": purpose,
            "allow_unknown_license": allow_unknown_license,
            "allow_unpinned": allow_unpinned,
            "include_evaluation": include_evaluation,
            "all_configs": all_configs,
            "config": config,
            "limit": limit,
            "max_download_bytes": max_download_bytes,
            "max_uncompressed_bytes": max_uncompressed_bytes,
        },
        "jobs": jobs,
        "decisions": decisions,
        "summary": {
            "source_records": len(entries),
            "download_jobs": len(jobs),
            "skipped_decisions": sum(d["status"] == "skipped" for d in decisions),
        },
        "warnings": [
            "This plan is not legal permission; review original terms.",
            "Different sources, configurations and official splits remain separate.",
            "Row limits do not cap Hub network transfers; the planner rejects known "
            "over-cap shards, but runtime range requests and buffering are controlled "
            "by the upstream library.",
            "Archives are opaque artifacts, never extracted or treated as training examples.",
        ],
    }


def verify_bundle(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("Export directory must not be a symlink")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if Path(name).name != name or (path / name).is_symlink():
            raise ValueError("Unsafe path in export manifest")
        if file_sha256(path / name) != info["sha256"]:
            raise ValueError(f"Export integrity check failed: {path / name}")
    return manifest


def _artifact(entry: dict, output: Path, *, cache_dir: Path, max_bytes: int) -> dict:
    path = _download(entry["download_url"], cache_dir, max_bytes)
    with tempfile.TemporaryDirectory(prefix=".artifact-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        shutil.copyfile(path, staging / "artifact.bin")
        manifest = {
            "schema_version": 1,
            "operation": "artifact",
            "source": entry,
            "files": {"artifact.bin": {"bytes": path.stat().st_size, "sha256": file_sha256(path)}},
        }
        save_json(staging / "manifest.json", manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(str(output))
        os.rename(staging, output)
    return manifest


def run_batch(
    plan: dict,
    output: str | Path,
    *,
    cache_dir: str | Path | None = None,
    resume: bool = False,
) -> dict:
    if plan.get("schema_version") != 1 or not isinstance(plan.get("jobs"), list):
        raise ValueError("Invalid plan schema")
    keys = [job["key"] for job in plan["jobs"]]
    directories = [job["directory"] for job in plan["jobs"]]
    if len(keys) != len(set(keys)) or len(directories) != len(set(directories)):
        raise ValueError("Duplicate job keys or directories in plan")
    for job in plan["jobs"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job["directory"]):
            raise ValueError("Unsafe job directory in plan")
    output = Path(output)
    plan_hash = fingerprint(plan)
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache/indonesian-datasets"
    if output.is_symlink():
        raise ValueError("Batch output cannot be a symlink")
    if output.exists() and not resume:
        raise FileExistsError("Batch output already exists; use --resume with the exact same plan")
    if resume:
        old = json.loads((output / "batch-report.json").read_text(encoding="utf-8"))
        if old["plan_sha256"] != plan_hash:
            raise ValueError("Resume requires the identical frozen plan")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".batch.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        report = {
            "schema_version": 1,
            "plan_sha256": plan_hash,
            "policy": plan["policy"],
            "decisions": plan["decisions"],
            "results": [],
        }
        save_json(output / "batch-report.json", report, overwrite=True)
        for job in plan["jobs"]:
            entry = job["source"]
            destination = output / job["directory"]
            result = {
                "key": job["key"],
                "id": entry["id"],
                "config": job["config"],
                "split": job["split"],
                "directory": job["directory"],
            }
            try:
                if destination.exists():
                    if not resume:
                        raise FileExistsError(str(destination))
                    manifest = verify_bundle(destination)
                    if manifest.get("source") != entry:
                        raise ValueError("Existing export source does not match the frozen plan")
                    result["status"] = "verified_existing"
                else:
                    policy = plan["policy"]
                    if entry["backend"] == "artifact":
                        manifest = _artifact(
                            entry,
                            destination,
                            cache_dir=cache,
                            max_bytes=policy["max_download_bytes"],
                        )
                    else:
                        records = iter_source(
                            entry,
                            config=job["config"],
                            split=job["split"],
                            limit=policy["limit"],
                            cache_dir=cache,
                            max_download_bytes=policy["max_download_bytes"],
                            max_uncompressed_bytes=policy["max_uncompressed_bytes"],
                        )
                        manifest = write_bundle(
                            destination,
                            {"data.jsonl": records},
                            {
                                "operation": "batch-download",
                                "source": entry,
                                "config": job["config"],
                                "split": job["split"],
                                "plan_sha256": plan_hash,
                                "policy": policy,
                            },
                        )
                    result["status"] = "completed"
                result["files"] = manifest["files"]
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["results"].append(result)
            report["summary"] = dict(Counter(r["status"] for r in report["results"]))
            save_json(output / "batch-report.json", report, overwrite=True)
        report.setdefault("summary", {})
        save_json(output / "batch-report.json", report, overwrite=True)
        return report
    finally:
        lock.unlink(missing_ok=True)
