"""Full source catalog, distinct from the small set of canonical task adapters."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path


def fingerprint(entry: object) -> str:
    return hashlib.sha256(
        json.dumps(entry, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def list_catalog(
    *,
    query: str = "",
    task: str | None = None,
    language: str | None = None,
    resource_type: str | None = None,
    audit_path: str | Path | None = None,
    include_audit: bool = True,
) -> list[dict]:
    package = files("indonesian_datasets")
    entries = json.loads(package.joinpath("catalog.json").read_text(encoding="utf-8"))
    snapshot = Path(audit_path) if audit_path else package.joinpath("audit.json")
    audit = (
        json.loads(snapshot.read_text(encoding="utf-8"))
        if include_audit and snapshot.is_file()
        else {}
    )
    verified = audit.get("sources", {})
    result = []
    for entry in entries:
        observation = verified.get(entry["id"], {})
        if observation.get("catalog_fingerprint") == fingerprint(entry):
            documented_access = entry["access"]
            entry.update(observation.get("updates", {}))
            if documented_access in {"permission_required", "credentials_required", "gated"}:
                # Readable metadata/files never waive an explicit upstream access requirement.
                entry["access"] = documented_access
            entry["audit"] = {k: v for k, v in observation.items() if k != "updates"}
        if (
            query.casefold() in json.dumps(entry, ensure_ascii=False).casefold()
            and (task is None or entry["task"] == task)
            and (language is None or entry["language"] == language)
            and (resource_type is None or entry["resource_type"] == resource_type)
        ):
            result.append(entry)
    return result


def get_source(name: str, *, audit_path: str | Path | None = None) -> dict:
    for entry in list_catalog(audit_path=audit_path):
        if entry["id"] == name:
            return entry
    raise ValueError(f"Unknown source {name!r}; use 'indo-datasets catalog' to find source IDs")


def coverage(*, audit_path: str | Path | None = None) -> dict:
    entries = list_catalog(audit_path=audit_path)
    counts: dict[str, dict[str, int]] = {}
    for field in ["resource_type", "backend", "access", "availability", "license"]:
        counts[field] = {}
        for entry in entries:
            value = entry.get(field) or "none"
            counts[field][value] = counts[field].get(value, 0) + 1
    return {
        "source_records": len(entries),
        "counts": counts,
        "note": (
            "Source records are not unique datasets. "
            "Configurations, mirrors and references are explicit."
        ),
    }
