"""Packaged loader definitions and a separate inventory of documentation links."""

import json
from importlib.resources import files


def _read(name: str) -> list[dict]:
    return json.loads(files("indonesian_datasets").joinpath(name).read_text(encoding="utf-8"))


def list_datasets(
    *, task: str | None = None, language: str | None = None, query: str = ""
) -> list[dict]:
    """Return registered loaders, not every resource mentioned in the awesome list."""
    return [
        entry
        for entry in _read("registry.json")
        if (task is None or entry["task"] == task)
        and (language is None or entry["language"] == language)
        and query.casefold() in json.dumps(entry, ensure_ascii=False).casefold()
    ]


def get_dataset(name: str) -> dict:
    for entry in list_datasets():
        if entry["id"] == name:
            return entry
    available = ", ".join(entry["id"] for entry in list_datasets())
    raise ValueError(f"Unknown dataset {name!r}. Available: {available}")


def list_sources(*, query: str = "", category: str | None = None) -> list[dict]:
    """Search all indexed links. An indexed link does not imply loader support."""
    return [
        entry
        for entry in _read("resources.json")
        if query.casefold() in json.dumps(entry, ensure_ascii=False).casefold()
        and (category is None or entry["category"] == category)
    ]
