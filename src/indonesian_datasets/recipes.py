"""Explicit task mappings and validation, with alignment-sensitive text left untouched."""

from collections.abc import Iterable, Iterator

from .adapters import json_safe
from .catalog import fingerprint
from .preprocessing import clean_text

REQUIRED = {
    "classification": ("text", "label"),
    "nli": ("premise", "hypothesis", "label"),
    "summarization": ("text", "summary"),
    "translation": ("source_text", "target_text"),
    "qa-extractive": ("context", "question", "answers"),
    "token-classification": ("tokens", "labels"),
    "chat": ("messages",),
    "audio": ("audio", "transcript"),
    "image-caption": ("image", "caption"),
    "knowledge-graph": ("head", "relation", "tail"),
    "lexicon": ("term", "definition"),
    "raw": ("data",),
}


def _lookup(record: dict, path: str):
    value = record
    for key in path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdecimal() and int(key) < len(value):
            value = value[int(key)]
        else:
            raise ValueError(f"Missing recipe field path {path!r}")
    return value


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_prepared(task: str, row: dict) -> None:
    if task not in REQUIRED:
        raise ValueError(f"Unknown recipe task {task!r}; available: {sorted(REQUIRED)}")
    if any(key not in row for key in REQUIRED[task]):
        raise ValueError(f"Task {task} requires fields {REQUIRED[task]}")
    text_fields = {
        "classification": ("text",),
        "nli": ("premise", "hypothesis"),
        "summarization": ("text", "summary"),
        "translation": ("source_text", "target_text"),
        "qa-extractive": ("context", "question"),
        "audio": ("transcript",),
        "image-caption": ("caption",),
        "knowledge-graph": ("head", "relation", "tail"),
        "lexicon": ("term", "definition"),
    }.get(task, ())
    if any(not isinstance(row[key], str) or not row[key].strip() for key in text_fields):
        raise ValueError("Required text fields must be non-empty strings")
    if task in {"classification", "nli"} and (
        row["label"] is None or not isinstance(row["label"], (str, int, list, dict))
    ):
        raise ValueError("Labels must be strings, integers, lists or aspect mappings")
    if task == "token-classification":
        if (
            not _strings(row["tokens"])
            or not row["tokens"]
            or not isinstance(row["labels"], list)
            or len(row["tokens"]) != len(row["labels"])
        ):
            raise ValueError("tokens and labels must be non-empty aligned lists")
        if not all(isinstance(label, (str, int)) for label in row["labels"]):
            raise ValueError("Token labels must be strings or integers")
    if task == "qa-extractive":
        answers = row["answers"]
        if (
            not isinstance(answers, dict)
            or not _strings(answers.get("text"))
            or not isinstance(answers.get("answer_start"), list)
            or len(answers["text"]) != len(answers["answer_start"])
        ):
            raise ValueError("answers must contain aligned text and answer_start lists")
        for text, start in zip(answers["text"], answers["answer_start"], strict=True):
            if (
                not text
                or isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or row["context"][start : start + len(text)] != text
            ):
                raise ValueError("Answer text does not match its original context offset")
    if task == "chat":
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("Each chat message needs a supported role and string content")
    for media in {"audio": ("audio",), "image-caption": ("image",)}.get(task, ()):
        if not isinstance(row[media], (str, dict)) or not row[media]:
            raise ValueError("Media must be a non-empty path/reference or undecoded bytes object")
    if task == "raw" and not isinstance(row["data"], dict):
        raise ValueError("Raw recipes require a data object")


def prepare_records(
    records: Iterable[dict], recipe: dict, stats: dict | None = None
) -> Iterator[dict]:
    if recipe.get("schema_version") != 1 or recipe.get("task") not in REQUIRED:
        raise ValueError("Recipe requires schema_version=1 and a supported task")
    mapping = recipe.get("fields")
    if not isinstance(mapping, dict) or any(
        not isinstance(path, str) or not path for path in mapping.values()
    ):
        raise ValueError("Recipe fields must map target names to explicit source paths")
    if any(key not in mapping for key in REQUIRED[recipe["task"]]):
        raise ValueError("Recipe is missing required target fields")
    cleaning = recipe.get("clean_fields", [])
    if not isinstance(cleaning, list) or any(key not in mapping for key in cleaning):
        raise ValueError("clean_fields must be a list of mapped target fields")
    if cleaning and recipe["task"] in {
        "qa-extractive",
        "token-classification",
        "chat",
        "audio",
        "image-caption",
        "raw",
    }:
        raise ValueError(
            "Generic cleaning is prohibited for alignment-sensitive/media/chat recipes"
        )
    counters = stats if stats is not None else {}
    counters.update(input=0, output=0)
    recipe_hash = fingerprint(recipe)
    for number, record in enumerate(records, 1):
        counters["input"] += 1
        try:
            prepared = {target: _lookup(record, source) for target, source in mapping.items()}
            for key in cleaning:
                prepared[key] = clean_text(prepared[key])
            validate_prepared(recipe["task"], prepared)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Record {number}: {exc}") from exc
        counters["output"] += 1
        yield {
            "id": record.get("id", fingerprint(json_safe(record))),
            "source": record.get("source", "local"),
            "split": record.get("split", "unspecified"),
            "task": recipe["task"],
            "recipe_sha256": recipe_hash,
            "prepared": json_safe(prepared),
            "original": record,
        }
