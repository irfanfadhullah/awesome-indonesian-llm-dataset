"""Conservative, task-independent cleaning. Destructive transformations are opt-in."""

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field


def clean_text(text: str, *, lowercase: bool = False, remove_urls: bool = False) -> str:
    """Normalize NFC and whitespace; preserve accents, emoji, punctuation and negation."""
    if not isinstance(text, str):
        raise TypeError("clean_text expects a string")
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width space and BOM, but keep ZWJ/ZWNJ used by scripts and emoji.
    text = text.replace("\u200b", "").replace("\ufeff", "")
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text.lower() if lowercase else text


@dataclass
class Pipeline:
    """Clean selected fields, reject invalid lengths, and optionally deduplicate.

    Counts reset for each call to ``process``. Deduplication uses an in-memory
    digest set (O(unique records)), not full texts. It is exact, not fuzzy.
    """

    text_fields: tuple[str, ...] = ("text",)
    lowercase: bool = False
    remove_urls: bool = False
    min_chars: int = 1
    max_chars: int | None = None
    deduplicate: bool = False
    dedup_fields: tuple[str, ...] | None = None
    stats: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.text_fields:
            raise ValueError("text_fields cannot be empty")
        if self.dedup_fields is not None and not self.dedup_fields:
            raise ValueError("dedup_fields cannot be empty")
        if self.min_chars < 0 or (self.max_chars is not None and self.max_chars < self.min_chars):
            raise ValueError("Invalid character length bounds")

    def config(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "stats"}

    def process(self, records: Iterable[dict]) -> Iterator[dict]:
        self.stats = dict(input=0, output=0, invalid=0, too_short=0, too_long=0, duplicates=0)
        seen: set[bytes] = set()
        for record in records:
            self.stats["input"] += 1
            if not isinstance(record, dict) or any(
                not isinstance(record.get(key), str) for key in self.text_fields
            ):
                self.stats["invalid"] += 1
                continue
            cleaned = dict(record)
            for key in self.text_fields:
                cleaned[key] = clean_text(
                    record[key], lowercase=self.lowercase, remove_urls=self.remove_urls
                )
            if any(len(cleaned[key]) < self.min_chars for key in self.text_fields):
                self.stats["too_short"] += 1
                continue
            if self.max_chars is not None and any(
                len(cleaned[key]) > self.max_chars for key in self.text_fields
            ):
                self.stats["too_long"] += 1
                continue
            if self.deduplicate:
                keys = self.dedup_fields or self.text_fields
                if any(key not in cleaned for key in keys):
                    raise ValueError("A deduplication field is missing from a record")
                value = json.dumps(
                    [cleaned[key] for key in keys],
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                digest = hashlib.sha256(value.encode()).digest()
                if digest in seen:
                    self.stats["duplicates"] += 1
                    continue
                seen.add(digest)
            self.stats["output"] += 1
            yield cleaned


def split_records(
    records: Iterable[dict],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    group_fields: tuple[str, ...] = ("text",),
) -> dict[str, list[dict]]:
    """Content-hash split for NEW corpora, not existing benchmark splits.

    Equal group values always land in the same split, independent of input
    order. Ratios are approximate, not exact or stratified. Returns lists in
    memory; group by document/user/parallel ID to avoid related-example leakage.
    Selected grouping fields must exist; nested paths are not supported.
    """
    if (
        len(ratios) != 3
        or any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios)
        or not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("ratios must contain three finite non-negative values summing to 1")
    if not group_fields:
        raise ValueError("group_fields cannot be empty")
    output: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for record in records:
        if any(key not in record or record[key] is None for key in group_fields):
            raise ValueError(f"Every record must contain non-null group fields {group_fields}")
        key = json.dumps(
            [seed, [record[field] for field in group_fields]],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        # Use 52 bits so the floating-point score stays strictly below 1.
        score = int(hashlib.sha256(key.encode()).hexdigest()[:13], 16) / (16**13)
        name = (
            "train"
            if score < ratios[0]
            else "validation"
            if score < ratios[0] + ratios[1]
            else "test"
        )
        output[name].append({**record, "split": name})
    return output
