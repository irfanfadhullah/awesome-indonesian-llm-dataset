"""Index Markdown links without mistaking every link for a downloadable dataset."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/indonesian_datasets/resources.json"


def build_inventory() -> list[dict]:
    by_url: dict[str, dict] = {}
    for path in sorted((ROOT / "data").glob("*.md")):
        category = path.stem
        heading = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                heading = line.lstrip("# ").strip()
            for label, url in re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", line):
                url = url.rstrip("/")
                host = urlparse(url).netloc
                kind = "reference"
                if host == "huggingface.co" and urlparse(url).path.startswith("/datasets/"):
                    kind = "huggingface_dataset"
                elif category.startswith("13_"):
                    kind = "model_resource"
                elif category.startswith("14_"):
                    kind = "tool_resource"
                elif url.lower().endswith(".pdf") or host in {"arxiv.org", "aclanthology.org"}:
                    kind = "paper"
                elif host == "github.com":
                    kind = "github_resource"
                entry = by_url.setdefault(url, {"url": url, "kind": kind, "mentions": []})
                mention = {
                    "label": label,
                    "heading": heading,
                    "category": category,
                    "source_file": path.relative_to(ROOT).as_posix(),
                }
                if mention not in entry["mentions"]:
                    entry["mentions"].append(mention)
    # One entry per URL/category keeps category filtering straightforward.
    result = []
    for url, entry in sorted(by_url.items()):
        categories = sorted({mention["category"] for mention in entry["mentions"]})
        for category in categories:
            result.append(
                {
                    "url": url,
                    "kind": entry["kind"],
                    "category": category,
                    "mentions": [m for m in entry["mentions"] if m["category"] == category],
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if packaged inventory is stale")
    args = parser.parse_args()
    text = json.dumps(build_inventory(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            print("Inventory is stale. Run: python scripts/build_inventory.py")
            return 1
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
