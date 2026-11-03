"""Generate a factual source-by-source report from the catalog and its audit snapshot."""

import argparse
from pathlib import Path

from indonesian_datasets.catalog import coverage, list_catalog

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_COVERAGE.md"


def render() -> str:
    report = coverage()
    lines = [
        "# Repository source coverage",
        "",
        "Generated from `catalog.json` and the fingerprint-matched `audit.json` snapshot.",
        "These are **source records, not a count of unique datasets**: mirrors, configurations,",
        "papers and tools are distinguished. A reachable index is not a successful sample test.",
        "",
        f"Total source records: **{report['source_records']}**.",
        "",
        "| Resource kind | Records |",
        "| --- | ---: |",
    ]
    for kind, count in sorted(report["counts"]["resource_type"].items()):
        lines.append(f"| {kind} | {count} |")
    lines += [
        "",
        "## Data-source records",
        "",
        "`registered` means a reviewed canonical loader. `file`/`hf_parquet` preserve raw",
        "task fields; they need an explicit preparation recipe before training. `artifact`",
        "downloads an opaque archive without extraction. Pending/none requires audit,",
        "permission, credentials, provider action or a custom adapter. GitHub-discovered",
        "file candidates are not asserted to be task-ready datasets. Unknown splits are",
        "excluded from default training plans. An HTTP/DNS failure describes this audit,",
        "not permanent unavailability. See each source's `instructions` and license evidence.",
        "",
        "| Source ID | Adapter | Availability observed | Access | Declared license |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in list_catalog(resource_type="dataset_source"):
        name = entry["id"]
        label = f"[`{name}`]({entry['homepage']})" if entry.get("homepage") else f"`{name}`"
        lines.append(
            f"| {label} | {entry['backend'] or 'none'} | {entry['availability']} | "
            f"{entry['access']} | {entry['license']} |"
        )
    lines += [
        "",
        "Regenerate with `python scripts/report_coverage.py` after refreshing the audit.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            print("Coverage report is stale. Run: python scripts/report_coverage.py")
            return 1
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
