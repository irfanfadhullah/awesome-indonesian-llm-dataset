"""Opt-in live sample checks. Reports selected variants only, never claims full validation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from indonesian_datasets.adapters import iter_source, source_variants
from indonesian_datasets.catalog import list_catalog
from indonesian_datasets.storage import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-download-mb", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 1 or args.max_download_mb < 1 or not 1 <= args.workers <= 8:
        parser.error("positive limits and 1–8 workers are required")
    entries = [
        e
        for e in list_catalog()
        if e["backend"] in {"registered", "file", "hf_parquet"}
        and e["availability"]
        not in {"not_found", "check_failed", "access_denied", "rate_limited", "http_error"}
    ]

    def check(entry):
        variants = source_variants(entry)
        variants.sort(
            key=lambda v: (
                v["config"] not in {"default", "id", "indonesian", "en-id"},
                v["split"] != "train",
                v["config"],
                v["split"],
            )
        )
        if not variants:
            return {"id": entry["id"], "status": "no_variants"}
        variant = variants[0]
        result = {
            "id": entry["id"],
            "config": variant["config"],
            "split": variant["split"],
            "revision": entry.get("revision"),
            "conversion_revision": entry.get("conversion_revision"),
        }
        if (
            max(variant.get("shard_sizes") or [entry.get("content_length") or 0])
            > args.max_download_mb * 1024 * 1024
        ):
            return {**result, "status": "skipped_size_cap"}
        try:
            records = list(
                iter_source(
                    entry,
                    config=variant["config"],
                    split=variant["split"],
                    limit=args.limit,
                    cache_dir=args.cache_dir,
                    max_download_bytes=args.max_download_mb * 1024 * 1024,
                )
            )
            return {
                **result,
                "status": "sample_ok" if records else "empty_sample",
                "rows": len(records),
            }
        except Exception as exc:
            if isinstance(exc, ValueError) and "exceeds" in str(exc) and "bytes" in str(exc):
                return {**result, "status": "skipped_size_cap", "error": str(exc)}
            return {**result, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checks = list(pool.map(check, entries))
    report = {
        "scope": (
            "Selected small variant per adapter, not all rows/configs. "
            "No content or credentials stored."
        ),
        "results": checks,
        "summary": dict(Counter(item["status"] for item in checks)),
    }
    save_json(args.output, report)
    print(report["summary"])
    return 1 if report["summary"].get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
