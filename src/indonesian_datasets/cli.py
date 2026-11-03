"""Command-line access. Listing and preprocessing work entirely offline."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError

from .loaders import load_dataset
from .preprocessing import Pipeline, split_records
from .registry import get_dataset, list_datasets, list_sources
from .storage import file_sha256, read_jsonl, write_bundle


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _fields(value: str) -> tuple[str, ...]:
    result = tuple(key.strip() for key in value.split(","))
    if not result or not all(result):
        raise argparse.ArgumentTypeError("use comma-separated non-empty field names")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and prepare Indonesian datasets")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List supported dataset loaders (offline)")
    listing.add_argument("--task")
    listing.add_argument("--language")
    listing.add_argument("--query", default="")
    listing.add_argument("--json", action="store_true")
    sources = sub.add_parser(
        "sources", help="Search documentation links, including unimplemented sources"
    )
    sources.add_argument("--query", default="")
    sources.add_argument("--category", help="Category filename without .md, e.g. 09_speech_audio")
    info = sub.add_parser("info", help="Show schema, pinned revision, license notes and citation")
    info.add_argument("name")
    download = sub.add_parser("download", help="Export one dataset split as canonical JSONL")
    download.add_argument("name")
    download.add_argument("--split", default="train")
    download.add_argument("--output", type=Path, required=True, help="NEW export directory")
    download.add_argument("--cache-dir", type=Path)
    limits = download.add_mutually_exclusive_group()
    limits.add_argument(
        "--limit", type=_positive, default=1000, help="Maximum rows (default: 1000)"
    )
    limits.add_argument(
        "--all", action="store_true", help="Explicitly export all rows of this split"
    )
    download.add_argument(
        "--max-download-mb", type=_positive, default=100, help="CSV/TSV byte cap only"
    )
    prepare = sub.add_parser(
        "preprocess", help="Clean local JSONL without changing official splits"
    )
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--output", type=Path, required=True, help="NEW export directory")
    prepare.add_argument("--text-fields", type=_fields, default=("text",))
    prepare.add_argument("--lowercase", action="store_true")
    prepare.add_argument("--remove-urls", action="store_true")
    prepare.add_argument("--min-chars", type=int, default=1)
    prepare.add_argument("--max-chars", type=int)
    prepare.add_argument("--deduplicate", action="store_true")
    prepare.add_argument("--dedup-fields", type=_fields)
    splitting = sub.add_parser("split", help="Hash-split NEW corpora; do not re-split benchmarks")
    splitting.add_argument("input", type=Path)
    splitting.add_argument("--output", type=Path, required=True, help="NEW export directory")
    splitting.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    splitting.add_argument("--seed", type=int, default=42)
    splitting.add_argument("--group-fields", type=_fields, default=("text",))
    splitting.add_argument(
        "--new-corpus",
        action="store_true",
        required=True,
        help="Acknowledge this is a new corpus, not an existing benchmark",
    )
    catalog = sub.add_parser(
        "catalog", help="Full source inventory, including manual/unavailable sources"
    )
    catalog.add_argument("--query", default="")
    catalog.add_argument("--task")
    catalog.add_argument("--language")
    catalog.add_argument(
        "--kind", choices=["dataset_source", "paper", "model", "tool", "reference"]
    )
    catalog.add_argument("--audit", type=Path)
    catalog.add_argument("--json", action="store_true")
    catalog.add_argument("--summary", action="store_true")
    source_info = sub.add_parser(
        "source-info", help="Inspect any catalog source and access instructions"
    )
    source_info.add_argument("name")
    source_info.add_argument("--audit", type=Path)
    audit = sub.add_parser(
        "audit", help="Explicit network audit: metadata/headers only, never remote scripts"
    )
    selection = audit.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", nargs="+")
    selection.add_argument(
        "--all-sources", action="store_true", help="Check all dataset sources, not papers/tools"
    )
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--workers", type=_positive, default=6)
    audit.add_argument(
        "--git-fallback",
        action="store_true",
        help=(
            "Use public Git partial-fetch metadata when the GitHub API is rate limited; "
            "requires git"
        ),
    )
    plan = sub.add_parser(
        "plan", help="Create a frozen batch plan, including skip reasons; no downloads"
    )
    selection = plan.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all-sources", action="store_true")
    plan.add_argument("--task")
    plan.add_argument("--language")
    plan.add_argument("--audit", type=Path)
    plan.add_argument("--purpose", choices=["research", "commercial"], default="research")
    plan.add_argument(
        "--allow-unknown-license",
        action="store_true",
        help="Acknowledge your own responsibility to establish permission",
    )
    plan.add_argument("--allow-unpinned", action="store_true")
    plan.add_argument(
        "--include-evaluation",
        action="store_true",
        help="Download held-out sets separately, never mix into training",
    )
    configurations = plan.add_mutually_exclusive_group()
    configurations.add_argument("--config")
    configurations.add_argument("--all-configs", action="store_true")
    limits = plan.add_mutually_exclusive_group()
    limits.add_argument("--limit", type=_positive, default=1000)
    limits.add_argument("--all-rows", action="store_true")
    plan.add_argument("--max-download-mb", type=_positive, default=100)
    plan.add_argument("--max-uncompressed-mb", type=_positive, default=256)
    plan.add_argument("--output", type=Path, required=True)
    batch = sub.add_parser(
        "batch", help="Execute a saved plan with independent exports and a resume report"
    )
    batch.add_argument("plan", type=Path)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--cache-dir", type=Path)
    batch.add_argument("--resume", action="store_true")
    importing = sub.add_parser(
        "import-file", help="Import an authorized local copy without guessing its task schema"
    )
    importing.add_argument("input", type=Path)
    importing.add_argument(
        "--encoding", default="utf-8-sig", help="Explicit source encoding; never guessed"
    )
    importing.add_argument(
        "--format",
        required=True,
        choices=[
            "csv",
            "csv_rows",
            "tsv_rows",
            "json",
            "jsonl",
            "lines",
            "conll",
            "conllu",
            "parquet",
        ],
    )
    importing.add_argument("--max-uncompressed-mb", type=_positive, default=256)
    importing.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser(
        "prepare", help="Map and validate a task schema with an explicit JSON recipe"
    )
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--recipe", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    return parser


def _input_metadata(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _run(args: argparse.Namespace) -> int | None:
    if args.command in {
        "catalog",
        "source-info",
        "audit",
        "plan",
        "batch",
        "import-file",
        "prepare",
    }:
        return _run_sources(args)
    if args.command == "list":
        entries = list_datasets(task=args.task, language=args.language, query=args.query)
        if args.json:
            print(json.dumps(entries, ensure_ascii=False, indent=2))
        else:
            print(f"{'ID':24} {'LANG':6} {'TASK':18} LICENSE")
            for entry in entries:
                print(
                    f"{entry['id']:24} {entry['language']:6} {entry['task']:18} {entry['license']}"
                )
        return
    if args.command == "sources":
        print(json.dumps(list_sources(query=args.query, category=args.category), indent=2))
        return
    if args.command == "info":
        print(json.dumps(get_dataset(args.name), ensure_ascii=False, indent=2))
        return
    if args.command == "download":
        entry = get_dataset(args.name)
        print(f"License: {entry['license']}. Review {entry['license_url']}", file=sys.stderr)
        limit = None if args.all else args.limit
        records = load_dataset(
            args.name,
            split=args.split,
            limit=limit,
            cache_dir=args.cache_dir,
            max_download_bytes=args.max_download_mb * 1024 * 1024,
        )
        manifest = write_bundle(
            args.output,
            {"data.jsonl": records},
            {
                "operation": "download",
                "dataset": entry,
                "split": args.split,
                "limit": limit,
                "max_download_bytes_csv_only": args.max_download_mb * 1024 * 1024,
            },
        )
    elif args.command == "preprocess":
        pipeline = Pipeline(
            text_fields=args.text_fields,
            lowercase=args.lowercase,
            remove_urls=args.remove_urls,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            deduplicate=args.deduplicate,
            dedup_fields=args.dedup_fields,
        )
        source = _input_metadata(args.input)

        def metadata():
            if pipeline.stats["input"] and pipeline.stats["invalid"] == pipeline.stats["input"]:
                raise ValueError(
                    "No usable configured text fields in the input. "
                    "For raw source exports, use 'prepare' with an explicit task recipe."
                )
            return {
                "operation": "preprocess",
                "input": source,
                "pipeline": pipeline.config(),
                "stats": pipeline.stats,
            }

        manifest = write_bundle(
            args.output,
            {"data.jsonl": pipeline.process(read_jsonl(args.input))},
            metadata,
        )
    else:
        source = _input_metadata(args.input)
        splits = split_records(
            read_jsonl(args.input),
            ratios=tuple(args.ratios),
            seed=args.seed,
            group_fields=args.group_fields,
        )
        manifest = write_bundle(
            args.output,
            {f"{name}.jsonl": records for name, records in splits.items()},
            {
                "operation": "split",
                "input": source,
                "seed": args.seed,
                "ratios": args.ratios,
                "group_fields": args.group_fields,
            },
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args) or 0
    except (ValueError, OSError, ImportError, URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled; no partial export was published.", file=sys.stderr)
        return 130
    return 0


def _select(entries: list[dict], ids: list[str] | None) -> list[dict]:
    if ids is None:
        return entries
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs in selection")
    by_id = {entry["id"]: entry for entry in entries}
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"Unknown or filtered-out source IDs: {sorted(missing)}")
    return [by_id[name] for name in ids]


def _run_sources(args: argparse.Namespace) -> int:
    from .adapters import iter_file, json_safe
    from .audit import audit_sources
    from .batch import plan_sources, run_batch
    from .catalog import coverage, fingerprint, get_source, list_catalog
    from .recipes import prepare_records
    from .storage import save_json

    if args.command == "catalog":
        if args.summary:
            print(json.dumps(coverage(audit_path=args.audit), indent=2))
        else:
            entries = list_catalog(
                query=args.query,
                task=args.task,
                language=args.language,
                resource_type=args.kind,
                audit_path=args.audit,
            )
            if args.json:
                print(json.dumps(entries, ensure_ascii=False, indent=2))
            else:
                for entry in entries:
                    print(
                        f"{entry['id']}\t{entry['resource_type']}\t{entry['backend'] or 'manual'}\t"
                        f"{entry['availability']}\t{entry['license']}"
                    )
        return 0
    if args.command == "source-info":
        print(
            json.dumps(get_source(args.name, audit_path=args.audit), ensure_ascii=False, indent=2)
        )
        return 0
    if args.command == "audit":
        entries = list_catalog(include_audit=False)
        entries = (
            _select(entries, args.ids)
            if args.ids
            else [e for e in entries if e["resource_type"] == "dataset_source"]
        )
        if args.output.exists():
            raise FileExistsError(str(args.output))
        report = audit_sources(entries, workers=args.workers, git_fallback=args.git_fallback)
        save_json(args.output, report)
        print(
            json.dumps(
                {
                    "checked": len(report["sources"]),
                    "errors": sum("error" in item for item in report["sources"].values()),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "plan":
        entries = _select(
            list_catalog(task=args.task, language=args.language, audit_path=args.audit), args.ids
        )
        plan = plan_sources(
            entries,
            purpose=args.purpose,
            allow_unknown_license=args.allow_unknown_license,
            allow_unpinned=args.allow_unpinned,
            include_evaluation=args.include_evaluation,
            all_configs=args.all_configs,
            config=args.config,
            limit=None if args.all_rows else args.limit,
            max_download_bytes=args.max_download_mb * 1024 * 1024,
            max_uncompressed_bytes=args.max_uncompressed_mb * 1024 * 1024,
        )
        save_json(args.output, plan)
        print(json.dumps(plan["summary"], indent=2))
        return 0
    if args.command == "batch":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        report = run_batch(plan, args.output, cache_dir=args.cache_dir, resume=args.resume)
        print(
            json.dumps(
                {
                    "summary": report["summary"],
                    "skipped_decisions": sum(d["status"] == "skipped" for d in report["decisions"]),
                    "report": str(args.output / "batch-report.json"),
                },
                indent=2,
            )
        )
        return 1 if report["summary"].get("failed") else 0
    source = _input_metadata(args.input)
    if args.command == "import-file":

        def imported():
            for index, row in enumerate(
                iter_file(
                    args.input,
                    format=args.format,
                    max_uncompressed_bytes=args.max_uncompressed_mb * 1024 * 1024,
                    encoding=args.encoding,
                )
            ):
                data = json_safe(row)
                yield {
                    "id": fingerprint([source["sha256"], index, data]),
                    "source": args.input.name,
                    "revision": source["sha256"],
                    "split": "unspecified",
                    "data": data,
                }

        manifest = write_bundle(
            args.output,
            {"data.jsonl": imported()},
            {
                "operation": "import-file",
                "input": source,
                "format": args.format,
                "encoding": args.encoding,
            },
        )
    else:
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        stats = {}
        manifest = write_bundle(
            args.output,
            {"data.jsonl": prepare_records(read_jsonl(args.input), recipe, stats)},
            lambda: {
                "operation": "prepare",
                "input": source,
                "recipe": recipe,
                "recipe_sha256": fingerprint(recipe),
                "stats": stats,
            },
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0
