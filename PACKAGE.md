# Indonesian dataset toolkit

An experimental Python package accompanying the [awesome list](README.md).
Discover sources, load a supported dataset, and prepare reproducible JSONL
without copying every upstream dataset into this repository.

## Repository-wide workflows (v0.2)

The package now indexes **every URL in the category files**, additional README
sources and code-only Hub references, plus named datasets with no download URL.
It distinguishes data sources from papers/models/tools. These are **source
records, not a count of unique datasets**: configurations, mirrors and reference
links may refer to the same underlying data.

See [SOURCE_COVERAGE.md](SOURCE_COVERAGE.md) for the generated source-by-source
coverage table. `catalog.json` contains access instructions and conservative
license status; `audit.json` is a bundled, timestamped network observation, not
a guarantee of current availability. Only fingerprint-matched observations are
applied, so a changed source cannot inherit stale verification.

```bash
python -m pip install -e ".[all]"
indo-datasets catalog --summary
indo-datasets catalog --query nusa
indo-datasets source-info nusax-mt
```

### Plan, inspect, then download

```bash
# Offline: plan all catalog records, reporting every excluded source and why.
indo-datasets plan --all-sources --limit 100 --output plans/sources.json
# Inspect plans/sources.json before executing it.
indo-datasets batch plans/sources.json --output downloads/source-batch
# Resume using the IDENTICAL plan; completed files are checksum-verified.
indo-datasets batch plans/sources.json --output downloads/source-batch --resume
```

Each job has its own directory and provenance manifest. Sources/configurations
are not concatenated; official held-out sets are not merged into training. A
failed source does not abort unrelated jobs. `batch-report.json` lists completed,
failed and verified-existing jobs alongside all skip decisions. The CLI returns
nonzero when any attempted job fails; skipped sources are never called successes.
Corrupt existing exports are reported, never overwritten. An interrupted process
may leave `.batch.lock`; verify the process is gone before removing that lock.

Default policy excludes unknown licenses, permission/credential-gated sources,
failed availability checks, moving/unpinned files, evaluation datasets and
non-training/unknown splits. Multiconfiguration datasets need an explicit
`--config NAME` or `--all-configs`; arbitrary language configurations are not
silently combined. Original NusaWrites, IndoNLI and UD split conventions have
explicit adapters. Generic GitHub file candidates retain `unspecified` splits.

Optional planning flags are explicit acknowledgements, **not legal clearance**:

- `--allow-unknown-license`: you are responsible for establishing permission.
- `--allow-unpinned`: accept a moving source, recording a checksum after download.
- `--include-evaluation`: fetch held-out/unknown splits separately for inspection
  or evaluation, not for accidental inclusion in training.
- `--all-rows`: remove the row sampling limit for selected jobs.
- `--purpose commercial`: rejects explicitly non-commercial sources even if
  `--allow-unknown-license` was supplied. Other declared licenses still need review.
- `--max-download-mb` / `--max-uncompressed-mb`: file and decoded-input caps.
  Known over-cap Hub shards are excluded at planning time. **There is no hard
  total-network or disk quota for Hub streaming**, and sizes can be unknown;
  upstream range requests/buffering are outside the file downloader's cap.

Archives are fetched only as opaque `artifact.bin` files with the source URL
and hash in a manifest. They are not automatically extracted, executed, decoded
or presented as training examples. Gated datasets, Kaggle, Drive, Mendeley,
OpenSLR and author-request-only sources retain their original access process.
Use an authorized local copy when an automated adapter is unavailable.

### Refresh source availability

```bash
# Explicit network operation; reads API metadata and HTTP headers, not data shards.
indo-datasets audit --all-sources --workers 6 --output reports/fresh-audit.json
indo-datasets catalog --audit reports/fresh-audit.json --summary
indo-datasets plan --all-sources --audit reports/fresh-audit.json \
  --limit 100 --output plans/refreshed.json
```

GitHub API rate limits can be raised with your own `GH_TOKEN`/`GITHUB_TOKEN`.
Alternatively `audit --git-fallback ...` uses public Git partial-fetch metadata
when the API is rate limited. This requires `git`; it disables hooks and prompts,
fetches commit/tree objects (not file blobs), and never executes repository code.
No credential values are stored in catalogs, plans or reports.

Legacy Hub dataset scripts are never executed. The auditor uses pinned native
data files or separately pinned, data-only Parquet conversion commits when
available. A conversion revision is recorded independently of the upstream
source revision. Missing conversion/native data leaves a manual-access source,
not a fabricated loader. A DNS timeout/rate limit is not permanent removal.

### Raw task schemas and explicit preparation

`load_dataset()` and the original seven canonical loaders remain unchanged.
`load_source()` adds repository-wide adapters and preserves unreviewed fields
under `data`, including token labels, nested QA structures and untranslated
columns. Audio/image decoding is disabled; media stays as references or explicitly
base64-encoded embedded bytes. This is not audio resampling, video processing,
image downloading or privacy filtering. Non-finite missing floats become null.
Generic CSV/TSV file candidates retain header rows as column arrays unless a
specific header convention was reviewed; do not count them as training examples.

```python
from indonesian_datasets import load_source

rows = load_source("nusax-mt", config="default", split="train", limit=10)
for row in rows:
    print(row["data"]["indonesian"], row["data"]["javanese"])
```

```bash
# Obtain a small, separately exported translation split.
indo-datasets plan --ids nusax-mt --limit 100 --output plans/mt.json
indo-datasets batch plans/mt.json --output downloads/mt

# For any authorized local copy, explicitly specify format and encoding.
indo-datasets import-file my-data.csv --format csv --output downloads/imported
indo-datasets prepare downloads/imported/data.jsonl \
  --recipe examples/recipes/classification.json --output downloads/prepared
```

`import-file` supports CSV, headerless CSV/TSV rows, JSON, JSONL, line-based text,
CoNLL token/tag sentences, CoNLL-U, and optional Parquet. Gzip/Bzip2/XZ/Zstandard
inputs have a decoded-byte cap to guard against decompression bombs. Encoding is
strict UTF-8 by default; use `--encoding` for an explicitly reviewed legacy
encoding. Arbitrary Python/pickle inputs are never loaded.

Recipes in `examples/recipes/` cover classification, NLI, summarization,
translation, extractive QA, token classification, chat, audio references, image
captions, knowledge triples and lexicons. Most are **schema templates**, not
claims that every source shares those column names. Inspect a raw record and
edit the mapping first. Dotted paths and list indices are supported; source
column names containing literal dots need a custom adapter. `indonlu-ner.json`
and `nusax-translation-id-jv.json` target the inspected adapters.

Preparation validates aligned token-label lengths and QA answer offsets against
the untouched context. Generic cleaning is prohibited for alignment-sensitive,
chat and media recipes. Failures abort the export with a record number; invalid
rows are not silently dropped. Prepared output retains `original`, plus mapped
fields under `prepared` and the recipe hash. No automatic mixing, tokenization,
language-ID, near-duplicate removal, PII redaction or licensing approval is implied.

### Coverage maintenance and opt-in live checks

```bash
python scripts/build_inventory.py
python scripts/build_catalog.py
python scripts/report_coverage.py
python scripts/smoke_sources.py --output reports/new-smoke.json \
  --max-download-mb 10 --limit 1
```

Live smoke checks can access public data with unverified licenses for inspection;
they are separate from the conservative batch-planning policy. They check only
one selected variant per adapter, not every configuration or full corpus. The
committed `reports/source-smoke.json` reports exactly what was checked or skipped
without retaining dataset content. Offline unit tests do not access the network.

## Install from this checkout

Requires Python 3.10+. Nothing is downloaded when importing or listing datasets.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Optional: required only for Hugging Face sources
python -m pip install -e ".[hf]"
```

This is a local package, not a published PyPI release. The base installation
has no runtime dependencies; Hugging Face support uses the `datasets` library.

## Discover

```bash
indo-datasets list
indo-datasets list --task sentiment --language id
indo-datasets info fineweb-edu-id
indo-datasets sources --query speech
indo-datasets sources --category 05_local_languages
```

`list` shows **implemented loaders**. `sources` searches an automatically
generated inventory of links from every `data/*.md` category, including papers,
models, tools, and sources that do not yet have loaders. Indexed does **not** mean
downloadable, license-verified, or currently reachable. Link kinds are heuristic.

### Canonical loader coverage (the original API)

| ID | Task | Language | Backend | Official splits |
| --- | --- | --- | --- | --- |
| `nusax-id` | Sentiment | Indonesian | CSV | train, validation, test |
| `nusax-jv` | Sentiment | Javanese | CSV | train, validation, test |
| `nusax-su` | Sentiment | Sundanese | CSV | train, validation, test |
| `smsa` | Sentiment | Indonesian | TSV | train, validation, test |
| `openwebtext-id` | Text generation | Indonesian translation | Hugging Face | train |
| `fineweb-edu-id` | Text generation | Indonesian translation | Hugging Face | train |
| `simple-summaries-id` | Summarization | Indonesian translation | Hugging Face | train |

Source revisions are pinned to upstream commits. `info` includes the field
mapping, license evidence, restrictions/uncertainties, and a citation link.
The translated datasets were inspected against the Hub schema when added.

## Download one split

```bash
indo-datasets download nusax-id --split train --limit 100 --output downloads/nusax-raw
indo-datasets download fineweb-edu-id --limit 100 --output downloads/fineweb-raw
# Explicit opt-in to all rows of ONE dataset split:
indo-datasets download nusax-id --split train --all --output downloads/nusax-full
```

Exports contain `data.jsonl` and `manifest.json`. A manifest records source
revision and license notes, requested split/limit, package version, row count,
and an output SHA-256. Existing output directories are never overwritten;
exports are staged and published only when complete.

- CLI defaults to **1,000 rows**. `--limit` limits rows, **not network bytes**.
- CSV/TSV sources download a complete file before yielding records, cache it,
  and verify its local checksum on reuse. The default file cap is **100 MiB**;
  change it with `--max-download-mb`. Checksums detect local corruption, not
  upstream authenticity. Missing or corrupt cache files are downloaded again.
- Hugging Face sources use upstream streaming; the CSV byte cap does not apply.
  Streaming can fetch buffers or shards larger than the requested row sample.
- Cache defaults to `$XDG_CACHE_HOME/indonesian-datasets`, or
  `~/.cache/indonesian-datasets`. Override with `--cache-dir`.
- Existing upstream splits are retained. Downloads never auto-merge benchmarks.
- Unknown licenses produce warnings, not legal clearance. Review terms first.
- Authentication uses the upstream `HF_TOKEN` mechanism. No credentials are
  written into manifests, and remote dataset scripts are not enabled.

### Canonical records

```json
{
  "id": "stable-sha256-of-original-row-and-source-identity",
  "text": "Saya suka makanan ini.",
  "label": "positive",
  "language": "id",
  "source": "nusax-id",
  "split": "train",
  "revision": "pinned-upstream-commit",
  "metadata": {"id": "original-example-id"}
}
```

Summarization also has `summary`. Unmapped original fields are retained in
`metadata`, including original English text for translated collections. For
FineWeb, canonical `text` comes from **`text_indonesian`**, not the English
`text` column. Missing or empty required fields fail explicitly; no silent
fallback to another language occurs. IDs identify original source records and
remain unchanged by cleaning; export checksums identify transformed content.

## Preprocess local JSONL

```bash
indo-datasets preprocess downloads/nusax-raw/data.jsonl \
  --output downloads/nusax-clean --deduplicate --min-chars 10

indo-datasets preprocess downloads/summaries-raw/data.jsonl \
  --output downloads/summaries-clean --text-fields text,summary --deduplicate
```

Defaults normalize Unicode NFC, remove zero-width spaces/BOMs, and collapse
whitespace. They **preserve casing, negation, punctuation, diacritics, emoji,
URLs, and regional-language content**. Lowercasing and URL removal require
`--lowercase` and `--remove-urls`. URL removal is a simple regex and can also
remove adjacent punctuation; do not enable it for tasks needing original spans.
Whitespace is flattened, so this pipeline is not suitable for code/layout data.

Rows with missing/non-string text or invalid lengths are counted and dropped.
Malformed JSON fails with a line number. The manifest records the input file
hash, cleaning configuration, output hashes, and rejection/deduplication counts.

Deduplication is opt-in, exact on cleaned selected fields, and keeps the first
record. For classification with conflicting annotations, use
`--dedup-fields text,label` or audit conflicts separately rather than silently
discarding labels. The digest set consumes O(unique records) memory. This is
not fuzzy deduplication or benchmark contamination detection.

**Do not apply this generic text pipeline to token-level labels, character
offsets, or NER spans without updating their alignments.** PII detection,
language identification, slang normalization, stemming, tokenization, and
multimodal processing are deliberately not implemented in this first version.
No data should be considered privacy-safe merely because it was cleaned.

## Python API

```python
from indonesian_datasets import Pipeline, list_datasets, load_dataset

print(list_datasets(language="id"))
records = load_dataset("nusax-id", split="train", limit=100)
pipeline = Pipeline(deduplicate=True, dedup_fields=("text", "label"))
for record in pipeline.process(records):
    print(record["text"], record["label"])
print(pipeline.stats)
```

`load_dataset` returns a lazy iterator; Python has no default row limit. Use
`limit=` for exploration. Complete iteration is needed for complete statistics.
For summaries, use `Pipeline(text_fields=("text", "summary"))`.

## Split a NEW corpus

Prefer official splits for existing datasets. The separate splitting command
requires acknowledgement that the input is a new corpus:

```bash
indo-datasets split my-new-corpus.jsonl --new-corpus \
  --output downloads/my-splits --ratios 0.8 0.1 0.1 --seed 42 --group-fields text
```

It writes train/validation/test JSONL and a manifest. Seeded content hashing
keeps equal groups in the same split regardless of input order. Ratios are
approximate, not exact or stratified. This command currently holds results in
memory; do not use it for an unbounded stream or a web-scale corpus.

Group on a document, user, or parallel-example ID when examples are related.
For parallel NusaX examples, extract `metadata.id` into a top-level `parallel_id`
and group on that ID, not on translated text. Nested grouping paths are not
supported. Hashing alone does not prevent near-duplicate or semantic leakage.

## Contribute and test

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python scripts/build_inventory.py --check
ruff check .
ruff format --check .
python -m build
```

Tests are offline and mock network access. CI runs Python 3.10–3.13, checks that
the link inventory is current, builds the distribution, and checks installed
package resources. Upstream availability needs separate, explicit live checks.

To add a loader:

1. Add an entry to `src/indonesian_datasets/registry.json`: unique ID, task,
   language, pinned revision, official splits, and explicit canonical field map.
2. Include source/citation links and dataset-specific license evidence. Use
   `unknown` when uncertain; do not substitute the repository's code license.
3. Use the existing CSV/TSV or Hugging Face backend; add a separate adapter for
   other formats. Add an offline fixture testing schema and split handling.
4. Verify a small live sample, then document coverage here. Avoid automated
   redistribution or downloads of gated/restricted sources.
5. When category Markdown changes, run `python scripts/build_inventory.py`
   and commit the generated `resources.json`.

For repository-wide source changes, also regenerate `catalog.json`, refresh
affected audit observations and regenerate `SOURCE_COVERAGE.md`.

### Suggested next milestones

1. **Metadata audit:** availability, exact size, language/configuration, dataset
   cards, commercial-use restrictions, citations, and per-source test fixtures.
2. **Task adapters:** NLI, extractive QA (preserve offsets), token classification
   (preserve alignment), instruction/chat formats, then speech and vision.
3. **Quality tooling:** Indonesian/regional-language identification with measured
   error rates, PII flagging/redaction, near-duplicate detection, and explicit
   train/evaluation decontamination reports.
4. **Reproducible recipes:** versioned YAML mixtures with sampling weights and
   input hashes, Parquet export, disk-backed deduplication, bounded-memory splits.
5. **Release:** choose a software license with the repository owner before
   publishing the package; add a changelog and release workflow. No software
   license has been selected by this implementation. Dataset licenses remain
   independent. Mirror data only where redistribution is explicitly permitted.
