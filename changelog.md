# Changelog

Historical features, improvements, fixes, validation results, and limitations of
Awesome Indonesian LLM Dataset and its Python toolkit.

History recorded on **2026-11-03**. This changelog reconstructs the development
milestones from the work performed on this checkout, the current implementation,
and saved validation reports. The checkout has no Git history, so earlier commit,
tag, and release dates cannot be verified and are left unspecified. Package
versions below describe **local implementation milestones**, not confirmed PyPI
publications. Counts describe the checks recorded at each milestone, not a
guarantee of present upstream availability.

## Unreleased

### Fixed — 2026-11-03

- **Hashtag tokens were silently dropped from CoNLL token/tag files.** Lines
  starting with `#` were treated as comments, although the format has no comment
  syntax. This removed 46 labelled tokens (for example `#BCA` tagged `B`) from
  the pinned IndoNLU BaPOS, KEPS, NERGrit, NERP, POSP and TermA files, 21 of them
  from KEPS train. Sentence counts were unaffected, so
  [reports/pinned-split-check.json](reports/pinned-split-check.json) remains valid.
- **CoNLL-U `tokens` included multiword-token ranges and empty nodes**, so it
  could not align with per-word annotations. The pinned UD Indonesian-GSD
  validation file has 238 multiword rows (e.g. `membelinya` → `membeli` + `nya`);
  `tokens` now lists exactly its 12,661 words. `rows` remains lossless.
- The CoNLL token/tag reader now accepts whitespace-separated and multi-column
  (CoNLL-2003-style) files, treats `-DOCSTART-` as a document boundary, and
  rejects rows with inconsistent column counts or an empty token or tag.
- **The seven canonical loaders now return the common record envelope through
  `load_source()` and batch exports**, with canonical fields under `data`.
  Previously their flat records made recipes such as
  `examples/recipes/classification.json` (`data.text`) fail. `load_dataset()`
  is unchanged. Batch exports of these loaders made before this fix keep the old
  flat shape (resume only verifies checksums); re-export them in a new directory.
- The default cache now honors `XDG_CACHE_HOME` for every backend; adapters and
  batches previously always used `~/.cache`, creating a second cache.
- Archive manifests now record `package_version`, `created_at`, `config`,
  `split`, `plan_sha256` and `policy`, like row exports.
- A leftover `.batch.lock` now reports its process ID and how to recover,
  instead of a bare "File exists" error.
- README: fixed the GitHub Issues link (`awesome-indonesia-…` did not resolve),
  removed the link to disabled Discussions, updated the toolkit roadmap, and
  removed a duplicate Citations heading. The regenerated catalog therefore has
  **283 source records** (one fewer project-reference link); the 224 data-source
  records and their audit observations are unchanged.

### Documentation

- **2026-11-03:** Added this changelog, covering the original curated repository,
  the initial Python package, and the repository-wide toolkit expansion.
- Recorded feature coverage, CLI and Python APIs, preprocessing behavior,
  licensing/access safeguards, validation evidence, and remaining limitations.

## 0.2.0 — Repository-wide toolkit expansion

Implementation date not recorded. Extends 0.1.0 while retaining its seven
canonical dataset loaders and original Python API.

### Added — Full source catalog

- Added `catalog.json` and catalog-building automation covering every URL in
  the category files, additional README sources, and code-only Hugging Face
  references, including the six NusaWrites collections.
- Added explicit records for named resources without a documented direct
  download URL, including IndoCloze, MinangNLP, the Indonesian unsupervised speech
  collection, and the described clean Indonesian mC4 corpus.
- Recorded **284 source records**, including **224 data-source candidates**.
  These are not 284 or 224 unique datasets: mirrors, configurations, papers,
  models, tools, and reference links are distinguished.
- Added source IDs, source URLs, documentation mentions, category/task/language
  metadata, modality, evaluation role, access status, license evidence,
  availability observations, and source-specific instructions.
- Added `catalog`, `source-info`, and catalog-summary CLI commands.
- Added Python APIs `list_catalog()`, `get_source()`, and `coverage()`.
- Added [SOURCE_COVERAGE.md](SOURCE_COVERAGE.md) and its deterministic generator
  for source-by-source adapter, availability, access, and license reporting.

### Added — Metadata and availability auditing

- Added the `audit` command and a bundled `audit.json` observation snapshot.
- Audited all **224 data-source records** using metadata, HTTP headers, and,
  where explicitly enabled, public Git metadata.
- Recorded timestamps, evidence URLs, source fingerprints, revisions, available
  configurations/splits, reported sizes, and diagnostic failures.
- Added fingerprint matching so changed catalog definitions cannot silently
  inherit stale audit observations.
- Distinguished missing resources, access denial, rate limits, HTTP errors,
  failed checks, and landing-page availability. A failed check is not treated
  as proof of permanent dataset removal.
- Added bounded audit concurrency and capped metadata responses.
- Added optional GitHub authentication through `GH_TOKEN`/`GITHUB_TOKEN`.
- Added `--git-fallback` for public commit/tree discovery when the GitHub API
  is rate limited. The fallback disables hooks and prompts and requests a
  shallow, blob-filtered fetch rather than executing repository code.
- Added data-only Hugging Face discovery through native files or converted
  Parquet, with conversion commits pinned separately from original revisions.
- Left sources requiring unavailable scripts, provider action, permission, or
  credentials visible with instructions rather than inventing working loaders.

### Added — Source adapters and format support

- Added `load_source()` for repository-wide source access, separate from the
  original canonical `load_dataset()` API.
- Added pinned profiles for all 12 NusaX sentiment languages, 11 bilingual
  lexicons, and the multilingual translation files.
- Added 11 additional IndoNLU task profiles alongside the existing SmSA loader,
  covering token classification, aspect sentiment, emotion, entailment, and QA.
- Added original-source file discovery, including explicit split conventions
  for NusaWrites, IndoNLI, and UD Indonesian-GSD.
- Added generic versioned GitHub data-file candidates without treating every
  repository file as a verified training dataset. Unknown partitions remain
  `unspecified` instead of being assigned fabricated train/test splits.
- Added the `import-file` command for authorized local copies of sources that
  cannot be downloaded automatically.
- Added CSV, headerless CSV/TSV rows, JSON, JSONL, line-oriented text, CoNLL
  token/tag sentences, CoNLL-U, and optional Parquet adapters.
- Added Gzip, Bzip2, XZ, and optional Zstandard input support with a decoded-byte
  limit to guard against decompression bombs.
- Preserved unreviewed task schemas under `data`, including nested QA content,
  token annotations, translation columns, and source-specific metadata.
- Preserved unknown CSV/TSV header conventions as raw column arrays rather than
  silently discarding the first row.
- Disabled audio/image decoding in the data-only Parquet route. Media remains
  references or explicitly base64-encoded embedded bytes; this does not imply
  media downloading, resampling, or vision/audio preprocessing.
- Added JSON-safe handling for embedded bytes, dates, and missing non-finite
  floating-point values.
- Added explicit source-encoding support and a documented, reversible Latin-1
  decoding profile for the non-UTF-8 names file found during live checks.
- Added opaque archive exports with a source manifest and checksum. Archives
  are not automatically extracted or interpreted as training records.
- Added optional dependency groups `compression` and `all`.

### Added — Policy-aware batch workflows

- Added `plan` to create a frozen, inspectable download plan without downloading
  data, and `batch` to execute that plan.
- Added selection by source ID, task, language, and explicit configuration,
  plus opt-in selection of all configurations.
- Added research/commercial purpose selection, with explicit non-commercial
  restrictions retained for commercial plans.
- Excluded unknown licenses, gated/permission-required sources, failed checks,
  unpinned files, evaluation datasets, and held-out/unknown splits by default.
- Added explicit acknowledgement flags for unknown licenses, unpinned sources,
  evaluation/unknown splits, and full-row exports. These flags do not grant
  legal permission or bypass access controls.
- Added file-size and decoded-input caps and preflight rejection of known
  over-cap data shards.
- Kept source, configuration, and split exports separate; no automatic corpus
  concatenation or benchmark-to-training mixing was introduced.
- Added per-job provenance manifests, atomic publication, independent failure
  handling, and `batch-report.json` with actual results and skip decisions.
- Added `--resume` with frozen-plan identity and completed-file checksum checks.
  Corrupt or mismatched existing exports are reported, not overwritten.
- Added batch locking and checks against unsafe output paths, duplicate job
  directories, and symlink-based output replacement.

### Added — Explicit task preparation recipes

- Added `prepare` with versioned JSON recipes, explicit source-field mappings,
  dotted paths/list indices, recipe hashes, and retained original records.
- Added validation for classification, NLI, summarization, translation,
  extractive QA, token classification, chat, audio references, image captions,
  knowledge triples, lexicons, and raw records.
- Added 11 example recipe files in [examples/recipes](examples/recipes).
  Most are schema templates requiring source-specific field review, not
  universal adapters for every dataset in the corresponding task.
- Added inspected examples for IndoNLU token labels and NusaX Indonesian–Javanese
  translation columns.
- Checked token/label alignment and QA answer offsets against untouched source
  text. Invalid preparation records abort the export with a record number.
- Prohibited generic cleaning for alignment-sensitive, chat, and media recipes.

### Fixed and hardened

- Verified NusaX's dataset-specific `datasets/LICENSE` as **CC BY-SA 4.0**,
  replacing the initial version-unspecified license notes. Kept code licensing
  separate from dataset licensing.
- Ensured readable metadata or file indexes cannot waive documented permission,
  credential, or gating requirements.
- Rejected HTML responses where a data file was expected.
- Prevented generic preprocessing from silently publishing an empty export when
  every input record lacks the configured text fields; directed raw-source
  users to explicit task preparation instead.
- Required a resolved feature schema before the Parquet route can proceed with
  media decoding disabled.
- Preserved CoNLL-U comments, multiword-token IDs, and empty-node rows rather
  than flattening them into plain text.
- Kept remote dataset scripts, pickle deserialization, automatic archive
  extraction, and execution of downloaded code out of the workflow.

### Documentation and development

- Expanded [PACKAGE.md](PACKAGE.md) with catalog, audit, policy, batching,
  resume, local import, preparation, and safety documentation.
- Added [CONTRIBUTING.md](CONTRIBUTING.md) with evidence, fixture, regeneration,
  source-audit, and validation requirements.
- Updated the README and Docsify sidebar to expose the toolkit and coverage
  report without removing the curated collection.
- Added catalog/coverage freshness checks and installed-package catalog/planning
  checks to the Python 3.10–3.13 CI workflow.
- Excluded generated build/distribution directories from Ruff checks.

### Validation recorded

- **70 offline unit tests passed**, including tests against the installed wheel
  in a clean environment without optional runtime dependencies.
- Ruff linting/format checks and generated inventory/catalog/coverage freshness
  checks passed.
- Built and installed the **0.2.0 wheel**; verified packaged catalog resources,
  whole-catalog planning, selected-source batch execution, checksum-verified
  resume, and translation preparation.
- Live sample checks: **110 successful source samples**, with **10 sources
  skipped under the test size cap**. These checks covered one selected variant
  per adapter, not every configuration or complete dataset. See
  [reports/source-smoke.json](reports/source-smoke.json).
- Parsed all **86 pinned NusaX/IndoNLU split/file profiles**, totaling
  **85,499 rows**. These are parser/schema checks, not a count of unique
  examples, semantic quality certification, or legal approval. See
  [reports/pinned-split-check.json](reports/pinned-split-check.json).

### Remaining limitations

- Full source visibility does not mean every dataset is accessible, licensed
  for a given purpose, automatically downloadable, or ready for training.
- Broken hosts, missing files, gated datasets, manual provider workflows, and
  uncertain terms remain explicit blockers in the coverage report.
- The repository does not contain a complete mirror of all listed datasets.
- Row limits do not impose a hard total network/disk quota on Hugging Face
  streaming; upstream buffering and range requests can exceed a row sample.
- No automatic PII redaction, language identification, fuzzy deduplication,
  evaluation decontamination, tokenization, or media processing was added.
- Software licensing still requires the repository owner's decision before
  publishing a release. Dataset licenses remain independent.

## 0.1.0 — Initial Python dataset toolkit

Implementation date not recorded. Introduced executable tooling alongside the
existing Markdown collection and Docsify website.

### Added — Packaging and discovery

- Added the `awesome-indonesian-datasets` package, the
  `indonesian_datasets` import namespace, and the `indo-datasets` CLI.
- Added `pyproject.toml`, a `src/` package layout, Python 3.10+ support, and a
  base installation with no runtime dependencies.
- Added optional Hugging Face and development dependencies.
- Added a pinned, machine-readable loader registry with field mappings,
  official splits, source links, license notes, and citation links.
- Added **seven canonical loaders**:

  | ID | Dataset/task | Language |
  | --- | --- | --- |
  | `nusax-id` | NusaX sentiment | Indonesian |
  | `nusax-jv` | NusaX sentiment | Javanese |
  | `nusax-su` | NusaX sentiment | Sundanese |
  | `smsa` | IndoNLU sentiment | Indonesian |
  | `openwebtext-id` | Translated OpenWebText corpus | Indonesian |
  | `fineweb-edu-id` | Translated FineWeb-Edu corpus | Indonesian |
  | `simple-summaries-id` | Translated article/summary pairs | Indonesian |

- Added a generated inventory of **246 category-file link records**, separately
  from implemented loaders; indexed links did not imply loader support.
- Added `list`, `info`, and `sources` commands and task/language/query filtering.
- Added Python APIs `list_datasets()`, `get_dataset()`, `list_sources()`, and
  lazy `load_dataset()` iteration.

### Added — Loading and reproducible exports

- Added cached CSV/TSV loading and optional Hugging Face streaming.
- Pinned source revisions and retained original train/validation/test boundaries.
- Added local cache checksums, corruption detection, bounded file downloads,
  and configurable cache locations.
- Defaulted CLI downloads to 1,000 rows, with explicit full-split opt-in.
  Documented that row limits are not network-byte limits.
- Added canonical records with stable source IDs, language, source, split,
  revision, mapped task fields, and retained unmapped metadata.
- Mapped translated FineWeb's Indonesian `text_indonesian` column to canonical
  `text`, rather than accidentally using its English `text` column.
- Added explicit errors for missing columns, malformed rows, and invalid
  required fields instead of silent language/schema fallbacks.
- Added the `download` command, JSONL output, provenance manifests, input/output
  hashes, row counts, atomic export directories, and refusal to overwrite
  existing output paths.
- Added license warnings without treating public availability as permission.

### Added — Cleaning, deduplication, and splitting

- Added `clean_text()`, `Pipeline`, and the `preprocess` command.
- Normalized Unicode NFC, removed zero-width spaces/BOMs, and collapsed
  whitespace while retaining casing, negation, punctuation, accents, emoji,
  URLs, and regional-language content by default.
- Made lowercasing and URL removal explicit opt-ins.
- Added configurable text fields, character-length filtering, exact
  deduplication keys, and input/output/rejection/duplicate statistics.
- Preserved original records rather than mutating callers' input dictionaries.
- Added strict JSONL parsing with line-numbered errors.
- Added `split_records()` and the separate `split` command for new corpora,
  requiring `--new-corpus` acknowledgement in the CLI.
- Used seeded, order-independent content hashing so equal grouping keys remain
  in the same split. Documented approximate ratios, in-memory storage, and the
  need to group related documents/users/parallel examples appropriately.
- Did not automatically re-split existing benchmarks or claim fuzzy/semantic
  leakage protection.

### Documentation, testing, and build checks

- Added [PACKAGE.md](PACKAGE.md), README installation/usage examples, a toolkit
  sidebar link, contribution instructions, and a prioritized improvement roadmap.
- Added offline network-mocked tests, inventory freshness checks, Ruff
  linting/formatting, wheel builds, and a Python 3.10–3.13 CI matrix.
- Recorded **32 passing offline tests** by the end of the milestone.
- Verified all **12 CSV/TSV splits** in the initial NusaX/SmSA coverage,
  totaling **15,760 rows**, and small live samples from all three Hugging Face
  sources. Data stayed in external test caches rather than being added to the
  repository as a mirrored corpus.
- Built and installed the 0.1.0 wheel in a clean environment and checked the
  download/preprocess CLI workflow.

## Original repository — Unversioned curated collection

Original implementation date not verified. This is the baseline observed before
the Python toolkit was added; individual earlier commits cannot be reconstructed
from this checkout.

### Existing features

- A curated README of Indonesian and regional-language AI datasets, source
  links, dataset descriptions, sizes, tasks, selected papers, and citations.
- An overview page and 14 detailed category guides covering NLU, token
  classification, knowledge graphs, text corpora, local languages, multimodal
  resources, similarity/paraphrasing, QA, speech, summarization, translation,
  dictionaries, pretrained models, and tools/libraries.
- A Docsify website with navigation/sidebar files, category browsing, search,
  image zoom, code highlighting, and GitHub Pages configuration.
- Standalone Hugging Face loading examples and dataset-repository clone
  commands in the README.
- Dataset contribution guidance and warnings that upstream datasets have
  independent licensing and usage restrictions.
- A feature roadmap proposing centralized dataset access, standardized formats,
  preprocessing, validation, documentation, benchmarks, and community workflows.

### Baseline limitations

- The collection was documentation-first: no installable Python package,
  unified loader registry, preprocessing implementation, automated batch
  workflow, or reproducible export-manifest system was present.
- Source references mixed datasets, collections, papers, models, and tools;
  availability and licensing needed individual review.
- Links and descriptions did not imply that all referenced data was hosted,
  downloadable, redistributable, or suitable for a shared training corpus.
