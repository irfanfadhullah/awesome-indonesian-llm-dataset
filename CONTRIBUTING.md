# Contributing data sources and adapters

Keep the awesome list readable and the toolkit's claims verifiable.

1. Add the source to the appropriate `data/*.md` category, with original source,
   dataset-specific terms, language/configuration, task, splits and citation.
   Do not substitute a repository's software license for the dataset license.
2. Regenerate `resources.json` and `catalog.json`:
   ```bash
   python scripts/build_inventory.py
   python scripts/build_catalog.py
   ```
3. For task-ready canonical loaders, add an explicitly pinned definition to
   `src/indonesian_datasets/registry.json` and an offline fixture. For source-level
   files, preserve the original schema. Generic file discovery is not evidence
   of correct labels, task suitability, license approval or official partitions.
4. Add an explicit JSON preparation recipe when the task schema has been
   reviewed. Test span offsets/token alignment where relevant. Never run remote
   scripts, deserialize pickle, decode media automatically or flatten nested
   supervision just to make it fit a text-only loader.
5. Audit changed source IDs into a **new** report:
   ```bash
   indo-datasets audit --ids SOURCE_ID --output reports/source-update.json
   ```
   Review observations before deliberately updating the bundled audit snapshot.
   Preserve other source observations. Changed catalog fingerprints invalidate
   old observations automatically; stale evidence must not be relabeled as fresh.
6. Verify small authorized live samples separately from offline tests, respecting
   access requirements and download caps. Do not commit source data or secrets.
   Report failures honestly; a HEAD request is not a validated dataset load.
7. Regenerate `SOURCE_COVERAGE.md`, then run:
   ```bash
   python -m unittest discover -s tests -v
   python scripts/build_inventory.py --check
   python scripts/build_catalog.py --check
   python scripts/report_coverage.py --check
   ruff check src scripts tests
   ruff format --check src scripts tests
   python -m build
   ```

Examples, aliases and mirror relationships should cite their evidence. Never
delete a source because one audit timed out, and never mark a gated source as
working by bypassing its access process. Software licensing must be selected by
the repository owner before distributing a release; dataset rights remain separate.
