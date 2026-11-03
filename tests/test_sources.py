import base64
import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from indonesian_datasets.adapters import (
    BoundedReader,
    choose_variant,
    iter_file,
    iter_source,
    json_safe,
)
from indonesian_datasets.audit import _github_files, _hub, _native_hub_files, audit_sources
from indonesian_datasets.batch import plan_sources, run_batch, verify_bundle
from indonesian_datasets.catalog import fingerprint, get_source, list_catalog
from indonesian_datasets.cli import main
from indonesian_datasets.recipes import prepare_records, validate_prepared
from indonesian_datasets.registry import list_sources
from indonesian_datasets.storage import read_jsonl, save_json


def source(**overrides):
    return {
        "id": "fixture",
        "name": "Fixture",
        "resource_type": "dataset_source",
        "backend": "file",
        "access": "public",
        "availability": "reachable",
        "license": "cc-by-4.0",
        "license_note": "fixture only",
        "revision": "a" * 40,
        "task": "classification",
        "role": "unspecified",
        "language": "id",
        "instructions": "fixture",
        "homepage": "https://example.org/file.csv",
        "variants": [
            {
                "config": "default",
                "split": "train",
                "format": "csv",
                "urls": ["https://example.org/train.csv"],
            }
        ],
        **overrides,
    }


class CatalogTests(unittest.TestCase):
    def test_all_documented_category_urls_covered(self):
        entries = list_catalog(include_audit=False)
        self.assertEqual(len(entries), len({e["id"] for e in entries}))
        urls = {url for entry in entries for url in entry["source_urls"]}
        self.assertTrue({s["url"] for s in list_sources()} <= urls)
        self.assertTrue(all(e["instructions"] and e["license_note"] for e in entries))
        self.assertTrue(all("resource_type" in e and "role" in e for e in entries))

    def test_missing_url_and_permission_sources_visible(self):
        speech = get_source("indonesian-unsupervised-speech")
        self.assertEqual(speech["access"], "permission_required")
        self.assertIsNone(speech["homepage"])
        self.assertIn("260", speech["instructions"])
        self.assertEqual(get_source("indocloze")["role"], "evaluation")

    def test_all_nusax_profiles_and_indonlu_tasks(self):
        entries = list_catalog(include_audit=False)
        self.assertEqual(len([e for e in entries if e["id"].startswith("nusax-lexicon-")]), 11)
        self.assertEqual(len([e for e in entries if e["id"].startswith("indonlu-")]), 11)
        self.assertEqual(get_source("nusax-mt")["task"], "translation")
        self.assertEqual(get_source("nusax-ace")["language"], "ace")

    def test_filters(self):
        self.assertTrue(list_catalog(resource_type="tool"))
        self.assertTrue(list_catalog(query="Minang"))
        self.assertTrue(all(e["language"] == "jv" for e in list_catalog(language="jv")))

    def test_stale_audit_does_not_override_catalog(self):
        entry = next(e for e in list_catalog(include_audit=False) if e["id"] == "nusax-id")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            report = {
                "sources": {
                    entry["id"]: {
                        "catalog_fingerprint": "old",
                        "updates": {"license": "fabricated"},
                    }
                }
            }
            save_json(path, report)
            self.assertNotEqual(get_source("nusax-id", audit_path=path)["license"], "fabricated")
            report["sources"][entry["id"]]["catalog_fingerprint"] = fingerprint(entry)
            save_json(path, report, overwrite=True)
            self.assertEqual(get_source("nusax-id", audit_path=path)["license"], "fabricated")

    def test_readable_index_does_not_waive_permission(self):
        entry = next(
            e for e in list_catalog(include_audit=False) if e["access"] == "permission_required"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            save_json(
                path,
                {
                    "sources": {
                        entry["id"]: {
                            "catalog_fingerprint": fingerprint(entry),
                            "updates": {"access": "public"},
                        }
                    }
                },
            )
            self.assertEqual(
                get_source(entry["id"], audit_path=path)["access"], "permission_required"
            )


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_csv_and_headerless_rows(self):
        path = self.write("data.csv", 'text,label\n"Halo, dunia",positive\n')
        self.assertEqual(
            list(iter_file(path, format="csv")), [{"text": "Halo, dunia", "label": "positive"}]
        )
        rows = list(iter_file(path, format="csv_rows"))
        self.assertEqual(rows[0], {"columns": ["text", "label"]})
        path = self.write("data.tsv", "a\tb\n1\t2\n")
        self.assertEqual(len(list(iter_file(path, format="tsv_rows"))), 2)

    def test_conll_alignment_and_final_sentence(self):
        path = self.write("tokens.txt", "Saya\tO\nJakarta\tB-LOC\n\nBandung\tB-LOC\n")
        rows = list(iter_file(path, format="conll"))
        self.assertEqual(rows[0]["tokens"], ["Saya", "Jakarta"])
        self.assertEqual(rows[0]["tags"], ["O", "B-LOC"])
        self.assertEqual(rows[1]["tokens"], ["Bandung"])

    def test_conllu_preserves_multitokens_empty_nodes_and_comments(self):
        path = self.write(
            "ud.conllu",
            "# sent_id = x\n1-2\ttakkan\t_\t_\t_\t_\t_\t_\t_\t_\n"
            "1\ttak\ttak\tPART\t_\t_\t0\troot\t_\t_\n"
            "1.1\takan\takan\tAUX\t_\t_\t1\tdep\t_\t_\n",
        )
        row = next(iter_file(path, format="conllu"))
        self.assertEqual(row["rows"][0][0], "1-2")
        self.assertEqual(row["rows"][2][0], "1.1")
        self.assertEqual(row["comments"], ["# sent_id = x"])

    def test_conllu_tokens_are_syntactic_words_only(self):
        path = self.write(
            "mwt.conllu",
            "1-2\tbukunya\t_\t_\t_\t_\t_\t_\t_\t_\n"
            "1\tbuku\tbuku\tNOUN\t_\t_\t0\troot\t_\t_\n"
            "2\tnya\tnya\tPRON\t_\t_\t1\tnmod\t_\t_\n"
            "2.1\tada\tada\tVERB\t_\t_\t_\t_\t_\t_\n",
        )
        row = next(iter_file(path, format="conllu"))
        self.assertEqual(row["tokens"], ["buku", "nya"])
        self.assertEqual(len(row["rows"]), 4)

    def test_conll_hash_lines_are_tokens_not_comments(self):
        # IndoNLU KEPS/NER/POS files contain hashtag tokens such as "#BCA\tB".
        path = self.write("tweets.txt", "#BCA\tB\nterbaik\tO\n#\tO\n")
        row = next(iter_file(path, format="conll"))
        self.assertEqual(row["tokens"], ["#BCA", "terbaik", "#"])
        self.assertEqual(row["tags"], ["B", "O", "O"])
        self.assertEqual(row["comments"], [])

    def test_conll_whitespace_columns_and_docstart(self):
        path = self.write(
            "conll2003.txt",
            "-DOCSTART- -X- -X- O\n\nBudi NNP B-NP B-PER\npergi VB B-VP O\n"
            "-DOCSTART- -X- -X- O\nAni NNP B-NP B-PER\n",
        )
        rows = list(iter_file(path, format="conll"))
        self.assertEqual([r["tokens"] for r in rows], [["Budi", "pergi"], ["Ani"]])
        self.assertEqual(rows[0]["tags"], ["B-PER", "O"])
        self.assertEqual(rows[0]["rows"][0], ["Budi", "NNP", "B-NP", "B-PER"])

    def test_conll_inconsistent_or_empty_columns_fail(self):
        for name, text in [
            ("ragged.txt", "Budi\tB-PER\npergi\tVB\tO\n"),
            ("empty-tag.txt", "Budi\t\n"),
            ("empty-token.txt", "\tO\n"),
        ]:
            with self.assertRaisesRegex(ValueError, "token/tag row at line"):
                list(iter_file(self.write(name, text), format="conll"))

    @patch("indonesian_datasets.adapters.load_dataset")
    def test_registered_sources_use_the_data_envelope(self, loader):
        loader.return_value = iter(
            [
                {
                    "id": "abc",
                    "text": "Saya suka.",
                    "label": "positive",
                    "language": "id",
                    "source": "nusax-id",
                    "split": "train",
                    "revision": "r" * 40,
                    "metadata": {"id": "7"},
                }
            ]
        )
        entry = source(backend="registered", splits=["train"], variants=None)
        record = next(iter_source(entry, split="train", cache_dir=self.root))
        self.assertEqual(
            record["data"], {"text": "Saya suka.", "label": "positive", "metadata": {"id": "7"}}
        )
        self.assertEqual((record["id"], record["revision"]), ("abc", "r" * 40))
        self.assertNotIn("text", record)
        recipe_path = Path(__file__).resolve().parents[1] / "examples/recipes/classification.json"
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        prepared = next(prepare_records([record], recipe))["prepared"]
        self.assertEqual(prepared, {"text": "Saya suka.", "label": "positive"})

    @patch("indonesian_datasets.adapters._download")
    def test_default_cache_follows_xdg_for_every_backend(self, download):
        from indonesian_datasets.loaders import default_cache_dir

        with patch.dict("os.environ", {"XDG_CACHE_HOME": str(self.root / "xdg")}):
            self.assertEqual(default_cache_dir(), self.root / "xdg" / "indonesian-datasets")
            download.return_value = self.write("c.csv", "text\nHalo\n")
            list(iter_source(source(), split="train"))
            self.assertEqual(download.call_args.args[1], default_cache_dir())
        with patch.dict("os.environ", {"XDG_CACHE_HOME": ""}):
            self.assertEqual(default_cache_dir().parent, Path.home() / ".cache")

    def test_nested_json_not_flattened(self):
        raw = {"data": [{"paragraphs": [{"context": "Jakarta"}]}]}
        path = self.write("qa.json", json.dumps(raw))
        self.assertEqual(list(iter_file(path, format="json")), [raw])
        path = self.write("qa.jsonl", '{"a": 1}\n\n{"a": 2}\n')
        self.assertEqual(len(list(iter_file(path, format="jsonl"))), 2)

    def test_compression_bomb_limit(self):
        path = self.root / "big.txt.gz"
        path.write_bytes(gzip.compress(b"x" * 100000))
        with self.assertRaisesRegex(ValueError, "Decompressed"):
            list(iter_file(path, format="lines", max_uncompressed_bytes=100))

    def test_bounded_reader_exact_limit(self):
        reader = io.BufferedReader(BoundedReader(io.BytesIO(b"12345"), 5))
        self.assertEqual(reader.read(), b"12345")

    def test_bad_formats_and_rows_fail(self):
        for name, text, fmt in [
            ("bad.csv", "a,b\n1\n", "csv"),
            ("bad.jsonl", "[]\n", "jsonl"),
            ("bad.txt", "onecolumn\n", "conll"),
        ]:
            with self.assertRaises(ValueError):
                list(iter_file(self.write(name, text), format=fmt))
        with self.assertRaises(ValueError):
            list(iter_file(self.root / "absent", format="pickle"))

    def test_bytes_nan_and_safe_values(self):
        encoded = json_safe({"audio": b"abc", "missing": float("nan")})
        self.assertEqual(base64.b64decode(encoded["audio"]["data"]), b"abc")
        self.assertIsNone(encoded["missing"])
        with self.assertRaises(TypeError):
            json_safe(object())

    def test_configuration_must_be_explicit(self):
        entry = source(
            variants=[
                {"config": "id", "split": "train"},
                {"config": "jv", "split": "train"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "explicit config"):
            choose_variant(entry, None, "train")
        self.assertEqual(choose_variant(entry, "id", "train")["config"], "id")

    @patch("indonesian_datasets.adapters._download")
    def test_raw_adapter_preserves_schema_and_limit(self, download):
        download.return_value = self.write(
            "source.csv", "question,answer\nApa?,Jakarta\nSiapa?,Saya\n"
        )
        records = list(iter_source(source(), split="train", limit=1, cache_dir=self.root))
        self.assertEqual(records[0]["data"], {"question": "Apa?", "answer": "Jakarta"})
        self.assertEqual(records[0]["split"], "train")
        self.assertNotIn("text", records[0])
        self.assertEqual(len(records), 1)


class AuditTests(unittest.TestCase):
    def test_native_hub_data_files_are_script_free_and_pinned(self):
        result = _native_hub_files(
            {
                "id": "owner/repo",
                "sha": "a" * 40,
                "siblings": [
                    {"rfilename": "repo.py"},
                    {"rfilename": "id/train-00000.parquet", "size": 123},
                ],
            },
            {"repo_id": "owner/repo"},
        )
        self.assertEqual(result["backend"], "hf_parquet")
        self.assertIn("a" * 40, result["variants"][0]["urls"][0])
        self.assertEqual(result["variants"][0]["split"], "train")

    @patch("indonesian_datasets.audit._github_tree")
    def test_github_index_preserves_unknown_partitions(self, tree):
        tree.return_value = {
            "tree": [
                {"type": "blob", "path": "data/train.csv", "size": 20},
                {"type": "blob", "path": "tests/fixture.csv", "size": 10},
                {"type": "blob", "path": "requirements.txt", "size": 10},
                {"type": "blob", "path": "data/unsafe.pkl", "size": 10},
            ]
        }
        result = _github_files({"github_repo": "owner/repo", "task": "classification"}, "a" * 40)
        self.assertEqual(len(result["variants"]), 1)
        self.assertEqual(result["variants"][0]["split"], "unspecified")

    @patch("indonesian_datasets.audit._github_tree")
    def test_nusawrites_known_split_convention(self, tree):
        tree.return_value = {
            "tree": [{"type": "blob", "path": "data/nusa_alinea-emot-jav-valid.csv", "size": 20}]
        }
        result = _github_files(
            {"github_repo": "IndoNLP/nusa-writes", "task": "regional-language"}, "a" * 40
        )
        self.assertEqual(result["variants"][0]["split"], "validation")
        self.assertEqual(result["variants"][0]["config"], "nusa_alinea-emot-jav")

    @patch("indonesian_datasets.audit._json")
    def test_hub_conversion_pins_separate_revision(self, get):
        get.side_effect = [
            {
                "sha": "a" * 40,
                "id": "owner/data",
                "cardData": {"license": "mit", "language": ["id"]},
            },
            {
                "parquet_files": [
                    {
                        "config": "default",
                        "split": "train",
                        "size": 123,
                        "url": "https://huggingface.co/datasets/owner/data/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
                    }
                ]
            },
            {"sha": "b" * 40},
        ]
        result = _hub({"repo_id": "owner/data"})
        self.assertEqual(result["revision"], "a" * 40)
        self.assertEqual(result["conversion_revision"], "b" * 40)
        self.assertIn("b" * 40, result["variants"][0]["urls"][0])
        self.assertEqual(result["variants"][0]["shard_sizes"], [123])

    @patch("indonesian_datasets.audit._json")
    def test_gates_not_bypassed(self, get):
        get.return_value = {"sha": "a" * 40, "gated": "auto", "cardData": {}}
        result = _hub({"repo_id": "owner/data"})
        self.assertEqual(result["access"], "gated")
        self.assertIsNone(result["backend"])
        get.assert_called_once()

    @patch("indonesian_datasets.audit._inspect", side_effect=TimeoutError("timed out"))
    def test_timeout_is_not_classified_missing(self, inspect):
        report = audit_sources([source()], workers=1)
        self.assertEqual(report["sources"]["fixture"]["updates"]["availability"], "check_failed")


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_default_policy(self):
        entries = [
            source(),
            source(id="unknown", license="unknown"),
            source(id="manual", access="manual"),
            source(id="eval", role="evaluation"),
            source(id="moving", revision=None),
            source(id="model", resource_type="model"),
        ]
        plan = plan_sources(entries)
        self.assertEqual(len(plan["jobs"]), 1)
        reasons = {d["id"]: d["reason"] for d in plan["decisions"]}
        self.assertEqual(reasons["unknown"], "license_review_required")
        self.assertEqual(reasons["eval"], "evaluation_excluded")
        self.assertEqual(reasons["moving"], "unpinned_source")
        self.assertEqual(reasons["model"], "not_a_dataset_source")

    def test_noncommercial_restrictions_not_overridden(self):
        plan = plan_sources(
            [source(license="cc-by-nc-sa-4.0")], purpose="commercial", allow_unknown_license=True
        )
        self.assertEqual(plan["decisions"][0]["reason"], "noncommercial_restriction")

    def test_official_heldout_splits_excluded_by_default(self):
        variants = [
            dict(source()["variants"][0], split=name) for name in ["train", "validation", "test"]
        ]
        self.assertEqual(len(plan_sources([source(variants=variants)])["jobs"]), 1)
        self.assertEqual(
            len(plan_sources([source(variants=variants)], include_evaluation=True)["jobs"]), 3
        )

    def test_large_shards_and_multiple_configs_are_not_implicit(self):
        entry = source(
            variants=[dict(source()["variants"][0], config=config) for config in ["id", "jv"]]
        )
        self.assertEqual(plan_sources([entry])["decisions"][0]["reason"], "configuration_required")
        self.assertEqual(len(plan_sources([entry], all_configs=True)["jobs"]), 2)
        entry = source(variants=[dict(source()["variants"][0], shard_sizes=[1000])])
        self.assertEqual(len(plan_sources([entry], max_download_bytes=100)["jobs"]), 0)

    @patch("indonesian_datasets.batch.iter_source")
    def test_batch_resume_verifies_exports(self, iterator):
        iterator.side_effect = lambda *args, **kwargs: iter([{"text": "Halo"}])
        plan = plan_sources([source()])
        output = self.root / "batch"
        report = run_batch(plan, output)
        self.assertEqual(report["summary"], {"completed": 1})
        report = run_batch(plan, output, resume=True)
        self.assertEqual(report["summary"], {"verified_existing": 1})
        iterator.assert_called_once()
        destination = output / plan["jobs"][0]["directory"]
        (destination / "data.jsonl").write_text("corrupted", encoding="utf-8")
        report = run_batch(plan, output, resume=True)
        self.assertEqual(report["summary"], {"failed": 1})
        self.assertEqual((destination / "data.jsonl").read_text(), "corrupted")

    @patch("indonesian_datasets.batch.iter_source")
    def test_one_failed_source_does_not_abort_others(self, iterator):
        def rows(entry, **kwargs):
            if entry["id"] == "bad":
                raise ValueError("unavailable")
            yield {"text": "Halo"}

        iterator.side_effect = rows
        report = run_batch(plan_sources([source(id="bad"), source(id="good")]), self.root / "batch")
        self.assertEqual(report["summary"], {"failed": 1, "completed": 1})

    def test_resume_requires_identical_plan(self):
        output = self.root / "batch"
        plan = plan_sources([])
        run_batch(plan, output)
        changed = {**plan, "policy": {**plan["policy"], "limit": 1}}
        with self.assertRaisesRegex(ValueError, "identical"):
            run_batch(changed, output, resume=True)

    def test_path_traversal_and_symlinks_refused(self):
        plan = plan_sources([source()])
        plan["jobs"][0]["directory"] = "../escape"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            run_batch(plan, self.root / "batch")
        link = self.root / "link"
        link.symlink_to(self.root / "absent")
        with self.assertRaises(ValueError):
            run_batch(plan_sources([]), link)

    @patch("indonesian_datasets.batch._download")
    def test_archive_is_opaque_not_extracted(self, download):
        archive = self.root / "malicious.zip"
        archive.write_bytes(b"fixture-not-an-extractable-archive")
        download.return_value = archive
        entry = source(
            backend="artifact", download_url="https://example.org/file.zip", variants=None
        )
        plan = plan_sources([entry])
        report = run_batch(plan, self.root / "batch")
        self.assertEqual(report["summary"], {"completed": 1})
        manifest = verify_bundle(self.root / "batch" / plan["jobs"][0]["directory"])
        self.assertIn("artifact.bin", manifest["files"])
        self.assertEqual(manifest["plan_sha256"], report["plan_sha256"])
        self.assertEqual(manifest["policy"], plan["policy"])
        self.assertEqual((manifest["config"], manifest["split"]), ("default", "all"))
        self.assertIn("package_version", manifest)

    def test_stale_lock_reports_owner_and_recovery(self):
        output = self.root / "batch"
        plan = plan_sources([])
        run_batch(plan, output)
        (output / ".batch.lock").write_text("4242", encoding="ascii")
        with self.assertRaisesRegex(FileExistsError, r"PID 4242.*--resume"):
            run_batch(plan, output, resume=True)
        self.assertTrue((output / ".batch.lock").exists())


class RecipeTests(unittest.TestCase):
    def test_tokens_preserved_and_alignment_checked(self):
        recipe = {
            "schema_version": 1,
            "task": "token-classification",
            "fields": {"tokens": "data.tokens", "labels": "data.tags"},
        }
        record = {"data": {"tokens": ["Jakarta", "!"], "tags": ["B-LOC", "O"]}}
        prepared = list(prepare_records([record], recipe))[0]
        self.assertEqual(prepared["prepared"]["tokens"], ["Jakarta", "!"])
        self.assertEqual(prepared["original"], record)
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_prepared("token-classification", {"tokens": ["a"], "labels": []})
        with self.assertRaisesRegex(ValueError, "prohibited"):
            list(prepare_records([record], {**recipe, "clean_fields": ["tokens"]}))

    def test_qa_offsets_validated_without_cleaning(self):
        row = {
            "context": "  Jakarta",
            "question": "Di mana?",
            "answers": {"text": ["Jakarta"], "answer_start": [2]},
        }
        validate_prepared("qa-extractive", row)
        row["answers"]["answer_start"] = [0]
        with self.assertRaisesRegex(ValueError, "offset"):
            validate_prepared("qa-extractive", row)

    def test_supported_task_schemas(self):
        rows = {
            "classification": {"text": "Baik", "label": 1},
            "nli": {"premise": "Hujan", "hypothesis": "Basah", "label": "entailment"},
            "summarization": {"text": "Artikel", "summary": "Ringkasan"},
            "translation": {"source_text": "Halo", "target_text": "Hello"},
            "chat": {"messages": [{"role": "user", "content": "Halo"}]},
            "audio": {"audio": {"path": "clip.wav"}, "transcript": "Halo"},
            "image-caption": {"image": "image.jpg", "caption": "Kucing"},
            "knowledge-graph": {"head": "a", "relation": "b", "tail": "c"},
            "lexicon": {"term": "aku", "definition": "saya"},
            "raw": {"data": {"unknown": "preserved"}},
        }
        for task, row in rows.items():
            with self.subTest(task=task):
                validate_prepared(task, row)

    def test_explicit_mapping_cleaning_and_original_retained(self):
        recipe = {
            "schema_version": 1,
            "task": "classification",
            "fields": {"text": "data.columns.0", "label": "data.columns.1"},
            "clean_fields": ["text"],
        }
        stats = {}
        original = {"data": {"columns": ["  TIDAK  buruk ", "positive"]}, "split": "test"}
        row = list(prepare_records([original], recipe, stats))[0]
        self.assertEqual(row["prepared"]["text"], "TIDAK buruk")
        self.assertEqual(row["split"], "test")
        self.assertEqual(original["data"]["columns"][0], "  TIDAK  buruk ")
        self.assertEqual(stats, {"input": 1, "output": 1})


class SourceCLITests(unittest.TestCase):
    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_catalog_and_plan_offline(self):
        self.assertEqual(self.invoke(["catalog", "--summary"])[0], 0)
        self.assertEqual(self.invoke(["source-info", "nusax-mt"])[0], 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            status, stdout, stderr = self.invoke(
                ["plan", "--ids", "nusax-id", "--limit", "2", "--output", str(path)]
            )
            self.assertEqual(status, 0, stderr)
            self.assertEqual(json.loads(stdout)["download_jobs"], 1)
            self.assertEqual(json.loads(path.read_text())["policy"]["limit"], 2)

    def test_import_prepare_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "local.csv"
            source_path.write_text("text,label\nBaik,positive\n", encoding="utf-8")
            status, _, stderr = self.invoke(
                [
                    "import-file",
                    str(source_path),
                    "--format",
                    "csv",
                    "--output",
                    str(root / "import"),
                ]
            )
            self.assertEqual(status, 0, stderr)
            recipe = root / "recipe.json"
            save_json(
                recipe,
                {
                    "schema_version": 1,
                    "task": "classification",
                    "fields": {"text": "data.text", "label": "data.label"},
                },
            )
            status, _, stderr = self.invoke(
                [
                    "prepare",
                    str(root / "import/data.jsonl"),
                    "--recipe",
                    str(recipe),
                    "--output",
                    str(root / "prepared"),
                ]
            )
            self.assertEqual(status, 0, stderr)
            self.assertEqual(
                next(read_jsonl(root / "prepared/data.jsonl"))["prepared"]["text"], "Baik"
            )

    def test_preprocess_does_not_silently_empty_raw_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.jsonl"
            source.write_text('{"data":{"text":"Halo"}}\n', encoding="utf-8")
            status, _, stderr = self.invoke(
                ["preprocess", str(source), "--output", str(root / "clean")]
            )
            self.assertEqual(status, 1)
            self.assertIn("task recipe", stderr)
            self.assertFalse((root / "clean").exists())


if __name__ == "__main__":
    unittest.main()
