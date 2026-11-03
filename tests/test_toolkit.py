import contextlib
import hashlib
import io
import json
import math
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

from indonesian_datasets import (
    Pipeline,
    clean_text,
    get_dataset,
    list_datasets,
    list_sources,
    load_dataset,
    split_records,
)
from indonesian_datasets.cli import main
from indonesian_datasets.loaders import _canonicalize, _download, _hub_rows
from indonesian_datasets.storage import file_sha256, read_jsonl, write_bundle


class Response(io.BytesIO):
    def __init__(self, content, declared_length=None):
        super().__init__(content)
        self.headers = {} if declared_length is None else {"Content-Length": str(declared_length)}


class RegistryTests(unittest.TestCase):
    def test_registry_invariants(self):
        entries = list_datasets()
        self.assertEqual(len(entries), len({entry["id"] for entry in entries}))
        for entry in entries:
            self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
            self.assertIn("text", entry["fields"])
            self.assertIn(entry["backend"], {"csv", "huggingface"})
            self.assertTrue(entry["license_url"].startswith("https://"))
            self.assertTrue(entry["license_note"])
            self.assertTrue(entry.get("splits", entry.get("files")))

    def test_filter_and_lookup(self):
        self.assertEqual([x["id"] for x in list_datasets(language="jv")], ["nusax-jv"])
        self.assertEqual(len(list_datasets(task="summarization")), 1)
        self.assertEqual(get_dataset("nusax-id")["language"], "id")
        self.assertTrue(list_datasets(query="FINEWEB"))
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            get_dataset("does-not-exist")

    def test_registry_is_not_mutable_global_state(self):
        get_dataset("nusax-id")["id"] = "changed"
        self.assertEqual(get_dataset("nusax-id")["id"], "nusax-id")

    def test_full_documentation_inventory(self):
        sources = list_sources()
        self.assertGreater(len(sources), 100)
        self.assertEqual(len(sources), len({(x["url"], x["category"]) for x in sources}))
        self.assertTrue(list_sources(query="nusax"))
        self.assertTrue(list_sources(category="09_speech_audio"))
        self.assertTrue(all(x["mentions"] for x in sources))


class CleaningTests(unittest.TestCase):
    def test_conservative_cleaning(self):
        self.assertEqual(clean_text("  Saya\tTIDAK\u00a0suka!\n 😊 "), "Saya TIDAK suka! 😊")
        self.assertEqual(clean_text("cafe\u0301"), "café")
        self.assertEqual(clean_text("a\u200bb\ufeff"), "ab")
        self.assertEqual(clean_text("👩\u200d💻"), "👩\u200d💻")

    def test_opt_in_transforms(self):
        self.assertEqual(clean_text("SAYA Tidak", lowercase=True), "saya tidak")
        self.assertEqual(clean_text("a https://example.com b", remove_urls=True), "a b")
        self.assertIn("https://", clean_text("https://example.com"))
        with self.assertRaises(TypeError):
            clean_text(None)

    def test_pipeline_counts_and_no_mutation(self):
        original = {"text": "  Halo  dunia ", "label": "positive"}
        pipeline = Pipeline(deduplicate=True, min_chars=3, max_chars=20)
        result = list(
            pipeline.process(
                [
                    original,
                    {"text": "Halo dunia"},
                    {"text": ""},
                    {"text": None},
                    {},
                    {"text": "x" * 21},
                ]
            )
        )
        self.assertEqual(result, [{"text": "Halo dunia", "label": "positive"}])
        self.assertEqual(original["text"], "  Halo  dunia ")
        self.assertEqual(
            pipeline.stats,
            {
                "input": 6,
                "output": 1,
                "invalid": 2,
                "too_short": 1,
                "too_long": 1,
                "duplicates": 1,
            },
        )
        list(pipeline.process([{"text": "Halo dunia"}]))
        self.assertEqual(pipeline.stats["output"], 1)
        self.assertEqual(pipeline.stats["duplicates"], 0)

    def test_deduplication_is_opt_in(self):
        self.assertEqual(len(list(Pipeline().process([{"text": "a"}] * 2))), 2)

    def test_multiple_fields_and_dedup_key(self):
        pipeline = Pipeline(text_fields=("text", "summary"), deduplicate=True)
        rows = [
            {"text": " article ", "summary": " one "},
            {"text": "article", "summary": "two"},
            {"text": "article", "summary": None},
        ]
        self.assertEqual(len(list(pipeline.process(rows))), 2)
        self.assertEqual(pipeline.stats["invalid"], 1)
        pipeline = Pipeline(deduplicate=True, dedup_fields=("text", "label"))
        rows = [{"text": "same", "label": label} for label in ["positive", "negative"]]
        self.assertEqual(len(list(pipeline.process(rows))), 2)
        with self.assertRaisesRegex(ValueError, "missing"):
            list(pipeline.process([{"text": "missing label"}]))

    def test_pipeline_config_validation(self):
        for kwargs in [
            {"min_chars": -1},
            {"min_chars": 5, "max_chars": 2},
            {"text_fields": ()},
            {"dedup_fields": ()},
        ]:
            with self.assertRaises(ValueError):
                Pipeline(**kwargs)


class SplittingTests(unittest.TestCase):
    def test_deterministic_grouped_splits(self):
        records = [{"text": str(i), "group": str(i // 2)} for i in range(1000)]
        a = split_records(records, group_fields=("group",))
        b = split_records(reversed(records), group_fields=("group",))
        membership_a = {row["text"]: name for name, rows in a.items() for row in rows}
        membership_b = {row["text"]: name for name, rows in b.items() for row in rows}
        self.assertEqual(membership_a, membership_b)
        self.assertEqual(sum(map(len, a.values())), 1000)
        for i in range(0, 1000, 2):
            self.assertEqual(membership_a[str(i)], membership_a[str(i + 1)])
        self.assertTrue(700 < len(a["train"]) < 900)
        self.assertNotIn("split", records[0])
        self.assertNotEqual(a, split_records(records, seed=43, group_fields=("group",)))

    def test_exact_duplicates_stay_together(self):
        splits = split_records([{"text": "same"}] * 10)
        self.assertEqual(sorted(map(len, splits.values())), [0, 0, 10])

    def test_split_validation(self):
        for ratios in [(1, 1, 1), (-0.1, 0.5, 0.6), (math.nan, 0, 1), (1, 0)]:
            with self.assertRaises(ValueError):
                split_records([], ratios=ratios)
        with self.assertRaises(ValueError):
            split_records([{}])
        with self.assertRaises(ValueError):
            split_records([{"text": None}])
        with self.assertRaises(ValueError):
            split_records([], group_fields=())
        self.assertEqual(len(split_records([{"text": "x"}], ratios=(0, 0, 1))["test"]), 1)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)

    @patch("indonesian_datasets.loaders.urlopen")
    def test_lazy_csv_cache_and_provenance(self, network):
        network.return_value = Response(
            b'id,text,label\n1,"Halo, dunia",positive\n2,Baik,neutral\n'
        )
        rows = load_dataset("nusax-id", cache_dir=self.cache, limit=1)
        network.assert_not_called()
        record = next(rows)
        rows.close()
        self.assertEqual(record["text"], "Halo, dunia")
        self.assertEqual(record["metadata"], {"id": "1"})
        self.assertEqual(record["source"], "nusax-id")
        self.assertEqual(record["split"], "train")
        self.assertEqual(record["revision"], get_dataset("nusax-id")["revision"])
        self.assertEqual(len(record["id"]), 64)
        again = list(load_dataset("nusax-id", cache_dir=self.cache, limit=1))
        self.assertEqual(again, [record])
        network.assert_called_once()

    @patch("indonesian_datasets.loaders.urlopen")
    def test_headerless_tsv_preserves_quotes(self, network):
        network.return_value = Response(b'"sangat bagus"\tpositive\nburuk\tnegative\n')
        with warnings.catch_warnings(record=True) as captured:
            records = list(load_dataset("smsa", cache_dir=self.cache))
        self.assertEqual(records[0]["text"], '"sangat bagus"')
        self.assertEqual(records[1]["label"], "negative")
        self.assertTrue(any("license unverified" in str(w.message) for w in captured))

    @patch("indonesian_datasets.loaders.urlopen")
    def test_zero_limit_and_invalid_arguments_do_not_fetch(self, network):
        self.assertEqual(list(load_dataset("nusax-id", limit=0)), [])
        for kwargs in [{"limit": -1}, {"split": "other"}, {"max_download_bytes": 0}]:
            with self.assertRaises(ValueError):
                load_dataset("nusax-id", **kwargs)
        network.assert_not_called()

    @patch("indonesian_datasets.loaders.urlopen")
    def test_download_caps_with_and_without_content_length(self, network):
        for declared_length in [None, 10]:
            network.return_value = Response(b"1234567890", declared_length=declared_length)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                _download("https://example.com/data", self.cache, 5)
            self.assertEqual(list(self.cache.iterdir()), [])

    @patch("indonesian_datasets.loaders.urlopen")
    def test_cache_corruption_redownloads(self, network):
        network.return_value = Response(b"original")
        path = _download("https://example.com", self.cache, 100)
        path.write_bytes(b"corrupt")
        network.return_value = Response(b"original")
        with self.assertWarnsRegex(UserWarning, "integrity"):
            repaired = _download("https://example.com", self.cache, 100)
        self.assertEqual(repaired.read_bytes(), b"original")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            _download("https://example.com", self.cache, 2)

    @patch("indonesian_datasets.loaders.urlopen")
    def test_malformed_csv_fails(self, network):
        network.return_value = Response(b"id,text,label\n1,hello\n")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            list(load_dataset("nusax-id", cache_dir=self.cache))

    def test_missing_columns_and_nulls_fail(self):
        entry = get_dataset("nusax-id")
        for row in [{"text": "x"}, {"text": None, "label": "positive"}]:
            with self.assertRaises(ValueError):
                _canonicalize(row, entry, "train")

    def test_translated_field_mappings(self):
        fineweb = _canonicalize(
            {"text": "English", "text_indonesian": "Indonesia"},
            get_dataset("fineweb-edu-id"),
            "train",
        )
        self.assertEqual(fineweb["text"], "Indonesia")
        self.assertEqual(fineweb["metadata"]["text"], "English")
        summaries = _canonicalize(
            {"translated_text": "Artikel", "translated_summary": "Ringkasan"},
            get_dataset("simple-summaries-id"),
            "train",
        )
        self.assertEqual(summaries["summary"], "Ringkasan")

    @patch("indonesian_datasets.loaders._hub_rows")
    def test_hub_is_lazy_and_limited(self, hub):
        def source():
            yield {"translated_text": "Halo"}
            raise AssertionError("Read beyond row limit")

        hub.return_value = source()
        rows = load_dataset("openwebtext-id", limit=1, cache_dir=self.cache)
        hub.assert_not_called()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = list(rows)
        self.assertEqual(result[0]["text"], "Halo")
        hub.assert_called_once()

    def test_hub_backend_pins_revision_and_streams(self):
        fake = types.ModuleType("datasets")
        fake.load_dataset = Mock(return_value=iter([{"translated_text": "Halo"}]))
        entry = get_dataset("openwebtext-id")
        with patch.dict("sys.modules", {"datasets": fake}):
            self.assertEqual(
                list(_hub_rows(entry, "train", self.cache)),
                [
                    {"translated_text": "Halo"},
                ],
            )
        fake.load_dataset.assert_called_once_with(
            entry["repo_id"],
            name=None,
            revision=entry["revision"],
            split="train",
            streaming=True,
            cache_dir=str(self.cache / "huggingface"),
        )

    def test_hub_dependency_is_optional(self):
        with patch.dict("sys.modules", {"datasets": None}):
            with self.assertRaisesRegex(ImportError, "Install Hub support"):
                list(_hub_rows(get_dataset("openwebtext-id"), "train", self.cache))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_unicode_export_checksums_and_no_overwrite(self):
        output = self.root / "export"
        manifest = write_bundle(
            output, {"data.jsonl": [{"text": "café 🇮🇩"}]}, {"operation": "test"}
        )
        data = output / "data.jsonl"
        self.assertEqual(manifest["files"]["data.jsonl"]["records"], 1)
        self.assertEqual(manifest["files"]["data.jsonl"]["sha256"], file_sha256(data))
        self.assertEqual(list(read_jsonl(data)), [{"text": "café 🇮🇩"}])
        before = data.read_bytes()
        with self.assertRaises(FileExistsError):
            write_bundle(output, {"data.jsonl": []}, {})
        self.assertEqual(data.read_bytes(), before)

    def test_failed_export_is_not_published(self):
        def failing():
            yield {"text": "first"}
            raise ValueError("upstream failed")

        with self.assertRaisesRegex(ValueError, "upstream"):
            write_bundle(self.root / "export", {"data.jsonl": failing()}, {})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unsafe_filenames_rejected(self):
        with self.assertRaises(ValueError):
            write_bundle(self.root / "export", {"../escape.jsonl": []}, {})

    def test_strict_jsonl_error_locations(self):
        path = self.root / "bad.jsonl"
        for content in ["{}\nnope", "{}\n[]", '{}\n{"x": NaN}']:
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2:"):
                list(read_jsonl(path))

    def test_broken_output_symlink_not_overwritten(self):
        output = self.root / "output"
        output.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            write_bundle(output, {"data.jsonl": []}, {})
        self.assertTrue(output.is_symlink())


class CLITests(unittest.TestCase):
    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_offline_discovery_and_errors(self):
        status, stdout, _ = self.invoke(["list", "--language", "jv", "--json"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout)[0]["id"], "nusax-jv")
        self.assertEqual(self.invoke(["info", "nusax-id"])[0], 0)
        self.assertTrue(json.loads(self.invoke(["sources", "--query", "speech"])[1]))
        status, _, stderr = self.invoke(["info", "nope"])
        self.assertEqual(status, 1)
        self.assertIn("Unknown dataset", stderr)

    def test_local_preprocess_split_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text('{"text":" Halo "}\n{"text":"Halo"}\n', encoding="utf-8")
            clean = root / "clean"
            status, stdout, stderr = self.invoke(
                [
                    "preprocess",
                    str(source),
                    "--output",
                    str(clean),
                    "--deduplicate",
                ]
            )
            self.assertEqual(status, 0, stderr)
            manifest = json.loads(stdout)
            self.assertEqual(manifest["stats"]["duplicates"], 1)
            self.assertEqual(
                manifest["input"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
            )
            status, _, stderr = self.invoke(
                [
                    "split",
                    str(clean / "data.jsonl"),
                    "--output",
                    str(root / "split"),
                    "--new-corpus",
                ]
            )
            self.assertEqual(status, 0, stderr)
            self.assertTrue((root / "split" / "manifest.json").exists())

    @patch("indonesian_datasets.loaders.urlopen")
    def test_cli_download(self, network):
        network.return_value = Response(b"id,text,label\n1,Halo,positive\n")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, stdout, stderr = self.invoke(
                [
                    "download",
                    "nusax-id",
                    "--limit",
                    "1",
                    "--output",
                    str(root / "export"),
                    "--cache-dir",
                    str(root / "cache"),
                ]
            )
            self.assertEqual(status, 0, stderr)
            manifest = json.loads(stdout)
            self.assertEqual(manifest["limit"], 1)
            self.assertEqual(manifest["files"]["data.jsonl"]["records"], 1)


if __name__ == "__main__":
    unittest.main()
