"""Build an offline, source-level catalog covering every documented URL.

The catalog intentionally distinguishes source records from unique datasets.
No source is marked live-verified by this generator; that is the auditor's job.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/indonesian_datasets"
OUTPUT = PACKAGE / "catalog.json"
DATA_SUFFIXES = (".csv", ".tsv", ".txt", ".jsonl", ".json", ".conllu", ".parquet")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.zst", ".tar.bz2")
TASKS = {
    "01": "classification",
    "02": "token-classification",
    "03": "knowledge-graph",
    "04": "text-generation",
    "05": "regional-language",
    "06": "multimodal",
    "07": "text-similarity",
    "08": "question-answering",
    "09": "speech",
    "10": "summarization",
    "11": "translation",
    "12": "lexicon",
}
GENERIC = {
    "github",
    "github repository",
    "repository",
    "download",
    "huggingface",
    "data",
    "data download",
    "links",
    "link",
    "paper",
    "original",
    "mendeley dataset",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:72] or "source"


def identity(url: str) -> str:
    return url.rstrip("/")


def infer_source(url: str, mentions: list[dict]) -> dict:
    parsed = urlparse(url)
    host, path = parsed.netloc, unquote(parsed.path)
    first = mentions[0]
    label = first["label"]
    name = first["heading"] if label.lower() in GENERIC else label
    category = first["category"]
    entry = {
        "id": f"{slug(host + path)}-{hashlib.sha256(url.encode()).hexdigest()[:8]}",
        "name": name or path.rsplit("/", 1)[-1] or host,
        "homepage": url,
        "source_urls": [url],
        "mentions": mentions,
        "categories": sorted({m["category"] for m in mentions}),
        "resource_type": "dataset_source",
        "task": TASKS.get(category[:2], "reference"),
        "language": "und",
        "modality": "text",
        "role": "unspecified",
        "access": "unverified",
        "availability": "not_checked",
        "backend": None,
        "license": "unknown",
        "license_evidence": [],
        "license_note": (
            "No dataset-specific terms have been verified. Public access is not a license."
        ),
        "instructions": (
            "Open the source and select a data file/configuration; a source link is not an adapter."
        ),
    }
    if category.startswith("09_"):
        entry["modality"] = "audio"
    elif category.startswith("06_"):
        entry["modality"] = "multimodal"
    if "voxlingua" in url.lower() or host == "openslr.org":
        entry["modality"] = "audio"
    if "lumina" in name.lower():
        entry["modality"] = "audio-video"
    if any(
        word in (name + " " + url).lower()
        for word in ["indommlu", "indocareer", "indoculture", "story_cloze", "su-csqa"]
    ):
        entry["role"] = "evaluation"
        entry["modality"] = "text"
    if host == "huggingface.co" and path.startswith("/datasets/"):
        repo_id = path.removeprefix("/datasets/").split("/tree/")[0].rstrip("/")
        entry.update(id="hf-" + slug(repo_id), repo_id=repo_id, backend="hf_pending")
        entry["instructions"] = (
            "Audit Hub metadata/configurations first. Remote dataset scripts are never run."
        )
    elif category.startswith("13_") or (
        host == "huggingface.co" and not path.startswith("/datasets/")
    ):
        entry.update(resource_type="model", access="not_applicable")
    elif category.startswith("14_") or host == "www.npmjs.com":
        entry.update(resource_type="tool", access="not_applicable")
    elif (
        path.endswith(".pdf")
        or host
        in {"arxiv.org", "aclanthology.org", "www.aclweb.org", "doi.org", "ieeexplore.ieee.org"}
        or "sciencedirect.com" in host
    ):
        entry.update(resource_type="paper", access="not_applicable")
    elif label.lower() == "original":
        entry.update(resource_type="reference", access="not_applicable")
    elif host == "github.com":
        parts = path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"blob", "tree"} and path.endswith(DATA_SUFFIXES):
            entry.update(
                backend="file_pending",
                github_repo="/".join(parts[:2]),
                github_ref=parts[3],
                github_path="/".join(parts[4:]),
                format=path.rsplit(".", 1)[-1],
            )
        elif any(part in parts for part in ["discussions", "issues"]):
            entry.update(resource_type="reference", access="not_applicable")
        elif len(parts) >= 2 and (len(parts) == 2 or (len(parts) >= 4 and parts[2] == "tree")):
            entry.update(
                backend="github_pending",
                github_repo="/".join(parts[:2]),
                github_ref=parts[3] if len(parts) >= 4 else "HEAD",
                github_directory="/".join(parts[4:]) if len(parts) >= 5 else "",
            )
            entry["instructions"] = (
                "Repository source: inspect its data files and original terms. "
                "No code is executed or repository blindly parsed as data."
            )
    elif host == "www.kaggle.com":
        entry.update(access="credentials_required")
        entry["instructions"] = (
            "Accept upstream terms and use your Kaggle credentials to download. "
            "Import authorized files with the local file adapter."
        )
    elif host == "drive.google.com" or host == "docs.google.com":
        entry.update(access="manual")
        entry["instructions"] = (
            "Use the source's access/download process; download an authorized copy, "
            "then import it locally. No access gates are bypassed."
        )
    elif host in {
        "data.mendeley.com",
        "zenodo.org",
        "lindat.mff.cuni.cz",
        "sourceforge.net",
        "openslr.org",
    }:
        entry.update(access="manual")
        entry["instructions"] = (
            "Select files and review dataset-specific terms at the landing page, "
            "then import an authorized local copy."
        )
    elif path.lower().endswith(ARCHIVE_SUFFIXES):
        entry.update(
            backend="artifact", download_url=url, access="public_unverified", revision=None
        )
        entry["instructions"] = (
            "Download as an opaque archive with a byte cap and checksum. "
            "No extraction, execution or automatic training ingestion."
        )
    elif path.lower().endswith(DATA_SUFFIXES + (".jsonl.zst", ".json.zst", ".txt.xz", ".jsonl.gz")):
        suffix = path.lower()
        fmt = next(
            (
                kind
                for extension, kind in [
                    (".jsonl", "jsonl"),
                    (".json", "json"),
                    (".csv", "csv_rows"),
                    (".tsv", "tsv_rows"),
                    (".conllu", "conllu"),
                    (".parquet", "parquet"),
                ]
                if extension in suffix
            ),
            "lines",
        )
        entry.update(
            backend="file", download_url=url, format=fmt, revision=None, access="public_unverified"
        )
        entry["instructions"] = (
            "Unversioned file: audit availability and explicitly accept unpinned provenance "
            "before downloading."
        )
    if "TITML" in name or "research.nii.ac.jp" in host:
        entry.update(access="permission_required", license="research-only")
        entry["license_note"] = (
            "Repository documentation requires formal academic/non-commercial permission; "
            "verify with the provider."
        )
    if "sum_liputan6" in path or ("Liputan6" in name and "forms" in path):
        entry.update(access="permission_required", license="research-only")
        entry["license_note"] = (
            "Repository documents non-commercial academic use and an access request process."
        )
    if "seuriously/genderprediction" in path and path.endswith("namatraining.txt"):
        entry["encoding"] = "latin-1"
        entry["encoding_note"] = (
            "Live source contains non-UTF-8 name bytes. Latin-1 is an explicit reversible "
            "byte-preserving decoding, not a claim about its original linguistic encoding."
        )
    return entry


def build_catalog() -> list[dict]:
    resources = json.loads((PACKAGE / "resources.json").read_text(encoding="utf-8"))
    by_url: dict[str, list[dict]] = {}
    for resource in resources:
        by_url.setdefault(identity(resource["url"]), []).extend(resource["mentions"])
    # The README contains a few datasets absent from the detailed category pages.
    category = "README"
    heading = ""
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            heading = line.lstrip("# ").strip()
        elif line.startswith("#### "):
            heading = line.lstrip("# ").strip()
        for label, url in re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", line):
            url = identity(url)
            if url not in by_url:
                by_url[url] = [
                    {
                        "label": label,
                        "heading": heading,
                        "category": category,
                        "source_file": "README.md",
                    }
                ]
        # Code-only Hub references (notably the six NusaWrites collections).
        for repo in re.findall(r"(?:datasets\.)?load_dataset\(['\"]([^'\"]+)['\"]", line):
            url = "https://huggingface.co/datasets/" + repo
            by_url.setdefault(
                url,
                [
                    {
                        "label": repo,
                        "heading": heading,
                        "category": "05_local_languages" if "nusa" in repo else category,
                        "source_file": "README.md",
                    }
                ],
            )
    entries = [infer_source(url, mentions) for url, mentions in sorted(by_url.items())]
    # Do not claim site navigation, the awesome badge or project maintenance URLs are data.
    for entry in entries:
        url = entry["homepage"]
        if "awesome-indones" in url or "awesome.re" in url or "github.io" in url:
            entry.update(resource_type="reference", access="not_applicable", backend=None)
    # Existing reviewed canonical loaders remain stable, with their actual revision/field map.
    for definition in json.loads((PACKAGE / "registry.json").read_text(encoding="utf-8")):
        entry = next(
            (e for e in entries if e["homepage"] == identity(definition["homepage"])), None
        )
        # NusaX variants deliberately are separate configurations of the same source family.
        if entry is None or definition["id"].startswith("nusax-"):
            entry = infer_source(
                definition["homepage"],
                [
                    {
                        "label": definition["name"],
                        "heading": definition["name"],
                        "category": "05_local_languages"
                        if definition["id"].startswith("nusax-")
                        else "01_natural_language_understanding",
                        "source_file": "src/indonesian_datasets/registry.json",
                    }
                ],
            )
            entries.append(entry)
        entry.update(
            id=definition["id"],
            name=definition["name"],
            task=definition["task"],
            language=definition["language"],
            backend="registered",
            access="public",
            availability="adapter_tested",
            revision=definition["revision"],
            license=definition["license"],
            license_note=definition["license_note"],
            license_evidence=[definition["license_url"]],
            splits=list(definition.get("splits", definition.get("files", {}))),
            definition=definition,
            record_mode="canonical",
        )
    add_task_profiles(entries)
    # No-URL and paper-only datasets must not disappear from coverage reports.
    for name, access, task, instructions in [
        (
            "Indonesian Unsupervised Speech",
            "permission_required",
            "speech",
            "Contact akmal@depia.wiki. Research-only; about 260 GB. "
            "No public file URL is documented.",
        ),
        (
            "IndoCloze",
            "manual",
            "evaluation",
            "The README describes the dataset but supplies no download URL. "
            "Obtain an authorized source from the authors.",
        ),
        (
            "MinangNLP",
            "manual",
            "regional-language",
            "Only a paper is linked. Obtain the original dictionary, sentiment "
            "and translation data from the authors.",
        ),
        (
            "Clean mC4 Indonesian",
            "manual",
            "text-generation",
            "The linked discussion is not a dataset download; "
            "an exact dataset revision and filtering recipe are required.",
        ),
    ]:
        entry = infer_source(
            "https://example.invalid/" + slug(name),
            [{"label": name, "heading": name, "category": "README", "source_file": "README.md"}],
        )
        entry.update(
            id=slug(name),
            homepage=None,
            source_urls=[],
            name=name,
            access=access,
            task=task,
            instructions=instructions,
        )
        if name == "IndoCloze":
            entry["role"] = "evaluation"
        entries.append(entry)
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog ID collision; add an explicit alias instead of silently merging")
    return sorted(entries, key=lambda e: e["id"])


def add_task_profiles(entries: list[dict]) -> None:
    """Pinned, inspected formats; preserve raw labels rather than guessing conversions."""
    nusax_revision = "75b0be5d982348332509f296e10d9bf9cb66d60f"
    indonlu_revision = "ce728f6926a36174b9923dfe49d6a6839b6e9bb7"
    languages = {
        "acehnese": "ace",
        "balinese": "ban",
        "banjarese": "bjn",
        "buginese": "bug",
        "english": "en",
        "indonesian": "id",
        "javanese": "jv",
        "madurese": "mad",
        "minangkabau": "min",
        "ngaju": "nij",
        "sundanese": "su",
        "toba_batak": "bbc",
    }

    def add(name, source_id, repo, revision, folder, fmt, task, language="id", split_files=None):
        homepage = f"https://github.com/{repo}/tree/{revision}/{folder}"
        entry = next(
            (
                e
                for e in entries
                if not e.get("variants") and any(url.endswith(folder) for url in e["source_urls"])
            ),
            None,
        )
        if entry is None:
            entry = infer_source(
                homepage,
                [
                    {
                        "label": name,
                        "heading": name,
                        "category": "05_local_languages"
                        if repo.endswith("nusax")
                        else "02_token_classification",
                        "source_file": "scripts/build_catalog.py",
                    }
                ],
            )
            entries.append(entry)
        variants = [
            {
                "config": "default",
                "split": split,
                "urls": [
                    f"https://raw.githubusercontent.com/{repo}/{revision}/{folder}/{filename}"
                ],
                "format": fmt,
            }
            for split, filename in split_files.items()
        ]
        entry.update(
            id=source_id,
            name=name,
            backend="file",
            availability="profile_inspected",
            access="public",
            revision=revision,
            variants=variants,
            task=task,
            language=language,
            record_mode="raw",
        )
        if repo.endswith("nusax"):
            entry.update(
                license="cc-by-sa-4.0",
                license_evidence=[f"https://github.com/{repo}/blob/{revision}/datasets/LICENSE"],
                license_note=(
                    "Dataset LICENSE inspected: CC BY-SA 4.0; "
                    "attribution and share-alike obligations apply."
                ),
            )
        return entry

    csv_splits = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}
    for language, code in languages.items():
        if code not in {"id", "jv", "su"}:
            add(
                f"NusaX {language} sentiment",
                f"nusax-{code}",
                "IndoNLP/nusax",
                nusax_revision,
                f"datasets/sentiment/{language}",
                "csv",
                "sentiment",
                code,
                csv_splits,
            )
        if code != "id":
            add(
                f"NusaX Indonesian–{language} lexicon",
                f"nusax-lexicon-{code}",
                "IndoNLP/nusax",
                nusax_revision,
                "datasets/lexicon",
                "csv",
                "lexicon",
                "mul",
                {"all": language + ".csv"},
            )
    add(
        "NusaX multilingual translation",
        "nusax-mt",
        "IndoNLP/nusax",
        nusax_revision,
        "datasets/mt",
        "csv",
        "translation",
        "mul",
        csv_splits,
    )
    for folder, fmt, task in [
        ("bapos_pos-idn", "conll", "token-classification"),
        ("posp_pos-prosa", "conll", "token-classification"),
        ("nerp_ner-prosa", "conll", "token-classification"),
        ("nergrit_ner-grit", "conll", "token-classification"),
        ("keps_keyword-extraction-prosa", "conll", "token-classification"),
        ("terma_term-extraction-airy", "conll", "token-classification"),
        ("casa_absa-prosa", "csv", "aspect-sentiment"),
        ("hoasa_absa-airy", "csv", "aspect-sentiment"),
        ("emot_emotion-twitter", "csv", "emotion"),
        ("wrete_entailment-ui", "csv", "nli"),
        ("facqa_qa-factoid-itb", "csv", "question-answering"),
    ]:
        suffix = "txt" if fmt == "conll" else "csv"
        add(
            "IndoNLU " + folder.split("_")[0],
            "indonlu-" + folder.split("_")[0],
            "IndoNLP/indonlu",
            indonlu_revision,
            "dataset/" + folder,
            fmt,
            task,
            split_files={
                "train": f"train_preprocess.{suffix}",
                "validation": f"valid_preprocess.{suffix}",
                "test": f"test_preprocess.{suffix}",
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build_catalog(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            print("Catalog is stale. Run: python scripts/build_catalog.py")
            return 1
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
