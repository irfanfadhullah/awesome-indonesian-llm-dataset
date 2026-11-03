"""Small, explicit building blocks for Indonesian dataset preparation."""

from .adapters import load_source
from .catalog import coverage, get_source, list_catalog
from .loaders import load_dataset
from .preprocessing import Pipeline, clean_text, split_records
from .registry import get_dataset, list_datasets, list_sources

__version__ = "0.2.0"
__all__ = [
    "Pipeline",
    "clean_text",
    "get_dataset",
    "list_datasets",
    "list_sources",
    "load_dataset",
    "split_records",
    "coverage",
    "get_source",
    "list_catalog",
    "load_source",
]
