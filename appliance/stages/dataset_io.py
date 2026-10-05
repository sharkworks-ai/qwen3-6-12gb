from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _one(source: dict[str, Any]):
    from datasets import load_dataset
    if source.get("dataset_path"):
        path = Path(source["dataset_path"])
        fmt = source.get("dataset_format") or ("json" if path.suffix in {".json", ".jsonl"} else "parquet")
        return load_dataset(fmt, data_files=str(path), split=source.get("dataset_split", "train"))
    name = source.get("dataset_name")
    if not name:
        raise ValueError("dataset_name or dataset_path is required")
    return load_dataset(name, source.get("dataset_config"), split=source.get("dataset_split", "train"), token=os.environ.get("HF_TOKEN") or None)


def load_training_dataset(config: dict[str, Any]):
    from datasets import interleave_datasets
    if config.get("datasets"):
        sources = config["datasets"]
        datasets = [_one(source) for source in sources]
        weights = [float(source.get("weight", 1.0)) for source in sources]
        total = sum(weights)
        return interleave_datasets(datasets, probabilities=[w / total for w in weights], seed=int(config.get("seed", 42)), stopping_strategy="all_exhausted")
    from datasets import load_dataset

    if config.get("dataset_path"):
        path = Path(config["dataset_path"])
        fmt = config.get("dataset_format") or ("json" if path.suffix in {".json", ".jsonl"} else "parquet")
        return load_dataset(fmt, data_files=str(path), split=config.get("dataset_split", "train"))

    name = config.get("dataset_name")
    if not name:
        raise ValueError("dataset_name or dataset_path is required")
    return load_dataset(
        name,
        config.get("dataset_config"),
        split=config.get("dataset_split", "train"),
        token=os.environ.get("HF_TOKEN") or None,
    )


def maybe_limit(dataset, config: dict[str, Any]):
    maximum = int(config.get("max_samples", 0) or 0)
    if maximum > 0:
        return dataset.select(range(min(maximum, len(dataset))))
    return dataset
