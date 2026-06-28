from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Iterable

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


JAVA_DATASETS = {"jodatime", "smack", "math", "official", "jenkov"}
ANDROID_DATASETS = {"resources", "data", "graphics", "text"}


@dataclass(slots=True)
class ApiSample:
    sample_id: int
    fragment: str
    api: str
    label: int
    source: str
    library: str
    language: str
    dataset: str

    @property
    def text(self) -> str:
        return self.fragment or ""


def infer_metadata(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    source, library = stem.split("_", 1)
    language = "java" if library in JAVA_DATASETS else "android"
    return source, library, language


def load_samples(data_dir: str | Path) -> list[ApiSample]:
    data_path = Path(data_dir)
    samples: list[ApiSample] = []
    sample_id = 0
    for csv_path in sorted(data_path.glob("*.csv")):
        source, library, language = infer_metadata(csv_path)
        frame = pd.read_csv(csv_path).fillna("")
        for row in frame.itertuples(index=False):
            samples.append(
                ApiSample(
                    sample_id=sample_id,
                    fragment=str(row.fragment),
                    api=str(row.api),
                    label=int(row.relevance),
                    source=source,
                    library=library,
                    language=language,
                    dataset=csv_path.stem,
                )
            )
            sample_id += 1
    return samples


def samples_to_frame(samples: Iterable[ApiSample]) -> pd.DataFrame:
    return pd.DataFrame([asdict(sample) for sample in samples])


def _stratify_keys(samples: list[ApiSample]) -> list[str]:
    return [f"{sample.source}_{sample.label}" for sample in samples]


def stratified_split(
    samples: list[ApiSample],
    train_size: float,
    dev_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[ApiSample], list[ApiSample], list[ApiSample]]:
    if round(train_size + dev_size + test_size, 6) != 1.0:
        raise ValueError("train/dev/test sizes must sum to 1.0")

    indices = list(range(len(samples)))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=seed,
        stratify=_stratify_keys(samples),
    )
    temp_samples = [samples[idx] for idx in temp_idx]
    temp_keys = _stratify_keys(temp_samples)
    dev_ratio = dev_size / (dev_size + test_size)
    dev_rel_idx, test_rel_idx = train_test_split(
        list(range(len(temp_samples))),
        train_size=dev_ratio,
        random_state=seed,
        stratify=temp_keys,
    )
    train_samples = [samples[idx] for idx in train_idx]
    dev_samples = [temp_samples[idx] for idx in dev_rel_idx]
    test_samples = [temp_samples[idx] for idx in test_rel_idx]
    return train_samples, dev_samples, test_samples


def build_kfold_splits(samples: list[ApiSample], n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    labels = [f"{sample.source}_{sample.label}" for sample in samples]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx.tolist(), test_idx.tolist()) for train_idx, test_idx in splitter.split(samples, labels)]


def stratified_holdout(
    samples: list[ApiSample],
    test_size: int,
    seed: int,
) -> tuple[list[ApiSample], list[ApiSample]]:
    if test_size <= 0 or test_size >= len(samples):
        raise ValueError("test_size must be between 1 and len(samples) - 1")
    grouped_indices: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        key = f"{sample.library}_{sample.source}_{sample.label}"
        grouped_indices.setdefault(key, []).append(idx)
    if test_size < len(grouped_indices):
        raise ValueError(
            f"test_size={test_size} cannot cover all {len(grouped_indices)} strata"
        )

    rng = random.Random(seed)
    for indices in grouped_indices.values():
        rng.shuffle(indices)

    allocations = {key: 1 for key in grouped_indices}
    remaining = test_size - len(grouped_indices)
    total = sum(len(indices) for indices in grouped_indices.values())
    fractional: list[tuple[float, str]] = []
    for key, indices in grouped_indices.items():
        exact = remaining * len(indices) / total
        extra = min(int(exact), len(indices) - 1)
        allocations[key] += extra
        fractional.append((exact - int(exact), key))

    unassigned = test_size - sum(allocations.values())
    for _, key in sorted(fractional, reverse=True):
        if unassigned == 0:
            break
        if allocations[key] < len(grouped_indices[key]):
            allocations[key] += 1
            unassigned -= 1
    if unassigned:
        raise RuntimeError(f"Unable to allocate {unassigned} stratified test samples")

    test_idx = [
        idx
        for key in sorted(grouped_indices)
        for idx in grouped_indices[key][: allocations[key]]
    ]
    test_idx_set = set(test_idx)
    pool_idx = [idx for idx in range(len(samples)) if idx not in test_idx_set]
    rng.shuffle(test_idx)
    return [samples[idx] for idx in pool_idx], [samples[idx] for idx in test_idx]


def filter_by_language(samples: list[ApiSample], language: str) -> list[ApiSample]:
    return [sample for sample in samples if sample.language == language]


def group_samples_by_library(samples: list[ApiSample]) -> dict[str, list[ApiSample]]:
    grouped: dict[str, list[ApiSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.library, []).append(sample)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))
