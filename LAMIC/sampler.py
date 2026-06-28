from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler

from .data import ApiSample


class ApiBalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, samples: list[ApiSample], batch_size: int, seed: int) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.seed = seed

        by_api: dict[str, list[int]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            by_api[sample.api].append(idx)
        self.groups = list(by_api.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        groups = [group[:] for group in self.groups]
        for group in groups:
            rng.shuffle(group)
        rng.shuffle(groups)

        batch: list[int] = []
        labels: set[int] = set()
        for group in groups:
            for idx in group:
                batch.append(idx)
                labels.add(self.samples[idx].label)
                if len(batch) == self.batch_size:
                    if len(labels) == 1:
                        replacement = self._find_opposite_label(batch, rng)
                        if replacement is not None:
                            batch[-1] = replacement
                    yield batch
                    batch = []
                    labels = set()
        if batch:
            yield batch

    def _find_opposite_label(self, batch: list[int], rng: random.Random) -> int | None:
        label = self.samples[batch[0]].label
        candidates = [idx for idx, sample in enumerate(self.samples) if sample.label != label and idx not in batch]
        return rng.choice(candidates) if candidates else None

    def __len__(self) -> int:
        return max(1, len(self.samples) // self.batch_size)
