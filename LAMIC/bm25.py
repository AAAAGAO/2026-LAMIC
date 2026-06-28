from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .data import ApiSample
from .preprocessing import build_lexical_document, normalize_minmax


@dataclass(slots=True)
class BM25Result:
    scores: list[float]
    normalized_scores: list[float]


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.samples: list[ApiSample] = []
        self.documents: list[list[str]] = []
        self.model: BM25Okapi | None = None

    def fit(self, samples: list[ApiSample]) -> None:
        self.samples = list(samples)
        self.documents = [build_lexical_document(sample) for sample in samples]
        self.model = BM25Okapi(self.documents, k1=self.k1, b=self.b)

    def score(self, sample: ApiSample) -> BM25Result:
        if self.model is None:
            raise RuntimeError("BM25Retriever.fit must be called before score().")
        query_tokens = build_lexical_document(sample)
        scores = list(self.model.get_scores(query_tokens))
        return BM25Result(scores=scores, normalized_scores=normalize_minmax(scores))

    def pair_score(self, query: ApiSample, candidate_index: int) -> float:
        result = self.score(query)
        return result.normalized_scores[candidate_index]
