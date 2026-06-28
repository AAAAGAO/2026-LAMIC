from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch
from rank_bm25 import BM25Okapi

from .bm25 import BM25Retriever
from .config import ModelConfig, RetrieverWeights
from .data import ApiSample
from .models import SentenceBertEncoder
from .preprocessing import semantic_input


@dataclass(slots=True)
class RetrievalRow:
    candidate: ApiSample
    bm25_score: float
    semantic_score: float
    sop_score: float
    fused_score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    structural_rank: int | None = None


def cosine_to_unit_interval(scores: np.ndarray) -> np.ndarray:
    return (scores + 1.0) / 2.0


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.int32)
    for position, idx in enumerate(order, start=1):
        ranks[idx] = position
    return ranks


class PosTagExtractor:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._nlp = None
        self._load_attempted = False

    @property
    def nlp(self):
        if self._load_attempted:
            return self._nlp
        self._load_attempted = True
        try:
            import spacy

            try:
                self._nlp = spacy.load(self.model_name, disable=["parser", "ner", "lemmatizer"])
            except OSError:
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        except Exception:
            self._nlp = None
        return self._nlp

    def tags(self, text: str) -> list[str]:
        nlp = self.nlp
        if nlp is not None:
            doc = nlp(text[:5000])
            tags = [token.pos_ or token.tag_ for token in doc if not token.is_space]
            if tags and any(tag for tag in tags):
                return tags
        return self._fallback_tags(text)

    def tags_many(self, texts: list[str], batch_size: int = 64) -> list[list[str]]:
        nlp = self.nlp
        if nlp is None:
            return [self._fallback_tags(text) for text in texts]
        results: list[list[str]] = []
        for text, doc in zip(
            texts,
            nlp.pipe((text[:5000] for text in texts), batch_size=batch_size),
            strict=True,
        ):
            tags = [token.pos_ or token.tag_ for token in doc if not token.is_space]
            results.append(tags if tags and any(tags) else self._fallback_tags(text))
        return results

    def _fallback_tags(self, text: str) -> list[str]:
        tags: list[str] = []
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[{}()[\].,;:=+\-*/<>]", text):
            lowered = token.lower()
            if lowered in {"if", "else", "for", "while", "try", "catch", "return"}:
                tags.append("CONTROL")
            elif lowered in {"new", "use", "call", "create", "set", "get", "parse", "convert"}:
                tags.append("VERB")
            elif re.fullmatch(r"\d+", token):
                tags.append("NUM")
            elif re.fullmatch(r"[{}()[\].,;:=+\-*/<>]", token):
                tags.append("PUNCT")
            elif token[:1].isupper() or "_" in token:
                tags.append("NOUN")
            elif lowered.endswith(("ing", "ed")):
                tags.append("VERB")
            else:
                tags.append("X")
        return tags or ["X"]


class PosBM25Retriever:
    def __init__(self, tagger: PosTagExtractor, k1: float = 1.2, b: float = 0.75) -> None:
        self.tagger = tagger
        self.k1 = k1
        self.b = b
        self.samples: list[ApiSample] = []
        self.documents: list[list[str]] = []
        self.model: BM25Okapi | None = None

    def fit(self, samples: list[ApiSample]) -> None:
        self.samples = list(samples)
        self.documents = self.tagger.tags_many([sample.fragment for sample in samples])
        self.model = BM25Okapi(self.documents, k1=self.k1, b=self.b)

    def score(self, sample: ApiSample) -> list[float]:
        if self.model is None:
            raise RuntimeError("PosBM25Retriever.fit must be called before score().")
        return list(self.model.get_scores(self.tagger.tags(sample.fragment)))


class MultiPerspectiveRetriever:
    def __init__(self, model_config: ModelConfig, batch_size: int, device: torch.device) -> None:
        self.model_config = model_config
        self.batch_size = batch_size
        self.device = device
        self.pool: list[ApiSample] = []
        self.lexical = BM25Retriever()
        self.pos_tagger = PosTagExtractor(model_config.spacy_model_name)
        self.structural = PosBM25Retriever(self.pos_tagger)
        self.semantic_encoder: SentenceBertEncoder | None = None
        self.pool_semantic: torch.Tensor | None = None
        if not any(
            (
                model_config.perspectives.lexical,
                model_config.perspectives.semantic,
                model_config.perspectives.structural,
            )
        ):
            raise ValueError("At least one retrieval perspective must be enabled.")

    def fit(self, pool: list[ApiSample]) -> None:
        self.pool = list(pool)
        if self.model_config.perspectives.lexical:
            self.lexical.fit(self.pool)
        if self.model_config.perspectives.structural:
            self.structural.fit(self.pool)
        if self.model_config.perspectives.semantic:
            self.semantic_encoder = SentenceBertEncoder(
                self.model_config.semantic_model_name,
                max_length=self.model_config.chunking.semantic_max_length,
            ).to(self.device)
            texts = [semantic_input(sample.api, sample.fragment) for sample in self.pool]
            self.pool_semantic = self.semantic_encoder.encode(texts, self.device, self.batch_size)

    def _semantic_scores(self, query: ApiSample) -> np.ndarray:
        if self.semantic_encoder is None or self.pool_semantic is None:
            return np.zeros(len(self.pool), dtype=np.float32)
        query_text = semantic_input(query.api, query.fragment)
        query_embedding = self.semantic_encoder.encode([query_text], self.device, self.batch_size)[0]
        return torch.matmul(self.pool_semantic, query_embedding).detach().cpu().numpy()

    def retrieve(self, query: ApiSample, top_k: int) -> list[RetrievalRow]:
        if not self.pool:
            return []

        lexical_scores = (
            np.asarray(self.lexical.score(query).scores, dtype=np.float32)
            if self.model_config.perspectives.lexical
            else np.zeros(len(self.pool), dtype=np.float32)
        )
        semantic_scores = (
            self._semantic_scores(query)
            if self.model_config.perspectives.semantic
            else np.zeros(len(self.pool), dtype=np.float32)
        )
        structural_scores = (
            np.asarray(self.structural.score(query), dtype=np.float32)
            if self.model_config.perspectives.structural
            else np.zeros(len(self.pool), dtype=np.float32)
        )

        fused_scores = np.zeros(len(self.pool), dtype=np.float32)
        lexical_ranks: np.ndarray | None = None
        semantic_ranks: np.ndarray | None = None
        structural_ranks: np.ndarray | None = None
        rank_k = self.model_config.rank_fusion_k
        weights = self.model_config.weights

        if self.model_config.perspectives.lexical:
            lexical_ranks = _rank_positions(lexical_scores)
            fused_scores += weights.lexical / (rank_k + lexical_ranks)
        if self.model_config.perspectives.semantic:
            semantic_ranks = _rank_positions(semantic_scores)
            fused_scores += weights.semantic / (rank_k + semantic_ranks)
        if self.model_config.perspectives.structural:
            structural_ranks = _rank_positions(structural_scores)
            fused_scores += weights.structural / (rank_k + structural_ranks)

        rows: list[RetrievalRow] = []
        for idx, candidate in enumerate(self.pool):
            if candidate.sample_id == query.sample_id:
                continue
            rows.append(
                RetrievalRow(
                    candidate=candidate,
                    bm25_score=float(lexical_scores[idx]),
                    semantic_score=float(semantic_scores[idx]),
                    sop_score=float(structural_scores[idx]),
                    fused_score=float(fused_scores[idx]),
                    lexical_rank=int(lexical_ranks[idx]) if lexical_ranks is not None else None,
                    semantic_rank=int(semantic_ranks[idx]) if semantic_ranks is not None else None,
                    structural_rank=int(structural_ranks[idx]) if structural_ranks is not None else None,
                )
            )
        rows.sort(key=lambda item: item.fused_score, reverse=True)
        return rows[:top_k]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, weights: RetrieverWeights) -> None:
        self.bm25 = bm25
        self.weights = weights

    def _embed_samples(self, encoder, samples: list[ApiSample], sop_strings: list[str], device: torch.device) -> torch.Tensor:
        encoder.eval()
        with torch.no_grad():
            if sop_strings:
                return encoder(sop_strings, device)
            return encoder([sample.api for sample in samples], [sample.fragment for sample in samples], device)

    def retrieve(
        self,
        query: ApiSample,
        pool: list[ApiSample],
        pool_semantic: torch.Tensor,
        pool_structural: torch.Tensor,
        semantic_encoder,
        structural_encoder,
        sop_string: str,
        top_k: int,
        device: torch.device,
    ) -> list[RetrievalRow]:
        semantic_query = self._embed_samples(semantic_encoder, [query], [], device)[0]
        structural_query = self._embed_samples(structural_encoder, [], [sop_string], device)[0]
        bm25_result = self.bm25.score(query)

        sem_scores = torch.matmul(pool_semantic, semantic_query).cpu().numpy()
        struct_scores = torch.matmul(pool_structural, structural_query).cpu().numpy()
        sem_unit = cosine_to_unit_interval(sem_scores)
        struct_unit = cosine_to_unit_interval(struct_scores)
        bm25_scores = np.asarray(bm25_result.normalized_scores, dtype=np.float32)

        fused_scores = (
            self.weights.lexical * bm25_scores
            + self.weights.semantic * sem_unit
            + self.weights.structural * struct_unit
        )

        rows: list[RetrievalRow] = []
        for idx, candidate in enumerate(pool):
            if candidate.sample_id == query.sample_id:
                continue
            rows.append(
                RetrievalRow(
                    candidate=candidate,
                    bm25_score=float(bm25_scores[idx]),
                    semantic_score=float(sem_scores[idx]),
                    sop_score=float(struct_scores[idx]),
                    fused_score=float(fused_scores[idx]),
                )
            )
        rows.sort(key=lambda item: item.fused_score, reverse=True)
        return rows[:top_k]


def recall_at_k(rankings: list[list[RetrievalRow]], query_labels: list[int], k: int) -> float:
    hit = 0
    for rows, label in zip(rankings, query_labels, strict=True):
        if any(row.candidate.label == label for row in rows[:k]):
            hit += 1
    return hit / max(len(rankings), 1)


def mean_reciprocal_rank(rankings: list[list[RetrievalRow]], query_labels: list[int]) -> float:
    rr_total = 0.0
    for rows, label in zip(rankings, query_labels, strict=True):
        rr = 0.0
        for rank, row in enumerate(rows, start=1):
            if row.candidate.label == label:
                rr = 1.0 / rank
                break
        rr_total += rr
    return rr_total / max(len(rankings), 1)


def same_api_hit_rate(rankings: list[list[RetrievalRow]], queries: list[ApiSample], k: int) -> float:
    hit = 0
    for rows, query in zip(rankings, queries, strict=True):
        if any(row.candidate.api == query.api for row in rows[:k]):
            hit += 1
    return hit / max(len(rankings), 1)
