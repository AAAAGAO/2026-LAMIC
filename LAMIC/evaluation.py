from __future__ import annotations

from sklearn.metrics import f1_score, precision_score, recall_score

from .data import ApiSample
from .retrieval import RetrievalRow, mean_reciprocal_rank, recall_at_k, same_api_hit_rate


def retrieval_metrics(queries: list[ApiSample], rankings: list[list[RetrievalRow]]) -> dict[str, float]:
    labels = [query.label for query in queries]
    return {
        "recall@1": recall_at_k(rankings, labels, 1),
        "recall@3": recall_at_k(rankings, labels, 3),
        "recall@5": recall_at_k(rankings, labels, 5),
        "recall@10": recall_at_k(rankings, labels, 10),
        "mrr": mean_reciprocal_rank(rankings, labels),
        "topk_same_label_hit_rate": recall_at_k(rankings, labels, 5),
        "topk_same_api_hit_rate": same_api_hit_rate(rankings, queries, 5),
    }


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def case_studies(queries: list[ApiSample], rankings: list[list[RetrievalRow]], limit: int) -> list[dict]:
    studies: list[dict] = []
    for query, rows in zip(queries, rankings, strict=True):
        studies.append(
            {
                "query": {
                    "api": query.api,
                    "label": query.label,
                    "fragment": query.fragment[:600],
                },
                "retrieved": [
                    {
                        "api": row.candidate.api,
                        "label": row.candidate.label,
                        "dataset": row.candidate.dataset,
                        "bm25_score": row.bm25_score,
                        "semantic_score": row.semantic_score,
                        "structural_score": row.sop_score,
                        "fused_score": row.fused_score,
                        "lexical_rank": row.lexical_rank,
                        "semantic_rank": row.semantic_rank,
                        "structural_rank": row.structural_rank,
                        "api_match": row.candidate.api == query.api,
                        "label_match": row.candidate.label == query.label,
                    }
                    for row in rows
                ],
            }
        )
        if len(studies) >= limit:
            break
    return studies
