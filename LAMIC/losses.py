from __future__ import annotations

import torch
from torch import nn


def cosine_scores(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(nn.functional.normalize(a, dim=-1), nn.functional.normalize(b, dim=-1).T)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    similarity = cosine_scores(embeddings, embeddings) / temperature
    logits_mask = torch.ones_like(similarity) - torch.eye(similarity.size(0), device=similarity.device)
    positives = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * logits_mask
    exp_logits = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_counts = positives.sum(dim=1).clamp(min=1.0)
    loss = -(positives * log_prob).sum(dim=1) / positive_counts
    return loss.mean()


def cross_view_info_nce(anchor_embeddings: torch.Tensor, target_embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = cosine_scores(anchor_embeddings, target_embeddings) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def margin_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.relu(margin - pos_scores + neg_scores).mean()
