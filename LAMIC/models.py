from __future__ import annotations

from collections.abc import Sequence
import os

import torch
from torch import nn

# TUNA removed its Hugging Face mirror on 2021-08-31, so default to a
# currently usable mirror while still allowing users to override via HF_ENDPOINT.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoModel, AutoTokenizer

from .preprocessing import build_semantic_chunks


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _load_pretrained_with_fallback(loader, model_name: str, **kwargs):
    last_exc: Exception | None = None
    for local_only in (True, False):
        try:
            return loader.from_pretrained(model_name, local_files_only=local_only, **kwargs)
        except Exception as exc:  # pragma: no cover - fallback path depends on runtime cache/network.
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def load_hf_tokenizer(model_name: str):
    try:
        return _load_pretrained_with_fallback(AutoTokenizer, model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HuggingFace tokenizer '{model_name}'. "
            "Please ensure local cache exists or HF mirror connectivity is available."
        ) from exc


def load_hf_backbone(model_name: str):
    try:
        return _load_pretrained_with_fallback(AutoModel, model_name, use_safetensors=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HuggingFace model '{model_name}' with safetensors.\n"
            "This project now prefers safetensors to avoid the torch<2.6 torch.load restriction.\n"
            "If the model has no safetensors weights, upgrade torch to >=2.6 or choose a model with safetensors."
        ) from exc


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(chunk_embeddings).squeeze(-1), dim=-1)
        return torch.sum(chunk_embeddings * weights.unsqueeze(-1), dim=1)


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        projection_dim: int,
        dropout: float,
        chunk_size: int,
        stride: int,
        semantic_max_length: int,
    ) -> None:
        super().__init__()
        self.tokenizer = load_hf_tokenizer(model_name)
        self.backbone = load_hf_backbone(model_name)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.config.hidden_size, projection_dim),
        )
        self.pooler = AttentionPooler(self.backbone.config.hidden_size)
        self.chunk_size = chunk_size
        self.stride = stride
        self.semantic_max_length = semantic_max_length

    def encode_texts(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.semantic_max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.backbone(**batch)
        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        return nn.functional.normalize(pooled, p=2, dim=-1)

    def forward(self, apis: Sequence[str], fragments: Sequence[str], device: torch.device) -> torch.Tensor:
        grouped_chunks = [
            build_semantic_chunks(
                api=api,
                fragment=fragment,
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                stride=self.stride,
                semantic_max_length=self.semantic_max_length,
            )
            for api, fragment in zip(apis, fragments, strict=True)
        ]
        flat_chunks = [chunk for chunks in grouped_chunks for chunk in chunks]
        chunk_embeddings = self.encode_texts(flat_chunks, device)

        grouped_embeddings: list[torch.Tensor] = []
        offset = 0
        for chunks in grouped_chunks:
            next_offset = offset + len(chunks)
            current = chunk_embeddings[offset:next_offset]
            offset = next_offset
            if current.size(0) == 1:
                grouped_embeddings.append(current.squeeze(0))
                continue
            pooled = self.pooler(current.unsqueeze(0)).squeeze(0)
            grouped_embeddings.append(pooled)
        stacked = torch.stack(grouped_embeddings, dim=0)
        projected = self.projection(stacked)
        return nn.functional.normalize(projected, p=2, dim=-1)


class SentenceBertEncoder(nn.Module):
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        super().__init__()
        self.tokenizer = load_hf_tokenizer(model_name)
        self.backbone = load_hf_backbone(model_name)
        self.max_length = max_length

    def forward(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.backbone(**batch)
        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        return nn.functional.normalize(pooled, p=2, dim=-1)

    def encode(
        self,
        texts: Sequence[str],
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        self.eval()
        encoded: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded.append(self.forward(batch, device))
        if not encoded:
            hidden_size = int(self.backbone.config.hidden_size)
            return torch.empty((0, hidden_size), device=device)
        return torch.cat(encoded, dim=0)


class StructuralEncoder(nn.Module):
    def __init__(self, model_name: str, projection_dim: int, dropout: float, max_length: int) -> None:
        super().__init__()
        self.tokenizer = load_hf_tokenizer(model_name)
        self.backbone = load_hf_backbone(model_name)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.config.hidden_size, projection_dim),
        )
        self.max_length = max_length

    def forward(self, sop_strings: Sequence[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            list(sop_strings),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.backbone(**batch)
        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        projected = self.projection(pooled)
        return nn.functional.normalize(projected, p=2, dim=-1)
