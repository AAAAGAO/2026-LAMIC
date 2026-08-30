from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _load_pretrained_with_fallback(loader, model_name: str, **kwargs):
    last_error: Exception | None = None
    for local_only in (True, False):
        try:
            return loader.from_pretrained(model_name, local_files_only=local_only, **kwargs)
        except Exception as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError(f"Unable to load pretrained model: {model_name}")
    raise last_error


def load_hf_tokenizer(model_name: str):
    try:
        return _load_pretrained_with_fallback(AutoTokenizer, model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Hugging Face tokenizer '{model_name}'.") from exc


def load_hf_backbone(model_name: str):
    try:
        return _load_pretrained_with_fallback(AutoModel, model_name, use_safetensors=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Hugging Face model '{model_name}'.") from exc


class SentenceBertEncoder(nn.Module):
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        super().__init__()
        self.tokenizer = load_hf_tokenizer(model_name)
        self.backbone = load_hf_backbone(model_name)
        self.max_length = max_length

    def forward(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.backbone(**batch)
        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        return nn.functional.normalize(pooled, p=2, dim=-1)

    def encode(self, texts: Sequence[str], device: torch.device, batch_size: int) -> torch.Tensor:
        self.eval()
        encoded: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded.append(self.forward(texts[start:start + batch_size], device))
        if not encoded:
            hidden_size = int(self.backbone.config.hidden_size)
            return torch.empty((0, hidden_size), device=device)
        return torch.cat(encoded, dim=0)
