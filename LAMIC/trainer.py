from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pathlib
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .bm25 import BM25Retriever
from .config import AppConfig
from .data import ApiSample
from .evaluation import retrieval_metrics
from .losses import cross_view_info_nce, margin_ranking_loss, supervised_contrastive_loss
from .models import SemanticEncoder, StructuralEncoder
from .retrieval import HybridRetriever
from .sampler import ApiBalancedBatchSampler
from .sop import SOPExtractor
from .utils import dump_json, ensure_dir, load_json, seed_everything


class SampleDataset(Dataset):
    def __init__(self, samples: list[ApiSample], sop_strings: dict[int, str]) -> None:
        self.samples = samples
        self.sop_strings = sop_strings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "sample": sample,
            "api": sample.api,
            "fragment": sample.fragment,
            "label": sample.label,
            "sop": self.sop_strings[sample.sample_id],
        }


@dataclass(slots=True)
class TrainingArtifacts:
    semantic_encoder: SemanticEncoder
    structural_encoder: StructuralEncoder
    bm25: BM25Retriever | None
    sop_strings: dict[int, str]


class RetrieverTrainer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        seed_everything(config.training.seed)

    def _sop_cache_path(self) -> str:
        model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.config.model.spacy_model_name)
        return str(ensure_dir(self.config.data_dir.parent / ".cache") / f"sop_cache_{model_tag}.json")

    def _load_checkpoint(self, checkpoint_path: str | Path) -> dict:
        # Some checkpoints were produced on POSIX and include pathlib.PosixPath
        # objects in the serialized config. Map them to WindowsPath while loading.
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        finally:
            pathlib.PosixPath = original_posix_path

    def _sop_cache_key(self, sample: ApiSample) -> str:
        digest = hashlib.sha1(f"{sample.api}\n{sample.fragment}".encode("utf-8")).hexdigest()
        return f"{sample.sample_id}:{digest}"

    def build_sop_strings(self, samples: list[ApiSample]) -> dict[int, str]:
        cache_path = self._sop_cache_path()
        payload = load_json(cache_path) if Path(cache_path).exists() else {}
        cached_items = payload.get("items", {}) if isinstance(payload, dict) else {}

        sop_strings: dict[int, str] = {}
        missing_samples: list[ApiSample] = []
        for sample in samples:
            cache_key = self._sop_cache_key(sample)
            cached = cached_items.get(cache_key)
            if cached is not None:
                sop_strings[sample.sample_id] = str(cached)
            else:
                missing_samples.append(sample)

        if not missing_samples:
            return sop_strings

        extractor = SOPExtractor(self.config.model.spacy_model_name)
        for idx, sample in enumerate(tqdm(missing_samples, desc="Extracting SOP"), start=1):
            sop_value = extractor.extract(sample.fragment, sample.api)
            sop_strings[sample.sample_id] = sop_value
            cached_items[self._sop_cache_key(sample)] = sop_value
            if idx % 100 == 0:
                dump_json(
                    {
                        "model_name": self.config.model.spacy_model_name,
                        "items": cached_items,
                    },
                    cache_path,
                )

        dump_json(
            {
                "model_name": self.config.model.spacy_model_name,
                "items": cached_items,
            },
            cache_path,
        )
        return sop_strings

    def _collate(self, batch: list[dict]) -> dict:
        return {
            "samples": [item["sample"] for item in batch],
            "apis": [item["api"] for item in batch],
            "fragments": [item["fragment"] for item in batch],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "sops": [item["sop"] for item in batch],
        }

    def _build_models(self) -> tuple[SemanticEncoder, StructuralEncoder]:
        semantic_encoder = SemanticEncoder(
            model_name=self.config.model.semantic_model_name,
            projection_dim=self.config.model.projection_dim,
            dropout=self.config.model.dropout,
            chunk_size=self.config.model.chunking.chunk_size,
            stride=self.config.model.chunking.stride,
            semantic_max_length=self.config.model.chunking.semantic_max_length,
        ).to(self.device)
        structural_encoder = StructuralEncoder(
            model_name=self.config.model.structural_model_name,
            projection_dim=self.config.model.projection_dim,
            dropout=self.config.model.dropout,
            max_length=self.config.model.chunking.structural_max_length,
        ).to(self.device)
        return semantic_encoder, structural_encoder

    def _mine_rank_pairs(
        self,
        samples: list[ApiSample],
        sem_embeddings: torch.Tensor,
        struct_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_scores = []
        neg_scores = []
        for idx, anchor in enumerate(samples):
            pos_idx = next(
                (
                    candidate_idx
                    for candidate_idx, candidate in enumerate(samples)
                    if candidate_idx != idx
                    and candidate.api == anchor.api
                    and candidate.label == anchor.label
                ),
                None,
            )
            if pos_idx is None:
                pos_idx = next(
                    (
                        candidate_idx
                        for candidate_idx, candidate in enumerate(samples)
                        if candidate_idx != idx and candidate.label == anchor.label
                    ),
                    idx,
                )
            neg_idx = next(
                (
                    candidate_idx
                    for candidate_idx, candidate in enumerate(samples)
                    if candidate.api == anchor.api and candidate.label != anchor.label
                ),
                None,
            )
            if neg_idx is None:
                neg_idx = next(
                    (
                        candidate_idx
                        for candidate_idx, candidate in enumerate(samples)
                        if candidate.label != anchor.label
                    ),
                    idx,
                )
            pos_score = 0.5 * torch.dot(sem_embeddings[idx], sem_embeddings[pos_idx]) + 0.2 * torch.dot(
                struct_embeddings[idx], struct_embeddings[pos_idx]
            )
            neg_score = 0.5 * torch.dot(sem_embeddings[idx], sem_embeddings[neg_idx]) + 0.2 * torch.dot(
                struct_embeddings[idx], struct_embeddings[neg_idx]
            )
            pos_scores.append(pos_score)
            neg_scores.append(neg_score)
        return torch.stack(pos_scores), torch.stack(neg_scores)

    def _encode_pool(
        self,
        samples: list[ApiSample],
        sop_strings: dict[int, str],
        semantic_encoder: SemanticEncoder,
        structural_encoder: StructuralEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_encoder.eval()
        structural_encoder.eval()
        batch_size = self.config.training.batch_size
        semantic_embeddings = []
        structural_embeddings = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                semantic_embeddings.append(
                    semantic_encoder([sample.api for sample in batch], [sample.fragment for sample in batch], self.device)
                )
                structural_embeddings.append(
                    structural_encoder([sop_strings[sample.sample_id] for sample in batch], self.device)
                )
        return torch.cat(semantic_embeddings, dim=0), torch.cat(structural_embeddings, dim=0)

    def evaluate(
        self,
        train_pool: list[ApiSample],
        dev_samples: list[ApiSample],
        sop_strings: dict[int, str],
        semantic_encoder: SemanticEncoder,
        structural_encoder: StructuralEncoder,
        bm25: BM25Retriever,
    ) -> dict[str, float]:
        pool_semantic, pool_structural = self._encode_pool(train_pool, sop_strings, semantic_encoder, structural_encoder)
        retriever = HybridRetriever(bm25=bm25, weights=self.config.model.weights)
        rankings = []
        for query in tqdm(dev_samples, desc="Evaluating", leave=False):
            rows = retriever.retrieve(
                query=query,
                pool=train_pool,
                pool_semantic=pool_semantic,
                pool_structural=pool_structural,
                semantic_encoder=semantic_encoder,
                structural_encoder=structural_encoder,
                sop_string=sop_strings[query.sample_id],
                top_k=10,
                device=self.device,
            )
            rankings.append(rows)
        return retrieval_metrics(dev_samples, rankings)

    def fit(
        self,
        train_samples: list[ApiSample],
        dev_samples: list[ApiSample],
        extra_samples: list[ApiSample] | None = None,
    ) -> TrainingArtifacts:
        all_samples = train_samples + dev_samples + (extra_samples or [])
        unique_samples = {sample.sample_id: sample for sample in all_samples}
        sop_strings = self.build_sop_strings(list(unique_samples.values()))
        dataset = SampleDataset(train_samples, sop_strings)
        batch_sampler = ApiBalancedBatchSampler(
            train_samples,
            batch_size=self.config.training.batch_size,
            seed=self.config.training.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate,
            num_workers=self.config.training.num_workers,
        )
        semantic_encoder, structural_encoder = self._build_models()
        bm25 = BM25Retriever()
        bm25.fit(train_samples)

        optimizer = torch.optim.AdamW(
            [
                {"params": semantic_encoder.parameters(), "lr": self.config.training.semantic_lr},
                {"params": structural_encoder.parameters(), "lr": self.config.training.structural_lr},
            ],
            weight_decay=self.config.training.weight_decay,
        )
        total_steps = max(1, len(dataloader) * self.config.training.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.training.warmup_ratio),
            num_training_steps=total_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.training.fp16 and self.device.type == "cuda")

        best_metric = -1.0
        patience = 0
        output_dir = ensure_dir(self.config.output_dir / "checkpoints")
        best_path = output_dir / "best.pt"

        for epoch in range(1, self.config.training.epochs + 1):
            semantic_encoder.train()
            structural_encoder.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}"), start=1):
                apis = batch["apis"]
                fragments = batch["fragments"]
                sops = batch["sops"]
                labels = batch["labels"].to(self.device)
                samples = batch["samples"]

                with torch.autocast(device_type=self.device.type, enabled=scaler.is_enabled()):
                    sem_embeddings = semantic_encoder(apis, fragments, self.device)
                    struct_embeddings = structural_encoder(sops, self.device)
                    sem_loss = supervised_contrastive_loss(
                        sem_embeddings, labels, self.config.model.temperature
                    )
                    struct_loss = supervised_contrastive_loss(
                        struct_embeddings, labels, self.config.model.temperature
                    )
                    cross_loss = cross_view_info_nce(
                        sem_embeddings, struct_embeddings, self.config.model.temperature
                    )
                    pos_scores, neg_scores = self._mine_rank_pairs(samples, sem_embeddings, struct_embeddings)
                    rank_loss = margin_ranking_loss(pos_scores, neg_scores, self.config.model.margin)
                    loss = 0.4 * sem_loss + 0.2 * struct_loss + 0.2 * cross_loss + 0.2 * rank_loss
                    loss = loss / self.config.training.grad_accumulation_steps

                scaler.scale(loss).backward()
                if step % self.config.training.grad_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                running_loss += float(loss.item())

            metrics = self.evaluate(train_samples, dev_samples, sop_strings, semantic_encoder, structural_encoder, bm25)
            metrics["train_loss"] = running_loss / max(len(dataloader), 1)

            if metrics["mrr"] > best_metric:
                best_metric = metrics["mrr"]
                patience = 0
                torch.save(
                    {
                        "semantic_encoder": semantic_encoder.state_dict(),
                        "structural_encoder": structural_encoder.state_dict(),
                        "metrics": metrics,
                        "config": self.config.to_dict(),
                    },
                    best_path,
                )
                dump_json(metrics, self.config.output_dir / "best_dev_metrics.json")
            else:
                patience += 1
                if patience >= self.config.training.early_stopping_patience:
                    break

        checkpoint = self._load_checkpoint(best_path)
        semantic_encoder.load_state_dict(checkpoint["semantic_encoder"])
        structural_encoder.load_state_dict(checkpoint["structural_encoder"])
        return TrainingArtifacts(
            semantic_encoder=semantic_encoder,
            structural_encoder=structural_encoder,
            bm25=bm25,
            sop_strings=sop_strings,
        )

    def load_from_checkpoint(self, checkpoint_path: str | Path, samples: list[ApiSample]) -> TrainingArtifacts:
        checkpoint = self._load_checkpoint(checkpoint_path)
        semantic_encoder, structural_encoder = self._build_models()
        semantic_encoder.load_state_dict(checkpoint["semantic_encoder"])
        structural_encoder.load_state_dict(checkpoint["structural_encoder"])
        sop_strings = self.build_sop_strings(samples)
        return TrainingArtifacts(
            semantic_encoder=semantic_encoder,
            structural_encoder=structural_encoder,
            bm25=None,
            sop_strings=sop_strings,
        )
