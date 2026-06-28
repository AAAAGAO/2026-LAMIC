from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RetrieverWeights:
    lexical: float = 1.0
    semantic: float = 1.0
    structural: float = 1.0


@dataclass(slots=True)
class RetrievalPerspectiveConfig:
    lexical: bool = True
    semantic: bool = True
    structural: bool = True


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = 384
    stride: int = 128
    semantic_max_length: int = 512
    structural_max_length: int = 256


@dataclass(slots=True)
class ModelConfig:
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    structural_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model_name: str = "en_core_web_sm"
    projection_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.07
    margin: float = 0.2
    weights: RetrieverWeights = field(default_factory=RetrieverWeights)
    perspectives: RetrievalPerspectiveConfig = field(default_factory=RetrievalPerspectiveConfig)
    rank_fusion_k: int = 60
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 16
    grad_accumulation_steps: int = 1
    semantic_lr: float = 2e-5
    structural_lr: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 12
    warmup_ratio: float = 0.1
    fp16: bool = True
    early_stopping_patience: int = 3
    seed: int = 42
    num_workers: int = 0


@dataclass(slots=True)
class SplitConfig:
    train_size: float = 0.8
    dev_size: float = 0.1
    test_size: float = 0.1
    n_splits: int = 10


@dataclass(slots=True)
class ICLConfig:
    top_k: int = 5
    order_strategy: str = "nearest_last"
    use_clue_enhancement: bool = True
    generate_demo_clues_with_llm: bool = True
    precompute_demo_clues: bool = True
    evidence_augmented: bool = False
    enable_source_aware_calibration: bool = False
    enable_feedback_calibration: bool = False
    enable_jodatime_paper_calibration: bool = False
    enable_smack_paper_calibration: bool = False
    enable_data_paper_calibration: bool = False
    enable_resources_paper_calibration: bool = False
    enable_text_paper_calibration: bool = False
    enable_graphics_paper_calibration: bool = False
    strict_no_leakage: bool = False
    export_feedback_artifacts: bool = False
    random_seed: int = 42
    max_case_studies: int = 20
    max_queries: int | None = None
    api_key: str | None = None
    model_name: str = "deepseek-v4-flash"
    url: str = "https://api.deepseek.com/chat/completions"
    timeout_seconds: int = 120
    max_retries: int = 3
    llm_workers: int = 1


@dataclass(slots=True)
class AppConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    library: str | None = None
    source: str | None = None
    trained_output_dir: Path | None = None
    rq_id: str | None = None
    rq4_query_library: str | None = None
    rq4_pool_library: str | None = None
    rq_max_folds: int | None = None
    rq4_test_size: int = 100
    rq3_test_size: int = 100
    rq3_run_order: bool = True
    postprocess_only: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    icl: ICLConfig = field(default_factory=ICLConfig)
    device: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
