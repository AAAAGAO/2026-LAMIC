from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LAMIC experiment runner")
    parser.add_argument("command", choices=["rq"], help="Command to run")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--library", default=None)
    parser.add_argument("--source", choices=["SO", "TU"], default=None)
    parser.add_argument("--rq-id", default=None)
    parser.add_argument("--rq4-query-library", default=None)
    parser.add_argument("--rq4-pool-library", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--rq4-test-size", type=int, default=100)
    parser.add_argument("--rq3-test-size", type=int, default=100)
    parser.add_argument("--rq3-sample-size-only", action="store_true")
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--llm-workers", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--order-strategy", choices=["nearest_last", "nearest_first", "random"], default=None)
    parser.add_argument("--no-llm-demo-clues", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        library=args.library,
        source=args.source,
        rq_id=args.rq_id,
        rq4_query_library=args.rq4_query_library,
        rq4_pool_library=args.rq4_pool_library,
        rq_max_folds=args.max_folds,
        rq4_test_size=max(1, args.rq4_test_size),
        rq3_test_size=max(1, args.rq3_test_size),
        rq3_run_order=not args.rq3_sample_size_only,
        device=args.device,
    )
    config.icl.api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    config.icl.model_name = args.model_name
    if args.timeout_seconds is not None:
        config.icl.timeout_seconds = args.timeout_seconds
    if args.max_retries is not None:
        config.icl.max_retries = args.max_retries
    if args.llm_workers is not None:
        config.icl.llm_workers = max(1, args.llm_workers)
    config.icl.top_k = args.top_k
    if args.order_strategy is not None:
        config.icl.order_strategy = args.order_strategy
    if args.no_llm_demo_clues:
        config.icl.generate_demo_clues_with_llm = False
    config.icl.max_queries = args.max_queries
    config.experiment.batch_size = args.batch_size
    config.experiment.seed = args.seed
    config.icl.random_seed = args.seed
    if args.n_splits is not None:
        config.split.n_splits = args.n_splits
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    if args.command == "rq":
        from .experiments import run_rq_experiment

        run_rq_experiment(config)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
