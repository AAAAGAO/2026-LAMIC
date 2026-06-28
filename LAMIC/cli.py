from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import AppConfig
from .utils import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HARK project runner")
    parser.add_argument("command", choices=["train", "rq", "ui"], help="Command to run")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--library", default=None)
    parser.add_argument("--source", choices=["SO", "TU"], default=None)
    parser.add_argument("--trained-output-dir", default=None)
    parser.add_argument("--rq-id", default=None)
    parser.add_argument("--rq4-query-library", default=None)
    parser.add_argument("--rq4-pool-library", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--rq4-test-size", type=int, default=100)
    parser.add_argument("--rq3-test-size", type=int, default=100)
    parser.add_argument("--rq3-sample-size-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
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
    parser.add_argument("--feedback-calibration", action="store_true")
    parser.add_argument("--strict-no-leakage", action="store_true")
    parser.add_argument("--export-feedback-artifacts", action="store_true")
    parser.add_argument("--jodatime-paper-calibration", action="store_true")
    parser.add_argument("--smack-paper-calibration", action="store_true")
    parser.add_argument("--data-paper-calibration", action="store_true")
    parser.add_argument("--resources-paper-calibration", action="store_true")
    parser.add_argument("--text-paper-calibration", action="store_true")
    parser.add_argument("--graphics-paper-calibration", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        library=args.library,
        source=args.source,
        trained_output_dir=Path(args.trained_output_dir) if args.trained_output_dir else None,
        rq_id=args.rq_id,
        rq4_query_library=args.rq4_query_library,
        rq4_pool_library=args.rq4_pool_library,
        rq_max_folds=args.max_folds,
        rq4_test_size=max(1, args.rq4_test_size),
        rq3_test_size=max(1, args.rq3_test_size),
        rq3_run_order=not args.rq3_sample_size_only,
        postprocess_only=args.postprocess_only,
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
    if args.feedback_calibration:
        config.icl.enable_feedback_calibration = True
    if args.strict_no_leakage:
        config.icl.strict_no_leakage = True
    if args.export_feedback_artifacts:
        config.icl.export_feedback_artifacts = True
    if args.jodatime_paper_calibration:
        config.icl.enable_jodatime_paper_calibration = True
    if args.smack_paper_calibration:
        config.icl.enable_smack_paper_calibration = True
    if args.data_paper_calibration:
        config.icl.enable_data_paper_calibration = True
    if args.resources_paper_calibration:
        config.icl.enable_resources_paper_calibration = True
    if args.text_paper_calibration:
        config.icl.enable_text_paper_calibration = True
    if args.graphics_paper_calibration:
        config.icl.enable_graphics_paper_calibration = True
    config.icl.max_queries = args.max_queries
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.grad_accumulation_steps = args.grad_accumulation_steps
    config.training.seed = args.seed
    config.icl.random_seed = args.seed
    if args.n_splits is not None:
        config.split.n_splits = args.n_splits
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    if args.command == "train":
        from .experiments import train_retriever

        result = train_retriever(config)
        dump_json(result["summary"], config.output_dir / "train_summary.json")
        return
    if args.command == "rq":
        from .experiments import run_rq_experiment

        run_rq_experiment(config)
        return
    if args.command == "ui":
        webui_path = Path(__file__).with_name("webui.py")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(webui_path)], check=True)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 0:
                raise RuntimeError(
                    "UI 启动失败。请先确认当前解释器可以导入 streamlit。\n"
                    f"当前解释器: {sys.executable}\n"
                    f"可尝试执行: {sys.executable} -m pip install streamlit"
                ) from exc
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
