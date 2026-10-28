"""
Model benchmarking and evaluation executable script.
Trains, evaluates, and compares ML, DL, and Multimodal architectures
under strict patient-level isolation and generates comparison tables and figures.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.utils import load_config, setup_logger, set_seed, get_device
from src.metadata_preprocessing import ClinicalFeaturePreprocessor
from src.benchmarking import BenchmarkRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Knee Osteoporosis Model Benchmarking Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Benchmarking mode: 'quick' for fast verification, 'full' for extensive epoch budget",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of deep learning and multimodal models from scratch",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to custom split manifest (e.g., outputs/reports/balanced_2400_manifest.csv)",
    )
    return parser.parse_args()

def run_benchmarks():
    args = parse_args()
    logger = setup_logger("benchmark")
    config = load_config()
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(8)
    set_seed(config.get("project", {}).get("random_seed", 42))

    logger.info(f"Running in '{args.mode.upper()}' benchmark mode.")
    logger.info(f"Using compute device: {device} ({'GPU Accelerated' if device.type == 'cuda' else f'CPU Mode ({torch.get_num_threads()} threads)'})")

    reports_dir = config["paths"]["reports_dir"]
    checkpoints_dir = config["paths"]["checkpoints_dir"]
    split_manifest_csv = args.manifest if args.manifest else os.path.join(reports_dir, "split_manifest.csv")
    preprocessor_path = os.path.join(checkpoints_dir, "clinical_preprocessor.pkl")

    if not os.path.exists(split_manifest_csv):
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_csv}.")

    split_df = pd.read_csv(split_manifest_csv)
    valid_df = split_df[split_df["split"].isin(["train", "val", "test"])].copy().reset_index(drop=True)
    logger.info(f"Loaded {len(valid_df)} verified, non-conflicting records for benchmarking.")

    preprocessor = ClinicalFeaturePreprocessor.load(preprocessor_path)
    clinical_X, feature_names = preprocessor.transform(valid_df)
    logger.info(f"Engineered {clinical_X.shape[1]} clinical features: {feature_names}")

    runner = BenchmarkRunner(
        split_df=valid_df,
        clinical_X=clinical_X,
        clinical_feature_names=feature_names,
        outputs_dir=config["paths"]["outputs_dir"],
        device=device,
        mode=args.mode,
        force_retrain=args.retrain,
        logger=logger,
    )

    # 1. Benchmark Traditional ML models on Clinical Features
    runner.benchmark_clinical_ml_models()

    # 2. Benchmark Deep Learning Models
    dl_models = runner.benchmark_deep_learning_models(
        backbones=["densenet121", "efficientnet_b0", "resnet50", "mobilenet_v3_large"]
    )

    # 3. Benchmark ML models on CNN image embeddings
    runner.benchmark_embedding_ml_models(encoder=dl_models["densenet121"])

    # 4. Benchmark Multimodal Fusion Model
    multimodal_model = runner.benchmark_multimodal_model(
        base_cnn=dl_models["densenet121"]
    )

    # 5. Generate comprehensive reports and comparison figures
    logger.info("Generating benchmarking comparison tables and visual diagnostic figures...")
    results_df = runner.generate_benchmark_reports_and_visualizations()

    logger.info("=== Benchmarking Complete ===")
    print("\nModel Baseline Summary Table:")
    cols_to_print = ["model_name", "model_family", "val_macro_f1", "test_macro_f1", "test_accuracy", "normal_recall", "osteopenia_recall", "osteoporosis_recall"]
    print(results_df[cols_to_print].to_string(index=False))

if __name__ == "__main__":
    run_benchmarks()
