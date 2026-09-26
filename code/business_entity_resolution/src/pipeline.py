"""
Main pipeline orchestration.

Usage:
    python -m src.pipeline --mode test --config configs/final.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from config import ensure_dirs, RANDOM_SEED
from io_utils import load_tsv
from normalization import add_normalized_columns
from blocking import generate_candidates, save_candidate_pairs
from features import build_feature_matrix, get_feature_columns, select_features
from train_lgbm import load_model as load_lgbm_model, evaluate_model, optimize_threshold
from decision import predict_entities, predict_entities_multi_match
from evaluation import compute_candidate_recall, compute_candidate_stats, compute_entity_metrics, generate_error_analysis
from submission import validate_submission, format_submission

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_profiling(config: Dict):
    """Run dataset profiling."""
    from profiling import run_profiling
    run_profiling()


def run_normalization(config: Dict):
    """Run normalization on all sources."""
    from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3

    for path in [TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3]:
        df = load_tsv(path)
        add_normalized_columns(df)
        logger.info(f"Normalized {path.name}: {len(df)} rows")


def run_blocking(config: Dict) -> pd.DataFrame:
    """Run candidate generation."""
    from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3

    sample_size = config.get("sample_size", None)
    if sample_size:
        s1 = load_tsv(TRAIN_SOURCE1).head(sample_size)
        s2 = load_tsv(TRAIN_SOURCE2).head(sample_size * 2)
        s3 = load_tsv(TRAIN_SOURCE3).head(sample_size * 2)
    else:
        s1 = load_tsv(TRAIN_SOURCE1)
        s2 = load_tsv(TRAIN_SOURCE2)
        s3 = load_tsv(TRAIN_SOURCE3)

    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    blocking_config = config.get("blocking", {})
    candidates = generate_candidates(s1, s2, s3, blocking_config)

    output_dir = Path(config.get("output_dir", "output"))
    save_candidate_pairs(candidates, output_dir / "candidate_pairs.tsv")

    return candidates


def run_features(config: Dict, candidates: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix."""
    from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3

    sample_size = config.get("sample_size", None)
    if sample_size:
        s1 = load_tsv(TRAIN_SOURCE1).head(sample_size)
        s2 = load_tsv(TRAIN_SOURCE2).head(sample_size * 2)
        s3 = load_tsv(TRAIN_SOURCE3).head(sample_size * 2)
    else:
        s1 = load_tsv(TRAIN_SOURCE1)
        s2 = load_tsv(TRAIN_SOURCE2)
        s3 = load_tsv(TRAIN_SOURCE3)

    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    features = build_feature_matrix(candidates, s1, s2, s3)
    return features


def run_train(config: Dict):
    """Train LightGBM model."""
    from train_lgbm import run_baseline_experiment

    result = run_baseline_experiment(config)
    logger.info(f"Training complete. Best threshold: {result['best_threshold']:.4f}")
    logger.info(f"Val F0.5: {result['val_metrics']['f05']:.4f}")
    return result


def run_evaluate(config: Dict):
    """Evaluate on validation set."""
    from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3
    from training_data import split_by_s1_entity
    from validation_split import ValidationSplit

    # Load data
    s1 = load_tsv(TRAIN_SOURCE1)
    s2 = load_tsv(TRAIN_SOURCE2)
    s3 = load_tsv(TRAIN_SOURCE3)
    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    # Generate candidates
    blocking_config = config.get("blocking", {})
    candidates = generate_candidates(s1, s2, s3, blocking_config)

    # Load ground truth and label
    from training_data import load_ground_truth_pairs
    gt_pairs = load_ground_truth_pairs()
    gt_keys = set(zip(gt_pairs["s1_id"], gt_pairs["candidate_id"]))
    candidates["label"] = candidates.apply(lambda r: 1 if (r["s1_id"], r["candidate_id"]) in gt_keys else 0, axis=1)

    # Split
    splitter = ValidationSplit(val_fraction=config.get("val_fraction", 0.2), seed=RANDOM_SEED)
    train_pairs, val_pairs = splitter.apply_to_pairs(candidates)

    # Build features
    train_feat = build_feature_matrix(train_pairs, s1, s2, s3)
    val_feat = build_feature_matrix(val_pairs, s1, s2, s3)

    # Load model
    model, feature_names = load_lgbm_model()

    # Evaluate
    val_metrics = evaluate_model(model, val_feat, feature_names, threshold=0.5)
    logger.info(f"Val metrics (thresh=0.5): {val_metrics}")

    # Optimize threshold
    best_thresh, best_metrics = optimize_threshold(model, val_feat, feature_names)
    val_metrics_opt = evaluate_model(model, val_feat, feature_names, threshold=best_thresh)
    logger.info(f"Val metrics (thresh={best_thresh:.2f}): {val_metrics_opt}")

    return val_metrics_opt


def run_test(config: Dict):
    """Run test inference and generate submission."""
    from inference import run_full_inference

    candidates, predictions = run_full_inference(config)

    # Format submission
    output_dir = Path(config.get("output_dir", "output"))
    cand_dict = candidates.groupby("s1_id")["candidate_id"].apply(set).to_dict()
    format_submission(predictions, cand_dict, output_dir)

    # Validate
    validate_submission(
        output_dir / "matching_results.tsv",
        output_dir / "candidate_pairs.tsv",
    )

    logger.info("Test inference complete!")
    return predictions


def run_error_analysis(config: Dict):
    """Generate error analysis on validation set."""
    from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3
    from training_data import load_ground_truth_pairs
    from validation_split import ValidationSplit

    s1 = load_tsv(TRAIN_SOURCE1)
    s2 = load_tsv(TRAIN_SOURCE2)
    s3 = load_tsv(TRAIN_SOURCE3)
    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    blocking_config = config.get("blocking", {})
    candidates = generate_candidates(s1, s2, s3, blocking_config)

    gt_pairs = load_ground_truth_pairs()
    gt_keys = set(zip(gt_pairs["s1_id"], gt_pairs["candidate_id"]))
    candidates["label"] = candidates.apply(lambda r: 1 if (r["s1_id"], r["candidate_id"]) in gt_keys else 0, axis=1)

    splitter = ValidationSplit(val_fraction=config.get("val_fraction", 0.2), seed=RANDOM_SEED)
    _, val_pairs = splitter.apply_to_pairs(candidates)

    val_feat = build_feature_matrix(val_pairs, s1, s2, s3)

    model, feature_names = load_lgbm_model()
    X_val, y_val, _ = select_features(val_feat, feature_names)
    val_scores = model.predict(X_val, num_iteration=model.best_iteration)

    # Optimize threshold
    best_thresh, _ = optimize_threshold(model, val_feat, feature_names)

    # Predict
    val_preds = predict_entities(
        val_scores,
        val_pairs["s1_id"].values,
        val_pairs["candidate_id"].values,
        val_pairs["candidate_source"].values,
        threshold=best_thresh,
    )

    # Load ground truth for val S1s
    gt_df = load_tsv("dataset/train/train_ground_truth.tsv")
    from io_utils import parse_ground_truth
    gt_dict = parse_ground_truth(gt_df)
    val_s1_ids = set(val_pairs["s1_id"].unique())
    val_gt = {s1_id: gt_dict.get(s1_id, set()) for s1_id in val_s1_ids}

    # Error analysis
    output_dir = Path(config.get("output_dir", "validation"))
    generate_error_analysis(
        val_preds, val_gt, val_pairs, s1, s2, s3,
        output_dir / "error_analysis.csv"
    )

    logger.info("Error analysis complete!")


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--mode", required=True, choices=[
        "profile", "normalize", "block", "features", "train", "evaluate", "test", "error_analysis"
    ], help="Pipeline mode")
    parser.add_argument("--config", default="configs/final.yaml", help="Config YAML file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    setup_logging(args.log_level)
    ensure_dirs()

    config = load_config(Path(args.config))
    logger.info(f"Running mode: {args.mode}")
    logger.info(f"Config: {config}")

    if args.mode == "profile":
        run_profiling(config)
    elif args.mode == "normalize":
        run_normalization(config)
    elif args.mode == "block":
        run_blocking(config)
    elif args.mode == "features":
        candidates = run_blocking(config)
        run_features(config, candidates)
    elif args.mode == "train":
        run_train(config)
    elif args.mode == "evaluate":
        run_evaluate(config)
    elif args.mode == "test":
        run_test(config)
    elif args.mode == "error_analysis":
        run_error_analysis(config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()