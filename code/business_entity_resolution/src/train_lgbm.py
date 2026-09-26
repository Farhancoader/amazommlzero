"""
LightGBM training for entity resolution baseline.

Binary classifier: P(match | candidate pair features)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, fbeta_score

from config import ARTIFACTS_DIR, RANDOM_SEED
from features import (
    ALL_FEATURES,
    build_feature_matrix,
    get_feature_columns,
    prepare_sources_for_features,
    save_feature_importance,
    select_features,
)
from training_data import load_ground_truth_pairs, split_by_s1_entity
from validation_split import compute_candidate_recall, compute_candidate_stats

logger = logging.getLogger(__name__)


# ── Default LightGBM parameters ────────────────────────────────────

DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
    "n_jobs": -1,
    "class_weight": "balanced",
}


def prepare_training_data(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    cache_dir: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare labeled training data from candidates + ground truth.

    Returns:
        train_features, val_features DataFrames with label column
    """
    # Load ground truth positive pairs
    gt_pairs = load_ground_truth_pairs()

    # Label candidates
    gt_keys = set(zip(gt_pairs["s1_id"], gt_pairs["candidate_id"]))

    def is_match(row):
        return 1 if (row["s1_id"], row["candidate_id"]) in gt_keys else 0

    candidates_labeled = candidates_df.copy()
    candidates_labeled["label"] = candidates_labeled.apply(is_match, axis=1)

    # Split by S1 entity
    train_pairs, val_pairs = split_by_s1_entity(candidates_labeled, val_fraction=0.2, seed=RANDOM_SEED)

    # Build features
    cache_dir = cache_dir or (ARTIFACTS_DIR.parent / "cache" / "features")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_features = build_feature_matrix(train_pairs, s1_df, s2_df, s3_df, cache_dir / "train_features.parquet")
    val_features = build_feature_matrix(val_pairs, s1_df, s2_df, s3_df, cache_dir / "val_features.parquet")

    return train_features, val_features


def train_lgbm(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    params: Dict = None,
    feature_list: List[str] = None,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """
    Train LightGBM model.

    Args:
        train_features: Training feature DataFrame with label column
        val_features: Validation feature DataFrame with label column
        params: LightGBM parameters
        feature_list: List of feature names to use
        early_stopping_rounds: Early stopping patience

    Returns:
        Trained LightGBM Booster
    """
    params = params or DEFAULT_LGBM_PARAMS.copy()

    if feature_list is None:
        feature_list = get_feature_columns(train_features)

    X_train, y_train, _ = select_features(train_features, feature_list)
    X_val, y_val, _ = select_features(val_features, feature_list)

    logger.info(f"Training on {len(X_train)} samples, {len(feature_list)} features")
    logger.info(f"Train positive rate: {y_train.mean():.4f}")
    logger.info(f"Val positive rate: {y_val.mean():.4f}")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_list)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_list, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=params.get("n_estimators", 500),
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(50),
        ],
    )

    return model


def evaluate_model(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    feature_list: List[str] = None,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate model on feature DataFrame."""
    if feature_list is None:
        feature_list = get_feature_columns(features_df)

    X, y, _ = select_features(features_df, feature_list)

    y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    f05 = fbeta_score(y, y_pred, beta=0.5, zero_division=0)

    # Singleton accuracy: among S1 with 0 true matches, how many correctly predicted 0
    singleton_acc = None
    if "s1_id" in features_df.columns:
        gt_per_s1 = features_df.groupby("s1_id")["label"].max()
        pred_per_s1 = features_df.groupby("s1_id").apply(
            lambda g: (g["label"] == 0).all() and (g["pred"] == 0).all() if "pred" in g.columns else False
        )
        # Add predictions to features_df temporarily
        features_df = features_df.copy()
        features_df["pred"] = y_pred
        pred_any = features_df.groupby("s1_id")["pred"].max()
        singleton_mask = (gt_per_s1 == 0)
        if singleton_mask.any():
            singleton_acc = (pred_any[singleton_mask] == 0).mean()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f05": float(f05),
        "singleton_accuracy": float(singleton_acc) if singleton_acc is not None else None,
        "threshold": threshold,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
    }


def optimize_threshold(
    model: lgb.Booster,
    val_features: pd.DataFrame,
    feature_list: List[str] = None,
    thresholds: List[float] = None,
) -> Tuple[float, Dict]:
    """
    Find optimal threshold on validation set for F0.5.

    Returns:
        best_threshold, best_metrics
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if feature_list is None:
        feature_list = get_feature_columns(val_features)

    X_val, y_val, _ = select_features(val_features, feature_list)
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)

    best_f05 = -1
    best_thresh = 0.5
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)

        metrics = {
            "threshold": thresh,
            "f05": float(f05),
            "precision": float(precision),
            "recall": float(recall),
        }

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = metrics

        logger.info(f"Threshold {thresh:.2f}: F0.5={f05:.4f}, P={precision:.4f}, R={recall:.4f}")

    logger.info(f"Best threshold: {best_thresh:.2f} (F0.5={best_f05:.4f})")
    return best_thresh, best_metrics


def save_model(model: lgb.Booster, feature_names: List[str], output_dir: Path = None):
    """Save model and feature names."""
    output_dir = output_dir or ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "lgbm_baseline.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)

    feat_path = output_dir / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved feature names to {feat_path}")


def load_model(model_dir: Path = None) -> Tuple[lgb.Booster, List[str]]:
    """Load model and feature names."""
    model_dir = model_dir or ARTIFACTS_DIR
    model = lgb.Booster(model_file=str(model_dir / "lgbm_baseline.txt"))
    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    return model, feature_names


def run_baseline_experiment(
    config: Dict = None,
) -> Dict:
    """
    Run complete baseline experiment:
    1. Load data
    2. Generate candidates
    3. Build features
    4. Train LightGBM
    5. Evaluate
    """
    config = config or {}
    logger.info("=" * 60)
    logger.info("BASELINE LIGHTGBM EXPERIMENT")
    logger.info("=" * 60)

    # 1. Load sources
    s1, s2, s3 = prepare_sources_for_features(train=True)

    # 2. Generate candidates (use subset for speed in development)
    from blocking import generate_candidates
    sample_size = config.get("sample_size", None)
    if sample_size:
        s1 = s1.head(sample_size)
        s2 = s2.head(sample_size * 2)
        s3 = s3.head(sample_size * 2)

    candidates = generate_candidates(s1, s2, s3, config.get("blocking", {}))

    # 3. Candidate stats
    cand_stats = compute_candidate_stats(candidates)
    logger.info(f"Candidate stats: {cand_stats}")

    # 4. Candidate recall (using full ground truth)
    cand_recall = compute_candidate_recall(candidates)
    logger.info(f"Candidate recall: {cand_recall}")

    # 5. Prepare training data
    train_features, val_features = prepare_training_data(candidates, s1, s2, s3)

    # 6. Train
    model = train_lgbm(train_features, val_features, params=config.get("lgbm_params"))

    # 7. Evaluate
    feature_list = get_feature_columns(train_features)
    val_metrics = evaluate_model(model, val_features, feature_list, threshold=0.5)
    logger.info(f"Val metrics (thresh=0.5): {val_metrics}")

    # 8. Optimize threshold
    best_thresh, best_metrics = optimize_threshold(model, val_features, feature_list)
    val_metrics_opt = evaluate_model(model, val_features, feature_list, threshold=best_thresh)
    logger.info(f"Val metrics (opt thresh={best_thresh}): {val_metrics_opt}")

    # 9. Save
    save_model(model, feature_list)
    save_feature_importance(feature_list, model.feature_importance(importance_type="gain"), ARTIFACTS_DIR / "feature_importance.csv")

    return {
        "model": model,
        "feature_list": feature_list,
        "best_threshold": best_thresh,
        "val_metrics": val_metrics_opt,
        "candidate_stats": cand_stats,
        "candidate_recall": cand_recall,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick test with small sample
    result = run_baseline_experiment({
        "sample_size": 5000,
        "blocking": {
            "tfidf_name_top_k": 20,
            "tfidf_addr_top_k": 20,
        },
    })
    print("Experiment complete!")
    print(f"Best threshold: {result['best_threshold']}")
    print(f"Val F0.5: {result['val_metrics']['f05']:.4f}")