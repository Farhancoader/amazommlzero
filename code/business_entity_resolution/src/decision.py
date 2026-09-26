"""
Decision logic for entity resolution.

Implements threshold optimization, singleton handling,
score margins, and entity-level prediction rules.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_fscore_support

from config import DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)


def optimize_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    s1_ids: np.ndarray,
    thresholds: List[float] = None,
) -> Tuple[float, Dict]:
    """
    Optimize decision threshold for F0.5.

    Args:
        scores: Model scores
        labels: True labels
        s1_ids: S1 entity IDs for each pair
        thresholds: List of thresholds to try

    Returns:
        (best_threshold, best_metrics)
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    best_f05 = -1
    best_thresh = DEFAULT_THRESHOLD
    best_metrics = {}

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f05 = fbeta_score(labels, preds, beta=0.5, zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

        # Also compute singleton accuracy
        pred_df = pd.DataFrame({"s1_id": s1_ids, "pred": preds, "label": labels})
        pred_any = pred_df.groupby("s1_id")["pred"].max()
        label_any = pred_df.groupby("s1_id")["label"].max()
        singleton_mask = (label_any == 0)
        singleton_acc = (pred_any[singleton_mask] == 0).mean() if singleton_mask.any() else 1.0

        metrics = {
            "threshold": thresh,
            "f05": float(f05),
            "precision": float(precision),
            "recall": float(recall),
            "singleton_accuracy": float(singleton_acc),
        }

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = metrics

        logger.info(f"Thresh {thresh:.2f}: F0.5={f05:.4f}, P={precision:.4f}, R={recall:.4f}, SingAcc={singleton_acc:.4f}")

    logger.info(f"Best threshold: {best_thresh:.2f} (F0.5={best_f05:.4f})")
    return best_thresh, best_metrics


def predict_with_threshold(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply threshold to get binary predictions."""
    return (scores >= threshold).astype(int)


def compute_entity_level_features(
    scores: np.ndarray,
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_sources: np.ndarray,
) -> pd.DataFrame:
    """
    Compute entity-level features for each S1.

    Features:
    - top_score, second_score, third_score
    - score_margin (top - second)
    - n_above_threshold (for various thresholds)
    - n_candidates
    """
    df = pd.DataFrame({
        "s1_id": s1_ids,
        "candidate_id": candidate_ids,
        "candidate_source": candidate_sources,
        "score": scores,
    })

    # Sort by score descending per S1
    df = df.sort_values(["s1_id", "score"], ascending=[True, False])

    # Compute features per S1
    entity_features = []

    for s1_id, group in df.groupby("s1_id"):
        scores_sorted = group["score"].values
        cand_ids = group["candidate_id"].values
        cand_sources = group["candidate_source"].values

        feat = {"s1_id": s1_id}
        feat["n_candidates"] = len(scores_sorted)

        if len(scores_sorted) > 0:
            feat["top_score"] = scores_sorted[0]
            feat["top_candidate"] = cand_ids[0]
            feat["top_source"] = cand_sources[0]
        else:
            feat["top_score"] = 0.0
            feat["top_candidate"] = ""
            feat["top_source"] = ""

        if len(scores_sorted) > 1:
            feat["second_score"] = scores_sorted[1]
            feat["score_margin"] = scores_sorted[0] - scores_sorted[1]
        else:
            feat["second_score"] = 0.0
            feat["score_margin"] = feat["top_score"]

        if len(scores_sorted) > 2:
            feat["third_score"] = scores_sorted[2]
        else:
            feat["third_score"] = 0.0

        # Count above thresholds
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            feat[f"n_above_{int(thresh*100)}"] = int((scores_sorted >= thresh).sum())

        entity_features.append(feat)

    return pd.DataFrame(entity_features)


def predict_entities(
    scores: np.ndarray,
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_sources: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    use_margin: bool = False,
    min_margin: float = 0.1,
    require_min_score: float = None,
) -> Dict[str, Set[str]]:
    """
    Make entity-level predictions from pair scores.

    Args:
        scores: Model scores for each candidate pair
        s1_ids: S1 entity IDs
        candidate_ids: Candidate entity IDs
        candidate_sources: Candidate sources (S2/S3)
        threshold: Absolute score threshold
        use_margin: Whether to use margin-based decision
        min_margin: Minimum margin (top - second) required
        require_min_score: Minimum absolute score for top candidate

    Returns:
        Dict mapping s1_id -> set of matched candidate_ids
    """
    df = pd.DataFrame({
        "s1_id": s1_ids,
        "candidate_id": candidate_ids,
        "candidate_source": candidate_sources,
        "score": scores,
    })

    # Sort by score descending
    df = df.sort_values(["s1_id", "score"], ascending=[True, False])

    predictions = {}

    for s1_id, group in df.groupby("s1_id"):
        group = group.reset_index(drop=True)

        if len(group) == 0:
            predictions[s1_id] = set()
            continue

        top_row = group.iloc[0]
        top_score = top_row["score"]

        # Check absolute threshold
        if require_min_score is not None and top_score < require_min_score:
            predictions[s1_id] = set()
            continue

        if top_score < threshold:
            predictions[s1_id] = set()
            continue

        # Check margin if enabled
        if use_margin and len(group) > 1:
            second_score = group.iloc[1]["score"]
            margin = top_score - second_score
            if margin < min_margin:
                predictions[s1_id] = set()
                continue

        # Predict top candidate (or all above threshold for multi-match)
        # For now: predict only top if it passes checks
        predictions[s1_id] = {top_row["candidate_id"]}

        # Optional: predict all above threshold
        # above_thresh = group[group["score"] >= threshold]
        # predictions[s1_id] = set(above_thresh["candidate_id"].tolist())

    return predictions


def predict_entities_multi_match(
    scores: np.ndarray,
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_sources: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    max_matches: int = 10,
) -> Dict[str, Set[str]]:
    """
    Predict multiple matches per S1 (all above threshold up to max_matches).
    """
    df = pd.DataFrame({
        "s1_id": s1_ids,
        "candidate_id": candidate_ids,
        "candidate_source": candidate_sources,
        "score": scores,
    })

    df = df.sort_values(["s1_id", "score"], ascending=[True, False])

    predictions = {}

    for s1_id, group in df.groupby("s1_id"):
        above = group[group["score"] >= threshold].head(max_matches)
        predictions[s1_id] = set(above["candidate_id"].tolist())

    return predictions


def evaluate_entity_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict:
    """
    Evaluate entity-level predictions.

    Computes: precision, recall, F0.5, singleton accuracy
    """
    all_s1 = set(predictions.keys()) | set(ground_truth.keys())

    tp = fp = fn = 0
    singleton_correct = 0
    singleton_total = 0

    for s1_id in all_s1:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())

        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)

        if len(true) == 0:
            singleton_total += 1
            if len(pred) == 0:
                singleton_correct += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f05": f05,
        "singleton_accuracy": singleton_acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "singleton_correct": singleton_correct,
        "singleton_total": singleton_total,
    }


def find_optimal_decision_rule(
    val_features: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    s1_col: str = "s1_id",
) -> Dict:
    """
    Search for optimal decision rule on validation data.

    Rules tested:
    - Absolute threshold
    - Margin-based
    - Combined (threshold + margin)
    - Top-K with threshold
    """
    scores = val_features[score_col].values
    labels = val_features[label_col].values
    s1_ids = val_features[s1_col].values

    gt_dict = val_features.groupby(s1_col)[label_col].apply(
        lambda x: set(val_features.loc[x.index, "candidate_id"][x == 1])
    ).to_dict()

    best_f05 = -1
    best_rule = {}

    # Test thresholds
    for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        for min_margin in [0.0, 0.05, 0.1, 0.15, 0.2]:
            preds = predict_entities(
                scores, s1_ids,
                val_features["candidate_id"].values,
                val_features["candidate_source"].values,
                threshold=thresh,
                use_margin=(min_margin > 0),
                min_margin=min_margin,
            )
            metrics = evaluate_entity_predictions(preds, gt_dict)

            if metrics["f05"] > best_f05:
                best_f05 = metrics["f05"]
                best_rule = {
                    "type": "threshold_margin",
                    "threshold": thresh,
                    "min_margin": min_margin,
                    "metrics": metrics,
                }
                logger.info(f"New best: thresh={thresh}, margin={min_margin}, F0.5={metrics['f05']:.4f}")

    return best_rule


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Test with synthetic data
    np.random.seed(42)
    n = 5000
    s1_ids = np.repeat([f"S1-{i}" for i in range(500)], 10)[:n]
    candidate_ids = np.array([f"S2-{i}" for i in range(n)])
    scores = np.random.rand(n)
    labels = (scores + np.random.randn(n) * 0.3 > 0.5).astype(int)

    threshold, metrics = optimize_threshold(scores, labels, s1_ids)
    print(f"Optimal threshold: {threshold:.2f}")
    print(f"Metrics: {metrics}")