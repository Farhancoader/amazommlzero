"""
Evaluation utilities for entity resolution.

Computes candidate recall, F0.5, precision, recall, singleton accuracy,
and generates error analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_fscore_support

from config import VALIDATION_DIR
from io_utils import parse_ground_truth, load_tsv

logger = logging.getLogger(__name__)


def compute_candidate_recall(
    candidate_pairs_df: pd.DataFrame,
    gt_df: pd.DataFrame = None,
    s1_ids: Set[str] = None,
) -> Dict:
    """
    Compute candidate recall: fraction of true matches included in candidates.

    Args:
        candidate_pairs_df: DataFrame with s1_id, candidate_id columns
        gt_df: Ground truth DataFrame (if None, loads from file)
        s1_ids: Optional subset of S1 entities to evaluate

    Returns:
        Dict with recall metrics
    """
    if gt_df is None:
        gt_df = load_tsv("dataset/train/train_ground_truth.tsv")
    gt_dict = parse_ground_truth(gt_df)

    if s1_ids is not None:
        gt_dict = {k: v for k, v in gt_dict.items() if k in s1_ids}

    # Build candidate lookup
    candidate_lookup = {}
    for _, row in candidate_pairs_df.iterrows():
        s1 = row["s1_id"]
        cand = row["candidate_id"]
        if s1 not in candidate_lookup:
            candidate_lookup[s1] = set()
        candidate_lookup[s1].add(cand)

    total_true = 0
    total_found = 0
    s1_recalls = []
    s1_with_matches = 0

    for s1_id, true_matches in gt_dict.items():
        if not true_matches:
            continue
        s1_with_matches += 1
        candidates = candidate_lookup.get(s1_id, set())
        found = len(true_matches & candidates)
        total_true += len(true_matches)
        total_found += found
        s1_recalls.append(found / len(true_matches))

    overall_recall = total_found / total_true if total_true > 0 else 1.0
    mean_s1_recall = np.mean(s1_recalls) if s1_recalls else 1.0

    return {
        "candidate_recall": overall_recall,
        "mean_s1_recall": mean_s1_recall,
        "total_true_matches": total_true,
        "total_found_matches": total_found,
        "s1_evaluated": len(s1_recalls),
        "s1_with_matches": s1_with_matches,
    }


def compute_candidate_stats(candidate_pairs_df: pd.DataFrame, s1_ids: Set[str] = None) -> Dict:
    """Compute candidate generation statistics."""
    if s1_ids is not None:
        candidate_pairs_df = candidate_pairs_df[candidate_pairs_df["s1_id"].isin(s1_ids)]

    candidates_per_s1 = candidate_pairs_df.groupby("s1_id").size()

    return {
        "avg_candidates_per_s1": float(candidates_per_s1.mean()),
        "median_candidates_per_s1": float(candidates_per_s1.median()),
        "p95_candidates_per_s1": float(candidates_per_s1.quantile(0.95)),
        "max_candidates_per_s1": int(candidates_per_s1.max()),
        "min_candidates_per_s1": int(candidates_per_s1.min()),
        "total_candidates": len(candidate_pairs_df),
        "unique_s1_with_candidates": int(candidates_per_s1.count()),
    }


def compute_pair_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
) -> Dict:
    """Compute pairwise metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f05": float(f05),
    }

    if y_score is not None:
        # AUC metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass

    return metrics


def compute_entity_metrics(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict:
    """
    Compute entity-level metrics.

    This is the competition metric: macro-averaged F0.5
    """
    all_s1 = set(predictions.keys()) | set(ground_truth.keys())

    per_s1_f05 = []

    for s1_id in all_s1:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())

        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if len(pred) == 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if len(true) == 0 else 0.0

        if precision + recall > 0:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            f05 = 1.0 if len(pred) == 0 and len(true) == 0 else 0.0

        per_s1_f05.append(f05)

    macro_f05 = np.mean(per_s1_f05) if per_s1_f05 else 0.0

    # Also compute micro metrics
    tp_total = sum(len(predictions.get(s, set()) & ground_truth.get(s, set())) for s in all_s1)
    fp_total = sum(len(predictions.get(s, set()) - ground_truth.get(s, set())) for s in all_s1)
    fn_total = sum(len(ground_truth.get(s, set()) - predictions.get(s, set())) for s in all_s1)

    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f05 = (1.25 * micro_precision * micro_recall) / (0.25 * micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Singleton accuracy
    singleton_correct = 0
    singleton_total = 0
    for s1_id in all_s1:
        true = ground_truth.get(s1_id, set())
        pred = predictions.get(s1_id, set())
        if len(true) == 0:
            singleton_total += 1
            if len(pred) == 0:
                singleton_correct += 1

    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

    return {
        "macro_f05": float(macro_f05),
        "micro_f05": float(micro_f05),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "singleton_accuracy": float(singleton_acc),
        "singleton_correct": singleton_correct,
        "singleton_total": singleton_total,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def generate_error_analysis(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate detailed error analysis CSV.

    Columns:
    - s1_id, s1_name, s1_address, s1_country
    - candidate_id, cand_name, cand_address, cand_country
    - candidate_source
    - true_label, pred_label
    - error_type: FP, FN, singleton_FP, singleton_FN
    - top_competing_candidates
    """
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

    error_rows = []

    all_s1 = set(predictions.keys()) | set(ground_truth.keys())

    for s1_id in all_s1:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())

        s1_row = s1_df[s1_df["entity_id"] == s1_id]
        if len(s1_row) == 0:
            continue
        s1_row = s1_row.iloc[0]

        # False positives
        for cand_id in pred - true:
            cand_row = s23_df[s23_df["entity_id"] == cand_id]
            if len(cand_row) == 0:
                continue
            cand_row = cand_row.iloc[0]

            # Get top competing candidates
            s1_cands = candidate_pairs_df[candidate_pairs_df["s1_id"] == s1_id]
            top_comp = s1_cands.nlargest(5, "score")["candidate_id"].tolist() if "score" in s1_cands.columns else []

            error_rows.append({
                "s1_id": s1_id,
                "s1_name": s1_row.get("business_name", ""),
                "s1_address": s1_row.get("business_address", ""),
                "s1_country": s1_row.get("country", ""),
                "candidate_id": cand_id,
                "cand_name": cand_row.get("business_name", ""),
                "cand_address": cand_row.get("business_address", ""),
                "cand_country": cand_row.get("country", ""),
                "candidate_source": cand_row.get("source", ""),
                "true_label": 0,
                "pred_label": 1,
                "error_type": "FP",
                "top_competing": "|".join(top_comp),
            })

        # False negatives
        for cand_id in true - pred:
            cand_row = s23_df[s23_df["entity_id"] == cand_id]
            if len(cand_row) == 0:
                continue
            cand_row = cand_row.iloc[0]

            s1_cands = candidate_pairs_df[candidate_pairs_df["s1_id"] == s1_id]
            top_comp = s1_cands.nlargest(5, "score")["candidate_id"].tolist() if "score" in s1_cands.columns else []

            error_rows.append({
                "s1_id": s1_id,
                "s1_name": s1_row.get("business_name", ""),
                "s1_address": s1_row.get("business_address", ""),
                "s1_country": s1_row.get("country", ""),
                "candidate_id": cand_id,
                "cand_name": cand_row.get("business_name", ""),
                "cand_address": cand_row.get("business_address", ""),
                "cand_country": cand_row.get("country", ""),
                "candidate_source": cand_row.get("source", ""),
                "true_label": 1,
                "pred_label": 0,
                "error_type": "FN",
                "top_competing": "|".join(top_comp),
            })

        # Singleton errors (predicted match but true is empty)
        if len(true) == 0 and len(pred) > 0:
            for cand_id in pred:
                cand_row = s23_df[s23_df["entity_id"] == cand_id]
                if len(cand_row) == 0:
                    continue
                cand_row = cand_row.iloc[0]
                error_rows.append({
                    "s1_id": s1_id,
                    "s1_name": s1_row.get("business_name", ""),
                    "s1_address": s1_row.get("business_address", ""),
                    "s1_country": s1_row.get("country", ""),
                    "candidate_id": cand_id,
                    "cand_name": cand_row.get("business_name", ""),
                    "cand_address": cand_row.get("business_address", ""),
                    "cand_country": cand_row.get("country", ""),
                    "candidate_source": cand_row.get("source", ""),
                    "true_label": 0,
                    "pred_label": 1,
                    "error_type": "singleton_FP",
                    "top_competing": "",
                })

        # Singleton FN (missed all - but this is just FN for entities with matches)
        # Already covered in FN

    error_df = pd.DataFrame(error_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(output_path, index=False)
    logger.info(f"Generated error analysis: {len(error_df)} errors to {output_path}")

    return error_df


def save_experiment_results(
    experiment_id: str,
    config: Dict,
    metrics: Dict,
    output_dir: Path = None,
):
    """Save experiment results for tracking."""
    output_dir = output_dir or (VALIDATION_DIR / "experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    from datetime import datetime

    result = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
    }

    output_path = output_dir / f"{experiment_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Saved experiment results to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Test
    gt = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}, "S1-3": set()}
    pred = {"S1-1": {"S2-1"}, "S1-2": {"S3-1", "S3-2"}, "S1-3": {"S2-3"}}

    metrics = compute_entity_metrics(pred, gt)
    print("Entity metrics:", metrics)