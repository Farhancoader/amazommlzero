"""
Score calibration for entity resolution.

Implements Platt scaling and isotonic regression.
Only keep if improves validation F0.5.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

logger = logging.getLogger(__name__)


def platt_scaling(
    scores: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """
    Fit Platt scaling (logistic regression on scores).

    Args:
        scores: Model scores (probabilities or raw scores)
        labels: True binary labels

    Returns:
        Fitted LogisticRegression calibrator
    """
    # Reshape for sklearn
    X = scores.reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(X, labels)

    return calibrator


def isotonic_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
) -> IsotonicRegression:
    """
    Fit isotonic regression calibration.

    Args:
        scores: Model scores
        labels: True binary labels

    Returns:
        Fitted IsotonicRegression calibrator
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(scores, labels)
    return calibrator


def apply_calibration(
    scores: np.ndarray,
    calibrator,
) -> np.ndarray:
    """Apply calibrator to scores."""
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    elif isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(scores)
    else:
        raise ValueError(f"Unknown calibrator type: {type(calibrator)}")


def evaluate_calibration(
    calibrator,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    thresholds: list = None,
) -> Dict:
    """Evaluate calibrator on validation set."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    cal_probs = apply_calibration(val_scores, calibrator)

    best_f05 = -1
    best_thresh = 0.5
    best_metrics = {}

    for thresh in thresholds:
        preds = (cal_probs >= thresh).astype(int)
        f05 = fbeta_score(val_labels, preds, beta=0.5, zero_division=0)

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, _, _ = precision_recall_fscore_support(
                val_labels, preds, average="binary", zero_division=0
            )
            best_metrics = {
                "threshold": thresh,
                "f05": float(f05),
                "precision": float(precision),
                "recall": float(recall),
            }

    return {
        "best_threshold": best_thresh,
        "best_f05": best_f05,
        "metrics": best_metrics,
    }


def calibrate_and_evaluate(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    method: str = "both",
) -> Dict:
    """
    Fit and evaluate calibration methods.

    Args:
        train_scores: Training scores for fitting
        train_labels: Training labels
        val_scores: Validation scores for evaluation
        val_labels: Validation labels
        method: "platt", "isotonic", or "both"

    Returns:
        Dict with calibration results
    """
    results = {}

    # Uncalibrated baseline
    uncal_results = evaluate_calibration(None, val_scores, val_labels)
    # For uncalibrated, just use raw scores with sigmoid if needed
    if val_scores.max() > 1 or val_scores.min() < 0:
        # Assume logits, apply sigmoid
        uncal_probs = 1 / (1 + np.exp(-val_scores))
    else:
        uncal_probs = val_scores

    from sklearn.metrics import fbeta_score, precision_recall_fscore_support
    best_f05 = -1
    best_thresh = 0.5
    for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        preds = (uncal_probs >= thresh).astype(int)
        f05 = fbeta_score(val_labels, preds, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            precision, recall, _, _ = precision_recall_fscore_support(val_labels, preds, average="binary", zero_division=0)

    results["uncalibrated"] = {
        "best_threshold": best_thresh,
        "best_f05": best_f05,
        "precision": float(precision),
        "recall": float(recall),
    }

    if method in ("platt", "both"):
        logger.info("Fitting Platt scaling...")
        platt = platt_scaling(train_scores, train_labels)
        platt_results = evaluate_calibration(platt, val_scores, val_labels)
        results["platt"] = {
            "calibrator": platt,
            **platt_results,
        }
        logger.info(f"Platt: F0.5={platt_results['best_f05']:.4f} at thresh={platt_results['best_threshold']:.2f}")

    if method in ("isotonic", "both"):
        logger.info("Fitting isotonic regression...")
        isotonic = isotonic_calibration(train_scores, train_labels)
        iso_results = evaluate_calibration(isotonic, val_scores, val_labels)
        results["isotonic"] = {
            "calibrator": isotonic,
            **iso_results,
        }
        logger.info(f"Isotonic: F0.5={iso_results['best_f05']:.4f} at thresh={iso_results['best_threshold']:.2f}")

    # Select best
    best_method = min(results.keys(), key=lambda k: -results[k]["best_f05"])
    results["best_method"] = best_method
    results["improvement"] = results[best_method]["best_f05"] - results["uncalibrated"]["best_f05"]

    logger.info(f"Best calibration: {best_method} (F0.5 improvement: {results['improvement']:.4f})")

    return results


def save_calibrator(calibrator, output_path: Path):
    """Save calibrator to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, output_path)
    logger.info(f"Saved calibrator to {output_path}")


def load_calibrator(input_path: Path):
    """Load calibrator from disk."""
    return joblib.load(input_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Test with synthetic data
    np.random.seed(42)
    n = 10000
    train_scores = np.random.randn(n)
    train_labels = (train_scores + np.random.randn(n) * 0.5 > 0).astype(int)

    val_scores = np.random.randn(2000)
    val_labels = (val_scores + np.random.randn(2000) * 0.5 > 0).astype(int)

    results = calibrate_and_evaluate(train_scores, train_labels, val_scores, val_labels)
    print("Calibration results:")
    for k, v in results.items():
        if k != "best_method" and k != "improvement":
            print(f"  {k}: F0.5={v['best_f05']:.4f} at thresh={v['best_threshold']:.2f}")
    print(f"Best: {results['best_method']}, improvement: {results['improvement']:.4f}")