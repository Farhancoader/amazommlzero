"""
Validation split strategy for entity resolution.

Implements entity-level splits to avoid data leakage.
Mimics competition: train split -> candidate generation -> features -> model -> F0.5
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    RANDOM_SEED, DEFAULT_VAL_FRACTION, VALIDATION_DIR,
)
from io_utils import load_tsv, parse_ground_truth
from training_data import load_ground_truth_pairs

logger = logging.getLogger(__name__)


class ValidationSplit:
    """Manages reproducible entity-level train/validation splits."""

    def __init__(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        seed: int = RANDOM_SEED,
        split_dir: Path = None,
    ):
        self.val_fraction = val_fraction
        self.seed = seed
        self.split_dir = split_dir or (VALIDATION_DIR / "splits" / f"seed_{seed}_val_{val_fraction}")
        self.train_s1_ids: Optional[Set[str]] = None
        self.val_s1_ids: Optional[Set[str]] = None

    def create_split(self, force: bool = False) -> Tuple[Set[str], Set[str]]:
        """
        Create entity-level split from training Source 1 entities.

        Args:
            force: Recreate split even if saved split exists

        Returns:
            (train_s1_ids, val_s1_ids)
        """
        if not force and self.split_dir.exists():
            logger.info(f"Loading existing split from {self.split_dir}")
            return self.load_split()

        logger.info("Creating new entity-level validation split...")

        # Load all S1 entities from training
        train_s1 = load_tsv(TRAIN_SOURCE1)
        all_s1_ids = train_s1["entity_id"].astype(str).unique()

        # Stratify by country to ensure both US/India in each split
        train_s1_indexed = train_s1.set_index("entity_id")
        countries = train_s1_indexed.loc[all_s1_ids, "country"].astype(str)

        np.random.seed(self.seed)

        train_ids = []
        val_ids = []

        for country in countries.unique():
            country_s1 = all_s1_ids[countries == country]
            n_country = len(country_s1)
            n_val = int(n_country * self.val_fraction)
            val_country = np.random.choice(country_s1, size=n_val, replace=False)
            train_country = [s for s in country_s1 if s not in val_country]

            val_ids.extend(val_country)
            train_ids.extend(train_country)

        self.train_s1_ids = set(train_ids)
        self.val_s1_ids = set(val_ids)

        # Save split
        self.save_split()

        logger.info(f"Split created: {len(self.train_s1_ids)} train S1, {len(self.val_s1_ids)} val S1")
        logger.info(f"Train countries: {countries[train_ids].value_counts().to_dict()}")
        logger.info(f"Val countries: {countries[list(self.val_s1_ids)].value_counts().to_dict()}")

        return self.train_s1_ids, self.val_s1_ids

    def save_split(self):
        """Save split IDs to disk."""
        self.split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"s1_id": sorted(self.train_s1_ids)}).to_csv(
            self.split_dir / "train_s1_ids.csv", index=False
        )
        pd.DataFrame({"s1_id": sorted(self.val_s1_ids)}).to_csv(
            self.split_dir / "val_s1_ids.csv", index=False
        )
        # Save metadata
        import json
        metadata = {
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "n_train": len(self.train_s1_ids),
            "n_val": len(self.val_s1_ids),
        }
        with open(self.split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved split to {self.split_dir}")

    def load_split(self) -> Tuple[Set[str], Set[str]]:
        """Load split IDs from disk."""
        train_s1 = set(pd.read_csv(self.split_dir / "train_s1_ids.csv")["s1_id"].astype(str))
        val_s1 = set(pd.read_csv(self.split_dir / "val_s1_ids.csv")["s1_id"].astype(str))
        self.train_s1_ids = train_s1
        self.val_s1_ids = val_s1
        return train_s1, val_s1

    def apply_to_pairs(self, pairs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply split to candidate pairs DataFrame."""
        if self.train_s1_ids is None:
            self.load_split()

        train_df = pairs_df[pairs_df["s1_id"].isin(self.train_s1_ids)].copy()
        val_df = pairs_df[pairs_df["s1_id"].isin(self.val_s1_ids)].copy()
        return train_df, val_df


def create_validation_data(
    candidate_pairs_df: pd.DataFrame,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = RANDOM_SEED,
    split_dir: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create validation split from candidate pairs.

    Returns:
        (train_pairs, val_pairs) - both with label column
    """
    splitter = ValidationSplit(val_fraction=val_fraction, seed=seed, split_dir=split_dir)
    splitter.create_split()
    return splitter.apply_to_pairs(candidate_pairs_df)


def get_ground_truth_for_s1(s1_ids: Set[str]) -> Dict[str, Set[str]]:
    """Get ground truth matches for specific S1 entities."""
    gt_df = load_tsv(TRAIN_GROUND_TRUTH)
    gt_dict = parse_ground_truth(gt_df)
    return {s1_id: gt_dict.get(s1_id, set()) for s1_id in s1_ids}


def compute_candidate_recall(
    candidate_pairs_df: pd.DataFrame,
    s1_ids: Set[str] = None,
) -> Dict[str, float]:
    """
    Compute candidate recall: fraction of true matches included in candidates.

    Args:
        candidate_pairs_df: DataFrame with s1_id, candidate_id columns
        s1_ids: Optional subset of S1 entities to evaluate (e.g., val set)

    Returns:
        Dict with recall metrics
    """
    gt_df = load_tsv(TRAIN_GROUND_TRUTH)
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

    # Compute recall
    total_true = 0
    total_found = 0
    s1_recalls = []

    for s1_id, true_matches in gt_dict.items():
        if not true_matches:
            continue  # Skip singletons for recall
        candidates = candidate_lookup.get(s1_id, set())
        found = len(true_matches & candidates)
        total_true += len(true_matches)
        total_found += found
        s1_recalls.append(found / len(true_matches) if true_matches else 1.0)

    overall_recall = total_found / total_true if total_true > 0 else 1.0
    mean_s1_recall = np.mean(s1_recalls) if s1_recalls else 1.0

    return {
        "candidate_recall": overall_recall,
        "mean_s1_recall": mean_s1_recall,
        "total_true_matches": total_true,
        "total_found_matches": total_found,
        "s1_evaluated": len(s1_recalls),
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Test split creation
    splitter = ValidationSplit(val_fraction=0.2, seed=42)
    train_s1, val_s1 = splitter.create_split()
    print(f"Train: {len(train_s1)}, Val: {len(val_s1)}")