"""
Ground truth parsing and training data preparation.

Parses train_ground_truth.tsv into positive pairs for model training.
Supports zero, one, and multiple matches per Source 1 entity.
Handles both S2 and S3 matches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from config import (
    TRAIN_GROUND_TRUTH, TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    S1_PREFIX, S2_PREFIX, S3_PREFIX,
)
from io_utils import load_tsv, parse_ground_truth

logger = logging.getLogger(__name__)


def load_ground_truth_pairs() -> pd.DataFrame:
    """
    Load ground truth and convert to positive pair DataFrame.

    Returns:
        DataFrame with columns:
        - s1_id: Source 1 entity ID
        - candidate_id: Source 2 or Source 3 entity ID
        - candidate_source: 'S2' or 'S3'
        - label: 1 (positive match)
    """
    gt_df = load_tsv(TRAIN_GROUND_TRUTH)
    gt_dict = parse_ground_truth(gt_df)

    rows = []
    for s1_id, matches in gt_dict.items():
        for match_id in matches:
            if match_id.startswith(S2_PREFIX):
                source = "S2"
            elif match_id.startswith(S3_PREFIX):
                source = "S3"
            else:
                logger.warning(f"Unknown prefix for match ID: {match_id}")
                continue
            rows.append({
                "s1_id": s1_id,
                "candidate_id": match_id,
                "candidate_source": source,
                "label": 1,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Created {len(df)} positive pairs from ground truth")
    logger.info(f"  S2 matches: {(df['candidate_source'] == 'S2').sum()}")
    logger.info(f"  S3 matches: {(df['candidate_source'] == 'S3').sum()}")
    logger.info(f"  Unique S1 entities: {df['s1_id'].nunique()}")

    return df


def get_s1_entities_with_matches() -> Dict[str, int]:
    """
    Get count of matches per S1 entity from ground truth.

    Returns:
        Dict mapping s1_id -> number of matches
    """
    gt_df = load_tsv(TRAIN_GROUND_TRUTH)
    gt_dict = parse_ground_truth(gt_df)
    return {s1_id: len(matches) for s1_id, matches in gt_dict.items()}


def get_singleton_s1_entities() -> List[str]:
    """Get list of S1 entity IDs with zero matches in ground truth."""
    gt_df = load_tsv(TRAIN_GROUND_TRUTH)
    gt_dict = parse_ground_truth(gt_df)
    return [s1_id for s1_id, matches in gt_dict.items() if len(matches) == 0]


def verify_ground_truth_integrity() -> Dict[str, any]:
    """
    Verify all ground truth IDs exist in source files.

    Returns:
        Dict with verification results
    """
    train_s1 = load_tsv(TRAIN_SOURCE1)
    train_s2 = load_tsv(TRAIN_SOURCE2)
    train_s3 = load_tsv(TRAIN_SOURCE3)
    gt_df = load_tsv(TRAIN_GROUND_TRUTH)
    gt_dict = parse_ground_truth(gt_df)

    s1_ids = set(train_s1["entity_id"].astype(str))
    s2_ids = set(train_s2["entity_id"].astype(str))
    s3_ids = set(train_s3["entity_id"].astype(str))

    missing_s1 = []
    missing_s2 = []
    missing_s3 = []

    for s1_id, matches in gt_dict.items():
        if s1_id not in s1_ids:
            missing_s1.append(s1_id)
        for m in matches:
            if m.startswith(S2_PREFIX) and m not in s2_ids:
                missing_s2.append(m)
            elif m.startswith(S3_PREFIX) and m not in s3_ids:
                missing_s3.append(m)

    result = {
        "missing_s1_in_source1": len(missing_s1),
        "missing_s2_in_source2": len(missing_s2),
        "missing_s3_in_source3": len(missing_s3),
        "missing_s1_examples": missing_s1[:10],
        "missing_s2_examples": missing_s2[:10],
        "missing_s3_examples": missing_s3[:10],
    }

    if any([missing_s1, missing_s2, missing_s3]):
        logger.warning(f"Ground truth integrity issues: {result}")
    else:
        logger.info("Ground truth integrity verified: all IDs exist in sources")

    return result


def create_training_pairs(
    candidate_pairs_df: pd.DataFrame,
    gt_pairs_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge candidate pairs with ground truth labels.

    Args:
        candidate_pairs_df: DataFrame with columns [s1_id, candidate_id, candidate_source, ...]
        gt_pairs_df: Optional pre-loaded ground truth pairs. If None, loads from file.

    Returns:
        DataFrame with label column (1 for match, 0 for non-match)
    """
    if gt_pairs_df is None:
        gt_pairs_df = load_ground_truth_pairs()

    # Create lookup set for fast matching
    gt_keys = set(zip(gt_pairs_df["s1_id"], gt_pairs_df["candidate_id"]))

    # Label candidates
    def is_match(row):
        return 1 if (row["s1_id"], row["candidate_id"]) in gt_keys else 0

    candidate_pairs_df = candidate_pairs_df.copy()
    candidate_pairs_df["label"] = candidate_pairs_df.apply(is_match, axis=1)

    n_pos = candidate_pairs_df["label"].sum()
    n_neg = len(candidate_pairs_df) - n_pos
    logger.info(f"Labeled {len(candidate_pairs_df)} candidate pairs: {n_pos} positive, {n_neg} negative")

    return candidate_pairs_df


def split_by_s1_entity(
    pairs_df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pairs by S1 entity (not by pair) to avoid leakage.

    All pairs for a given S1 entity go to either train or val.

    Args:
        pairs_df: DataFrame with s1_id column
        val_fraction: Fraction of unique S1 entities for validation
        seed: Random seed for reproducibility

    Returns:
        (train_df, val_df)
    """
    import numpy as np
    np.random.seed(seed)

    unique_s1 = pairs_df["s1_id"].unique()
    n_val = int(len(unique_s1) * val_fraction)
    val_s1 = set(np.random.choice(unique_s1, size=n_val, replace=False))

    train_df = pairs_df[~pairs_df["s1_id"].isin(val_s1)].copy()
    val_df = pairs_df[pairs_df["s1_id"].isin(val_s1)].copy()

    logger.info(f"Entity-level split: {len(train_df)} train pairs, {len(val_df)} val pairs")
    logger.info(f"  Train S1 entities: {train_df['s1_id'].nunique()}")
    logger.info(f"  Val S1 entities: {val_df['s1_id'].nunique()}")

    return train_df, val_df


def save_split_ids(train_s1_ids: Set[str], val_s1_ids: Set[str], output_dir: Path):
    """Save split S1 entity IDs for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"s1_id": sorted(train_s1_ids)}).to_csv(
        output_dir / "train_s1_ids.csv", index=False
    )
    pd.DataFrame({"s1_id": sorted(val_s1_ids)}).to_csv(
        output_dir / "val_s1_ids.csv", index=False
    )
    logger.info(f"Saved split IDs to {output_dir}")


def load_split_ids(input_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Load previously saved split S1 entity IDs."""
    train_s1 = set(pd.read_csv(input_dir / "train_s1_ids.csv")["s1_id"].astype(str))
    val_s1 = set(pd.read_csv(input_dir / "val_s1_ids.csv")["s1_id"].astype(str))
    return train_s1, val_s1


def apply_saved_split(
    pairs_df: pd.DataFrame,
    split_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a previously saved entity-level split."""
    train_s1, val_s1 = load_split_ids(split_dir)
    train_df = pairs_df[pairs_df["s1_id"].isin(train_s1)].copy()
    val_df = pairs_df[pairs_df["s1_id"].isin(val_s1)].copy()
    return train_df, val_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Test ground truth loading
    gt_pairs = load_ground_truth_pairs()
    print(f"Positive pairs: {len(gt_pairs)}")
    print(gt_pairs.head())

    # Verify integrity
    verify_ground_truth_integrity()

    # Test singleton detection
    singletons = get_singleton_s1_entities()
    print(f"Singleton S1 entities (0 matches): {len(singletons)}")