"""
High-level feature engineering pipeline.

Combines similarity features with metadata for model training/inference.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CACHE_DIR
from io_utils import load_tsv
from normalization import add_normalized_columns
from similarity import (
    compute_all_features,
    compute_name_features,
    compute_address_features,
    compute_general_features,
)

logger = logging.getLogger(__name__)

# Feature column groups for reference
NAME_FEATURES = [
    "name_exact", "name_ratio", "name_wratio", "name_qratio",
    "name_token_sort", "name_token_set", "name_partial",
    "name_levenshtein", "name_jaccard_char", "name_dice",
    "name_len_diff", "name_len_ratio", "name_tok_diff",
    "name_shared_toks", "name_jaccard_tok",
    "name_tfidf_char", "name_tfidf_tok",
]

ADDR_FEATURES = [
    "addr_exact", "addr_ratio", "addr_wratio", "addr_qratio",
    "addr_token_sort", "addr_token_set", "addr_partial",
    "addr_levenshtein", "addr_jaccard_char", "addr_dice",
    "addr_len_diff", "addr_len_ratio", "addr_tok_diff",
    "addr_shared_toks", "addr_jaccard_tok",
    "addr_tfidf_char", "addr_tfidf_tok",
    "postal_match", "house_num_match", "digit_overlap",
    "digit_jaccard", "shared_numeric_toks", "addr_tok_overlap",
]

GENERAL_FEATURES = [
    "country_exact", "country_norm_match",
    "name_missing", "cand_name_missing",
    "addr_missing", "cand_addr_missing",
    "source_s2_s3",
]

ALL_FEATURES = NAME_FEATURES + ADDR_FEATURES + GENERAL_FEATURES


def prepare_sources_for_features(train: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize all sources for feature computation."""
    from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3

    if train:
        s1 = load_tsv(TRAIN_SOURCE1)
        s2 = load_tsv(TRAIN_SOURCE2)
        s3 = load_tsv(TRAIN_SOURCE3)
    else:
        s1 = load_tsv(TEST_SOURCE1)
        s2 = load_tsv(TEST_SOURCE2)
        s3 = load_tsv(TEST_SOURCE3)

    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    return s1, s2, s3


def build_feature_matrix(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build complete feature matrix for candidate pairs.

    Args:
        candidates_df: DataFrame with s1_id, candidate_id, candidate_source, [label], [blocking_methods]
        s1_df, s2_df, s3_df: Normalized source DataFrames
        cache_path: Optional path to cache computed features

    Returns:
        Feature DataFrame with all features + identifiers
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Building features for {len(candidates_df)} candidate pairs...")

    features = compute_all_features(candidates_df, s1_df, s2_df, s3_df)

    # Ensure feature columns are float
    for col in ALL_FEATURES:
        if col in features.columns:
            features[col] = features[col].astype(float)

    # Fill any NaN
    features = features.fillna(0.0)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path)
        logger.info(f"Cached features to {cache_path}")

    logger.info(f"Feature matrix shape: {features.shape}")
    return features


def get_feature_columns(features_df: pd.DataFrame) -> List[str]:
    """Get list of feature column names (excludes identifier/label columns)."""
    exclude = {"s1_id", "candidate_id", "candidate_source", "label", "blocking_methods"}
    return [c for c in features_df.columns if c not in exclude]


def select_features(
    features_df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Select features for modeling.

    Returns:
        X: Feature matrix
        y: Labels (or zeros if not present)
        feature_names: List of feature names used
    """
    if feature_list is None:
        feature_list = get_feature_columns(features_df)

    X = features_df[feature_list].values
    y = features_df["label"].values if "label" in features_df.columns else np.zeros(len(features_df))

    return X, y, feature_list


def add_embedding_features(
    features_df: pd.DataFrame,
    s1_embeddings: Dict[str, np.ndarray],
    cand_embeddings: Dict[str, np.ndarray],
    prefix: str = "emb",
) -> pd.DataFrame:
    """
    Add embedding cosine similarity features.

    Args:
        features_df: Feature DataFrame with s1_id, candidate_id columns
        s1_embeddings: Dict mapping s1_id -> embedding vector
        cand_embeddings: Dict mapping candidate_id -> embedding vector
        prefix: Prefix for new feature columns

    Returns:
        DataFrame with added embedding features
    """
    out = features_df.copy()

    def cosine_sim(a, b):
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    out[f"{prefix}_cosine"] = out.apply(
        lambda r: cosine_sim(s1_embeddings.get(r["s1_id"]), cand_embeddings.get(r["candidate_id"])),
        axis=1,
    )

    return out


def add_entity_level_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add entity-level features (computed per S1 entity).

    Features:
    - top_score, second_score, third_score (requires model scores)
    - score_margin (top - second)
    - n_above_threshold
    - n_high_confidence
    """
    # This is a placeholder - requires model scores to be computed first
    # Actual implementation in decision.py after scoring
    return features_df


def save_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: Path,
):
    """Save feature importance for analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(output_path, index=False)
    logger.info(f"Saved feature importance to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Test feature computation on small sample
    s1, s2, s3 = prepare_sources_for_features(train=True)
    s1_s = s1.head(100)
    s2_s = s2.head(500)
    s3_s = s3.head(500)

    from blocking import generate_candidates
    candidates = generate_candidates(s1_s, s2_s, s3_s)

    features = build_feature_matrix(candidates, s1_s, s2_s, s3_s)
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {get_feature_columns(features)}")
    print(features.head())