"""
Hard negative mining for entity resolution.

Generates difficult negative pairs using:
- TF-IDF nearest neighbors
- Name/address similarity
- Embedding similarity (later)

Removes known positives to avoid false negatives.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    DEFAULT_TFIDF_NAME_TOP_K, DEFAULT_TFIDF_ADDRESS_TOP_K,
)
from io_utils import load_tsv, parse_ground_truth
from normalization import add_normalized_columns

logger = logging.getLogger(__name__)


def load_positive_pairs() -> Set[Tuple[str, str]]:
    """Load ground truth positive pairs as set for fast lookup."""
    gt_df = load_tsv("dataset/train/train_ground_truth.tsv")
    gt_dict = parse_ground_truth(gt_df)

    pairs = set()
    for s1_id, matches in gt_dict.items():
        for m in matches:
            pairs.add((s1_id, m))
    return pairs


def build_tfidf_index(
    df: pd.DataFrame,
    text_col: str,
    ngram_range: Tuple[int, int] = (2, 5),
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """Build TF-IDF vectorizer and matrix for a text column."""
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(""))
    return vectorizer, tfidf_matrix


def mine_tfidf_hard_negatives(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    positive_pairs: Set[Tuple[str, str]],
    top_k: int = 50,
    text_col: str = "business_name_norm",
    ngram_range: Tuple[int, int] = (2, 5),
    name: str = "name",
) -> pd.DataFrame:
    """
    Mine hard negatives using TF-IDF nearest neighbors.

    For each S1, find top-k similar S2/S3 by TF-IDF cosine.
    Remove known positives.
    """
    logger.info(f"Mining {name} TF-IDF hard negatives (top_k={top_k})...")

    # Build TF-IDF on S2+S3
    vectorizer, tfidf_matrix = build_tfidf_index(s23_df, text_col, ngram_range)

    # Transform S1
    s1_tfidf = vectorizer.transform(s1_df[text_col].fillna(""))

    # Nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(top_k, len(s23_df)), metric="cosine")
    nn.fit(tfidf_matrix)

    distances, indices = nn.kneighbors(s1_tfidf)

    rows = []
    for i, s1_idx in enumerate(s1_df.index):
        s1_id = s1_df.loc[s1_idx, "entity_id"]
        for dist, idx in zip(distances[i], indices[i]):
            cand_id = s23_df.iloc[idx]["entity_id"]
            cand_source = s23_df.iloc[idx].get("source", "S2")

            # Skip known positives
            if (s1_id, cand_id) in positive_pairs:
                continue

            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": cand_source,
                "blocking_method": f"hard_neg_tfidf_{name}_{ngram_range[0]}_{ngram_range[1]}",
                "tfidf_score": 1.0 - dist,
            })

    df = pd.DataFrame(rows)
    logger.info(f"  Mined {len(df)} hard negatives from {name} TF-IDF")
    return df


def mine_similarity_hard_negatives(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    positive_pairs: Set[Tuple[str, str]],
    top_k: int = 50,
    similarity_func=None,
    text_col: str = "business_name_norm",
    name: str = "sim",
) -> pd.DataFrame:
    """
    Mine hard negatives using custom similarity function.
    (Slower, use for smaller subsets)
    """
    logger.info(f"Mining {name} similarity hard negatives (top_k={top_k})...")

    rows = []
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        s1_text = s1_row[text_col]

        # Compute similarity to all candidates
        similarities = []
        for _, cand_row in s23_df.iterrows():
            cand_id = cand_row["entity_id"]
            if (s1_id, cand_id) in positive_pairs:
                continue
            cand_text = cand_row[text_col]
            sim = similarity_func(s1_text, cand_text)
            similarities.append((sim, cand_id, cand_row.get("source", "S2")))

        # Top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        for sim, cand_id, source in similarities[:top_k]:
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": source,
                "blocking_method": f"hard_neg_{name}",
                "similarity_score": sim,
            })

    df = pd.DataFrame(rows)
    logger.info(f"  Mined {len(df)} hard negatives from {name} similarity")
    return df


def mine_embedding_hard_negatives(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    positive_pairs: Set[Tuple[str, str]],
    s1_embeddings: Dict[str, np.ndarray],
    cand_embeddings: Dict[str, np.ndarray],
    top_k: int = 50,
) -> pd.DataFrame:
    """Mine hard negatives using embedding nearest neighbors."""
    logger.info(f"Mining embedding hard negatives (top_k={top_k})...")

    # Build candidate embedding matrix
    cand_ids = s23_df["entity_id"].tolist()
    cand_matrix = np.vstack([cand_embeddings.get(cid, np.zeros(768)) for cid in cand_ids])

    # Nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(top_k, len(cand_ids)), metric="cosine")
    nn.fit(cand_matrix)

    rows = []
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        s1_emb = s1_embeddings.get(s1_id)
        if s1_emb is None:
            continue

        distances, indices = nn.kneighbors(s1_emb.reshape(1, -1))

        for dist, idx in zip(distances[0], indices[0]):
            cand_id = cand_ids[idx]
            cand_source = s23_df.iloc[idx].get("source", "S2")

            if (s1_id, cand_id) in positive_pairs:
                continue

            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": cand_source,
                "blocking_method": "hard_neg_embedding",
                "embedding_score": 1.0 - dist,
            })

    df = pd.DataFrame(rows)
    logger.info(f"  Mined {len(df)} hard negatives from embeddings")
    return df


def generate_hard_negatives(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    positive_pairs: Set[Tuple[str, str]] = None,
    config: Dict = None,
) -> pd.DataFrame:
    """
    Generate hard negatives using multiple strategies.

    Args:
        s1_df, s2_df, s3_df: Normalized source DataFrames
        positive_pairs: Set of (s1_id, cand_id) known positives
        config: Dict with mining parameters

    Returns:
        DataFrame of hard negative candidate pairs
    """
    config = config or {}

    if positive_pairs is None:
        positive_pairs = load_positive_pairs()

    # Combine S2 and S3
    s23_df = pd.concat([
        s2_df[["entity_id", "business_name_norm", "business_address_norm"]].assign(source="S2"),
        s3_df[["entity_id", "business_name_norm", "business_address_norm"]].assign(source="S3"),
    ], ignore_index=True)

    all_negatives = []

    # Strategy 1: Name TF-IDF
    if config.get("use_tfidf_name", True):
        neg = mine_tfidf_hard_negatives(
            s1_df, s23_df, positive_pairs,
            top_k=config.get("tfidf_name_top_k", DEFAULT_TFIDF_NAME_TOP_K),
            text_col="business_name_norm",
            ngram_range=config.get("tfidf_name_ngram", (2, 5)),
            name="name",
        )
        all_negatives.append(neg)

    # Strategy 2: Address TF-IDF
    if config.get("use_tfidf_addr", True):
        neg = mine_tfidf_hard_negatives(
            s1_df, s23_df, positive_pairs,
            top_k=config.get("tfidf_addr_top_k", DEFAULT_TFIDF_ADDRESS_TOP_K),
            text_col="business_address_norm",
            ngram_range=config.get("tfidf_addr_ngram", (2, 6)),
            name="addr",
        )
        all_negatives.append(neg)

    # Strategy 3: Embedding (if available)
    if config.get("use_embeddings", False):
        s1_emb = config.get("s1_embeddings", {})
        cand_emb = config.get("cand_embeddings", {})
        if s1_emb and cand_emb:
            neg = mine_embedding_hard_negatives(
                s1_df, s23_df, positive_pairs,
                s1_emb, cand_emb,
                top_k=config.get("embedding_top_k", 50),
            )
            all_negatives.append(neg)

    # Combine and deduplicate
    if all_negatives:
        combined = pd.concat(all_negatives, ignore_index=True)
        # Keep highest score per pair
        score_cols = [c for c in combined.columns if c.endswith("_score")]
        if score_cols:
            combined = combined.sort_values(score_cols[0], ascending=False)
        combined = combined.drop_duplicates(subset=["s1_id", "candidate_id", "candidate_source"])
    else:
        combined = pd.DataFrame()

    logger.info(f"Total hard negatives generated: {len(combined)}")
    return combined


def create_training_set_with_hard_negatives(
    candidates_df: pd.DataFrame,
    hard_negatives_df: pd.DataFrame,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create training set combining original candidates + hard negatives.

    Args:
        candidates_df: Original candidate pairs (with labels)
        hard_negatives_df: Hard negative pairs (no labels, assumed 0)
        negative_ratio: Ratio of negatives to positives (e.g., 1.0 = 1:1, 4.0 = 4:1)
        seed: Random seed

    Returns:
        Combined training DataFrame with labels
    """
    np.random.seed(seed)

    # Separate positives and existing negatives
    pos = candidates_df[candidates_df["label"] == 1].copy()
    existing_neg = candidates_df[candidates_df["label"] == 0].copy()

    logger.info(f"Original: {len(pos)} positive, {len(existing_neg)} negative")

    # Sample hard negatives to achieve target ratio
    target_neg = int(len(pos) * negative_ratio)
    available_hard = len(hard_negatives_df)

    if available_hard >= target_neg:
        sampled_hard = hard_negatives_df.sample(n=target_neg, random_state=seed)
    else:
        sampled_hard = hard_negatives_df.copy()
        logger.warning(f"Only {available_hard} hard negatives available, requested {target_neg}")

    sampled_hard["label"] = 0

    # Combine all negatives
    all_neg = pd.concat([existing_neg, sampled_hard], ignore_index=True)

    # Deduplicate (keep positive if conflict)
    all_neg = all_neg.drop_duplicates(subset=["s1_id", "candidate_id", "candidate_source"])

    # Remove any that are actually positive
    pos_keys = set(zip(pos["s1_id"], pos["candidate_id"]))
    all_neg = all_neg[~all_neg.apply(lambda r: (r["s1_id"], r["candidate_id"]) in pos_keys, axis=1)]

    # Combine
    combined = pd.concat([pos, all_neg], ignore_index=True)

    logger.info(f"Final training set: {len(pos)} positive, {len(all_neg)} negative (ratio={len(all_neg)/len(pos):.2f}:1)")
    return combined


def run_hard_negative_experiment(
    negative_ratios: List[float] = None,
    sample_size: int = None,
) -> Dict:
    """
    Run experiment comparing different negative ratios.

    Returns dict with results per ratio.
    """
    negative_ratios = negative_ratios or [1.0, 2.0, 4.0, 8.0]

    # Load data
    s1 = load_tsv(TRAIN_SOURCE1)
    s2 = load_tsv(TRAIN_SOURCE2)
    s3 = load_tsv(TRAIN_SOURCE3)
    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    if sample_size:
        s1 = s1.head(sample_size)
        s2 = s2.head(sample_size * 2)
        s3 = s3.head(sample_size * 2)

    # Generate base candidates
    from blocking import generate_candidates
    candidates = generate_candidates(s1, s2, s3)

    # Label with ground truth
    gt_pairs = load_positive_pairs()
    gt_keys = set(gt_pairs)
    candidates["label"] = candidates.apply(lambda r: 1 if (r["s1_id"], r["candidate_id"]) in gt_keys else 0, axis=1)

    # Generate hard negatives
    hard_negs = generate_hard_negatives(s1, s2, s3, gt_keys)

    results = {}
    for ratio in negative_ratios:
        logger.info(f"\n--- Training with negative ratio {ratio}:1 ---")
        train_df = create_training_set_with_hard_negatives(candidates, hard_negs, negative_ratio=ratio)

        # Split
        from training_data import split_by_s1_entity
        train_split, val_split = split_by_s1_entity(train_df)

        # Build features
        from features import build_feature_matrix
        train_feat = build_feature_matrix(train_split, s1, s2, s3)
        val_feat = build_feature_matrix(val_split, s1, s2, s3)

        # Train
        from train_lgbm import train_lgbm, evaluate_model, optimize_threshold, get_feature_columns
        model = train_lgbm(train_feat, val_feat)
        feat_list = get_feature_columns(train_feat)
        best_thresh, _ = optimize_threshold(model, val_feat, feat_list)
        metrics = evaluate_model(model, val_feat, feat_list, threshold=best_thresh)

        results[ratio] = {
            "threshold": best_thresh,
            "metrics": metrics,
            "train_pos": len(train_split[train_split["label"] == 1]),
            "train_neg": len(train_split[train_split["label"] == 0]),
        }
        logger.info(f"Ratio {ratio}: F0.5={metrics['f05']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick test
    s1 = load_tsv(TRAIN_SOURCE1).head(1000)
    s2 = load_tsv(TRAIN_SOURCE2).head(5000)
    s3 = load_tsv(TRAIN_SOURCE3).head(5000)
    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    pos_pairs = load_positive_pairs()
    hard_negs = generate_hard_negatives(s1, s2, s3, pos_pairs)
    print(f"Generated {len(hard_negs)} hard negatives")