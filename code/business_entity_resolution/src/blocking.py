"""
Multi-stage candidate generation (blocking) for entity resolution.

Implements independent blocking methods:
1. Exact normalized name match
2. Name character TF-IDF (n-gram)
3. Address character TF-IDF (n-gram)
4. Token similarity (Jaccard)
5. Country-compatible candidates
6. Embedding nearest neighbors (placeholder for later)

Final candidate set = UNION of all blockers.
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
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    S1_PREFIX, S2_PREFIX, S3_PREFIX,
    DEFAULT_TFIDF_NAME_TOP_K, DEFAULT_TFIDF_ADDRESS_TOP_K,
    DEFAULT_TOKEN_MIN_SHARED, DEFAULT_EMBEDDING_TOP_K,
)
from io_utils import load_tsv
from normalization import add_normalized_columns, get_tokens, jaccard_tokens

logger = logging.getLogger(__name__)


# ── Data loading ────────────────────────────────────────────────────

def load_all_sources(train: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three sources with normalization applied."""
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


# ── Block 1: Exact Normalized Name ──────────────────────────────────

def block_exact_name(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    name_col: str = "business_name_norm",
) -> pd.DataFrame:
    """Block 1: Exact match on normalized business name."""
    # Build lookup from S2/S3 normalized name -> entity_ids
    name_to_s2 = defaultdict(list)
    for _, row in s2.iterrows():
        name = row[name_col]
        if name:
            name_to_s2[name].append(row["entity_id"])

    name_to_s3 = defaultdict(list)
    for _, row in s3.iterrows():
        name = row[name_col]
        if name:
            name_to_s3[name].append(row["entity_id"])

    rows = []
    for _, s1_row in s1.iterrows():
        s1_id = s1_row["entity_id"]
        name = s1_row[name_col]
        if not name:
            continue

        for cand_id in name_to_s2.get(name, []):
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": "S2",
                "blocking_method": "name_exact",
            })
        for cand_id in name_to_s3.get(name, []):
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": "S3",
                "blocking_method": "name_exact",
            })

    df = pd.DataFrame(rows)
    logger.info(f"Block 1 (exact name): {len(df)} candidate pairs")
    return df


# ── Block 2: Name Character TF-IDF ──────────────────────────────────

def block_tfidf_name(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k: int = DEFAULT_TFIDF_NAME_TOP_K,
    ngram_range: Tuple[int, int] = (2, 5),
    name_col: str = "business_name_norm",
) -> pd.DataFrame:
    """Block 2: Character n-gram TF-IDF on normalized business name."""
    # Combine S2 and S3 for single index
    s23 = pd.concat([
        s2[["entity_id", name_col]].assign(source="S2"),
        s3[["entity_id", name_col]].assign(source="S3"),
    ], ignore_index=True)
    s23 = s23[s23[name_col].str.len() > 0].reset_index(drop=True)

    # Fit TF-IDF on S2+S3 names
    ngram_range = tuple(ngram_range) if isinstance(ngram_range, list) else ngram_range
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(s23[name_col])

    # Transform S1 names
    s1_valid = s1[s1[name_col].str.len() > 0].reset_index(drop=True)
    s1_tfidf = vectorizer.transform(s1_valid[name_col])

    # Nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(top_k, len(s23)), metric="cosine")
    nn.fit(tfidf_matrix)

    distances, indices = nn.kneighbors(s1_tfidf)

    rows = []
    for i, s1_idx in enumerate(s1_valid.index):
        s1_id = s1_valid.loc[s1_idx, "entity_id"]
        for rank, (dist, idx) in enumerate(zip(distances[i], indices[i])):
            cand_row = s23.iloc[idx]
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_row["entity_id"],
                "candidate_source": cand_row["source"],
                "blocking_method": f"tfidf_name_{ngram_range[0]}_{ngram_range[1]}",
                "tfidf_score": 1.0 - dist,  # cosine similarity
            })

    df = pd.DataFrame(rows)
    logger.info(f"Block 2 (TF-IDF name {ngram_range}): {len(df)} candidate pairs")
    return df


# ── Block 3: Address Character TF-IDF ───────────────────────────────

def block_tfidf_address(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k: int = DEFAULT_TFIDF_ADDRESS_TOP_K,
    ngram_range: Tuple[int, int] = (2, 6),
    addr_col: str = "business_address_norm",
) -> pd.DataFrame:
    """Block 3: Character n-gram TF-IDF on normalized address."""
    s23 = pd.concat([
        s2[["entity_id", addr_col]].assign(source="S2"),
        s3[["entity_id", addr_col]].assign(source="S3"),
    ], ignore_index=True)
    s23 = s23[s23[addr_col].str.len() > 0].reset_index(drop=True)

    ngram_range = tuple(ngram_range) if isinstance(ngram_range, list) else ngram_range
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(s23[addr_col])

    s1_valid = s1[s1[addr_col].str.len() > 0].reset_index(drop=True)
    s1_tfidf = vectorizer.transform(s1_valid[addr_col])

    nn = NearestNeighbors(n_neighbors=min(top_k, len(s23)), metric="cosine")
    nn.fit(tfidf_matrix)

    distances, indices = nn.kneighbors(s1_tfidf)

    rows = []
    for i, s1_idx in enumerate(s1_valid.index):
        s1_id = s1_valid.loc[s1_idx, "entity_id"]
        for dist, idx in zip(distances[i], indices[i]):
            cand_row = s23.iloc[idx]
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_row["entity_id"],
                "candidate_source": cand_row["source"],
                "blocking_method": f"tfidf_addr_{ngram_range[0]}_{ngram_range[1]}",
                "tfidf_score": 1.0 - dist,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Block 3 (TF-IDF address {ngram_range}): {len(df)} candidate pairs")
    return df


# ── Block 4: Token Jaccard Similarity ───────────────────────────────

def block_token_jaccard(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    min_shared: int = DEFAULT_TOKEN_MIN_SHARED,
    name_col: str = "business_name_tokens",
    addr_col: str = "business_address_tokens",
) -> pd.DataFrame:
    """Block 4: Token-based Jaccard similarity on name and address tokens."""
    # Build inverted index: token -> set of entity_ids for S2/S3
    token_to_s2 = defaultdict(set)
    token_to_s3 = defaultdict(set)

    for _, row in s2.iterrows():
        tokens = set(get_tokens(row[name_col])) if row[name_col] else set()
        addr_tokens = set(get_tokens(row[addr_col])) if row[addr_col] else set()
        all_tokens = tokens | addr_tokens
        for tok in all_tokens:
            token_to_s2[tok].add(row["entity_id"])

    for _, row in s3.iterrows():
        tokens = set(get_tokens(row[name_col])) if row[name_col] else set()
        addr_tokens = set(get_tokens(row[addr_col])) if row[addr_col] else set()
        all_tokens = tokens | addr_tokens
        for tok in all_tokens:
            token_to_s3[tok].add(row["entity_id"])

    rows = []
    for _, s1_row in s1.iterrows():
        s1_id = s1_row["entity_id"]
        s1_name_tokens = set(get_tokens(s1_row[name_col])) if s1_row[name_col] else set()
        s1_addr_tokens = set(get_tokens(s1_row[addr_col])) if s1_row[addr_col] else set()
        s1_all_tokens = s1_name_tokens | s1_addr_tokens

        if not s1_all_tokens:
            continue

        # Find candidates sharing tokens
        s2_candidates = defaultdict(int)
        s3_candidates = defaultdict(int)

        for tok in s1_all_tokens:
            for cid in token_to_s2.get(tok, set()):
                s2_candidates[cid] += 1
            for cid in token_to_s3.get(tok, set()):
                s3_candidates[cid] += 1

        # Keep only those with minimum shared tokens
        for cid, count in s2_candidates.items():
            if count >= min_shared:
                rows.append({
                    "s1_id": s1_id,
                    "candidate_id": cid,
                    "candidate_source": "S2",
                    "blocking_method": "token_jaccard",
                    "shared_tokens": count,
                })
        for cid, count in s3_candidates.items():
            if count >= min_shared:
                rows.append({
                    "s1_id": s1_id,
                    "candidate_id": cid,
                    "candidate_source": "S3",
                    "blocking_method": "token_jaccard",
                    "shared_tokens": count,
                })

    df = pd.DataFrame(rows)
    logger.info(f"Block 4 (token Jaccard, min_shared={min_shared}): {len(df)} candidate pairs")
    return df


# ── Block 5: Country-Compatible ─────────────────────────────────────

def block_by_country(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    country_col: str = "country_norm",
) -> pd.DataFrame:
    """Block 5: Country as soft blocking signal (same country)."""
    # Build country -> entity_ids mapping for S2/S3
    country_to_s2 = defaultdict(list)
    for _, row in s2.iterrows():
        country = row[country_col] if country_col in row else row.get("country", "")
        if country:
            country_to_s2[country].append(row["entity_id"])

    country_to_s3 = defaultdict(list)
    for _, row in s3.iterrows():
        country = row[country_col] if country_col in row else row.get("country", "")
        if country:
            country_to_s3[country].append(row["entity_id"])

    rows = []
    for _, s1_row in s1.iterrows():
        s1_id = s1_row["entity_id"]
        country = s1_row[country_col] if country_col in s1_row else s1_row.get("country", "")
        if not country:
            continue

        for cand_id in country_to_s2.get(country, []):
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": "S2",
                "blocking_method": "country",
            })
        for cand_id in country_to_s3.get(country, []):
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": "S3",
                "blocking_method": "country",
            })

    df = pd.DataFrame(rows)
    logger.info(f"Block 5 (country): {len(df)} candidate pairs")
    return df


# ── Block 6: Embedding Nearest Neighbors (placeholder) ──────────────

def block_embedding_knn(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    top_k: int = DEFAULT_EMBEDDING_TOP_K,
    embedding_col: str = "embedding",  # Will be added later
) -> pd.DataFrame:
    """Block 6: Embedding-based nearest neighbor retrieval."""
    # Placeholder - will be implemented after embeddings.py
    logger.warning("Block 6 (embedding KNN) not yet implemented - returning empty")
    return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source", "blocking_method"])


# ── Candidate Union & Deduplication ─────────────────────────────────

def union_candidates(block_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Union all blocking results, deduplicate, and aggregate blocking methods.

    Returns:
        DataFrame with columns: s1_id, candidate_id, candidate_source, blocking_methods
    """
    if not block_dfs:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source", "blocking_methods"])

    # Concatenate all
    all_candidates = pd.concat(block_dfs, ignore_index=True)

    # Group by (s1_id, candidate_id, candidate_source) and aggregate methods
    agg = all_candidates.groupby(["s1_id", "candidate_id", "candidate_source"]).agg(
        blocking_methods=("blocking_method", lambda x: "|".join(sorted(set(x)))),
    ).reset_index()

    # Add score columns if present
    score_cols = [c for c in all_candidates.columns if c not in ["s1_id", "candidate_id", "candidate_source", "blocking_method"]]
    if score_cols:
        # Take max score per method
        for col in score_cols:
            score_agg = all_candidates.groupby(["s1_id", "candidate_id", "candidate_source"])[col].max().reset_index()
            score_agg.columns = ["s1_id", "candidate_id", "candidate_source", col]
            agg = agg.merge(score_agg, on=["s1_id", "candidate_id", "candidate_source"], how="left")

    logger.info(f"Union of {len(block_dfs)} blocks: {len(agg)} unique candidate pairs")
    return agg


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    config: Dict = None,
) -> pd.DataFrame:
    """
    Generate candidates using all blocking methods.

    Args:
        s1, s2, s3: Normalized source DataFrames
        config: Dict with blocking parameters

    Returns:
        Candidate pairs DataFrame
    """
    config = config or {}

    # Block 1: Exact name
    b1 = block_exact_name(s1, s2, s3)

    # Block 2: TF-IDF name
    b2 = block_tfidf_name(
        s1, s2, s3,
        top_k=config.get("tfidf_name_top_k", DEFAULT_TFIDF_NAME_TOP_K),
        ngram_range=config.get("tfidf_name_ngram_range", (2, 5)),
    )

    # Block 3: TF-IDF address
    b3 = block_tfidf_address(
        s1, s2, s3,
        top_k=config.get("tfidf_addr_top_k", DEFAULT_TFIDF_ADDRESS_TOP_K),
        ngram_range=config.get("tfidf_addr_ngram_range", (2, 6)),
    )

    # Block 4: Token Jaccard
    b4 = block_token_jaccard(
        s1, s2, s3,
        min_shared=config.get("token_min_shared", DEFAULT_TOKEN_MIN_SHARED),
    )

    # Block 5: Country
    b5 = block_by_country(s1, s2, s3)

    # Block 6: Embeddings (optional)
    b6 = pd.DataFrame()
    if config.get("use_embeddings", False):
        b6 = block_embedding_knn(s1, s2, s3, top_k=config.get("embedding_top_k", DEFAULT_EMBEDDING_TOP_K))

    # Union all
    candidates = union_candidates([b1, b2, b3, b4, b5, b6])

    # Ensure all S1 entities appear (even with 0 candidates)
    all_s1 = set(s1["entity_id"].astype(str))
    candidates_s1 = set(candidates["s1_id"].astype(str))
    missing_s1 = all_s1 - candidates_s1

    if missing_s1:
        missing_rows = pd.DataFrame([{
            "s1_id": s1_id,
            "candidate_id": "",
            "candidate_source": "",
            "blocking_methods": "",
        } for s1_id in missing_s1])
        candidates = pd.concat([candidates, missing_rows], ignore_index=True)
        logger.info(f"Added {len(missing_s1)} S1 entities with 0 candidates")

    return candidates


def save_candidate_pairs(candidates: pd.DataFrame, output_path: Path, is_test: bool = False):
    """Save candidate pairs in required TSV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by s1_id and join candidate_ids
    grouped = candidates[candidates["candidate_id"] != ""].groupby("s1_id")["candidate_id"].apply(
        lambda x: ",".join(sorted(set(x)))
    ).reset_index()

    # Ensure all S1 entities present
    all_s1 = set(candidates["s1_id"].unique())
    grouped_s1 = set(grouped["s1_id"])
    missing = all_s1 - grouped_s1

    if missing:
        missing_df = pd.DataFrame({"s1_id": list(missing), "candidate_id": ""})
        grouped = pd.concat([grouped, missing_df], ignore_index=True)

    # Sort and write
    grouped = grouped.sort_values("s1_id")
    grouped.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved candidate pairs to {output_path} ({len(grouped)} S1 entities)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick test with small sample
    s1, s2, s3 = load_all_sources(train=True)
    s1_sample = s1.head(1000)
    s2_sample = s2.head(5000)
    s3_sample = s3.head(5000)

    candidates = generate_candidates(s1_sample, s2_sample, s3_sample)
    print(f"Generated {len(candidates)} candidate pairs")
    print(candidates.head())