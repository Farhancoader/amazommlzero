"""
Inference pipeline for test data.

Loads trained models, generates candidates, computes features,
runs predictions, and applies decision logic.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    ARTIFACTS_DIR, CACHE_DIR,
    DEFAULT_THRESHOLD,
)
from io_utils import load_tsv, write_matching_results, write_candidate_pairs
from normalization import add_normalized_columns
from blocking import generate_candidates, save_candidate_pairs
from features import build_feature_matrix, get_feature_columns, select_features
from decision import predict_entities, predict_entities_multi_match
from evaluation import compute_candidate_recall, compute_candidate_stats

logger = logging.getLogger(__name__)


def load_test_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize test sources."""
    s1 = load_tsv(TEST_SOURCE1)
    s2 = load_tsv(TEST_SOURCE2)
    s3 = load_tsv(TEST_SOURCE3)

    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    return s1, s2, s3


def load_lgbm_model(model_dir: Path = None) -> Tuple[lgb.Booster, List[str]]:
    """Load LightGBM model and feature names."""
    model_dir = model_dir or ARTIFACTS_DIR
    model = lgb.Booster(model_file=str(model_dir / "lgbm_baseline.txt"))
    import json
    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    return model, feature_names


def run_lgbm_inference(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model: lgb.Booster,
    feature_names: List[str],
    batch_size: int = 10000,
) -> np.ndarray:
    """
    Run LightGBM inference on candidate pairs.

    Returns:
        Array of scores (probabilities)
    """
    logger.info(f"Building features for {len(candidates_df)} candidate pairs...")

    # Build features
    features = build_feature_matrix(candidates_df, s1_df, s2_df, s3_df)

    # Select features
    X, _, _ = select_features(features, feature_names)

    # Predict in batches
    n_samples = len(X)
    scores = np.zeros(n_samples)

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        scores[i:end] = model.predict(X[i:end], num_iteration=model.best_iteration)

    return scores


def run_siamese_inference(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model_dir: Path,
    batch_size: int = 256,
) -> np.ndarray:
    """Run Siamese network inference."""
    from train_neural import load_siamese_model, predict_siamese, PairDataset

    model, saved = load_siamese_model(model_dir)
    config = saved["config"]
    device = config.get("device", "cuda")

    tokenizer = AutoTokenizer.from_pretrained(saved["model_key"])
    prefix = EMBEDDING_MODELS[saved["model_key"]].get("prefix", "")

    # Merge candidate pairs with source data for text
    s23_df = pd.concat([
        s2_df[["entity_id", "business_name_norm", "business_address_norm"]].assign(source="S2"),
        s3_df[["entity_id", "business_name_norm", "business_address_norm"]].assign(source="S3"),
    ], ignore_index=True)

    merged = candidates_df.merge(s1_df.add_suffix("_s1"), left_on="s1_id", right_on="entity_id_s1", how="left")
    merged = merged.merge(s23_df.add_suffix("_cand"), left_on="candidate_id", right_on="entity_id_cand", how="left")

    dataset = PairDataset(merged, tokenizer, config.get("max_length", 256), prefix=prefix)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["s1_input_ids"],
                batch["s1_attention_mask"],
                batch["cand_input_ids"],
                batch["cand_attention_mask"],
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


def run_cross_encoder_inference(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    model_dir: Path,
    top_k: int = 10,
    score_col: str = "lgbm_score",
    batch_size: int = 64,
) -> np.ndarray:
    """Run cross-encoder reranking on top-K candidates."""
    from cross_encoder import load_cross_encoder, predict_cross_encoder, CrossEncoderDataset

    model, saved = load_cross_encoder(model_dir)
    config = saved["config"]
    device = config.get("device", "cuda")

    tokenizer = AutoTokenizer.from_pretrained(saved["model_key"])
    prefix = EMBEDDING_MODELS[saved["model_key"]].get("prefix", "")

    # Select top-K per S1
    top_k_df = candidates_df.groupby("s1_id").apply(
        lambda x: x.nlargest(top_k, score_col)
    ).reset_index(drop=True)

    # Merge with source data
    s23_df = pd.concat([
        s2_df[["entity_id", "business_name_norm", "business_address_norm"]].assign(source="S2"),
        s3_df[["entity_id", "business_name_norm", "business_address_norm"]].assign(source="S3"),
    ], ignore_index=True)

    merged = top_k_df.merge(s1_df.add_suffix("_s1"), left_on="s1_id", right_on="entity_id_s1", how="left")
    merged = merged.merge(s23_df.add_suffix("_cand"), left_on="candidate_id", right_on="entity_id_cand", how="left")

    # Predict
    ce_scores = predict_cross_encoder(model, merged, tokenizer, device,
                                      max_length=config.get("max_length", 512),
                                      batch_size=batch_size, prefix=prefix)

    # Map back to all candidates
    all_scores = np.zeros(len(candidates_df))
    score_map = dict(zip(zip(top_k_df["s1_id"], top_k_df["candidate_id"]), ce_scores))

    for i, row in candidates_df.iterrows():
        all_scores[i] = score_map.get((row["s1_id"], row["candidate_id"]), 0.0)

    return all_scores


def fuse_scores(
    scores_dict: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Fuse multiple model scores.

    Args:
        scores_dict: Dict of model_name -> scores array
        weights: Dict of model_name -> weight

    Returns:
        Fused scores
    """
    fused = np.zeros_like(list(scores_dict.values())[0])
    total_weight = sum(weights.values())

    for name, scores in scores_dict.items():
        weight = weights.get(name, 0.0)
        if weight > 0:
            fused += (weight / total_weight) * scores

    return fused


def run_full_inference(
    config: Dict = None,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Run complete inference pipeline on test data.

    Returns:
        (candidates_df, predictions_dict)
    """
    config = config or {}

    # 1. Load test data
    logger.info("Loading test sources...")
    s1, s2, s3 = load_test_sources()

    # 2. Generate candidates
    logger.info("Generating candidates...")
    blocking_config = config.get("blocking", {})
    candidates = generate_candidates(s1, s2, s3, blocking_config)

    # Save candidate pairs
    output_dir = Path(config.get("output_dir", "output"))
    save_candidate_pairs(candidates, output_dir / "candidate_pairs.tsv")

    # 3. Run LightGBM (baseline)
    logger.info("Running LightGBM inference...")
    lgbm_model, lgbm_features = load_lgbm_model()
    lgbm_scores = run_lgbm_inference(candidates, s1, s2, s3, lgbm_model, lgbm_features)
    candidates = candidates.copy()
    candidates["lgbm_score"] = lgbm_scores

    # 4. Optional: Siamese
    siamese_scores = None
    if config.get("use_siamese", False):
        logger.info("Running Siamese inference...")
        siamese_scores = run_siamese_inference(candidates, s1, s2, s3, Path(config["siamese_model_dir"]))
        candidates["siamese_score"] = siamese_scores

    # 5. Optional: Cross-encoder reranking
    cross_scores = None
    if config.get("use_cross_encoder", False):
        logger.info("Running Cross-Encoder reranking...")
        cross_scores = run_cross_encoder_inference(
            candidates, s1, s2, s3,
            Path(config["cross_encoder_model_dir"]),
            top_k=config.get("cross_encoder_top_k", 10),
        )
        candidates["cross_encoder_score"] = cross_scores

    # 6. Score fusion
    score_dict = {"lgbm": lgbm_scores}
    if siamese_scores is not None:
        score_dict["siamese"] = siamese_scores
    if cross_scores is not None:
        score_dict["cross"] = cross_scores

    weights = config.get("fusion_weights", {"lgbm": 1.0})
    final_scores = fuse_scores(score_dict, weights)
    candidates["final_score"] = final_scores

    # 7. Apply decision logic
    logger.info("Applying decision logic...")
    threshold = config.get("threshold", DEFAULT_THRESHOLD)
    use_margin = config.get("use_margin", False)
    min_margin = config.get("min_margin", 0.1)
    multi_match = config.get("multi_match", False)

    if multi_match:
        predictions = predict_entities_multi_match(
            final_scores,
            candidates["s1_id"].values,
            candidates["candidate_id"].values,
            candidates["candidate_source"].values,
            threshold=threshold,
        )
    else:
        predictions = predict_entities(
            final_scores,
            candidates["s1_id"].values,
            candidates["candidate_id"].values,
            candidates["candidate_source"].values,
            threshold=threshold,
            use_margin=use_margin,
            min_margin=min_margin,
        )

    # 8. Save matching results
    write_matching_results(predictions, output_dir / "matching_results.tsv")

    # 9. Log stats
    cand_stats = compute_candidate_stats(candidates)
    logger.info(f"Candidate stats: {cand_stats}")
    logger.info(f"Predictions: {len(predictions)} S1 entities")
    matched_count = sum(1 for v in predictions.values() if v)
    logger.info(f"S1 entities with matches: {matched_count}")

    return candidates, predictions


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick test with small sample
    from config import TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
    from io_utils import load_tsv
    from normalization import add_normalized_columns
    from blocking import generate_candidates

    s1 = load_tsv(TEST_SOURCE1).head(100)
    s2 = load_tsv(TEST_SOURCE2).head(500)
    s3 = load_tsv(TEST_SOURCE3).head(500)

    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    candidates = generate_candidates(s1, s2, s3)
    print(f"Generated {len(candidates)} candidate pairs")
    print(candidates.head())