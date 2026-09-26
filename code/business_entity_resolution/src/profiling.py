"""
Dataset profiling for the entity resolution challenge.

Generates a comprehensive data profile report covering:
- Row counts, column names, dtypes, missing values
- Country distribution
- Business name / address length distributions
- Duplicate analysis
- Ground truth match statistics
- ID format validation

Saves output to validation/data_profile.json.
Does NOT modify source files.
"""

import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    VALIDATION_DIR, S1_PREFIX, S2_PREFIX, S3_PREFIX, VALID_PREFIXES,
)
from io_utils import load_tsv, parse_ground_truth

logger = logging.getLogger(__name__)


def _basic_normalize(text: str) -> str:
    """Minimal normalize for duplicate detection: lowercase, strip, collapse ws."""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text


def profile_source(df: pd.DataFrame, name: str) -> dict:
    """Profile a single source DataFrame."""
    profile = {
        "name": name,
        "row_count": len(df),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    # Missing values
    missing = {}
    for col in df.columns:
        n_missing = df[col].isna().sum() + (df[col] == "").sum()
        missing[col] = int(n_missing)
    profile["missing_values"] = missing

    # Unique counts
    unique = {}
    for col in df.columns:
        unique[col] = int(df[col].nunique())
    profile["unique_counts"] = unique

    # Country distribution
    if "country" in df.columns:
        country_counts = df["country"].value_counts().to_dict()
        profile["country_distribution"] = {str(k): int(v) for k, v in country_counts.items()}

    # Business name stats
    if "business_name" in df.columns:
        name_lens = df["business_name"].apply(lambda x: len(str(x)) if x and str(x) else 0)
        profile["business_name_length"] = {
            "mean": float(name_lens.mean()),
            "median": float(name_lens.median()),
            "min": int(name_lens.min()),
            "max": int(name_lens.max()),
            "p5": float(name_lens.quantile(0.05)),
            "p25": float(name_lens.quantile(0.25)),
            "p75": float(name_lens.quantile(0.75)),
            "p95": float(name_lens.quantile(0.95)),
        }

        # Duplicate normalized names
        norm_names = df["business_name"].apply(_basic_normalize)
        dup_name_count = int(norm_names.duplicated().sum())
        profile["duplicate_normalized_names"] = dup_name_count

    # Address stats
    if "business_address" in df.columns:
        addr_lens = df["business_address"].apply(lambda x: len(str(x)) if x and str(x) else 0)
        profile["business_address_length"] = {
            "mean": float(addr_lens.mean()),
            "median": float(addr_lens.median()),
            "min": int(addr_lens.min()),
            "max": int(addr_lens.max()),
            "p5": float(addr_lens.quantile(0.05)),
            "p25": float(addr_lens.quantile(0.25)),
            "p75": float(addr_lens.quantile(0.75)),
            "p95": float(addr_lens.quantile(0.95)),
        }

        norm_addrs = df["business_address"].apply(_basic_normalize)
        dup_addr_count = int(norm_addrs.duplicated().sum())
        profile["duplicate_normalized_addresses"] = dup_addr_count

    # ID format validation (vectorized)
    if "entity_id" in df.columns:
        ids = df["entity_id"].astype(str)
        has_valid_prefix = (
            ids.str.startswith(S1_PREFIX) |
            ids.str.startswith(S2_PREFIX) |
            ids.str.startswith(S3_PREFIX)
        )
        unexpected_prefix_mask = ~has_valid_prefix
        unexpected_prefix_count = int(unexpected_prefix_mask.sum())

        # Check numeric part after prefix for those with valid prefix
        id_pattern = re.compile(r"^S[123]-\d+$")
        well_formed = ids.str.match(r"^S[123]-\d+$")
        malformed_mask = has_valid_prefix & ~well_formed
        malformed_count = int(malformed_mask.sum())

        profile["malformed_ids"] = malformed_count
        profile["unexpected_source_prefixes"] = unexpected_prefix_count
        if malformed_count > 0:
            profile["malformed_id_examples"] = ids[malformed_mask].head(10).tolist()
        if unexpected_prefix_count > 0:
            profile["unexpected_prefix_examples"] = ids[unexpected_prefix_mask].head(10).tolist()

    return profile


def profile_ground_truth(gt_df: pd.DataFrame, train_s1_ids: set,
                         train_s2_ids: set, train_s3_ids: set) -> dict:
    """Profile the ground truth file."""
    gt = parse_ground_truth(gt_df)

    profile = {
        "row_count": len(gt_df),
        "columns": gt_df.columns.tolist(),
        "total_s1_entities": len(gt),
    }

    # Match statistics
    zero_matches = 0
    one_match = 0
    multi_matches = 0
    s2_match_count = 0
    s3_match_count = 0
    total_matches = 0
    match_count_dist = Counter()

    missing_s1_ids = []
    missing_match_ids = []

    for s1_id, matches in gt.items():
        n = len(matches)
        match_count_dist[n] += 1

        if n == 0:
            zero_matches += 1
        elif n == 1:
            one_match += 1
        else:
            multi_matches += 1

        total_matches += n

        for m in matches:
            if m.startswith(S2_PREFIX):
                s2_match_count += 1
            elif m.startswith(S3_PREFIX):
                s3_match_count += 1

        # Verify S1 ID exists in source1
        if s1_id not in train_s1_ids:
            missing_s1_ids.append(s1_id)

        # Verify match IDs exist in source2/source3
        for m in matches:
            if m.startswith(S2_PREFIX) and m not in train_s2_ids:
                missing_match_ids.append(m)
            elif m.startswith(S3_PREFIX) and m not in train_s3_ids:
                missing_match_ids.append(m)

    profile["zero_matches"] = zero_matches
    profile["one_match"] = one_match
    profile["multi_matches"] = multi_matches
    profile["total_match_links"] = total_matches
    profile["s2_match_count"] = s2_match_count
    profile["s3_match_count"] = s3_match_count

    # Match count distribution (top entries)
    profile["match_count_distribution"] = {
        str(k): int(v) for k, v in sorted(match_count_dist.items())
    }

    if total_matches > 0:
        profile["avg_matches_per_entity"] = round(total_matches / len(gt), 3)
        # Only among entities with matches
        entities_with_matches = len(gt) - zero_matches
        if entities_with_matches > 0:
            profile["avg_matches_per_matched_entity"] = round(
                total_matches / entities_with_matches, 3
            )

    profile["missing_s1_ids_in_source1"] = len(missing_s1_ids)
    profile["missing_match_ids_in_sources"] = len(missing_match_ids)
    if missing_s1_ids:
        profile["missing_s1_examples"] = missing_s1_ids[:10]
    if missing_match_ids:
        profile["missing_match_examples"] = missing_match_ids[:10]

    return profile


def run_profiling():
    """Run the complete data profiling pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("DATASET PROFILING")
    logger.info("=" * 60)

    report = {}

    # ── Load all sources ─────────────────────────────────────────
    logger.info("Loading training data...")
    train_s1 = load_tsv(TRAIN_SOURCE1)
    train_s2 = load_tsv(TRAIN_SOURCE2)
    train_s3 = load_tsv(TRAIN_SOURCE3)
    train_gt = load_tsv(TRAIN_GROUND_TRUTH)

    logger.info("Loading test data...")
    test_s1 = load_tsv(TEST_SOURCE1)
    test_s2 = load_tsv(TEST_SOURCE2)
    test_s3 = load_tsv(TEST_SOURCE3)

    # ── Profile each source ──────────────────────────────────────
    logger.info("Profiling train_source1...")
    report["train_source1"] = profile_source(train_s1, "train_source1")

    logger.info("Profiling train_source2...")
    report["train_source2"] = profile_source(train_s2, "train_source2")

    logger.info("Profiling train_source3...")
    report["train_source3"] = profile_source(train_s3, "train_source3")

    logger.info("Profiling test_source1...")
    report["test_source1"] = profile_source(test_s1, "test_source1")

    logger.info("Profiling test_source2...")
    report["test_source2"] = profile_source(test_s2, "test_source2")

    logger.info("Profiling test_source3...")
    report["test_source3"] = profile_source(test_s3, "test_source3")

    # ── Profile ground truth ─────────────────────────────────────
    logger.info("Profiling ground truth...")
    train_s1_ids = set(train_s1["entity_id"].astype(str))
    train_s2_ids = set(train_s2["entity_id"].astype(str))
    train_s3_ids = set(train_s3["entity_id"].astype(str))
    report["train_ground_truth"] = profile_ground_truth(
        train_gt, train_s1_ids, train_s2_ids, train_s3_ids
    )

    # ── Summary ──────────────────────────────────────────────────
    report["summary"] = {
        "train_s1_rows": len(train_s1),
        "train_s2_rows": len(train_s2),
        "train_s3_rows": len(train_s3),
        "train_gt_rows": len(train_gt),
        "test_s1_rows": len(test_s1),
        "test_s2_rows": len(test_s2),
        "test_s3_rows": len(test_s3),
        "train_countries": sorted(train_s1["country"].unique().tolist()),
        "test_countries": sorted(test_s1["country"].unique().tolist()),
    }

    # ── Save ─────────────────────────────────────────────────────
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VALIDATION_DIR / "data_profile.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Profile saved to {output_path}")

    # ── Print summary ────────────────────────────────────────────
    _print_summary(report)

    return report


def _print_summary(report: dict):
    """Print a human-readable summary."""
    s = report["summary"]
    gt = report["train_ground_truth"]

    print("\n" + "=" * 60)
    print("DATASET PROFILE SUMMARY")
    print("=" * 60)

    print(f"\n{'Dataset':<25} {'Rows':>10}")
    print("-" * 36)
    print(f"{'Train Source 1':<25} {s['train_s1_rows']:>10,}")
    print(f"{'Train Source 2':<25} {s['train_s2_rows']:>10,}")
    print(f"{'Train Source 3':<25} {s['train_s3_rows']:>10,}")
    print(f"{'Train Ground Truth':<25} {s['train_gt_rows']:>10,}")
    print(f"{'Test Source 1':<25} {s['test_s1_rows']:>10,}")
    print(f"{'Test Source 2':<25} {s['test_s2_rows']:>10,}")
    print(f"{'Test Source 3':<25} {s['test_s3_rows']:>10,}")

    print(f"\nTrain countries: {s['train_countries']}")
    print(f"Test countries:  {s['test_countries']}")

    print(f"\n--- Ground Truth ---")
    print(f"S1 entities with 0 matches: {gt['zero_matches']:,}")
    print(f"S1 entities with 1 match:   {gt['one_match']:,}")
    print(f"S1 entities with 2+ matches:{gt['multi_matches']:,}")
    print(f"Total match links:          {gt['total_match_links']:,}")
    print(f"S2 match links:             {gt['s2_match_count']:,}")
    print(f"S3 match links:             {gt['s3_match_count']:,}")

    if "avg_matches_per_entity" in gt:
        print(f"Avg matches per S1 entity:  {gt['avg_matches_per_entity']}")
    if "avg_matches_per_matched_entity" in gt:
        print(f"Avg matches (matched only): {gt['avg_matches_per_matched_entity']}")

    print(f"\nMissing S1 IDs in source1:     {gt['missing_s1_ids_in_source1']}")
    print(f"Missing match IDs in S2/S3:    {gt['missing_match_ids_in_sources']}")

    # Per-source stats
    for source_key in ["train_source1", "train_source2", "train_source3",
                        "test_source1", "test_source2", "test_source3"]:
        sp = report[source_key]
        print(f"\n--- {sp['name']} ---")
        print(f"  Missing values: {sp['missing_values']}")
        if "duplicate_normalized_names" in sp:
            print(f"  Duplicate normalized names:     {sp['duplicate_normalized_names']:,}")
        if "duplicate_normalized_addresses" in sp:
            print(f"  Duplicate normalized addresses: {sp['duplicate_normalized_addresses']:,}")
        if "malformed_ids" in sp:
            print(f"  Malformed IDs:      {sp['malformed_ids']}")
        if "unexpected_source_prefixes" in sp:
            print(f"  Unexpected prefixes:{sp['unexpected_source_prefixes']}")
        if "country_distribution" in sp:
            print(f"  Country dist: {sp['country_distribution']}")
        if "business_name_length" in sp:
            nl = sp["business_name_length"]
            print(f"  Name length: mean={nl['mean']:.1f}, median={nl['median']:.0f}, "
                  f"min={nl['min']}, max={nl['max']}, p5={nl['p5']:.0f}, p95={nl['p95']:.0f}")
        if "business_address_length" in sp:
            al = sp["business_address_length"]
            print(f"  Addr length: mean={al['mean']:.1f}, median={al['median']:.0f}, "
                  f"min={al['min']}, max={al['max']}, p5={al['p5']:.0f}, p95={al['p95']:.0f}")

    print("\nMatch count distribution:")
    for k, v in sorted(gt["match_count_distribution"].items(), key=lambda x: int(x[0])):
        print(f"  {k} matches: {v:,} entities")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_profiling()
