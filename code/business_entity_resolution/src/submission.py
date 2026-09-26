"""
Submission validation and formatting.

Ensures output files match competition requirements exactly.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from config import (
    TEST_SOURCE1,
    S1_PREFIX, S2_PREFIX, S3_PREFIX,
)
from io_utils import load_tsv, write_matching_results, write_candidate_pairs

logger = logging.getLogger(__name__)


def validate_matching_results(
    matching_path: Path,
    candidate_path: Path,
    test_s1_path: Path = TEST_SOURCE1,
) -> Tuple[bool, List[str]]:
    """
    Validate matching_results.tsv against competition rules.

    Rules:
    1. Every S1 test entity appears exactly once
    2. All matched IDs must be in candidate_pairs.tsv
    3. No duplicate IDs
    4. Only S2/S3 IDs allowed as matches
    5. TSV format with comma-separated IDs

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Load files
    matching = pd.read_csv(matching_path, sep="\t")
    candidates = pd.read_csv(candidate_path, sep="\t")
    test_s1 = load_tsv(test_s1_path)

    test_s1_ids = set(test_s1["entity_id"].astype(str))

    # 1. Check all S1 entities present exactly once
    matching_s1 = set(matching["source1_entity_id"].astype(str))
    missing = test_s1_ids - matching_s1
    extra = matching_s1 - test_s1_ids
    duplicates = matching["source1_entity_id"].duplicated()

    if missing:
        errors.append(f"Missing {len(missing)} S1 entities in matching_results: {sorted(missing)[:10]}")
    if extra:
        errors.append(f"Extra {len(extra)} S1 entities not in test set: {sorted(extra)[:10]}")
    if duplicates.any():
        dup_ids = matching[duplicates]["source1_entity_id"].unique()
        errors.append(f"Duplicate S1 entities in matching_results: {dup_ids[:10]}")

    # 2. Check candidate_pairs has all S1 entities
    cand_s1 = set(candidates["source1_entity_id"].astype(str))
    missing_cand = test_s1_ids - cand_s1
    if missing_cand:
        errors.append(f"Missing {len(missing_cand)} S1 entities in candidate_pairs: {sorted(missing_cand)[:10]}")

    # 3. Check matches are subset of candidates
    cand_lookup = {}
    for _, row in candidates.iterrows():
        s1 = row["source1_entity_id"]
        cands = row["candidate_entity_ids"]
        if cands:
            cand_lookup[s1] = set(cands.split(","))
        else:
            cand_lookup[s1] = set()

    for _, row in matching.iterrows():
        s1 = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if not matched:
            continue
        matched_set = set(matched.split(","))
        allowed = cand_lookup.get(s1, set())
        invalid = matched_set - allowed
        if invalid:
            errors.append(f"S1 {s1}: matched IDs not in candidates: {sorted(invalid)[:10]}")

    # 4. Check only S2/S3 IDs in matches
    for _, row in matching.iterrows():
        matched = row["matched_entity_ids"]
        if not matched:
            continue
        for mid in matched.split(","):
            if not (mid.startswith(S2_PREFIX) or mid.startswith(S3_PREFIX)):
                errors.append(f"Invalid ID prefix in matches: {mid} (must be S2- or S3-)")

    # 5. Check candidate_pairs format
    for _, row in candidates.iterrows():
        cands = row["candidate_entity_ids"]
        if cands:
            for cid in cands.split(","):
                if not (cid.startswith(S2_PREFIX) or cid.startswith(S3_PREFIX)):
                    errors.append(f"Invalid ID prefix in candidates: {cid} (must be S2- or S3-)")

    # 6. Check no duplicate candidate IDs within same S1
    for _, row in candidates.iterrows():
        cands = row["candidate_entity_ids"]
        if cands:
            cand_list = cands.split(",")
            if len(cand_list) != len(set(cand_list)):
                errors.append(f"Duplicate candidate IDs for S1 {row['source1_entity_id']}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_submission(
    matching_path: str,
    candidate_path: str,
    test_dir: str = "dataset/test",
) -> bool:
    """
    Full submission validation.

    Args:
        matching_path: Path to matching_results.tsv
        candidate_path: Path to candidate_pairs.tsv
        test_dir: Directory containing test source files

    Returns:
        True if valid, raises ValueError with errors if invalid
    """
    matching_path = Path(matching_path)
    candidate_path = Path(candidate_path)
    test_s1_path = Path(test_dir) / "test_source1.tsv"

    if not matching_path.exists():
        raise ValueError(f"Matching results not found: {matching_path}")
    if not candidate_path.exists():
        raise ValueError(f"Candidate pairs not found: {candidate_path}")
    if not test_s1_path.exists():
        raise ValueError(f"Test source1 not found: {test_s1_path}")

    is_valid, errors = validate_matching_results(matching_path, candidate_path, test_s1_path)

    if not is_valid:
        error_msg = "Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("✓ Submission validation PASSED")
    return True


def format_submission(
    predictions: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Format and save submission files.

    Args:
        predictions: s1_id -> set of matched IDs
        candidates: s1_id -> set of candidate IDs
        output_dir: Output directory

    Returns:
        (matching_path, candidate_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    write_matching_results(predictions, matching_path)
    write_candidate_pairs(candidates, candidate_path)

    logger.info(f"Saved matching_results to {matching_path}")
    logger.info(f"Saved candidate_pairs to {candidate_path}")

    return matching_path, candidate_path


def load_submission(matching_path: Path, candidate_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Load submission files back into dicts."""
    matching = pd.read_csv(matching_path, sep="\t")
    candidates = pd.read_csv(candidate_path, sep="\t")

    pred_dict = {}
    for _, row in matching.iterrows():
        s1 = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        pred_dict[s1] = set(matched.split(",")) if matched else set()

    cand_dict = {}
    for _, row in candidates.iterrows():
        s1 = row["source1_entity_id"]
        cands = row["candidate_entity_ids"]
        cand_dict[s1] = set(cands.split(",")) if cands else set()

    return pred_dict, cand_dict


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Test validation with dummy files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create dummy matching results
        matching_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-1", "S2-2", ""],
        })
        matching_path = tmp / "matching_results.tsv"
        matching_df.to_csv(matching_path, sep="\t", index=False)

        # Create dummy candidate pairs
        cand_df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_ids": ["S2-1,S2-2,S3-1", "S2-2,S3-2", "S2-3"],
        })
        candidate_path = tmp / "candidate_pairs.tsv"
        cand_df.to_csv(candidate_path, sep="\t", index=False)

        # Test validation (will fail due to missing test S1)
        try:
            validate_submission(matching_path, candidate_path, "dataset/test")
        except ValueError as e:
            print(f"Expected validation error: {e}")

        print("Submission module ready!")