"""
I/O utilities for loading and writing data files.

All data loading goes through this module to ensure consistent
TSV handling and encoding.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_tsv(path: str | Path, dtype: dict = None) -> pd.DataFrame:
    """
    Load a TSV file with proper tab separator.

    Args:
        path: Path to the TSV file.
        dtype: Optional column dtype dict. Defaults to str for all columns.

    Returns:
        DataFrame with string columns (NaN replaced with empty string).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if dtype is None:
        dtype = str

    df = pd.read_csv(path, sep="\t", dtype=dtype)
    df = df.fillna("")
    logger.info(f"Loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_source(path: str | Path) -> pd.DataFrame:
    """Load a source file (source1/2/3)."""
    return load_tsv(path)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load the ground truth file."""
    return load_tsv(path)


def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    """
    Parse ground truth DataFrame into a dict.

    Returns:
        dict: source1_entity_id -> set of matched entity ids
    """
    s1_ids = gt_df["source1_entity_id"].astype(str).str.strip().values
    matched_col = gt_df["matched_entity_ids"].astype(str).str.strip().values

    gt = {}
    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        matched = matched_col[i]
        if matched and matched != "nan" and matched != "":
            gt[s1_id] = set(m.strip() for m in matched.split(",") if m.strip())
        else:
            gt[s1_id] = set()
    return gt


def write_matching_results(predictions: dict, output_path: str | Path):
    """
    Write matching_results.tsv in the required format.

    Args:
        predictions: dict s1_id -> set of matched ids
        output_path: Path to write the TSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in sorted(predictions.keys()):
        matched = predictions[s1_id]
        matched_str = ",".join(sorted(matched)) if matched else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote matching results: {output_path} ({len(df)} rows)")


def write_candidate_pairs(candidates: dict, output_path: str | Path):
    """
    Write candidate_pairs.tsv in the required format.

    Args:
        candidates: dict s1_id -> set of candidate ids
        output_path: Path to write the TSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in sorted(candidates.keys()):
        cands = candidates[s1_id]
        cands_str = ",".join(sorted(cands)) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote candidate pairs: {output_path} ({len(df)} rows)")
