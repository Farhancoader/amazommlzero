"""
Central configuration for the entity resolution pipeline.

All paths, parameters, and constants are defined here.
No competition-specific values should be hard-coded in other modules.
"""

import os
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────

# Project root: code/business_entity_resolution/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset root: ../../dataset relative to project root
DATASET_ROOT = PROJECT_ROOT.parent.parent / "dataset"

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT.parent.parent / "output"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / "cache"
VALIDATION_DIR = PROJECT_ROOT / "validation"
EXPERIMENTS_DIR = VALIDATION_DIR / "experiments"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ── Reproducibility ──────────────────────────────────────────────────

RANDOM_SEED = 42

# ── Blocking defaults ────────────────────────────────────────────────

DEFAULT_TFIDF_NAME_TOP_K = 50
DEFAULT_TFIDF_ADDRESS_TOP_K = 50
DEFAULT_TFIDF_CHAR_TOP_K = 30
DEFAULT_TOKEN_MIN_SHARED = 2
DEFAULT_EMBEDDING_TOP_K = 50

# ── Model defaults ───────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.80
DEFAULT_BATCH_SIZE = 64
DEFAULT_VAL_FRACTION = 0.2

# ── Source prefixes ──────────────────────────────────────────────────

S1_PREFIX = "S1-"
S2_PREFIX = "S2-"
S3_PREFIX = "S3-"
VALID_PREFIXES = {S1_PREFIX, S2_PREFIX, S3_PREFIX}


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [OUTPUT_DIR, ARTIFACTS_DIR, CACHE_DIR, VALIDATION_DIR, EXPERIMENTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
