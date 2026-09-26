"""
Business Entity Resolution Pipeline for Amazon ML Challenge 2026.

Package structure:
- config: Central configuration
- io_utils: Data loading/writing
- profiling: Dataset profiling
- normalization: Text normalization
- training_data: Ground truth parsing
- validation_split: Entity-level validation splits
- blocking: Multi-stage candidate generation
- similarity: String similarity features
- features: Feature engineering pipeline
- train_lgbm: LightGBM baseline
- hard_negative_mining: Hard negative generation
- embeddings: Pretrained embeddings
- train_neural: Siamese/bi-encoder training
- cross_encoder: Cross-encoder reranker
- calibration: Score calibration
- decision: Threshold optimization & prediction
- evaluation: Metrics & error analysis
- inference: Test inference pipeline
- submission: Submission validation
- pipeline: Main orchestration
"""

__version__ = "1.0.0"
__author__ = "Amazon ML Challenge 2026"

# Main modules
from . import (
    config,
    io_utils,
    profiling,
    normalization,
    training_data,
    validation_split,
    blocking,
    similarity,
    features,
    train_lgbm,
    hard_negative_mining,
    embeddings,
    train_neural,
    cross_encoder,
    calibration,
    decision,
    evaluation,
    inference,
    submission,
    pipeline,
)

__all__ = [
    "config",
    "io_utils",
    "profiling",
    "normalization",
    "training_data",
    "validation_split",
    "blocking",
    "similarity",
    "features",
    "train_lgbm",
    "hard_negative_mining",
    "embeddings",
    "train_neural",
    "cross_encoder",
    "calibration",
    "decision",
    "evaluation",
    "inference",
    "submission",
    "pipeline",
]