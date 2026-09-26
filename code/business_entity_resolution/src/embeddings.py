"""
Pretrained embedding generation and caching.

Supports:
- intfloat/e5-base-v2 (MIT license, ~110M params)
- BAAI/bge-base-en-v1.5 (MIT license, ~110M params)

Both are under 8B parameter limit and MIT licensed.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import CACHE_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# ── Model configurations ────────────────────────────────────────────

EMBEDDING_MODELS = {
    "e5-base-v2": {
        "model_name": "intfloat/e5-base-v2",
        "version": "main",
        "params": 110_000_000,
        "license": "MIT",
        "dim": 768,
        "pooling": "mean",
        "prefix": "query: ",
    },
    "bge-base-en-v1.5": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "version": "main",
        "params": 110_000_000,
        "license": "MIT",
        "dim": 768,
        "pooling": "cls",
        "prefix": "",
    },
}


def verify_model_license(model_key: str) -> bool:
    """Verify model license is compatible (MIT/Apache-2.0)."""
    model_info = EMBEDDING_MODELS[model_key]
    return model_info["license"] in ("MIT", "Apache-2.0")


def get_cache_key(texts: List[str], model_key: str, text_type: str) -> str:
    """Generate deterministic cache key for embeddings."""
    content = f"{model_key}|{text_type}|{len(texts)}|" + "|".join(texts[:100])
    return hashlib.md5(content.encode()).hexdigest()[:16]


class EmbeddingGenerator:
    """Generates and caches embeddings using pretrained transformers."""

    def __init__(
        self,
        model_key: str = "e5-base-v2",
        device: str = None,
        batch_size: int = 64,
        max_length: int = 512,
        cache_dir: Path = None,
    ):
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(EMBEDDING_MODELS.keys())}")

        if not verify_model_license(model_key):
            raise ValueError(f"Model {model_key} license not verified as MIT/Apache-2.0")

        self.model_key = model_key
        self.model_info = EMBEDDING_MODELS[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir or (CACHE_DIR / "embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._tokenizer = None

        logger.info(f"Initialized {model_key} on {self.device}")
        logger.info(f"  Params: {self.model_info['params']:,} ({self.model_info['license']} license)")

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading model: {self.model_info['model_name']}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_name"])
            self._model = AutoModel.from_pretrained(self.model_info["model_name"])
            self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_info["model_name"])
        return self._tokenizer

    def _pool_embeddings(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """Apply pooling strategy."""
        if self.model_info["pooling"] == "cls":
            # Use [CLS] token
            embeddings = last_hidden_state[:, 0]
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            embeddings = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def encode(
        self,
        texts: List[str],
        text_type: str = "text",
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of strings to encode
            text_type: Type of text (for caching: "name", "address", "combined")
            use_cache: Whether to use/load from cache
            show_progress: Show tqdm progress bar

        Returns:
            Embeddings array of shape (n_texts, dim)
        """
        # Check cache
        cache_key = get_cache_key(texts, self.model_key, text_type)
        cache_path = self.cache_dir / f"{self.model_key}_{text_type}_{cache_key}.npy"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

        logger.info(f"Encoding {len(texts)} texts with {self.model_key}...")

        # Prepare texts with prefix (for E5)
        prefix = self.model_info.get("prefix", "")
        prefixed_texts = [prefix + t for t in texts]

        # Tokenize in batches
        all_embeddings = []

        for i in tqdm(range(0, len(prefixed_texts), self.batch_size), disable=not show_progress):
            batch = prefixed_texts[i:i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                batch_emb = self._pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"])
                all_embeddings.append(batch_emb)

        embeddings = np.vstack(all_embeddings)

        # Cache
        if use_cache:
            np.save(cache_path, embeddings)
            logger.info(f"Cached embeddings to {cache_path}")

        return embeddings

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        name_col: str = "business_name_norm",
        addr_col: str = "business_address_norm",
        id_col: str = "entity_id",
        use_cache: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Encode all texts in a DataFrame.

        Returns:
            Dict with keys: 'name', 'address', 'combined' -> embeddings
        """
        texts_name = df[name_col].fillna("").tolist()
        texts_addr = df[addr_col].fillna("").tolist()
        texts_combined = [f"{n} [SEP] {a}" for n, a in zip(texts_name, texts_addr)]

        ids = df[id_col].astype(str).tolist()

        emb_name = self.encode(texts_name, "name", use_cache)
        emb_addr = self.encode(texts_addr, "address", use_cache)
        emb_combined = self.encode(texts_combined, "combined", use_cache)

        return {
            "ids": ids,
            "name": emb_name,
            "address": emb_addr,
            "combined": emb_combined,
        }

    def get_embedding_dict(self, df: pd.DataFrame, text_type: str = "combined") -> Dict[str, np.ndarray]:
        """Get embeddings as dict mapping entity_id -> embedding."""
        result = self.encode_dataframe(df, text_type=text_type)
        return dict(zip(result["ids"], result[text_type]))


def load_all_embeddings(
    model_key: str = "e5-base-v2",
    train: bool = True,
    use_cache: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load/generate embeddings for all three sources.

    Returns:
        (s1_embeddings, s2_embeddings, s3_embeddings) as dicts
    """
    from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
    from io_utils import load_tsv
    from normalization import add_normalized_columns

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

    generator = EmbeddingGenerator(model_key=model_key)

    s1_emb = generator.get_embedding_dict(s1, "combined")
    s2_emb = generator.get_embedding_dict(s2, "combined")
    s3_emb = generator.get_embedding_dict(s3, "combined")

    return s1_emb, s2_emb, s3_emb


def compute_embedding_similarities(
    candidates_df: pd.DataFrame,
    s1_embeddings: Dict[str, np.ndarray],
    cand_embeddings: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Add embedding cosine similarity features to candidates.

    Adds: emb_cosine (combined), emb_name_cosine, emb_addr_cosine
    """
    def cosine(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))  # Already normalized

    out = candidates_df.copy()
    out["emb_cosine"] = out.apply(
        lambda r: cosine(s1_embeddings.get(r["s1_id"]), cand_embeddings.get(r["candidate_id"])),
        axis=1,
    )
    return out


def add_embedding_features_to_dataframe(
    features_df: pd.DataFrame,
    s1_embeddings: Dict[str, np.ndarray],
    cand_embeddings: Dict[str, np.ndarray],
    prefix: str = "e5",
) -> pd.DataFrame:
    """Add embedding features to feature DataFrame."""
    out = features_df.copy()

    def cosine(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    out[f"{prefix}_cosine"] = out.apply(
        lambda r: cosine(s1_embeddings.get(r["s1_id"]), cand_embeddings.get(r["candidate_id"])),
        axis=1,
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick test with small sample
    gen = EmbeddingGenerator("e5-base-v2")
    texts = [
        "google llc",
        "microsoft corporation",
        "apple inc",
        "amazon com inc",
    ]
    emb = gen.encode(texts, "test")
    print(f"Embeddings shape: {emb.shape}")
    print(f"Cosine similarity (0,1): {np.dot(emb[0], emb[1]):.4f}")