"""
Siamese/Bi-encoder neural model training.

Uses shared pretrained encoder for S1 and candidate representations.
Input: S1 representation + S2/S3 representation
Output: P(match)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from config import ARTIFACTS_DIR, CACHE_DIR, RANDOM_SEED
from embeddings import EMBEDDING_MODELS, verify_model_license

logger = logging.getLogger(__name__)


# ── Model Architecture ──────────────────────────────────────────────

class SiameseEncoder(nn.Module):
    """Siamese network with shared pretrained encoder."""

    def __init__(
        self,
        model_key: str = "e5-base-v2",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        model_info = EMBEDDING_MODELS[model_key]
        self.encoder = AutoModel.from_pretrained(model_info["model_name"])
        self.pooling = model_info["pooling"]
        self.prefix = model_info.get("prefix", "")
        self.encoder_dim = model_info["dim"]

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head (takes concatenated [u, v, |u-v|, u*v])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def encode(self, input_ids, attention_mask):
        """Encode single sequence."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return self.projection(pooled)

    def forward(self, s1_input_ids, s1_attention_mask, cand_input_ids, cand_attention_mask):
        """Forward pass for pair."""
        s1_emb = self.encode(s1_input_ids, s1_attention_mask)
        cand_emb = self.encode(cand_input_ids, cand_attention_mask)

        # Concatenate: [u, v, |u-v|, u*v]
        diff = torch.abs(s1_emb - cand_emb)
        prod = s1_emb * cand_emb
        combined = torch.cat([s1_emb, cand_emb, diff, prod], dim=1)

        logits = self.classifier(combined)
        return logits.squeeze(-1)


# ── Dataset ────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """Dataset for pairwise training."""

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
        text_col_s1: str = "business_name_norm",
        text_col_cand: str = "business_name_norm",
        prefix: str = "",
    ):
        self.pairs = pairs_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col_s1 = text_col_s1
        self.text_col_cand = text_col_cand
        self.prefix = prefix

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        s1_text = self.prefix + str(row.get(self.text_col_s1, ""))
        cand_text = self.prefix + str(row.get(self.text_col_cand, ""))

        s1_enc = self.tokenizer(
            s1_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        cand_enc = self.tokenizer(
            cand_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        label = row.get("label", 0)

        return {
            "s1_input_ids": s1_enc["input_ids"].squeeze(0),
            "s1_attention_mask": s1_enc["attention_mask"].squeeze(0),
            "cand_input_ids": cand_enc["input_ids"].squeeze(0),
            "cand_attention_mask": cand_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


# ── Training ────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    scaler: torch.cuda.amp.GradScaler = None,
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(
                    batch["s1_input_ids"],
                    batch["s1_attention_mask"],
                    batch["cand_input_ids"],
                    batch["cand_attention_mask"],
                )
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(
                batch["s1_input_ids"],
                batch["s1_attention_mask"],
                batch["cand_input_ids"],
                batch["cand_attention_mask"],
            )
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["label"])
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["s1_input_ids"],
                batch["s1_attention_mask"],
                batch["cand_input_ids"],
                batch["cand_attention_mask"],
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())

    from sklearn.metrics import precision_recall_fscore_support, fbeta_score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    f05 = fbeta_score(all_labels, all_preds, beta=0.5, zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f05": float(f05),
    }


def train_siamese(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    model_key: str = "e5-base-v2",
    config: Dict = None,
) -> Tuple[nn.Module, Dict]:
    """
    Train Siamese network.

    Args:
        train_pairs: DataFrame with s1_id, candidate_id, label, and text columns
        val_pairs: Validation DataFrame
        model_key: Pretrained model key
        config: Training config

    Returns:
        (model, history)
    """
    config = config or {}
    model_info = EMBEDDING_MODELS[model_key]

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 2e-5)
    epochs = config.get("epochs", 10)
    max_length = config.get("max_length", 256)
    hidden_dim = config.get("hidden_dim", 256)
    dropout = config.get("dropout", 0.2)
    freeze_encoder = config.get("freeze_encoder", True)
    warmup_steps = config.get("warmup_steps", 500)
    weight_decay = config.get("weight_decay", 0.01)
    mixed_precision = config.get("mixed_precision", True)

    # Seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_info["model_name"])
    prefix = model_info.get("prefix", "")

    # Datasets
    train_dataset = PairDataset(
        train_pairs, tokenizer, max_length,
        prefix=prefix,
    )
    val_dataset = PairDataset(
        val_pairs, tokenizer, max_length,
        prefix=prefix,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = SiameseEncoder(
        model_key=model_key,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device == "cuda" else None

    # Training loop
    history = {"train_loss": [], "val_f05": [], "val_precision": [], "val_recall": []}
    best_f05 = 0.0
    best_state = None
    patience = config.get("patience", 3)
    patience_counter = 0

    logger.info(f"Training Siamese ({model_key}) on {device}")
    logger.info(f"  Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    logger.info(f"  Batch size: {batch_size}, LR: {lr}, Epochs: {epochs}")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_metrics = evaluate_model(model, val_loader, device, threshold=0.5)

        history["train_loss"].append(train_loss)
        history["val_f05"].append(val_metrics["f05"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        logger.info(f"Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                    f"val_f05={val_metrics['f05']:.4f}, P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}")

        # Early stopping
        if val_metrics["f05"] > best_f05:
            best_f05 = val_metrics["f05"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    return model, history


def save_siamese_model(model: nn.Module, model_key: str, config: Dict, history: Dict, output_dir: Path = None):
    """Save model, config, and history."""
    output_dir = output_dir or (ARTIFACTS_DIR / f"siamese_{model_key}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Save config
    save_config = {
        "model_key": model_key,
        "config": config,
        "history": history,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(save_config, f, indent=2, default=str)

    logger.info(f"Saved Siamese model to {output_dir}")


def load_siamese_model(model_dir: Path, device: str = None) -> Tuple[nn.Module, Dict]:
    """Load Siamese model and config."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_dir / "config.json") as f:
        saved = json.load(f)

    model_key = saved["model_key"]
    config = saved["config"]

    model = SiameseEncoder(
        model_key=model_key,
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.2),
        freeze_encoder=config.get("freeze_encoder", True),
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    return model, saved


def predict_siamese(
    model: nn.Module,
    pairs_df: pd.DataFrame,
    tokenizer,
    device: str,
    max_length: int = 256,
    batch_size: int = 256,
    prefix: str = "",
) -> np.ndarray:
    """Generate predictions for pairs."""
    model.eval()

    # Create dataset
    dataset = PairDataset(pairs_df, tokenizer, max_length, prefix=prefix)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick test
    from io_utils import load_tsv
    from normalization import add_normalized_columns
    from training_data import load_ground_truth_pairs, split_by_s1_entity

    s1 = load_tsv("dataset/train/train_source1.tsv").head(1000)
    s2 = load_tsv("dataset/train/train_source2.tsv").head(2000)
    s3 = load_tsv("dataset/train/train_source3.tsv").head(2000)

    for df in [s1, s2, s3]:
        add_normalized_columns(df)

    # Generate some candidates
    gt_pairs = load_ground_truth_pairs()
    gt_keys = set(zip(gt_pairs["s1_id"], gt_pairs["candidate_id"]))

    # Simple candidate: all S2/S3 for first 100 S1
    from blocking import generate_candidates
    candidates = generate_candidates(s1.head(100), s2, s3)
    candidates["label"] = candidates.apply(lambda r: 1 if (r["s1_id"], r["candidate_id"]) in gt_keys else 0, axis=1)

    train_pairs, val_pairs = split_by_s1_entity(candidates)

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    print(f"Train pos: {train_pairs['label'].sum()}, Val pos: {val_pairs['label'].sum()}")

    # Train (commented out - takes time)
    # model, history = train_siamese(train_pairs, val_pairs, "e5-base-v2", {"epochs": 2})
    # print("Done!")