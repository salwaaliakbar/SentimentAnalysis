"""
Production Multi-Task Training Script
====================================

Upgrades:
1. Joint learning: rating regression + 3-way sentiment classification
2. Weighted sampling to mitigate class imbalance
3. Composite loss: Huber/MSE + CrossEntropy + overconfidence penalty
4. Neutral-aware diagnostics during validation
5. Continuous bounded outputs in [1, 5] via scaled sigmoid
"""

import json
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import DistilBertModel, DistilBertTokenizerFast, get_cosine_schedule_with_warmup


# ====== SEED ======
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====== CONFIG ======
CSV_PATH = "employee_reviews_processed.csv"
OUT_DIR = "model_output_v3"

BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 384
HIDDEN_DIM = 384
DROPOUT = 0.30

EPOCHS = 65
BATCH_SIZE = 24
GRADIENT_ACCUMULATION = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION

LR = 2e-5
WEIGHT_DECAY = 0.02
WARMUP_RATIO = 0.10

RATING_MIN = 1.0
RATING_MAX = 5.0

ASPECT_LOSS_WEIGHT = 0.35
CLASSIFICATION_LOSS_WEIGHT = 0.40
OVERCONF_PENALTY_WEIGHT = 0.05

USE_HUBER = True
HUBER_DELTA = 0.5

PATIENCE = 8
MIN_DELTA = 1e-4

ASPECT_NAMES = ["work_life_balance", "company_culture", "career_growth", "salary_benefits"]


def rating_to_sentiment_class(rating: float) -> int:
    if pd.isna(rating):
        return 1
    if rating <= 2.5:
        return 0
    if rating >= 3.5:
        return 2
    return 1


def sentiment_from_text_rule(text: str) -> int:
    """
    Lightweight heuristic used for calibration labels only.
    0=negative, 1=neutral, 2=positive
    """
    if not isinstance(text, str):
        return 1
    lower = text.lower()

    neg_terms = ["terrible", "awful", "toxic", "bad", "burnout", "worst", "low pay", "poor"]
    pos_terms = ["great", "excellent", "amazing", "supportive", "best", "fantastic", "good"]

    neg_hits = sum(term in lower for term in neg_terms)
    pos_hits = sum(term in lower for term in pos_terms)

    if neg_hits >= pos_hits + 1:
        return 0
    if pos_hits >= neg_hits + 1:
        return 2
    return 1


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.5):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.delta, 0.5 * diff.pow(2) / self.delta, diff - 0.5 * self.delta)
        return loss.mean()


def masked_regression_loss(pred: torch.Tensor, target: torch.Tensor, use_huber: bool, delta: float) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    if use_huber:
        diff = torch.abs(pred[mask] - target[mask])
        return torch.where(diff < delta, 0.5 * diff.pow(2) / delta, diff - 0.5 * delta).mean()
    return nn.functional.mse_loss(pred[mask], target[mask])


def confidence_penalty(class_logits: torch.Tensor) -> torch.Tensor:
    """Penalize overconfident class distributions with low entropy."""
    probs = torch.softmax(class_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    max_entropy = float(np.log(3.0))
    penalty = torch.clamp(0.65 * max_entropy - entropy, min=0.0)
    return penalty.mean()


class ReviewDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        for k, v in self.labels.items():
            item[k] = v[idx]
        return item


class MultiTaskDistilBert(nn.Module):
    def __init__(self, base_model: str, aspect_names: List[str], hidden_dim: int = 384, dropout: float = 0.30):
        super().__init__()
        self.aspect_names = aspect_names
        self.encoder = DistilBertModel.from_pretrained(base_model)

        for i, param in enumerate(self.encoder.parameters()):
            if i < len(list(self.encoder.parameters())) - 8:
                param.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size

        self.feature_extractor = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.overall_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.aspect_heads = nn.ModuleDict(
            {
                a: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for a in aspect_names
            }
        )

        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    @staticmethod
    def _scale_output(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (RATING_MAX - RATING_MIN) + RATING_MIN

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.last_hidden_state[:, 0]
        features = self.feature_extractor(pooled)

        out = {
            "overall": self._scale_output(self.overall_head(features).squeeze(-1)),
            "sentiment_logits": self.sentiment_head(features),
        }

        for aspect, head in self.aspect_heads.items():
            out[aspect] = self._scale_output(head(features).squeeze(-1))

        return out


def build_labels(frame: pd.DataFrame) -> Dict[str, torch.Tensor]:
    labels = {
        "overall": torch.tensor(frame["overall_rating"].values, dtype=torch.float32),
        "sentiment_class": torch.tensor(frame["sentiment_class"].values, dtype=torch.long),
    }
    for a in ASPECT_NAMES:
        if a in frame.columns:
            labels[a] = torch.tensor(frame[a].values, dtype=torch.float32)
        else:
            labels[a] = torch.full((len(frame),), float("nan"), dtype=torch.float32)
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("MULTI-TASK DISTILBERT TRAINING (REGRESSION + SENTIMENT CLASSIFICATION)")
    print("=" * 80)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run preprocess_augment.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if "text" not in df.columns or "overall_rating" not in df.columns:
        print("ERROR: required columns missing (text, overall_rating)")
        return

    for col in ["overall_rating"] + ASPECT_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["text"].fillna("").str.len() > 20].copy()
    df = df[~df["overall_rating"].isna()].copy()

    df["rating_sentiment_class"] = df["overall_rating"].apply(rating_to_sentiment_class)
    df["text_sentiment_class"] = df["text"].apply(sentiment_from_text_rule)

    disagreement_mask = df["rating_sentiment_class"] != df["text_sentiment_class"]
    df["sentiment_class"] = np.where(
        disagreement_mask,
        1,
        df["rating_sentiment_class"],
    )

    print(f"Samples after filtering: {len(df):,}")
    print(f"Sentiment class distribution: {df['sentiment_class'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=df["sentiment_class"],
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    train_enc = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    train_dataset = ReviewDataset(train_enc, build_labels(train_df))
    val_dataset = ReviewDataset(val_enc, build_labels(val_df))

    class_counts = train_df["sentiment_class"].value_counts().to_dict()
    class_weights = {
        k: len(train_df) / (len(class_counts) * v)
        for k, v in class_counts.items()
    }
    sample_weights = train_df["sentiment_class"].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskDistilBert(BASE_MODEL, ASPECT_NAMES, HIDDEN_DIM, DROPOUT).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) * EPOCHS) // GRADIENT_ACCUMULATION
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    reg_loss_fn = HuberLoss(delta=HUBER_DELTA) if USE_HUBER else nn.MSELoss()
    ce_weights = torch.tensor(
        [class_weights.get(0, 1.0), class_weights.get(1, 1.0), class_weights.get(2, 1.0)],
        dtype=torch.float32,
        device=device,
    )
    cls_loss_fn = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=0.05)

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_val_loss = float("inf")
    best_val_r2 = float("-inf")
    patience_ctr = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_overall_r2": [],
        "val_overall_mae": [],
        "val_sentiment_acc": [],
        "val_neutral_mae": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}

            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(input_ids, attention_mask)
                    reg_loss = reg_loss_fn(out["overall"], labels["overall"])
                    aspect_loss = torch.tensor(0.0, device=device)
                    for a in ASPECT_NAMES:
                        aspect_loss = aspect_loss + masked_regression_loss(out[a], labels[a], USE_HUBER, HUBER_DELTA)
                    cls_loss = cls_loss_fn(out["sentiment_logits"], labels["sentiment_class"])
                    conf_penalty = confidence_penalty(out["sentiment_logits"])

                    loss = (
                        reg_loss
                        + ASPECT_LOSS_WEIGHT * (aspect_loss / max(1, len(ASPECT_NAMES)))
                        + CLASSIFICATION_LOSS_WEIGHT * cls_loss
                        + OVERCONF_PENALTY_WEIGHT * conf_penalty
                    )
                    loss = loss / GRADIENT_ACCUMULATION

                scaler.scale(loss).backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                out = model(input_ids, attention_mask)
                reg_loss = reg_loss_fn(out["overall"], labels["overall"])
                aspect_loss = torch.tensor(0.0, device=device)
                for a in ASPECT_NAMES:
                    aspect_loss = aspect_loss + masked_regression_loss(out[a], labels[a], USE_HUBER, HUBER_DELTA)
                cls_loss = cls_loss_fn(out["sentiment_logits"], labels["sentiment_class"])
                conf_penalty = confidence_penalty(out["sentiment_logits"])

                loss = (
                    reg_loss
                    + ASPECT_LOSS_WEIGHT * (aspect_loss / max(1, len(ASPECT_NAMES)))
                    + CLASSIFICATION_LOSS_WEIGHT * cls_loss
                    + OVERCONF_PENALTY_WEIGHT * conf_penalty
                )
                loss = loss / GRADIENT_ACCUMULATION

                loss.backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * GRADIENT_ACCUMULATION
            steps += 1

        avg_train_loss = running_loss / max(1, steps)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        preds, trues = [], []
        pred_cls, true_cls = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}

                out = model(input_ids, attention_mask)
                reg_loss = reg_loss_fn(out["overall"], labels["overall"])
                aspect_loss = torch.tensor(0.0, device=device)
                for a in ASPECT_NAMES:
                    aspect_loss = aspect_loss + masked_regression_loss(out[a], labels[a], USE_HUBER, HUBER_DELTA)
                cls_loss = cls_loss_fn(out["sentiment_logits"], labels["sentiment_class"])
                conf_penalty = confidence_penalty(out["sentiment_logits"])

                loss = (
                    reg_loss
                    + ASPECT_LOSS_WEIGHT * (aspect_loss / max(1, len(ASPECT_NAMES)))
                    + CLASSIFICATION_LOSS_WEIGHT * cls_loss
                    + OVERCONF_PENALTY_WEIGHT * conf_penalty
                )

                val_loss += loss.item()
                val_steps += 1

                preds.append(out["overall"].cpu().numpy())
                trues.append(labels["overall"].cpu().numpy())

                pred_cls.append(out["sentiment_logits"].argmax(dim=-1).cpu().numpy())
                true_cls.append(labels["sentiment_class"].cpu().numpy())

        avg_val_loss = val_loss / max(1, val_steps)

        all_preds = np.concatenate(preds)
        all_trues = np.concatenate(trues)
        all_pred_cls = np.concatenate(pred_cls)
        all_true_cls = np.concatenate(true_cls)

        val_r2 = r2_score(all_trues, all_preds)
        val_mae = mean_absolute_error(all_trues, all_preds)
        val_sent_acc = float((all_pred_cls == all_true_cls).mean())

        neutral_mask = all_true_cls == 1
        if neutral_mask.any():
            neutral_mae = float(np.mean(np.abs(all_preds[neutral_mask] - all_trues[neutral_mask])))
        else:
            neutral_mae = float("nan")

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_overall_r2"].append(val_r2)
        history["val_overall_mae"].append(val_mae)
        history["val_sentiment_acc"].append(val_sent_acc)
        history["val_neutral_mae"].append(neutral_mae)

        improved = ""
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            best_val_r2 = val_r2
            patience_ctr = 0
            improved = " <- BEST"
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train={avg_train_loss:.4f} | val={avg_val_loss:.4f} "
            f"| R2={val_r2:.4f} | MAE={val_mae:.4f} | SentAcc={val_sent_acc:.3f} "
            f"| NeutralMAE={neutral_mae:.4f}{improved}"
        )

        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    tokenizer.save_pretrained(OUT_DIR)

    calibration_rules = {
        "neutral_range": [2.5, 3.5],
        "negative_cap": 2.2,
        "positive_floor": 3.9,
    }

    config = {
        "base_model": BASE_MODEL,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "max_len": MAX_LEN,
        "aspect_names": ASPECT_NAMES,
        "rating_min": RATING_MIN,
        "rating_max": RATING_MAX,
        "epochs_trained": int(history["epoch"][-1]) if history["epoch"] else 0,
        "best_val_loss": float(best_val_loss),
        "best_val_r2": float(best_val_r2),
        "loss_config": {
            "use_huber": USE_HUBER,
            "huber_delta": HUBER_DELTA,
            "aspect_loss_weight": ASPECT_LOSS_WEIGHT,
            "classification_loss_weight": CLASSIFICATION_LOSS_WEIGHT,
            "overconfidence_penalty_weight": OVERCONF_PENALTY_WEIGHT,
        },
        "calibration": calibration_rules,
        "hyperparameters": {
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "patience": PATIENCE,
        },
    }

    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)

    print("=" * 80)
    print(f"Training complete. Best val loss={best_val_loss:.4f}, best R2={best_val_r2:.4f}")
    print(f"Artifacts saved to {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
