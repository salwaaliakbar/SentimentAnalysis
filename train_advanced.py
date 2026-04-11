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
"""DistilBERT multi-task training for five continuous employee-review ratings."""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import DistilBertModel, DistilBertTokenizerFast, get_cosine_schedule_with_warmup


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CSV_PATH = "employee_reviews_processed.csv"
OUT_DIR = "model_output_v3"
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 384
DROPOUT = 0.25
EPOCHS = 25
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 5
MIN_DELTA = 1e-4
RATING_MIN = 1.0
RATING_MAX = 5.0
HUBER_DELTA = 0.5

OUTPUT_NAMES = [
    "overall_rating",
    "work_life_balance",
    "company_culture",
    "career_opportunities",
    "salary_benefits",
]

CSV_COLUMN_MAP = {
    "career_opportunities": "career_growth",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_review_input(text: str) -> str:
    return f"[REVIEW] {str(text).strip()}"


def clip_ratings(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(RATING_MIN, RATING_MAX)
    return frame


def rating_class(rating: float) -> int:
    if rating <= 2.5:
        return 0
    if rating >= 3.5:
        return 2
    return 1


def row_target_class(row: pd.Series) -> int:
    return rating_class(float(row["overall_rating"]))


def build_labels(frame: pd.DataFrame) -> Dict[str, torch.Tensor]:
    labels = {}
    for output_name in OUTPUT_NAMES:
        if output_name not in frame.columns:
            raise ValueError(f"Missing required column: {output_name}")
        labels[output_name] = torch.tensor(frame[output_name].values, dtype=torch.float32)
    return labels


class ReviewDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        for key, value in self.labels.items():
            item[key] = value[idx]
        return item


class DistilBertMultiHeadRegressor(nn.Module):
    def __init__(self, base_model: str, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, 1) for name in OUTPUT_NAMES})

    @staticmethod
    def scale_output(logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit) * (RATING_MAX - RATING_MIN) + RATING_MIN

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = self.scale_output(head(pooled).squeeze(-1))
        return outputs


def masked_huber_loss(prediction: torch.Tensor, target: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.zeros((), device=prediction.device)
    loss_fn = nn.HuberLoss(delta=delta, reduction="none")
    return loss_fn(prediction[mask], target[mask]).mean()


def compute_metrics(targets: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for name in OUTPUT_NAMES:
        y_true = targets[name]
        y_pred = predictions[name]
        pearson_r = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
        metrics[name] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "pearson_r": pearson_r,
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred)),
            "bias": float(np.mean(y_pred - y_true)),
        }
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    all_targets = {name: [] for name in OUTPUT_NAMES}
    all_predictions = {name: [] for name in OUTPUT_NAMES}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            for name in OUTPUT_NAMES:
                all_predictions[name].append(outputs[name].detach().cpu().numpy())
                all_targets[name].append(batch[name].cpu().numpy())

    metrics_targets = {}
    metrics_predictions = {}
    for name in OUTPUT_NAMES:
        metrics_targets[name] = np.concatenate(all_targets[name])
        metrics_predictions[name] = np.concatenate(all_predictions[name])

    return compute_metrics(metrics_targets, metrics_predictions), metrics_targets, metrics_predictions


def main() -> None:
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("DISTILBERT MULTI-HEAD REGRESSION TRAINING")
    print("=" * 80)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run preprocess_augment.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if "text" not in df.columns or "overall_rating" not in df.columns:
        print("ERROR: required columns missing (text, overall_rating)")
        return

    for source_name, target_name in CSV_COLUMN_MAP.items():
        if target_name in df.columns and source_name not in df.columns:
            df[source_name] = df[target_name]

    required_columns = ["overall_rating", "work_life_balance", "company_culture", "career_opportunities", "salary_benefits"]
    if "career_opportunities" not in df.columns and "career_growth" in df.columns:
        df["career_opportunities"] = df["career_growth"]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        print(f"ERROR: missing required rating columns: {missing}")
        return

    df = df[df["text"].fillna("").str.len() > 20].copy()
    df = clip_ratings(df, required_columns)
    df = df.dropna(subset=["overall_rating"]).copy()

    df["target_class"] = df["overall_rating"].apply(row_target_class)
    df["review_input"] = df["text"].apply(build_review_input)

    print(f"Samples after filtering: {len(df):,}")
    print(f"Class distribution: {df['target_class'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=df["target_class"],
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    train_encodings = tokenizer(
        train_df["review_input"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        val_df["review_input"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    train_dataset = ReviewDataset(train_encodings, build_labels(train_df))
    val_dataset = ReviewDataset(val_encodings, build_labels(val_df))

    class_counts = train_df["target_class"].value_counts().to_dict()
    class_weights = {cls: len(train_df) / (len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = train_df["target_class"].map(class_weights).astype(float).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertMultiHeadRegressor(BASE_MODEL, dropout=DROPOUT).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, (len(train_loader) * EPOCHS) // GRADIENT_ACCUMULATION)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        step_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {name: batch[name].to(device) for name in OUTPUT_NAMES}

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    losses = [masked_huber_loss(outputs[name], labels[name]) for name in OUTPUT_NAMES]
                    loss = sum(losses)
                    loss = loss / GRADIENT_ACCUMULATION
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                losses = [masked_huber_loss(outputs[name], labels[name]) for name in OUTPUT_NAMES]
                loss = sum(losses) / GRADIENT_ACCUMULATION
                loss.backward()

            if batch_index % GRADIENT_ACCUMULATION == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * GRADIENT_ACCUMULATION
            step_count += 1

        train_loss = running_loss / max(1, step_count)
        val_metrics, val_targets, val_predictions = evaluate(model, val_loader, device)

        overall_mae = val_metrics["overall_rating"]["mae"]
        mean_head_mae = float(np.mean([val_metrics[name]["mae"] for name in OUTPUT_NAMES]))
        neutral_mask = (val_targets["overall_rating"] >= 2.5) & (val_targets["overall_rating"] <= 3.5)
        neutral_mae = float(mean_absolute_error(
            val_targets["overall_rating"][neutral_mask],
            val_predictions["overall_rating"][neutral_mask],
        )) if neutral_mask.any() else float("nan")

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_mae={mean_head_mae:.4f} | overall_mae={overall_mae:.4f} | "
            f"overall_r2={val_metrics['overall_rating']['r2']:.4f} | "
            f"neutral_mae={neutral_mae:.4f}"
        )

        if mean_head_mae < best_val_mae - MIN_DELTA:
            best_val_mae = mean_head_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
            print("  saved new best checkpoint")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    tokenizer.save_pretrained(OUT_DIR)

    config = {
        "base_model": BASE_MODEL,
        "max_len": MAX_LEN,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "rating_min": RATING_MIN,
        "rating_max": RATING_MAX,
        "output_names": OUTPUT_NAMES,
        "best_val_mae": best_val_mae,
    }

    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best validation mean head MAE: {best_val_mae:.4f}")
    print(f"Checkpoint saved to: {os.path.join(OUT_DIR, 'best.pt')}")
    print(f"Tokenizer and config saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
