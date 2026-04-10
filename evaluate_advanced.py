"""
Production Evaluation Script
============================

Adds:
1. Full regression metrics
2. Neutral-review specific error analysis
3. Prediction bias checks by sentiment class
4. Distribution and scatter plots for calibration monitoring
"""

import json
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizerFast


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CSV_PATH = "employee_reviews_processed.csv"
MODEL_DIR = "model_output_v3"
BATCH_SIZE = 64


def rating_to_sentiment_class(rating: float) -> int:
    if rating <= 2.5:
        return 0
    if rating >= 3.5:
        return 2
    return 1


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
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
    def __init__(
        self,
        base_model: str,
        aspect_names: List[str],
        hidden_dim: int,
        dropout: float,
        rating_min: float,
        rating_max: float,
    ):
        super().__init__()
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.aspect_names = aspect_names

        self.encoder = DistilBertModel.from_pretrained(base_model)
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

    def _scale_output(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (self.rating_max - self.rating_min) + self.rating_min

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    errors = np.abs(y_pred - y_true)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "acc_within_0_5": float(np.mean(errors <= 0.5) * 100),
        "acc_within_1_0": float(np.mean(errors <= 1.0) * 100),
        "acc_exact": float(np.mean(np.round(y_pred) == np.round(y_true)) * 100),
        "mean_pred": float(np.mean(y_pred)),
        "mean_true": float(np.mean(y_true)),
    }


def save_plots(y_true: np.ndarray, y_pred: np.ndarray, sentiment_cls: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bins = np.linspace(1.0, 5.0, 25)
    plt.hist(y_true, bins=bins, alpha=0.55, label="Actual", color="#3d5a80")
    plt.hist(y_pred, bins=bins, alpha=0.55, label="Predicted", color="#e07a5f")
    plt.title("Prediction vs Actual Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rating_distribution.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(y_true, y_pred, c=sentiment_cls, cmap="viridis", alpha=0.45)
    plt.plot([1, 5], [1, 5], "r--", linewidth=1.3)
    plt.title("Predicted vs Actual Ratings")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.colorbar(scatter, label="Sentiment Class (0/1/2)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_actual.png"), dpi=140)
    plt.close()


def main():
    config_path = os.path.join(MODEL_DIR, "config.json")
    best_pt_path = os.path.join(MODEL_DIR, "best.pt")

    if not os.path.exists(config_path) or not os.path.exists(best_pt_path):
        print("Missing model artifacts. Ensure training completed and best.pt/config.json exist.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_model = cfg.get("base_model", "distilbert-base-uncased")
    hidden_dim = int(cfg.get("hidden_dim", 384))
    dropout = float(cfg.get("dropout", 0.30))
    max_len = int(cfg.get("max_len", 384))
    rating_min = float(cfg.get("rating_min", 1.0))
    rating_max = float(cfg.get("rating_max", 5.0))
    aspect_names = cfg.get("aspect_names", ["work_life_balance", "company_culture", "career_growth", "salary_benefits"])

    if not os.path.exists(CSV_PATH):
        print(f"Missing {CSV_PATH}. Run preprocess_augment.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    for col in ["overall_rating"] + aspect_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["text"].fillna("").str.len() > 20]
    df = df[~df["overall_rating"].isna()]
    df["sentiment_class"] = df["overall_rating"].apply(rating_to_sentiment_class)

    _, val_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["sentiment_class"])

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    val_enc = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )

    labels = {
        "overall": torch.tensor(val_df["overall_rating"].values, dtype=torch.float32),
        "sentiment_class": torch.tensor(val_df["sentiment_class"].values, dtype=torch.long),
    }
    for a in aspect_names:
        labels[a] = torch.tensor(val_df[a].values, dtype=torch.float32) if a in val_df.columns else torch.full((len(val_df),), float("nan"))

    val_loader = DataLoader(ReviewDataset(val_enc, labels), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskDistilBert(base_model, aspect_names, hidden_dim, dropout, rating_min, rating_max).to(device)
    model.load_state_dict(torch.load(best_pt_path, map_location=device))
    model.eval()

    all_pred = []
    all_true = []
    all_sent_true = []
    all_sent_pred = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids, attention_mask)
            pred_rating = out["overall"].cpu().numpy()
            true_rating = batch["overall"].cpu().numpy()
            sent_pred = out["sentiment_logits"].argmax(dim=-1).cpu().numpy()
            sent_true = batch["sentiment_class"].cpu().numpy()

            all_pred.append(pred_rating)
            all_true.append(true_rating)
            all_sent_pred.append(sent_pred)
            all_sent_true.append(sent_true)

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    sent_pred = np.concatenate(all_sent_pred)
    sent_true = np.concatenate(all_sent_true)

    metrics = compute_metrics(y_true, y_pred)
    sent_acc = float((sent_true == sent_pred).mean() * 100)

    neutral_mask = sent_true == 1
    neutral_mae = float(np.mean(np.abs(y_pred[neutral_mask] - y_true[neutral_mask]))) if neutral_mask.any() else float("nan")
    neutral_overpredict_rate = (
        float(np.mean(y_pred[neutral_mask] > 3.6) * 100) if neutral_mask.any() else float("nan")
    )

    class_bias = {}
    for cls in [0, 1, 2]:
        mask = sent_true == cls
        if mask.any():
            class_bias[str(cls)] = float(np.mean(y_pred[mask] - y_true[mask]))

    print("=" * 80)
    print("PRODUCTION EVALUATION")
    print("=" * 80)
    print(f"Samples: {len(y_true)}")
    print(f"±0.5 Star Accuracy: {metrics['acc_within_0_5']:.2f}%")
    print(f"±1.0 Star Accuracy: {metrics['acc_within_1_0']:.2f}%")
    print(f"Exact Match Accuracy: {metrics['acc_exact']:.2f}%")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Pearson r: {metrics['pearson_r']:.4f}")
    print(f"Sentiment Classification Accuracy: {sent_acc:.2f}%")
    print(f"Neutral MAE: {neutral_mae:.4f}")
    print(f"Neutral Overprediction Rate (>3.6): {neutral_overpredict_rate:.2f}%")
    print(f"Prediction Bias by Class (pred-true): {class_bias}")

    save_plots(y_true, y_pred, sent_true, MODEL_DIR)

    result = {
        **metrics,
        "sentiment_acc": sent_acc,
        "neutral_mae": neutral_mae,
        "neutral_overpredict_rate": neutral_overpredict_rate,
        "class_bias": class_bias,
    }
    pd.DataFrame([result]).to_csv(os.path.join(MODEL_DIR, "evaluation_results.csv"), index=False)
    print(f"Saved metrics to {os.path.join(MODEL_DIR, 'evaluation_results.csv')}")
    print(f"Saved plots: {os.path.join(MODEL_DIR, 'rating_distribution.png')}, {os.path.join(MODEL_DIR, 'pred_vs_actual.png')}")


if __name__ == "__main__":
    main()
