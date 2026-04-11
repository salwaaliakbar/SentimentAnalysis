"""Evaluation utilities for the five-head DistilBERT regression model."""

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
BATCH_SIZE = 32
RATING_MIN = 1.0
RATING_MAX = 5.0

OUTPUT_NAMES = [
    "overall_rating",
    "work_life_balance",
    "company_culture",
    "career_opportunities",
    "salary_benefits",
]


def build_review_input(text: str) -> str:
    return f"[REVIEW] {str(text).strip()}"


def class_from_rating(rating: float) -> int:
    if rating <= 2.5:
        return 0
    if rating >= 3.5:
        return 2
    return 1


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
    def __init__(self, base_model: str, dropout: float):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, 1) for name in OUTPUT_NAMES})

    @staticmethod
    def scale_output(logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logit) * (RATING_MAX - RATING_MIN) + RATING_MIN

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(encoded.last_hidden_state[:, 0])
        return {name: self.scale_output(head(pooled).squeeze(-1)) for name, head in self.heads.items()}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) > 1:
        pearson_r = float(pearsonr(y_true, y_pred)[0])
    else:
        pearson_r = float("nan")
    errors = np.abs(y_pred - y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": pearson_r,
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "std_true": float(np.std(y_true)),
        "std_pred": float(np.std(y_pred)),
    }


def save_plots(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bins = np.linspace(1.0, 5.0, 25)
    plt.hist(y_true, bins=bins, alpha=0.55, label="Actual", color="#264653")
    plt.hist(y_pred, bins=bins, alpha=0.55, label="Predicted", color="#e76f51")
    plt.title("Overall Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overall_distribution.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, color="#1d3557")
    plt.plot([1, 5], [1, 5], "r--", linewidth=1.2)
    plt.title("Predicted vs Actual Overall Rating")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overall_scatter.png"), dpi=140)
    plt.close()


def main() -> None:
    config_path = os.path.join(MODEL_DIR, "config.json")
    weights_path = os.path.join(MODEL_DIR, "best.pt")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print("Missing model artifacts. Ensure training completed and best.pt/config.json exist.")
        return

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    base_model = cfg.get("base_model", "distilbert-base-uncased")
    dropout = float(cfg.get("dropout", 0.25))
    max_len = int(cfg.get("max_len", 384))

    if not os.path.exists(CSV_PATH):
        print(f"Missing {CSV_PATH}. Run preprocess_augment.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if "career_opportunities" not in df.columns and "career_growth" in df.columns:
        df["career_opportunities"] = df["career_growth"]

    for column in OUTPUT_NAMES:
        if column not in df.columns:
            print(f"Missing required column: {column}")
            return
        df[column] = pd.to_numeric(df[column], errors="coerce").clip(RATING_MIN, RATING_MAX)

    df = df[df["text"].fillna("").str.len() > 20].copy()
    df = df.dropna(subset=["overall_rating"]).copy()
    df["rating_class"] = df["overall_rating"].apply(class_from_rating)
    df["review_input"] = df["text"].apply(build_review_input)

    _, val_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["rating_class"])

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    val_encodings = tokenizer(
        val_df["review_input"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )

    labels = {name: torch.tensor(val_df[name].values, dtype=torch.float32) for name in OUTPUT_NAMES}
    val_loader = DataLoader(ReviewDataset(val_encodings, labels), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertMultiHeadRegressor(base_model=base_model, dropout=dropout).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    collected_targets = {name: [] for name in OUTPUT_NAMES}
    collected_predictions = {name: [] for name in OUTPUT_NAMES}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            for name in OUTPUT_NAMES:
                collected_targets[name].append(batch[name].cpu().numpy())
                collected_predictions[name].append(outputs[name].cpu().numpy())

    targets = {name: np.concatenate(values) for name, values in collected_targets.items()}
    predictions = {name: np.concatenate(values) for name, values in collected_predictions.items()}

    per_head_metrics = {name: compute_metrics(targets[name], predictions[name]) for name in OUTPUT_NAMES}

    overall_true = targets["overall_rating"]
    overall_pred = predictions["overall_rating"]
    overall_metrics = per_head_metrics["overall_rating"]

    neutral_mask = (overall_true >= 2.5) & (overall_true <= 3.5)
    neutral_mae = float(mean_absolute_error(overall_true[neutral_mask], overall_pred[neutral_mask])) if neutral_mask.any() else float("nan")
    neutral_bias = float(np.mean(overall_pred[neutral_mask] - overall_true[neutral_mask])) if neutral_mask.any() else float("nan")

    distribution_summary = {
        name: {
            "mean_true": per_head_metrics[name]["mean_true"],
            "mean_pred": per_head_metrics[name]["mean_pred"],
            "std_true": per_head_metrics[name]["std_true"],
            "std_pred": per_head_metrics[name]["std_pred"],
            "bias": per_head_metrics[name]["bias"],
        }
        for name in OUTPUT_NAMES
    }

    print("=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    print(f"Samples: {len(overall_true)}")
    print(f"Overall MAE: {overall_metrics['mae']:.4f}")
    print(f"Overall RMSE: {overall_metrics['rmse']:.4f}")
    print(f"Overall R2: {overall_metrics['r2']:.4f}")
    print(f"Overall Pearson r: {overall_metrics['pearson_r']:.4f}")
    print(f"Neutral Overall MAE: {neutral_mae:.4f}")
    print(f"Neutral Overall Bias: {neutral_bias:.4f}")
    print("Per-head metrics:")
    for name in OUTPUT_NAMES:
        metrics = per_head_metrics[name]
        print(
            f"  {name}: MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
            f"R2={metrics['r2']:.4f} | Pearson={metrics['pearson_r']:.4f} | Bias={metrics['bias']:.4f}"
        )

    print("Distribution mismatch summary:")
    print(json.dumps(distribution_summary, indent=2))

    save_plots(overall_true, overall_pred, MODEL_DIR)

    summary_row = {
        "overall_mae": overall_metrics["mae"],
        "overall_rmse": overall_metrics["rmse"],
        "overall_r2": overall_metrics["r2"],
        "overall_pearson_r": overall_metrics["pearson_r"],
        "neutral_mae": neutral_mae,
        "neutral_bias": neutral_bias,
    }
    for name in OUTPUT_NAMES:
        summary_row[f"{name}_mae"] = per_head_metrics[name]["mae"]
        summary_row[f"{name}_rmse"] = per_head_metrics[name]["rmse"]
        summary_row[f"{name}_r2"] = per_head_metrics[name]["r2"]
        summary_row[f"{name}_pearson_r"] = per_head_metrics[name]["pearson_r"]
        summary_row[f"{name}_bias"] = per_head_metrics[name]["bias"]

    pd.DataFrame([summary_row]).to_csv(os.path.join(MODEL_DIR, "evaluation_results.csv"), index=False)
    print(f"Saved metrics to {os.path.join(MODEL_DIR, 'evaluation_results.csv')}")
    print(f"Saved plots to {os.path.join(MODEL_DIR, 'overall_distribution.png')} and {os.path.join(MODEL_DIR, 'overall_scatter.png')}")


if __name__ == "__main__":
    main()
