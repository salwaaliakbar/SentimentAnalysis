"""
Colab-friendly multi-task fine-tuning script.
- Overall sentiment: 3-class classification (0/1/2)
- Aspect stars: regression (1-5)

Example (Colab):
  !pip install -r requirements.txt
  !python train_multitask_colab.py --csv /content/employee_reviews.csv --out /content/model
"""

import os
import json
import random
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_column(df: pd.DataFrame, name: str, fallbacks: List[str] = None) -> str:
    """Find a column by case-insensitive name; optional fallbacks."""
    candidates = [name] + (fallbacks or [])
    lowered = {c.strip().lower(): c for c in df.columns}
    for key in candidates:
        if key.strip().lower() in lowered:
            return lowered[key.strip().lower()]
    return ""


def normalize_text(*parts: str) -> str:
    """Join text parts safely."""
    cleaned = [p for p in parts if isinstance(p, str) and p.strip()]
    return " ".join(cleaned)


class ReviewDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        for key, tensor in self.labels.items():
            item[key] = tensor[idx]
        return item


class MultiTaskDistilBert(nn.Module):
    def __init__(self, base_model: str, aspect_names: List[str]):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.overall_head = nn.Linear(hidden_size, 3)
        self.aspect_heads = nn.ModuleDict({
            aspect: nn.Linear(hidden_size, 1) for aspect in aspect_names
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        logits = {"overall": self.overall_head(pooled)}
        for aspect, head in self.aspect_heads.items():
            logits[aspect] = head(pooled).squeeze(-1)
        return logits


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE that ignores NaN targets."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean((pred[mask] - target[mask]) ** 2)


def rating_to_class(rating: float) -> int:
    """Map 1-5 star rating to 3-class label (0/1/2)."""
    if rating <= 2.0:
        return 0
    if rating >= 4.0:
        return 2
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/content/employee_reviews.csv")
    parser.add_argument("--out", default="/content/model")
    parser.add_argument("--base", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)

    col_company = find_column(df, "Company")
    col_summary = find_column(df, "summary")
    col_pros = find_column(df, "pros")
    col_cons = find_column(df, "cons")
    col_advice = find_column(df, "advice-to-mgmt")
    col_overall = find_column(df, "overall-ratings")

    col_wlb = find_column(df, "work-balance-stars")
    col_culture = find_column(df, "culture-values-stars")
    col_career = find_column(df, "carrer-opportunities-stars", ["career-opportunities-stars"])
    col_comp = find_column(df, "comp-benefit-stars", ["comp-benefit-stars.1"])

    required = [col_summary, col_pros, col_cons, col_advice, col_overall]
    if not all(required):
        missing = [name for name, col in zip(
            ["summary", "pros", "cons", "advice-to-mgmt", "overall-ratings"],
            required
        ) if not col]
        raise ValueError(f"Missing required columns: {missing}")

    df["text"] = df.apply(
        lambda r: normalize_text(
            r[col_summary],
            r[col_pros],
            r[col_cons],
            r[col_advice],
        ),
        axis=1
    )

    df["overall_rating"] = pd.to_numeric(df[col_overall], errors="coerce")
    df["work_life_balance"] = pd.to_numeric(df[col_wlb], errors="coerce") if col_wlb else np.nan
    df["company_culture"] = pd.to_numeric(df[col_culture], errors="coerce") if col_culture else np.nan
    df["career_growth"] = pd.to_numeric(df[col_career], errors="coerce") if col_career else np.nan
    df["salary_benefits"] = pd.to_numeric(df[col_comp], errors="coerce") if col_comp else np.nan

    df = df[df["text"].str.len() > 0]
    df = df[~df["overall_rating"].isna()]
    df["overall"] = df["overall_rating"].apply(rating_to_class).astype(int)

    if col_company:
        companies = df[col_company].dropna().unique()
        train_comp, val_comp = train_test_split(companies, test_size=0.2, random_state=SEED)
        train_df = df[df[col_company].isin(train_comp)].reset_index(drop=True)
        val_df = df[df[col_company].isin(val_comp)].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.base)

    train_enc = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.maxlen,
        return_tensors="pt"
    )
    val_enc = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.maxlen,
        return_tensors="pt"
    )

    aspect_names = ["work_life_balance", "company_culture", "career_growth", "salary_benefits"]

    def build_labels(frame: pd.DataFrame) -> Dict[str, torch.Tensor]:
        labels = {
            "overall": torch.tensor(frame["overall"].values, dtype=torch.long),
        }
        for aspect in aspect_names:
            labels[aspect] = torch.tensor(frame[aspect].values, dtype=torch.float)
        return labels

    train_labels = build_labels(train_df)
    val_labels = build_labels(val_df)

    train_dataset = ReviewDataset(train_enc, train_labels)
    val_dataset = ReviewDataset(val_enc, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskDistilBert(args.base, aspect_names).to(device)
    overall_loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = overall_loss_fn(outputs["overall"], labels["overall"])
            for aspect in aspect_names:
                loss = loss + masked_mse(outputs[aspect], labels[aspect])

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_losses = []
        overall_preds = []
        overall_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                loss = overall_loss_fn(outputs["overall"], labels["overall"])
                for aspect in aspect_names:
                    loss = loss + masked_mse(outputs[aspect], labels[aspect])

                val_losses.append(loss.item())

                preds = torch.argmax(outputs["overall"], dim=-1)
                overall_preds.extend(preds.cpu().numpy().tolist())
                overall_true.extend(labels["overall"].cpu().numpy().tolist())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        acc = accuracy_score(overall_true, overall_preds) if overall_true else 0.0

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Overall Acc: {acc:.3f}")

    torch.save(model.state_dict(), os.path.join(args.out, "multitask_distilbert.pt"))
    tokenizer.save_pretrained(args.out)

    meta = {
        "base_model": args.base,
        "max_length": args.maxlen,
        "aspect_names": aspect_names,
        "label_type": "classification_0_1_2",
        "text_columns": [col_summary, col_pros, col_cons, col_advice],
        "overall_column": col_overall,
        "aspect_columns": {
            "work_life_balance": col_wlb,
            "company_culture": col_culture,
            "career_growth": col_career,
            "salary_benefits": col_comp,
        }
    }

    with open(os.path.join(args.out, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to: {args.out}")


if __name__ == "__main__":
    main()
