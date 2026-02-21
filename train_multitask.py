# ====== IMPORTS ======
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup

# ====== SEED ======
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ====== CONFIG ======
CSV_PATH = "employee_reviews.csv"   # local CSV path
OUT_DIR = "model_output"

BASE_MODEL = "distilbert-base-uncased"
EPOCHS = 8
BATCH_SIZE = 16
MAX_LEN = 384
LR = 3e-5
WARMUP_RATIO = 0.1
PATIENCE = 2
ASPECT_LOSS_WEIGHT = 0.25

os.makedirs(OUT_DIR, exist_ok=True)

# ====== HELPERS ======
def find_column(df: pd.DataFrame, name: str, fallbacks: List[str] = None) -> str:
    candidates = [name] + (fallbacks or [])
    lowered = {c.strip().lower(): c for c in df.columns}
    for key in candidates:
        if key.strip().lower() in lowered:
            return lowered[key.strip().lower()]
    return ""

def normalize_text(*parts: str) -> str:
    cleaned = [p for p in parts if isinstance(p, str) and p.strip()]
    return " ".join(cleaned)

def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean((pred[mask] - target[mask]) ** 2)

# ====== DATASET ======
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

# ====== MODEL ======
class MultiTaskDistilBert(nn.Module):
    def __init__(self, base_model: str, aspect_names: List[str]):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.overall_head = nn.Linear(hidden_size, 1)  # regression
        self.aspect_heads = nn.ModuleDict({a: nn.Linear(hidden_size, 1) for a in aspect_names})

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        logits = {"overall": self.overall_head(pooled).squeeze(-1)}
        for a, head in self.aspect_heads.items():
            logits[a] = head(pooled).squeeze(-1)
        return logits

# ====== LOAD DATA ======
df = pd.read_csv(CSV_PATH)

col_summary = find_column(df, "summary")
col_pros = find_column(df, "pros")
col_cons = find_column(df, "cons")
col_advice = find_column(df, "advice-to-mgmt")
col_overall = find_column(df, "overall-ratings")

col_wlb = find_column(df, "work-balance-stars")
col_culture = find_column(df, "culture-values-stars")
col_career = find_column(df, "career-opportunities-stars", ["carrer-opportunities-stars"])
col_comp = find_column(df, "comp-benefit-stars", ["comp-benefit-stars.1"])

df["text"] = df.apply(lambda r: normalize_text(r[col_summary], r[col_pros], r[col_cons], r[col_advice]), axis=1)
df["overall_rating"] = pd.to_numeric(df[col_overall], errors="coerce")
df["work_life_balance"] = pd.to_numeric(df[col_wlb], errors="coerce") if col_wlb else np.nan
df["company_culture"] = pd.to_numeric(df[col_culture], errors="coerce") if col_culture else np.nan
df["career_growth"] = pd.to_numeric(df[col_career], errors="coerce") if col_career else np.nan
df["salary_benefits"] = pd.to_numeric(df[col_comp], errors="coerce") if col_comp else np.nan

df = df[df["text"].str.len() > 0]
df = df[~df["overall_rating"].isna()]

# Split train/val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

# ====== TOKENIZE ======
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
train_enc = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
val_enc = tokenizer(val_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

aspect_names = ["work_life_balance", "company_culture", "career_growth", "salary_benefits"]

def build_labels(frame):
    labels = {"overall": torch.tensor(frame["overall_rating"].values, dtype=torch.float)}
    for a in aspect_names:
        labels[a] = torch.tensor(frame[a].values, dtype=torch.float)
    return labels

train_dataset = ReviewDataset(train_enc, build_labels(train_df))
val_dataset = ReviewDataset(val_enc, build_labels(val_df))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# ====== TRAIN ======
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskDistilBert(BASE_MODEL, aspect_names).to(device)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
loss_fn = nn.MSELoss()  # regression

scaler = torch.amp.GradScaler()
best_val_loss = float("inf")
patience_ctr = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}

        with torch.amp.autocast(device_type="cuda"):
            out = model(input_ids, attention_mask)
            loss = loss_fn(out["overall"], labels["overall"])
            for a in aspect_names:
                loss += ASPECT_LOSS_WEIGHT * masked_mse(out[a], labels[a])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    # ====== VALIDATION ======
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids, attention_mask)
            batch_labels = {k: v.to(device) for k, v in batch.items() if k not in ["input_ids", "attention_mask"]}
            loss = loss_fn(out["overall"], batch_labels["overall"])
            for a in aspect_names:
                loss += ASPECT_LOSS_WEIGHT * masked_mse(out[a], batch_labels[a])
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping triggered.")
            break

tokenizer.save_pretrained(OUT_DIR)
print("Training complete. Best model saved to:", OUT_DIR)