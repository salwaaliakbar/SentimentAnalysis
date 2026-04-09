"""
RoBERTa Model Evaluation Script

This is the evaluation script for the RoBERTa model.
Use this after train_roberta.py completes to evaluate results.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel
from torch import nn
from typing import List, Dict
import json

# ====== SEED ======
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ====== CONFIG  (only things that don't come from config.json) ======
CSV_PATH  = "employee_reviews_processed.csv"
MODEL_DIR = "model_output_roberta"  # ← RoBERTa model folder
MAX_LEN   = 384
BATCH_SIZE = 64


# ====== DATASET ======
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        for k, v in self.labels.items():
            item[k] = v[idx]
        return item


# ====== MODEL  — must exactly match train_roberta.py ======
class MultiTaskRoBERTa(nn.Module):
    """
    Identical architecture to the training script:
    - GELU activations  (NOT ReLU)
    - Scaled sigmoid output  →  predictions in [rating_min, rating_max]
    - HIDDEN_DIM loaded from config.json at runtime
    - RoBERTa encoder (125M params)
    """

    def __init__(self, base_model: str, aspect_names: List[str],
                 hidden_dim: int = 384, dropout: float = 0.35,
                 rating_min: float = 1.0, rating_max: float = 5.0):
        super().__init__()
        self.rating_min = rating_min
        self.rating_max = rating_max

        self.encoder   = AutoModel.from_pretrained(base_model)
        encoder_dim    = self.encoder.config.hidden_size          # 768

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

        self.aspect_heads = nn.ModuleDict({
            a: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for a in aspect_names
        })

    def _scale_output(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid → [rating_min, rating_max]  e.g. 3.1, 4.2, 2.8"""
        return torch.sigmoid(x) * (self.rating_max - self.rating_min) + self.rating_min

    def forward(self, input_ids, attention_mask):
        encoded  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled   = encoded.last_hidden_state[:, 0]        # [CLS] token
        features = self.feature_extractor(pooled)

        out = {"overall": self._scale_output(self.overall_head(features).squeeze(-1))}
        for aspect, head in self.aspect_heads.items():
            out[aspect] = self._scale_output(head(features).squeeze(-1))
        return out


# ====== METRICS ======
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Dict:
    """
    Full metric suite for continuous rating prediction.

    PRIMARY ACCURACY = within ±0.5 stars
      pred=3.8, true=4.0  → error 0.2  → CORRECT  ✓
      pred=2.5, true=4.0  → error 1.5  → WRONG     ✗
    """
    mask   = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"name": name, "n_samples": 0}

    errors = np.abs(y_pred - y_true)

    r2        = r2_score(y_true, y_pred)
    mae       = mean_absolute_error(y_true, y_pred)
    rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson_r, pearson_p = pearsonr(y_true, y_pred)

    acc_0_3   = float(np.mean(errors <= 0.3) * 100)   # strict
    acc_0_5   = float(np.mean(errors <= 0.5) * 100)   # PRIMARY accuracy
    acc_1_0   = float(np.mean(errors <= 1.0) * 100)   # lenient
    acc_exact = float(np.mean(np.round(y_pred) == np.round(y_true)) * 100)

    return {
        "name":             name,
        "n_samples":        int(len(y_true)),
        "r2":               float(r2),
        "mae":              float(mae),
        "rmse":             rmse,
        "pearson_r":        float(pearson_r),
        "pearson_p":        float(pearson_p),
        "acc_within_0.3":   acc_0_3,
        "acc_within_0.5":   acc_0_5,
        "acc_within_1.0":   acc_1_0,
        "acc_exact_star":   acc_exact,
        "mean_pred":        float(np.mean(y_pred)),
        "mean_true":        float(np.mean(y_true)),
    }


def r2_label(r2: float) -> str:
    if r2 >= 0.90: return "Excellent"
    if r2 >= 0.80: return "Good"
    if r2 >= 0.70: return "Acceptable"
    if r2 >= 0.60: return "Moderate"
    if r2 >= 0.50: return "Fair"
    return "Poor"

def acc_label(acc: float) -> str:
    if acc >= 75: return "Excellent"
    if acc >= 60: return "Good"
    if acc >= 45: return "Acceptable"
    if acc >= 35: return "Fair"
    return "Poor"

def print_divider(char="=", width=68):
    print(char * width)

def print_metric_block(m: Dict, title: str):
    """Prints a clean metric block — fixed formatting"""
    W = 68
    print()
    print_divider("-", W)
    print(f"  {title}")
    print_divider("-", W)

    if m.get("n_samples", 0) == 0:
        print("  No samples found.")
        return

    # Accuracy section
    print(f"  ACCURACY METRICS  (primary = within ±0.5 stars)")
    print()
    print(f"    Within ±0.3 stars : {m['acc_within_0.3']:6.2f}%   [{acc_label(m['acc_within_0.3'])}]")
    print(f"    Within ±0.5 stars : {m['acc_within_0.5']:6.2f}%   [{acc_label(m['acc_within_0.5'])}]  ← quote this")
    print(f"    Within ±1.0 stars : {m['acc_within_1.0']:6.2f}%   [{acc_label(m['acc_within_1.0'])}]")
    print(f"    Exact star match  : {m['acc_exact_star']:6.2f}%")
    print()

    # Regression section
    print(f"  REGRESSION METRICS")
    print()
    print(f"    R² Score          : {m['r2']:7.4f}   [{r2_label(m['r2'])}]")
    print(f"    MAE               : {m['mae']:7.4f}   (avg stars off per prediction)")
    print(f"    RMSE              : {m['rmse']:7.4f}")
    print(f"    Pearson r         : {m['pearson_r']:7.4f}   (p = {m['pearson_p']:.2e})")
    print()
    print(f"    Mean predicted    : {m['mean_pred']:.3f}   (mean true: {m['mean_true']:.3f})")
    print_divider("-", W)


# ====== MAIN ======
def main():
    print_divider()
    print("  ROBERTA MODEL EVALUATION  —  ACCURACY & REGRESSION METRICS")
    print_divider()

    # ── Step 0: Load config.json ──────────────────────────────────────────
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(config_path):
        print(f"\n  ERROR: {config_path} not found.")
        print("  Make sure train_roberta.py has completed successfully.")
        exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    # Read everything from config — safe defaults if key missing
    hidden_dim   = cfg.get("hidden_dim",    384)
    dropout      = cfg.get("dropout",       0.35)
    rating_min   = cfg.get("rating_min",    1.0)
    rating_max   = cfg.get("rating_max",    5.0)
    aspect_names = cfg.get("aspect_names",  ["work_life_balance", "company_culture",
                                              "career_growth", "salary_benefits"])
    saved_r2     = cfg.get("best_val_r2",   None)
    base_model   = cfg.get("base_model",    "roberta-base")

    print(f"\n  Config loaded from : {config_path}")
    print(f"  base_model         : {base_model}")
    print(f"  hidden_dim         : {hidden_dim}")
    print(f"  dropout            : {dropout}")
    print(f"  rating range       : [{rating_min}, {rating_max}]")
    print(f"  aspects            : {aspect_names}")
    if saved_r2 is not None:
        print(f"  Best R² (training) : {saved_r2:.4f}")

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"  ERROR: {CSV_PATH} not found. Run preprocess_augment.py first.")
        exit(1)

    df = pd.read_csv(CSV_PATH)

    for col in ["overall_rating"] + aspect_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["text"].str.len() > 0]
    df = df[~df["overall_rating"].isna()]
    print(f"  Total samples : {len(df):,}")

    # Use same split as training
    _, val_df = train_test_split(df, test_size=0.15, random_state=SEED)
    print(f"  Val samples   : {len(val_df):,}")

    # Confirm ratings are continuous
    sample_true = val_df["overall_rating"].sample(8, random_state=SEED).round(1).tolist()
    print(f"  Sample true ratings : {sample_true}")
    all_int = all(v == int(v) for v in val_df["overall_rating"].dropna())
    if all_int:
        print("  WARNING: All ratings are integers — run preprocess_augment.py for continuous ratings")
    else:
        print("  ✓ Ratings are continuous (good)")

    # ── Step 2: Tokenize ──────────────────────────────────────────────────
    print("\n[2/5] Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    val_enc = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    def build_labels(frame):
        labels = {
            "overall": torch.tensor(frame["overall_rating"].values, dtype=torch.float)
        }
        for a in aspect_names:
            if a in frame.columns:
                labels[a] = torch.tensor(frame[a].values, dtype=torch.float)
            else:
                labels[a] = torch.full((len(frame),), float("nan"))
        return labels

    val_dataset = ReviewDataset(val_enc, build_labels(val_df))
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             pin_memory=True, num_workers=0)

    # ── Step 3: Load model ────────────────────────────────────────────────
    print("\n[3/5] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    best_pt = os.path.join(MODEL_DIR, "best.pt")
    if not os.path.exists(best_pt):
        print(f"  ERROR: {best_pt} not found.")
        print("  Training may not have completed a full epoch yet.")
        exit(1)

    model = MultiTaskRoBERTa(
        base_model, aspect_names,
        hidden_dim=hidden_dim,
        dropout=dropout,
        rating_min=rating_min,
        rating_max=rating_max,
    ).to(device)

    state_dict = torch.load(best_pt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  ✓ Model loaded successfully!")

    # ── Step 4: Predict ───────────────────────────────────────────────────
    print("\n[4/5] Running predictions...")

    all_preds  = {k: [] for k in ["overall"] + aspect_names}
    all_labels = {k: [] for k in ["overall"] + aspect_names}

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids, attention_mask)

            for key in ["overall"] + aspect_names:
                all_preds[key].extend(outputs[key].cpu().numpy().tolist())
                all_labels[key].extend(batch[key].numpy().tolist())

    for key in all_preds:
        all_preds[key]  = np.array(all_preds[key],  dtype=np.float32)
        all_labels[key] = np.array(all_labels[key], dtype=np.float32)

    # Print 10 sample pred/true pairs with ✓ / ~ / ✗ indicators
    print("\n  Sample predictions  (✓ = within ±0.5   ~ = within ±1.0   ✗ = off by >1.0)")
    sample_idx = np.random.default_rng(SEED).choice(len(all_preds["overall"]), 10, replace=False)
    for i in sample_idx:
        p    = float(all_preds["overall"][i])
        t    = float(all_labels["overall"][i])
        err  = abs(p - t)
        flag = "✓" if err <= 0.5 else ("~" if err <= 1.0 else "✗")
        print(f"    {flag}  Predicted: {p:.2f}   True: {t:.1f}   Error: {err:.2f}")

    # ── Step 5: Metrics ───────────────────────────────────────────────────
    print("\n[5/5] Computing metrics...")
    print_divider()
    print("  RESULTS")
    print_divider()

    all_metrics = []

    # Overall rating
    m_overall = compute_metrics(all_labels["overall"], all_preds["overall"], "Overall Rating")
    all_metrics.append(m_overall)
    print_metric_block(m_overall, "OVERALL RATING")

    # Aspects summary table
    print(f"\n  ASPECT SCORES — SUMMARY")
    print()
    print(f"  {'Aspect':<26} {'±0.5 Acc':>10} {'±1.0 Acc':>10} {'MAE':>8} {'R²':>8}")
    print(f"  {'-' * 64}")

    for aspect in aspect_names:
        label = aspect.replace("_", " ").title()
        m     = compute_metrics(all_labels[aspect], all_preds[aspect], label)
        all_metrics.append(m)
        if m["n_samples"] > 0:
            print(f"  {label:<26} {m['acc_within_0.5']:>9.2f}%"
                  f" {m['acc_within_1.0']:>9.2f}%"
                  f" {m['mae']:>8.4f}"
                  f" {m['r2']:>8.4f}")

    # ── Plain English Summary ─────────────────────────────────────────────
    acc  = m_overall["acc_within_0.5"]
    acc1 = m_overall["acc_within_1.0"]
    mae  = m_overall["mae"]
    r2   = m_overall["r2"]

    print()
    print_divider()
    print("  PLAIN ENGLISH SUMMARY")
    print_divider()
    print(f"""
  ► {acc:.2f}% of predictions are within ±0.5 stars of the true rating
      e.g. predicting 3.8 when true is 4.0  →  correct  ✓

  ► {acc1:.2f}% of predictions are within ±1.0 stars of the true rating

  ► On average the model is off by {mae:.3f} stars per review

  ► R² of {r2:.4f} means the model explains {r2*100:.1f}% of rating variance

  ✅ Quote as accuracy:  "{acc:.1f}% within ±0.5 stars"
""")

    # ── Recommendations ───────────────────────────────────────────────────
    print_divider()
    print("  COMPARISON WITH DISTILBERT")
    print_divider()
    print(f"""
  DistilBERT Results:   R²=0.4938, Accuracy=51.23%
  RoBERTa Results:      R²={r2:.4f}, Accuracy={acc:.2f}%
  
  Improvement:          ΔR²={r2-0.4938:+.4f}, ΔAcc={acc-51.23:+.2f}%
""")

    if r2 > 0.55:
        print("  🟢 EXCELLENT! RoBERTa shows significant improvement!")
        print("     → Use RoBERTa for production")
    elif r2 > 0.50:
        print("  🟡 Modest improvement, but within noise margin.")
        print("     → Can use RoBERTa or stick with DistilBERT (faster)")
    else:
        print("  🔴 No improvement or worse than DistilBERT.")
        print("     → Stick with DistilBERT (faster, same accuracy)")

    # ── Save ──────────────────────────────────────────────────────────────
    print()
    print_divider()
    out_path = os.path.join(MODEL_DIR, "evaluation_results.csv")
    pd.DataFrame(all_metrics).to_csv(out_path, index=False)
    print(f"  ✓ Results saved to : {out_path}")
    print_divider()


if __name__ == "__main__":
    main()
