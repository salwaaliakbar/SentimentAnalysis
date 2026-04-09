"""
RoBERTa Training Script - Continuous Rating Prediction
=======================================================

This is the ROBERTA version of train_advanced.py

Key improvements over DistilBERT:
1. RoBERTa-base has 125M parameters (vs DistilBERT's 66M)
2. Better language understanding = higher R² (expected +0.04-0.08)
3. Same multi-task architecture and loss functions
4. Output still clamped to [1.0, 5.0] range via sigmoid scaling
5. Everything else identical to train_advanced.py

Expected Results:
  - DistilBERT: R² ≈ 0.49, Accuracy ≈ 51%
  - RoBERTa:    R² ≈ 0.53-0.58, Accuracy ≈ 54-58% (estimated)
  - Improvement: +4-8% relative gain

Training Time: ~4-6 hours (CUDA GPU)
"""

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
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import json

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
OUT_DIR = "model_output_roberta"  # ← NEW: RoBERTa output directory

# Model — ONE source of truth for HIDDEN_DIM (saved to config.json for eval script)
BASE_MODEL = "roberta-base"  # ← CHANGED: roberta-base instead of distilbert-base-uncased
MAX_LEN = 384
DROPOUT = 0.35
HIDDEN_DIM = 384           # ← Single definition. Eval script reads this from config.json

# Training
EPOCHS = 60
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 2  # Effective batch = 64
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION

# Optimization
LR = 8e-6                  # Lower LR → more stable for continuous regression
WEIGHT_DECAY = 0.02
WARMUP_RATIO = 0.10

# Loss
ASPECT_LOSS_WEIGHT = 0.4   # Slightly lower — overall_rating is primary target
USE_HUBER = True           # Huber loss is better than FocalMSE for continuous regression
HUBER_DELTA = 0.5

# Early stopping
PATIENCE = 10
MIN_DELTA = 5e-5

# Rating range (for output clamping)
RATING_MIN = 1.0
RATING_MAX = 5.0


# ====== LOSS FUNCTIONS ======
class HuberLoss(nn.Module):
    """
    Huber loss: MSE for small errors, MAE for large errors.
    Better than FocalMSE for continuous regression — does NOT push
    predictions toward the mean as aggressively.
    """
    def __init__(self, delta: float = 0.5):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2 / self.delta,
            diff - 0.5 * self.delta
        )
        return loss.mean()


def masked_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    """Huber loss with NaN masking for aspect scores"""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = torch.abs(pred[mask] - target[mask])
    loss = torch.where(diff < delta, 0.5 * diff ** 2 / delta, diff - 0.5 * delta)
    return loss.mean()


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
class MultiTaskRoBERTa(nn.Module):
    """
    Multi-task RoBERTa for continuous rating prediction.

    Key design choices:
    - Output uses scaled sigmoid: sigmoid(x) * 4 + 1 → range [1, 5]
      This ensures predictions like 3.1, 4.2, 2.8 (not just integers)
    - Separate heads per aspect for better specialization
    - HIDDEN_DIM is passed in and saved to config — no more mismatches
    - RoBERTa-base encoder for better language understanding
    """

    def __init__(self, base_model: str, aspect_names: List[str],
                 hidden_dim: int = 384, dropout: float = 0.35):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model)
        encoder_dim = self.encoder.config.hidden_size  # 768 for RoBERTa-base

        # Freeze early transformer layers (keep last 8 unfrozen)
        for i, param in enumerate(self.encoder.parameters()):
            if i < len(list(self.encoder.parameters())) - 8:
                param.requires_grad = False

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Overall rating head
        self.overall_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Aspect heads
        self.aspect_heads = nn.ModuleDict({
            a: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for a in aspect_names
        })

    def _scale_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale raw logit to [1.0, 5.0] range using sigmoid.
        sigmoid(x) maps (-inf, +inf) → (0, 1)
        * 4 + 1 maps (0, 1) → (1, 5)
        This produces continuous predictions like 3.1, 4.2, 2.8
        """
        return torch.sigmoid(x) * (RATING_MAX - RATING_MIN) + RATING_MIN

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.last_hidden_state[:, 0]  # [CLS] token

        features = self.feature_extractor(pooled)

        logits = {
            "overall": self._scale_output(self.overall_head(features).squeeze(-1))
        }
        for aspect, head in self.aspect_heads.items():
            logits[aspect] = self._scale_output(head(features).squeeze(-1))

        return logits


# ====== HELPERS ======
def find_column(df: pd.DataFrame, name: str, fallbacks: List[str] = None) -> str:
    candidates = [name] + (fallbacks or [])
    lowered = {c.strip().lower(): c for c in df.columns}
    for key in candidates:
        if key.strip().lower() in lowered:
            return lowered[key.strip().lower()]
    return ""


# ====== SETUP ======
os.makedirs(OUT_DIR, exist_ok=True)

print("="*80)
print("MULTI-TASK ROBERTA — CONTINUOUS RATING PREDICTION")
print("="*80)
print(f"\nConfiguration:")
print(f"  Dataset:             {CSV_PATH}")
print(f"  Model:               {BASE_MODEL}")
print(f"  Epochs:              {EPOCHS}")
print(f"  Batch Size:          {BATCH_SIZE} (effective: {EFFECTIVE_BATCH_SIZE})")
print(f"  Learning Rate:       {LR}")
print(f"  Warmup Ratio:        {WARMUP_RATIO}")
print(f"  Dropout:             {DROPOUT}")
print(f"  Hidden Dim:          {HIDDEN_DIM}")
print(f"  Loss:                {'Huber' if USE_HUBER else 'MSE'} (delta={HUBER_DELTA})")
print(f"  Aspect Loss Weight:  {ASPECT_LOSS_WEIGHT}")
print(f"  Output Range:        [{RATING_MIN}, {RATING_MAX}] (continuous via scaled sigmoid)")
print(f"  Device:              {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("="*80)

# Load data
print("\n[1/4] Loading dataset...")
if not os.path.exists(CSV_PATH):
    print(f"  ERROR: {CSV_PATH} not found!")
    print(f"  Run: python preprocess_augment.py")
    exit(1)

df = pd.read_csv(CSV_PATH)
print(f"  Loaded {len(df):,} samples")

if 'text' not in df.columns:
    print("  ERROR: 'text' column not found")
    exit(1)

aspect_names = ["work_life_balance", "company_culture", "career_growth", "salary_benefits"]

# Ensure numeric
for col in ['overall_rating'] + aspect_names:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["text"].str.len() > 0]
df = df[~df["overall_rating"].isna()]
print(f"  After filtering: {len(df):,} samples")

# Show sample ratings to confirm continuity
sample_ratings = df['overall_rating'].sample(10, random_state=SEED).round(1).tolist()
print(f"  Sample overall_ratings: {sample_ratings}")
print(f"  Rating range: [{df['overall_rating'].min():.1f}, {df['overall_rating'].max():.1f}]")
print(f"  Rating mean: {df['overall_rating'].mean():.3f}, std: {df['overall_rating'].std():.3f}")

# Verify ratings are actually continuous (not all integers)
is_all_int = all(df['overall_rating'].dropna().apply(lambda x: x == int(x)))
if is_all_int:
    print("\n  WARNING: All ratings appear to be integers!")
    print("  Continuous predictions may be limited. Run preprocess_augment.py first.")
else:
    print(f"  ✓ Ratings are continuous (not all integers)")

# Split
train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)
print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

# Tokenize
print("\n[2/4] Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

train_enc = tokenizer(
    train_df["text"].tolist(),
    truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
)
val_enc = tokenizer(
    val_df["text"].tolist(),
    truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
)

def build_labels(frame):
    labels = {"overall": torch.tensor(frame["overall_rating"].values, dtype=torch.float)}
    for a in aspect_names:
        if a in frame.columns:
            labels[a] = torch.tensor(frame[a].values, dtype=torch.float)
    return labels

train_dataset = ReviewDataset(train_enc, build_labels(train_df))
val_dataset = ReviewDataset(val_enc, build_labels(val_df))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=0)

# Initialize model
print("\n[3/4] Initializing model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Using device: {device}")

model = MultiTaskRoBERTa(BASE_MODEL, aspect_names, HIDDEN_DIM, DROPOUT).to(device)

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,}")

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8
)

total_steps = (len(train_loader) * EPOCHS) // GRADIENT_ACCUMULATION
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
print(f"  Total steps: {total_steps:,} | Warmup: {warmup_steps:,}")

loss_fn = HuberLoss(delta=HUBER_DELTA)

# Tracking
best_val_loss = float("inf")
best_val_r2 = float("-inf")
patience_ctr = 0
training_history = {
    'epoch': [], 'train_loss': [], 'val_loss': [],
    'val_overall_r2': [], 'val_overall_mae': []
}

# Training
print(f"\n[4/4] Training...")
print("="*80)

scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

for epoch in range(1, EPOCHS + 1):
    # ====== TRAIN ======
    model.train()
    total_loss = 0
    step_count = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in batch.items()
                  if k not in ["input_ids", "attention_mask"]}

        if scaler:
            with torch.cuda.amp.autocast():
                out = model(input_ids, attention_mask)
                loss = loss_fn(out["overall"], labels["overall"])
                for a in aspect_names:
                    if a in labels:
                        loss = loss + ASPECT_LOSS_WEIGHT * masked_huber(out[a], labels[a], HUBER_DELTA)
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
            loss = loss_fn(out["overall"], labels["overall"])
            for a in aspect_names:
                if a in labels:
                    loss = loss + ASPECT_LOSS_WEIGHT * masked_huber(out[a], labels[a], HUBER_DELTA)
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION
        step_count += 1

    avg_train_loss = total_loss / step_count

    # ====== VALIDATION ======
    model.eval()
    val_loss = 0
    val_count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = {k: v.to(device) for k, v in batch.items()
                            if k not in ["input_ids", "attention_mask"]}

            out = model(input_ids, attention_mask)
            loss = loss_fn(out["overall"], batch_labels["overall"])
            for a in aspect_names:
                if a in batch_labels:
                    loss = loss + ASPECT_LOSS_WEIGHT * masked_huber(out[a], batch_labels[a], HUBER_DELTA)

            val_loss += loss.item()
            val_count += 1

            all_preds.append(out["overall"].cpu().numpy())
            all_labels.append(batch_labels["overall"].cpu().numpy())

    avg_val_loss = val_loss / val_count

    # Metrics
    try:
        from sklearn.metrics import r2_score, mean_absolute_error
        all_preds_np = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)
        val_r2 = r2_score(all_labels_np, all_preds_np)
        val_mae = mean_absolute_error(all_labels_np, all_preds_np)

        # Show sample predictions every 10 epochs
        if epoch % 10 == 0 or epoch <= 3:
            sample_idx = np.random.choice(len(all_preds_np), 5, replace=False)
            sample_pairs = [(round(float(all_preds_np[i]), 2), round(float(all_labels_np[i]), 1))
                            for i in sample_idx]
            print(f"  Sample (pred→true): {sample_pairs}")
    except Exception:
        val_r2 = 0.0
        val_mae = 0.0

    # Track
    training_history['epoch'].append(epoch)
    training_history['train_loss'].append(avg_train_loss)
    training_history['val_loss'].append(avg_val_loss)
    training_history['val_overall_r2'].append(val_r2)
    training_history['val_overall_mae'].append(val_mae)

    # Save best
    improvement = ""
    if avg_val_loss < best_val_loss - MIN_DELTA:
        improvement = " ← BEST!"
        best_val_loss = avg_val_loss
        best_val_r2 = val_r2
        patience_ctr = 0
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
    else:
        patience_ctr += 1

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train: {avg_train_loss:.4f} | "
          f"Val: {avg_val_loss:.4f} | R²: {val_r2:.4f} | MAE: {val_mae:.3f}{improvement}")

    if patience_ctr >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}.")
        break

# ====== SAVE ======
print("\n" + "="*80)
tokenizer.save_pretrained(OUT_DIR)

# Save config — HIDDEN_DIM is stored here so evaluate script reads it automatically
config = {
    'base_model': BASE_MODEL,
    'hidden_dim': HIDDEN_DIM,        # ← Eval script must read this!
    'dropout': DROPOUT,
    'aspect_names': aspect_names,
    'rating_min': RATING_MIN,
    'rating_max': RATING_MAX,
    'epochs_trained': epoch,
    'best_val_loss': float(best_val_loss),
    'best_val_r2': float(best_val_r2),
    'hyperparameters': {
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'effective_batch_size': EFFECTIVE_BATCH_SIZE,
        'dropout': DROPOUT,
        'hidden_dim': HIDDEN_DIM,
        'aspect_loss_weight': ASPECT_LOSS_WEIGHT,
        'huber_delta': HUBER_DELTA,
    }
}

with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

pd.DataFrame(training_history).to_csv(os.path.join(OUT_DIR, 'training_history.csv'), index=False)

print(f"✓ Training complete!")
print(f"  Best validation loss: {best_val_loss:.4f}")
print(f"  Best validation R²:   {best_val_r2:.4f}")
print(f"  Best validation MAE:  {min(training_history['val_overall_mae']):.4f}")
print(f"  Model saved to: {OUT_DIR}")
print(f"\n  config.json contains HIDDEN_DIM={HIDDEN_DIM} — your evaluate script")
print(f"  MUST read this value instead of hardcoding it!")
print("="*80)
