"""
Data Preprocessing & Augmentation Script
==========================================
Comprehensive preprocessing including:
- Text normalization & cleaning
- Data augmentation (synonym replacement)
- Continuous rating generation (e.g., 3.1, 4.2)
- Outlier handling
- Soft balancing (NO hard integer binning - preserves continuous distribution)
- Dataset statistics
"""

import os
import random
import numpy as np
import pandas as pd
from typing import List
import re

# ====== CONFIG ======
CSV_PATH = "employee_reviews.csv"
OUTPUT_PATH = "employee_reviews_processed.csv"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ====== SYNONYMS FOR AUGMENTATION ======
SYNONYMS = {
    'great': ['excellent', 'wonderful', 'fantastic', 'amazing', 'superb', 'outstanding'],
    'good': ['nice', 'fine', 'decent', 'okay', 'satisfactory', 'pleasant'],
    'bad': ['poor', 'terrible', 'awful', 'horrible', 'dismal', 'inferior'],
    'low': ['minimal', 'small', 'insufficient', 'inadequate', 'limited'],
    'high': ['generous', 'substantial', 'significant', 'considerable'],
    'work': ['job', 'role', 'position', 'task', 'assignment', 'duties'],
    'team': ['group', 'crew', 'staff', 'colleagues', 'department'],
    'company': ['firm', 'organization', 'business', 'employer', 'corporation'],
    'culture': ['environment', 'atmosphere', 'vibe', 'ethos', 'setting'],
    'management': ['leadership', 'administration', 'direction', 'supervision', 'oversight'],
    'manager': ['boss', 'supervisor', 'leader', 'director', 'superior'],
    'pay': ['salary', 'compensation', 'wage', 'earnings', 'income'],
    'benefits': ['perks', 'rewards', 'incentives', 'allowances', 'bonuses'],
    'growth': ['advancement', 'progression', 'development', 'improvement', 'expansion'],
    'opportunity': ['chance', 'possibility', 'opening', 'prospect'],
    'learning': ['education', 'training', 'knowledge', 'skill', 'development'],
    'flexibility': ['freedom', 'autonomy', 'independence', 'control'],
    'challenging': ['difficult', 'demanding', 'taxing', 'rigorous'],
    'support': ['help', 'assistance', 'backing', 'aid'],
    'communication': ['interaction', 'dialogue', 'discussion', 'feedback'],
}

# ====== HELPERS ======
def find_column(df: pd.DataFrame, name: str, fallbacks: List[str] = None) -> str:
    candidates = [name] + (fallbacks or [])
    lowered = {c.strip().lower(): c for c in df.columns}
    for key in candidates:
        if key.strip().lower() in lowered:
            return lowered[key.strip().lower()]
    return ""


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    contractions_dict = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "can not", "couldn't": "could not", "won't": "will not",
        "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have",
        "they've": "they have", "i'll": "i will", "you'll": "you will",
        "he'll": "he will", "she'll": "she will", "it'll": "it will",
        "we'll": "we will", "they'll": "they will"
    }
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-z0-9\s\.\!\?\,\-\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def make_continuous_rating(rating: float, noise_std: float = 0.15) -> float:
    """
    Convert discrete integer rating (1-5) to continuous float.
    Adds small Gaussian noise to simulate real-world continuous scores
    like 3.1, 4.2, 2.8 etc. Clamps to [1.0, 5.0].

    Why this works:
    - Your source data has integer stars (1, 2, 3, 4, 5)
    - The model needs to learn to predict continuous values
    - Without this, the model will always predict near integers
    - Noise std of 0.15 means ~95% of values stay within ±0.3 of original
    """
    if pd.isna(rating):
        return np.nan
    noise = np.random.normal(0, noise_std)
    continuous = float(rating) + noise
    continuous = round(continuous, 1)  # 3.1, 3.2, 4.7 etc
    continuous = max(1.0, min(5.0, continuous))
    return continuous


def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3.5) -> pd.DataFrame:
    if column not in df.columns or df[column].isna().all():
        return df
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    mask = z_scores < threshold
    original_len = len(df)
    df_clean = df[df[column].isna() | (df.index.isin(df[df[column].notna()].index[mask]))]
    removed = original_len - len(df_clean)
    if removed > 0:
        print(f"  Removed {removed} outliers from {column}")
    return df_clean


def augment_text(text: str) -> str:
    if not text or len(text.split()) < 5:
        return text
    words = text.split()
    augmented_words = []
    for word in words:
        word_lower = word.lower().strip('.,!?')
        if word_lower in SYNONYMS and random.random() < 0.3:
            replacement = random.choice(SYNONYMS[word_lower])
            if word[0].isupper():
                replacement = replacement.capitalize()
            augmented_words.append(replacement)
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)


def rating_to_sentiment_class(rating: float) -> int:
    """
    Map rating to 3-way sentiment class.
    0=negative, 1=neutral, 2=positive
    """
    if pd.isna(rating):
        return 1
    if rating <= 2.5:
        return 0
    if rating >= 3.5:
        return 2
    return 1


def create_augmented_samples(df: pd.DataFrame, augmentation_factor: float = 0.25) -> pd.DataFrame:
    print("\n[3/5] Creating augmented samples...")
    augmented_rows = []
    n_augmented = int(len(df) * augmentation_factor)
    indices_to_augment = np.random.choice(len(df), size=n_augmented, replace=False)
    for idx in indices_to_augment:
        row = df.iloc[idx].copy()
        if 'text' in row.index:
            row['text'] = augment_text(row['text'])
            # Slightly perturb continuous ratings for augmented samples
            for col in ['overall_rating', 'work_life_balance', 'company_culture',
                        'career_growth', 'salary_benefits']:
                if col in row.index and not pd.isna(row[col]):
                    noise = np.random.normal(0, 0.08)
                    row[col] = round(max(1.0, min(5.0, row[col] + noise)), 1)
            augmented_rows.append(row)
    df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print(f"  Created {len(augmented_rows)} augmented samples")
    print(f"  Total samples: {len(df)} → {len(df_augmented)}")
    return df_augmented


def create_targeted_neutral_mixed_augmentation(df: pd.DataFrame, target_factor: float = 0.45) -> pd.DataFrame:
    """
    Oversample neutral and mixed-sentiment samples.

    mixed-sentiment heuristic:
    - neutral overall rating, or
    - large disagreement across aspect ratings.
    """
    print("\n[4/6] Creating targeted neutral/mixed augmentation...")
    work_df = df.copy()
    work_df["sentiment_class"] = work_df["overall_rating"].apply(rating_to_sentiment_class)

    aspect_cols = ["work_life_balance", "company_culture", "career_growth", "salary_benefits"]
    available_aspects = [c for c in aspect_cols if c in work_df.columns]

    aspect_std = work_df[available_aspects].std(axis=1, skipna=True) if available_aspects else pd.Series(np.zeros(len(work_df)))
    neutral_mask = work_df["sentiment_class"] == 1
    mixed_mask = neutral_mask | (aspect_std >= 0.95)
    target_pool = work_df[mixed_mask]

    if len(target_pool) == 0:
        print("  No neutral/mixed samples detected; skipping targeted augmentation")
        return df

    n_target = int(len(df) * target_factor)
    sampled = target_pool.sample(n=n_target, replace=True, random_state=SEED).copy()

    for i in range(len(sampled)):
        sampled.iloc[i, sampled.columns.get_loc("text")] = augment_text(sampled.iloc[i]["text"])
        for col in ["overall_rating"] + available_aspects:
            if not pd.isna(sampled.iloc[i][col]):
                sampled.iloc[i, sampled.columns.get_loc(col)] = round(
                    max(1.0, min(5.0, sampled.iloc[i][col] + np.random.normal(0, 0.06))),
                    1,
                )

    combined = pd.concat([df, sampled.drop(columns=["sentiment_class"])], ignore_index=True)
    print(f"  Added targeted samples: {len(sampled)}")
    print(f"  Total samples: {len(df)} → {len(combined)}")
    return combined


def soft_balance_ratings(df: pd.DataFrame, column: str = 'overall_rating') -> pd.DataFrame:
    """
    Soft balancing using continuous bins.
    Does NOT use hard integer bins — that would destroy continuous distributions.
    Only lightly upsamples extremely underrepresented ranges.
    """
    print("\n[5/6] Soft-balancing rating distribution (continuous-aware)...")
    if column not in df.columns:
        return df

    bins = np.arange(0.75, 5.51, 0.5)
    bin_labels = [f"{b:.2f}-{b+0.5:.2f}" for b in bins[:-1]]
    df = df.copy()
    df['_bin'] = pd.cut(df[column], bins=bins, labels=bin_labels)

    bin_counts = df['_bin'].value_counts()
    print(f"  Rating bin distribution:")
    for b in sorted(bin_counts.index):
        print(f"    {b}: {bin_counts[b]} samples")

    median_count = bin_counts.median()
    threshold = median_count * 0.3  # only upsample bins with < 30% of median

    balanced_dfs = [df]
    for bin_label in bin_counts.index:
        count = bin_counts[bin_label]
        if count < threshold:
            bin_df = df[df['_bin'] == bin_label]
            needed = int(threshold) - int(count)
            if len(bin_df) > 0 and needed > 0:
                upsampled = bin_df.sample(n=needed, replace=True, random_state=SEED)
                balanced_dfs.append(upsampled)
                print(f"    Upsampled bin {bin_label}: +{needed} samples")

    df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=SEED)
    df_balanced = df_balanced.drop(columns=['_bin'])
    print(f"  Total after balancing: {len(df_balanced):,}")
    return df_balanced


def balance_sentiment_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Lightly rebalance 3-way sentiment classes derived from rating."""
    print("\n[6/6] Balancing sentiment classes (negative/neutral/positive)...")
    work_df = df.copy()
    work_df["sentiment_class"] = work_df["overall_rating"].apply(rating_to_sentiment_class)

    class_counts = work_df["sentiment_class"].value_counts().to_dict()
    print(f"  Class distribution before: {class_counts}")

    target = int(np.percentile(list(class_counts.values()), 75))
    buckets = []
    for cls in [0, 1, 2]:
        bucket = work_df[work_df["sentiment_class"] == cls]
        if len(bucket) == 0:
            continue
        if len(bucket) < target:
            n_add = target - len(bucket)
            bucket = pd.concat([bucket, bucket.sample(n=n_add, replace=True, random_state=SEED)], ignore_index=True)
        buckets.append(bucket)

    balanced = pd.concat(buckets, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    after_counts = balanced["sentiment_class"].value_counts().to_dict()
    print(f"  Class distribution after:  {after_counts}")
    return balanced.drop(columns=["sentiment_class"])


def compute_statistics(df: pd.DataFrame) -> None:
    print("\n[5/5] Dataset Statistics")
    print("="*70)
    print(f"\nTotal samples: {len(df):,}")
    print(f"Average text length: {df['text'].str.len().mean():.1f} characters")
    print(f"Min text length: {df['text'].str.len().min()}")
    print(f"Max text length: {df['text'].str.len().max()}")
    print(f"\nOverall Rating (continuous) Stats:")
    print(f"  Mean:  {df['overall_rating'].mean():.3f}")
    print(f"  Std:   {df['overall_rating'].std():.3f}")
    print(f"  Min:   {df['overall_rating'].min():.1f}")
    print(f"  Max:   {df['overall_rating'].max():.1f}")
    sample_vals = df['overall_rating'].dropna().sample(min(10, len(df)), random_state=SEED).round(1).tolist()
    print(f"  Sample values: {sample_vals}")
    aspects = ['work_life_balance', 'company_culture', 'career_growth', 'salary_benefits']
    print(f"\nAspect Scores (continuous):")
    for aspect in aspects:
        if aspect in df.columns:
            valid = df[aspect].notna().sum()
            mean_val = df[aspect].mean()
            std_val = df[aspect].std()
            print(f"  {aspect:20s}: {valid:6,} samples, mean={mean_val:.2f}, std={std_val:.2f}")


# ====== MAIN ======
def main():
    print("="*70)
    print("DATA PREPROCESSING & AUGMENTATION (Continuous Ratings)")
    print("="*70)

    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Original samples: {len(df):,}")

    col_summary = find_column(df, "summary")
    col_pros = find_column(df, "pros")
    col_cons = find_column(df, "cons")
    col_advice = find_column(df, "advice-to-mgmt")
    col_overall = find_column(df, "overall-ratings")
    col_wlb = find_column(df, "work-balance-stars")
    col_culture = find_column(df, "culture-values-stars")
    col_career = find_column(df, "career-opportunities-stars", ["carrer-opportunities-stars"])
    col_comp = find_column(df, "comp-benefit-stars", ["comp-benefit-stars.1"])

    def combine_text(row):
        parts = []
        for col in [col_summary, col_pros, col_cons, col_advice]:
            if col and pd.notna(row[col]):
                parts.append(str(row[col]))
        return " ".join(parts)

    df["text"] = df.apply(combine_text, axis=1)
    df["text"] = df["text"].apply(normalize_text)

    # Convert raw columns
    df["overall_rating_raw"] = pd.to_numeric(df[col_overall], errors="coerce")
    for col_name, col_src in [
        ("work_life_balance", col_wlb),
        ("company_culture", col_culture),
        ("career_growth", col_career),
        ("salary_benefits", col_comp),
    ]:
        if col_src:
            df[col_name + "_raw"] = pd.to_numeric(df[col_src], errors="coerce")
        else:
            df[col_name + "_raw"] = np.nan

    # Filter
    df = df[df["text"].str.len() > 30]
    df = df[~df["overall_rating_raw"].isna()]
    print(f"  After filtering: {len(df):,}")

    # ====== KEY STEP: Make continuous ratings ======
    print("\n[2/5] Converting discrete star ratings → continuous floats...")
    print("  Integer stars (1,2,3,4,5) → continuous (3.1, 4.2, 2.8, ...)")

    df["overall_rating"] = df["overall_rating_raw"].apply(make_continuous_rating)
    for col_name in ["work_life_balance", "company_culture", "career_growth", "salary_benefits"]:
        df[col_name] = df[col_name + "_raw"].apply(make_continuous_rating)

    raw_cols = [c for c in df.columns if c.endswith("_raw")]
    df = df.drop(columns=raw_cols)

    # Remove outliers
    df = remove_outliers(df, 'overall_rating', threshold=3.5)

    # Generic augmentation
    df = create_augmented_samples(df, augmentation_factor=0.25)

    # Target neutral/mixed samples to improve calibration around 3-star region
    df = create_targeted_neutral_mixed_augmentation(df, target_factor=0.40)

    # Soft balance (continuous bins)
    df = soft_balance_ratings(df, 'overall_rating')

    # Balance sentiment groups derived from rating for multi-task training stability
    df = balance_sentiment_classes(df)

    # Stats
    compute_statistics(df)

    # Save
    print("\n" + "="*70)
    print(f"Saving to: {OUTPUT_PATH}")
    df[['text', 'overall_rating', 'work_life_balance', 'company_culture',
        'career_growth', 'salary_benefits']].to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Done! Ratings are now continuous: e.g. 3.1, 4.2, 2.8")
    print("="*70)


if __name__ == "__main__":
    main()