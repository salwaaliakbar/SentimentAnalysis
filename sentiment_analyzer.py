"""
Sentiment Analysis Module
========================
Inference wrapper for the multi-task DistilBERT model trained on:
- overall rating
- work-life balance
- company culture
- career growth
- salary & benefits

The model expects a local model_output directory with best.pt and tokenizer files.
Architecture matches train_advanced.py with feature extractor and proper heads.
"""

from pathlib import Path
from typing import Dict, List
import logging
import json

import torch
from torch import nn
from transformers import DistilBertTokenizerFast, DistilBertModel

logger = logging.getLogger(__name__)

ASPECT_NAMES = [
    "work_life_balance",
    "company_culture",
    "career_growth",
    "salary_benefits",
]

# Default hyperparameters (will be overridden by config.json if available)
DEFAULT_HIDDEN_DIM = 384
DEFAULT_DROPOUT = 0.35
DEFAULT_RATING_MIN = 1.0
DEFAULT_RATING_MAX = 5.0


class MultiTaskDistilBert(nn.Module):
    """
    Multi-task DistilBERT with feature extractor and sequential heads.
    Matches train_advanced.py architecture.
    
    Architecture:
    - Encoder: DistilBERT
    - Feature Extractor: 2 Linear layers with LayerNorm + GELU + Dropout
    - Heads: Sequential with hidden_dim//2 intermediate layer
    - Output: Scaled sigmoid to [rating_min, rating_max]
    """
    
    def __init__(self, base_model: str, aspect_names: List[str],
                 hidden_dim: int = DEFAULT_HIDDEN_DIM, 
                 dropout: float = DEFAULT_DROPOUT,
                 rating_min: float = DEFAULT_RATING_MIN,
                 rating_max: float = DEFAULT_RATING_MAX):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.calibration_rules = config.get(
            "calibration",
            {"neutral_range": [2.5, 3.5], "negative_cap": 2.2, "positive_floor": 3.9},
        )
        
        self.encoder = DistilBertModel.from_pretrained(base_model)
        encoder_dim = self.encoder.config.hidden_size  # 768

        # Freeze early transformer layers
        for i, param in enumerate(self.encoder.parameters()):
            if i < len(list(self.encoder.parameters())) - 8:
                param.requires_grad = False

        # Shared feature extractor (matches train_advanced.py)
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

        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )

    def _scale_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale raw logit to [rating_min, rating_max] range using sigmoid.
        sigmoid(x) maps (-inf, +inf) → (0, 1)
        * (rating_max - rating_min) + rating_min maps (0, 1) → (rating_min, rating_max)
        """
        return torch.sigmoid(x) * (self.rating_max - self.rating_min) + self.rating_min

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.last_hidden_state[:, 0]  # [CLS] token

        features = self.feature_extractor(pooled)

        logits = {
            "overall": self._scale_output(self.overall_head(features).squeeze(-1)),
            "sentiment_logits": self.sentiment_head(features)
        }
        for aspect, head in self.aspect_heads.items():
            logits[aspect] = self._scale_output(head(features).squeeze(-1))

        return logits


class SentimentAnalyzer:
    """
    Multi-task sentiment and aspect rating inference.

    Outputs:
    - overall_rating: float in [1, 5]
    - aspect_scores: per-aspect rating in [1, 5]
    - sentiment_signal: mapped to [-1, 1]
    - label: NEGATIVE / NEUTRAL / POSITIVE
    """

    def __init__(
        self,
        model_dir: str = "model_output_v3",
        base_model: str = "distilbert-base-uncased",
        device: str = None,
        max_len: int = 384,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_dir = model_dir
        self.base_model = base_model
        self.max_len = max_len

        state_path = Path(model_dir) / "best.pt"
        if not state_path.exists():
            raise FileNotFoundError(
                f"Missing trained weights at {state_path}. "
                "Place your model_output folder in the project root."
            )

        logger.info(f"Loading multitask model from {model_dir} on device: {self.device}")

        # Load config.json to get hyperparameters
        config_path = Path(model_dir) / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            hidden_dim = config.get('hidden_dim', DEFAULT_HIDDEN_DIM)
            dropout = config.get('dropout', DEFAULT_DROPOUT)
            rating_min = config.get('rating_min', DEFAULT_RATING_MIN)
            rating_max = config.get('rating_max', DEFAULT_RATING_MAX)
            self.base_model = config.get('base_model', base_model)
            logger.info(f"Loaded config: hidden_dim={hidden_dim}, dropout={dropout}, rating_range=[{rating_min},{rating_max}]")
        else:
            logger.warning(f"config.json not found in {model_dir}, using defaults")
            hidden_dim = DEFAULT_HIDDEN_DIM
            dropout = DEFAULT_DROPOUT
            rating_min = DEFAULT_RATING_MIN
            rating_max = DEFAULT_RATING_MAX

        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        except Exception:
            logger.warning("Tokenizer files not found in model_output. Falling back to base model tokenizer.")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.base_model)

        self.model = MultiTaskDistilBert(
            self.base_model, 
            ASPECT_NAMES,
            hidden_dim=hidden_dim,
            dropout=dropout,
            rating_min=rating_min,
            rating_max=rating_max
        )
        state = torch.load(state_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        # Store rating range for later use
        self.rating_min = rating_min
        self.rating_max = rating_max

    @staticmethod
    def _clamp_score(score: float) -> float:
        return max(1.0, min(5.0, float(score)))

    @staticmethod
    def _score_to_signal(score: float) -> float:
        signal = (float(score) - 3.0) / 2.0
        return max(-1.0, min(1.0, signal))

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 3.5:
            return "POSITIVE"
        if score <= 2.5:
            return "NEGATIVE"
        return "NEUTRAL"

    @staticmethod
    def _class_to_label(cls_idx: int) -> str:
        return {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}.get(int(cls_idx), "NEUTRAL")

    @staticmethod
    def _signal_confidence(signal: float) -> float:
        return float(min(1.0, 0.5 + 0.5 * abs(signal)))

    def _apply_calibration(self, raw_rating: float, sentiment_label: str, sentiment_conf: float) -> float:
        """Rule-based calibration layer to reduce sentiment-rating mismatch."""
        calibrated = float(raw_rating)
        neutral_range = self.calibration_rules.get("neutral_range", [2.5, 3.5])
        negative_cap = float(self.calibration_rules.get("negative_cap", 2.2))
        positive_floor = float(self.calibration_rules.get("positive_floor", 3.9))

        if sentiment_label == "NEUTRAL" and sentiment_conf >= 0.55:
            calibrated = min(max(calibrated, neutral_range[0]), neutral_range[1])
        elif sentiment_label == "NEGATIVE" and sentiment_conf >= 0.60:
            calibrated = min(calibrated, negative_cap)
        elif sentiment_label == "POSITIVE" and sentiment_conf >= 0.60:
            calibrated = max(calibrated, positive_floor)

        return self._clamp_score(calibrated)

    def _predict_batch(self, texts: List[str]) -> List[Dict]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        class_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
        class_ids = class_probs.argmax(dim=-1)

        results = []
        for idx in range(len(texts)):
            raw_overall_score = self._clamp_score(outputs["overall"][idx].item())
            aspect_scores = {
                aspect: self._clamp_score(outputs[aspect][idx].item())
                for aspect in ASPECT_NAMES
            }

            sentiment_class = int(class_ids[idx].item())
            sentiment_conf = float(class_probs[idx][sentiment_class].item())
            label = self._class_to_label(sentiment_class)

            overall_score = self._apply_calibration(raw_overall_score, label, sentiment_conf)

            if label == "NEUTRAL":
                for aspect in aspect_scores:
                    aspect_scores[aspect] = min(max(aspect_scores[aspect], 2.4), 3.6)

            overall_signal = self._score_to_signal(overall_score)
            aspect_signals = {
                aspect: self._score_to_signal(score)
                for aspect, score in aspect_scores.items()
            }
            confidence = max(self._signal_confidence(overall_signal), sentiment_conf)

            results.append(
                {
                    "label": label,
                    "score": confidence,
                    "confidence": confidence,
                    "overall_rating": overall_score,
                    "overall_rating_raw": raw_overall_score,
                    "sentiment_signal": overall_signal,
                    "sentiment_class": sentiment_class,
                    "sentiment_probs": {
                        "negative": float(class_probs[idx][0].item()),
                        "neutral": float(class_probs[idx][1].item()),
                        "positive": float(class_probs[idx][2].item()),
                    },
                    "aspect_scores": aspect_scores,
                    "aspect_sentiments": aspect_signals,
                    "text": texts[idx],
                }
            )

        return results

    def predict(self, text: str) -> Dict:
        text = text[:512]
        return self._predict_batch([text])[0]

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []
        return self._predict_batch(texts)

    def calibration_check(self, prediction: Dict) -> bool:
        confidence = prediction["confidence"]
        return confidence >= 0.60

    def get_model_info(self) -> Dict:
        return {
            "model_dir": self.model_dir,
            "base_model": self.base_model,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "aspects": ASPECT_NAMES,
            "max_length": self.max_len,
        }


def demo_sentiment_analysis():
    analyzer = SentimentAnalyzer(device="cpu")

    sample_comments = [
        "Amazing culture, love the team!",
        "Pay is low",
        "Decent place to work",
        "Manager was awful, no growth opportunities",
        "Interview process was smooth, very professional",
        "Terrible leadership and poor compensation",
        "Great benefits but no work-life balance",

    ]

    print("=" * 80)
    print("SENTIMENT ANALYSIS DEMO")
    print("=" * 80)

    results = analyzer.batch_predict(sample_comments)

    for i, (comment, result) in enumerate(zip(sample_comments, results)):
        print(f"\n[{i+1}] Comment: '{comment}'")
        print(f"    Label: {result['label']}")
        print(f"    Overall Rating: {result['overall_rating']:.2f}/5.0")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Sentiment Signal (S_c): {result['sentiment_signal']:+.3f}")
        print(f"    Calibrated: {analyzer.calibration_check(result)}")

    print("\n" + "=" * 80)
    print(f"Model info: {analyzer.get_model_info()}")
    print("=" * 80)


if __name__ == "__main__":
    demo_sentiment_analysis()
