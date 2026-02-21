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
"""

from pathlib import Path
from typing import Dict, List
import logging

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


class MultiTaskDistilBert(nn.Module):
    def __init__(self, base_model: str, aspect_names: List[str]):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.overall_head = nn.Linear(hidden_size, 1)
        self.aspect_heads = nn.ModuleDict({a: nn.Linear(hidden_size, 1) for a in aspect_names})

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        logits = {"overall": self.overall_head(pooled).squeeze(-1)}
        for aspect, head in self.aspect_heads.items():
            logits[aspect] = head(pooled).squeeze(-1)
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
        model_dir: str = "model_output",
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

        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        except Exception:
            logger.warning("Tokenizer files not found in model_output. Falling back to base model tokenizer.")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(base_model)

        self.model = MultiTaskDistilBert(base_model, ASPECT_NAMES)
        state = torch.load(state_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

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
    def _signal_confidence(signal: float) -> float:
        return float(min(1.0, 0.5 + 0.5 * abs(signal)))

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

        results = []
        for idx in range(len(texts)):
            overall_score = self._clamp_score(outputs["overall"][idx].item())
            aspect_scores = {
                aspect: self._clamp_score(outputs[aspect][idx].item())
                for aspect in ASPECT_NAMES
            }

            overall_signal = self._score_to_signal(overall_score)
            aspect_signals = {
                aspect: self._score_to_signal(score)
                for aspect, score in aspect_scores.items()
            }
            confidence = self._signal_confidence(overall_signal)
            label = self._score_to_label(overall_score)

            results.append(
                {
                    "label": label,
                    "score": confidence,
                    "confidence": confidence,
                    "overall_rating": overall_score,
                    "sentiment_signal": overall_signal,
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
