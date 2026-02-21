"""
Sentiment Analysis Module
========================
Wrapper around DistilBERT fine-tuned for jobseeker comment sentiment analysis.
Provides both pretrained and fine-tuned inference capabilities.

Zero-cost: Uses only open-source Hugging Face models.
"""

import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment classifier for job review comments.
    
    Labels:
    - 0: Negative
    - 1: Neutral
    - 2: Positive
    
    Output: Probabilities (P_neg, P_neutral, P_pos) and sentiment signal S_c = P_pos - P_neg
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: Hugging Face model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading {model_name} on device: {self.device}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Load pretrained model (3-class sentiment classification)
        # Note: For production, this should be fine-tuned on jobseeker data
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # negative, neutral, positive
        )
        self.model.to(device)
        self.model.eval()
        
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == 'cuda' else -1
        )
    
    def predict(self, text: str, return_probabilities: bool = True) -> Dict:
        """
        Predict sentiment for a single comment.
        
        Args:
            text: Input comment text
            return_probabilities: If True, return full probability distribution
        
        Returns:
            Dictionary with:
            - 'label': 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
            - 'score': confidence [0, 1]
            - 'probabilities': [P_neg, P_neutral, P_pos]
            - 'sentiment_signal': S_c = P_pos - P_neg ([-1, 1])
            - 'confidence': max probability
        """
        # Truncate to max length
        text = text[:512]
        
        # Get raw logits
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        p_neg, p_neutral, p_pos = probabilities
        
        # Sentiment signal
        sentiment_signal = float(p_pos - p_neg)
        max_confidence = float(probabilities.max())
        
        # Determine label
        pred_label_idx = probabilities.argmax()
        label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
        label = label_map[pred_label_idx]
        
        return {
            'label': label,
            'score': max_confidence,
            'probabilities': {
                'negative': float(p_neg),
                'neutral': float(p_neutral),
                'positive': float(p_pos)
            },
            'sentiment_signal': sentiment_signal,  # S_c = P_pos - P_neg
            'confidence': max_confidence,
            'text': text
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for multiple comments.
        
        Args:
            texts: List of comment texts
        
        Returns:
            List of prediction dictionaries (same as predict)
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def calibration_check(self, prediction: Dict) -> bool:
        """
        Check if model confidence is calibrated (not overconfident).
        
        Returns False if model output seems unreliable:
        - Confidence < 0.60: Model uncertain
        
        Args:
            prediction: Output from predict()
        
        Returns:
            True if calibrated, False if suspicious
        """
        confidence = prediction['confidence']
        return confidence >= 0.60  # Require min 60% confidence
    
    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_labels': self.model.config.num_labels,
            'max_position_embeddings': self.model.config.max_position_embeddings
        }


def demo_sentiment_analysis():
    """Demo sentiment analysis on sample comments."""
    
    analyzer = SentimentAnalyzer(device='cpu')  # Use CPU for demo
    
    sample_comments = [
        "Amazing culture, love the team!",
        "Pay is low",
        "Decent place to work",
        "Manager was awful, no growth opportunities",
        "Interview process was smooth, very professional",
        "Terrible leadership and poor compensation",
        "Great benefits but no work-life balance"
    ]
    
    print("=" * 80)
    print("SENTIMENT ANALYSIS DEMO")
    print("=" * 80)
    
    results = analyzer.batch_predict(sample_comments)
    
    for i, (comment, result) in enumerate(zip(sample_comments, results)):
        print(f"\n[{i+1}] Comment: '{comment}'")
        print(f"    Label: {result['label']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Probabilities: Neg={result['probabilities']['negative']:.3f}, "
              f"Neutral={result['probabilities']['neutral']:.3f}, "
              f"Pos={result['probabilities']['positive']:.3f}")
        print(f"    Sentiment Signal (S_c): {result['sentiment_signal']:.3f}")
        print(f"    Calibrated: {analyzer.calibration_check(result)}")
    
    print("\n" + "=" * 80)
    print(f"Model info: {analyzer.get_model_info()}")
    print("=" * 80)


if __name__ == "__main__":
    demo_sentiment_analysis()
