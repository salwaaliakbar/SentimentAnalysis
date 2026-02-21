"""
Aspect-Based Sentiment Analysis (ABSA)
========================================
Extracts aspects (pay, culture, management, etc.) from comments
and links sentiment to specific aspects.

Uses keyword dictionaries and dependency parsing.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class AspectExtractor:
    """
    Extract aspects (pay, culture, etc.) from comments and associate sentiments.
    
    Approach:
    1. Dictionary-based aspect identification (keywords)
    2. Sentiment linkage (associate nearby sentiment words to aspects)
    3. Statistical aggregation (aspect-level sentiment scores)
    """
    
    ASPECT_KEYWORDS = {
        'pay': ['salary', 'pay', 'compensation', 'wages', 'bonus', 'benefits', 
                'paycheck', 'income', 'money', 'rate', 'hourly', 'annual'],
        'culture': ['culture', 'team', 'colleagues', 'environment', 'workplace', 
                    'atmosphere', 'vibe', 'coworkers', 'people', 'community'],
        'management': ['manager', 'boss', 'leadership', 'supervisor', 'director',
                       'management', 'executive', 'leader', 'management style'],
        'growth': ['growth', 'learning', 'development', 'career', 'advancement',
                   'opportunities', 'training', 'promotion', 'progression', 'skill'],
        'interviews': ['interview', 'hiring', 'recruitment', 'screening', 'application',
                       'hiring process', 'interview process', 'recruiter'],
        'work_life_balance': ['work-life', 'balance', 'hours', 'workload', 'overtime',
                              'flexible', 'remote', 'relocation', 'schedule', 'burnout'],
    }
    
    # Sentiment lexicons (simple sentiment words)
    POSITIVE_WORDS = {
        'love', 'awesome', 'amazing', 'great', 'excellent', 'perfect', 'fantastic',
        'wonderful', 'best', 'good', 'nice', 'enjoy', 'happy', 'satisfied',
        'impressed', 'grateful', 'supportive', 'comfortable', 'fun', 'friendly',
        'professional', 'rewarding', 'challenging'
    }
    
    NEGATIVE_WORDS = {
        'hate', 'terrible', 'awful', 'bad', 'poor', 'worse', 'worst', 'horrible',
        'useless', 'sucks', 'hate', 'dislike', 'frustrating', 'stressful',
        'disappointing', 'unfair', 'toxic', 'problematic', 'difficult', 'painful',
        'exhausting', 'annoying', 'chaotic', 'miserable'
    }
    
    def __init__(self):
        """Initialize aspect extractor."""
        self.aspect_map = self.ASPECT_KEYWORDS
        self.positive_lexicon = self.POSITIVE_WORDS
        self.negative_lexicon = self.NEGATIVE_WORDS
    
    def extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """
        Extract mentioned aspects from text.
        
        Args:
            text: Input comment
        
        Returns:
            Dictionary mapping aspect name to list of identified keywords
            Example: {'pay': ['salary', 'bonus'], 'culture': ['team']}
        """
        text_lower = text.lower()
        found_aspects = defaultdict(list)
        
        for aspect_name, keywords in self.aspect_map.items():
            for keyword in keywords:
                # Word boundary matching to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    found_aspects[aspect_name].append(keyword)
        
        return dict(found_aspects)
    
    def link_sentiment_to_aspects(self, text: str) -> Dict[str, float]:
        """
        Link sentiment words to nearby aspects.
        
        Approach:
        1. Find all sentiment words (positive/negative)
        2. For each sentiment word, find nearest aspect within window
        3. Aggregate sentiment signals per aspect
        
        Args:
            text: Input comment
        
        Returns:
            Dictionary mapping aspect to sentiment signal ([-1, 1])
            Example: {'pay': -0.7, 'culture': +0.9}
        """
        text_lower = text.lower()
        tokens = text_lower.split()
        
        aspect_sentiments = defaultdict(list)
        
        for idx, token in enumerate(tokens):
            sentiment_value = None
            
            # Check if this token is a sentiment word
            if token in self.positive_lexicon:
                sentiment_value = 1.0
            elif token in self.negative_lexicon:
                sentiment_value = -1.0
            
            if sentiment_value is not None:
                # Look for nearby aspects (within ±5 tokens)
                window_start = max(0, idx - 5)
                window_end = min(len(tokens), idx + 6)
                window_text = ' '.join(tokens[window_start:window_end])
                
                aspects_in_window = self.extract_aspects(window_text)
                
                # Assign sentiment to found aspects
                for aspect_name in aspects_in_window:
                    # Discount sentiment strength by distance
                    distance_discount = 1.0 - (abs(idx - (window_start + 5)) / 10.0)
                    distance_discount = max(0.3, distance_discount)  # Min 0.3 even far away
                    
                    weighted_sentiment = sentiment_value * distance_discount
                    aspect_sentiments[aspect_name].append(weighted_sentiment)
        
        # Aggregate per-aspect sentiments
        aggregated = {}
        for aspect, sentiments in aspect_sentiments.items():
            aggregated[aspect] = float(np.mean(sentiments))
        
        return aggregated
    
    def extract_full(self, text: str, external_sentiment_signal: float = None) -> Dict:
        """
        Full aspect extraction pipeline.
        
        Args:
            text: Input comment
            external_sentiment_signal: Overall sentiment from main classifier (S_c)
        
        Returns:
            Dictionary with:
            - 'mentions': aspects mentioned
            - 'sentiments': aspect-level sentiment signals
            - 'has_aspects': whether comments contain aspects
        """
        mentions = self.extract_aspects(text)
        sentiments = self.link_sentiment_to_aspects(text)
        
        # If no aspects extracted, use external signal for general sentiment
        if not sentiments and external_sentiment_signal is not None:
            sentiments = {'general': external_sentiment_signal}
        
        return {
            'mentions': mentions,
            'sentiments': sentiments,
            'has_aspects': len(mentions) > 0,
            'text_preview': text[:100] + ('...' if len(text) > 100 else '')
        }


def normalize_aspect_scores(aspect_sentiments: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize aspect scores to [-1, 1] range.
    
    Args:
        aspect_sentiments: Raw aspect sentiments from extraction
    
    Returns:
        Normalized scores (clamped to [-1, 1])
    """
    normalized = {}
    for aspect, score in aspect_sentiments.items():
        # Clamp to [-1, 1]
        normalized[aspect] = max(-1.0, min(1.0, float(score)))
    return normalized


def aggregate_aspect_scores(
    comment_sentiments: List[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Aggregate aspect scores across multiple comments.
    
    Args:
        comment_sentiments: List of aspect sentiment dicts from extract_full()
        weights: Optional aspect weights (default: uniform)
    
    Returns:
        Aggregated aspect scores
    """
    import numpy as np
    
    # Collect all aspect scores
    aspect_scores = defaultdict(list)
    
    for comment_dict in comment_sentiments:
        sentiments = comment_dict.get('sentiments', {})
        for aspect, score in sentiments.items():
            aspect_scores[aspect].append(score)
    
    # Average per aspect
    aggregated = {}
    for aspect, scores in aspect_scores.items():
        aggregated[aspect] = float(np.mean(scores))
    
    # Apply weights if provided
    if weights:
        for aspect in aggregated:
            if aspect in weights:
                aggregated[aspect] *= weights[aspect]
    
    return aggregated


# Demo
import numpy as np

def demo_aspect_extraction():
    """Demo aspect-based sentiment extraction."""
    
    extractor = AspectExtractor()
    
    sample_comments = [
        "I love the team and culture, but the salary is low",
        "Manager is terrible but interview process was smooth",
        "Great flexible working hours but no growth opportunities",
        "Awesome benefits and supportive leadership"
    ]
    
    print("=" * 80)
    print("ASPECT-BASED SENTIMENT ANALYSIS DEMO")
    print("=" * 80)
    
    all_sentiments = []
    
    for i, comment in enumerate(sample_comments):
        print(f"\n[{i+1}] Comment: '{comment}'")
        
        result = extractor.extract_full(comment, external_sentiment_signal=0.0)
        all_sentiments.append(result)
        
        print(f"    Mentioned aspects: {result['mentions']}")
        print(f"    Aspect sentiments:")
        for aspect, score in result['sentiments'].items():
            sentiment_word = "positive" if score > 0.3 else ("negative" if score < -0.3 else "neutral")
            print(f"      - {aspect}: {score:.2f} ({sentiment_word})")
    
    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATED ASPECT SCORES (across all comments)")
    print("=" * 80)
    
    aggregated = aggregate_aspect_scores(all_sentiments)
    for aspect, score in sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {aspect:20s}: {score:+.3f}")
    
    print("=" * 80)


if __name__ == "__main__":
    demo_aspect_extraction()
