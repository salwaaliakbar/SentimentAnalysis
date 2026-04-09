"""
End-to-End Sentiment Analysis & Reputation Scoring - Rating Prediction
====================================================================
Shows only:
1. Per-comment JSON details with ratings
2. Company reputation summary with aspect breakdown (ratings only)
"""

from datetime import datetime, timedelta
import json
import numpy as np
from sentiment_analyzer import SentimentAnalyzer
from reputation_scorer import ReputationScorer
from anti_manipulation import AntiManipulationEngine


def main():
    """Run end-to-end workflow."""
    
    print("\n" + "=" * 80)
    print("SENTIMENT ANALYSIS & REPUTATION SCORING REPORT")
    print("=" * 80)
    
    # ────────────────────────────────────────────────────────────────────────
    # Initialize components
    # ────────────────────────────────────────────────────────────────────────
    
    sentiment_analyzer = SentimentAnalyzer(model_dir='model_output_v3', device='cpu')
    reputation_scorer = ReputationScorer(temporal_half_life_days=90, bayesian_alpha=20)
    anti_manipulation = AntiManipulationEngine()

    
    # ────────────────────────────────────────────────────────────────────────
    # Sample comments
    # ────────────────────────────────────────────────────────────────────────
    
    sample_comments = [
        "Amazing culture, love the team! Very supportive colleagues.",
        "Pay is low compared to competitors. Not worth it.",
        "Decent place to work overall. Nothing exceptional.",
        "Manager was terrible, no career growth opportunities at all.",
        "Interview process was smooth and very professional.",
        "Leadership is chaotic. Management practices are outdated.",
        "Great flexible working hours! Work-life balance is excellent.",
        "Benefits are okay but salary is not competitive.",
        "Best company I've worked for. Culture is outstanding!",
        "Too much overtime, burnout is real here.",
        "Strong mentorship program and clear growth paths.",
        "Office politics are draining and slow down work.",
        "Good pay and bonuses, but limited learning opportunities.",
        "Supportive manager, but the team is understaffed.",
        "Workload is fair and deadlines are reasonable.",
        "Company vision is unclear and communication is poor.",
        "Great benefits package and health coverage.",
        "Flexible remote policy makes life easier.",
        "Promotion process feels unfair and inconsistent.",
        "Very Positive Review. Great companyculture, excellent work-life balance, amazing career growth opportunities, and competitive salary."
    ]
    
    company_id = 12345
    company_name = "TechCorp Inc."
    
    # ────────────────────────────────────────────────────────────────────────
    # Sentiment analysis (batch processing)
    # ────────────────────────────────────────────────────────────────────────
    
    sentiment_results = []
    aspect_results = []
    per_comment_details = []
    
    batch_size = 5
    for batch_start in range(0, len(sample_comments), batch_size):
        batch = sample_comments[batch_start:batch_start + batch_size]
        batch_results = sentiment_analyzer.batch_predict(batch)
        
        for offset, (comment, sentiment) in enumerate(zip(batch, batch_results)):
            i = batch_start + offset
            sentiment_results.append(sentiment)
            
            aspect_scores = sentiment.get("aspect_scores", {})
            aspect_signals = sentiment.get("aspect_sentiments", {})
            aspect_results.append(aspect_signals)
            
            per_comment_details.append({
                "index": i + 1,
                "text": comment,
                "sentiment": {
                    "label": sentiment["label"],
                    "confidence": round(sentiment["confidence"], 3),
                    "overall_rating": round(sentiment["overall_rating"], 2),
                    "sentiment_signal": round(sentiment["sentiment_signal"], 3),
                },
                "aspects": {
                    "scores": {k: round(v, 2) for k, v in aspect_scores.items()},
                    "signals": {k: round(v, 3) for k, v in aspect_signals.items()},
                },
            })
    
    # ────────────────────────────────────────────────────────────────────────
    # Anti-manipulation checks
    # ────────────────────────────────────────────────────────────────────────
    
    manipulation_checks = []
    for i, comment in enumerate(sample_comments):
        result = anti_manipulation.check_submission(
            text=comment,
            ip_hash=f"ip_hash_{i % 5}",
            recent_submissions=sample_comments[:i],
            sentiment_confidence=sentiment_results[i]['confidence']
        )
        manipulation_checks.append(result)
        per_comment_details[i]["anti_manipulation"] = result
    
    # ────────────────────────────────────────────────────────────────────────
    # OUTPUT 1: Per-Comment JSON
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("PART 1: PER-COMMENT ANALYSIS")
    print("=" * 80)
    print(json.dumps(per_comment_details, indent=2))
    
    # ────────────────────────────────────────────────────────────────────────
    # Compute reputation score
    # ────────────────────────────────────────────────────────────────────────
    
    approved_submissions = [
        i for i, check in enumerate(manipulation_checks)
        if check['recommendation'] != 'reject'
    ]
    
    sentiment_signals = [
        {
            'signal': sentiment_results[i]['sentiment_signal'],
            'confidence': sentiment_results[i]['confidence']
        }
        for i in approved_submissions
    ]
    
    now = datetime.now()
    timestamps = [
        now - timedelta(days=np.random.uniform(0, 90))
        for _ in approved_submissions
    ]
    
    score_5star = reputation_scorer.compute_reputation_score(
        sentiment_signals,
        submission_timestamps=timestamps,
        current_time=now,
        scale='5star'
    )
    
    # Extract rating values and convert to percentage
    rating_5star = score_5star['score']
    rating_percent = (rating_5star - 1) / 4 * 100  # Convert 1-5 scale to 0-100%
    ci_lower_percent = (score_5star['ci_lower'] - 1) / 4 * 100
    ci_upper_percent = (score_5star['ci_upper'] - 1) / 4 * 100
    
    # Compute aspect scores
    aspect_scores_5star = reputation_scorer.compute_aspect_scores(
        [aspect_results[i] for i in approved_submissions],
        submission_timestamps=timestamps,
        current_time=now,
        scale='5star'
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # OUTPUT 2: Company Reputation Summary
    # ────────────────────────────────────────────────────────────────────────
    
    summary = {
        "company": company_name,
        "overall_rating": {
            "stars": round(rating_5star, 1),
            "percentage": round(rating_percent, 0)
        },
        "reliability": {
            "confidence_interval_stars": [
                round(score_5star["ci_lower"], 1),
                round(score_5star["ci_upper"], 1)
            ],
            "confidence_interval_percent": [
                round(ci_lower_percent, 0),
                round(ci_upper_percent, 0)
            ],
            "sample_size": score_5star["sample_size"],
            "interpretation": f"True reputation likely between {round(score_5star['ci_lower'], 1)} and {round(score_5star['ci_upper'], 1)} stars"
        },
        "aspect_breakdown": {
            k: {
                "rating": round(v, 1),
                "status": "Strong" if v >= 4.0 else "Good" if v >= 3.5 else "Neutral" if v >= 2.5 else "Weak"
            }
            for k, v in sorted(aspect_scores_5star.items())
        }
    }
    
    print("\n" + "=" * 80)
    print("PART 2: COMPANY REPUTATION SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()