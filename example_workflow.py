"""
End-to-End Example Workflow
============================
Demonstrates the complete sentiment analysis and reputation scoring pipeline.

This script shows:
1. Sentiment analysis on sample job reviews
2. Aspect-based sentiment extraction
3. Reputation score computation with safeguards
4. Anti-manipulation detection
5. Final reputation dashboard
"""

from datetime import datetime, timedelta
import numpy as np
from sentiment_analyzer import SentimentAnalyzer
from reputation_scorer import ReputationScorer
from anti_manipulation import AntiManipulationEngine

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n{title}")
    print("─" * 80)

def main():
    """Run end-to-end workflow."""
    
    print_header("END-TO-END SENTIMENT ANALYSIS & REPUTATION SCORING WORKFLOW")
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 1: Initialize components
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 1: Initializing Components")
    
    print("  Loading sentiment analyzer...")
    sentiment_analyzer = SentimentAnalyzer(device='cpu')
    print("  ✅ Sentiment analyzer ready")
    
    print("  Loading reputation scorer...")
    reputation_scorer = ReputationScorer(
        temporal_half_life_days=90,
        bayesian_alpha=20
    )
    print("  ✅ Reputation scorer ready")
    
    print("  Loading anti-manipulation engine...")
    anti_manipulation = AntiManipulationEngine()
    print("  ✅ Anti-manipulation engine ready")
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 2: Sample comments for a company
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 2: Sample Comments from Job Seekers")
    
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
        "Too much overtime, burnout is real here."
    ]
    
    company_id = 12345
    company_name = "TechCorp Inc."
    
    print(f"  Company: {company_name} (ID: {company_id})")
    print(f"  Number of comments: {len(sample_comments)}")
    print(f"  Collection period: Last 90 days")
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 3: Sentiment analysis on each comment
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 3: Sentiment Analysis (Per Comment)")
    
    sentiment_results = []
    aspect_results = []
    
    for i, comment in enumerate(sample_comments):
        # Sentiment prediction
        sentiment = sentiment_analyzer.predict(comment)
        sentiment_results.append(sentiment)
        
        aspect_results.append(sentiment.get("aspect_sentiments", {}))
        
        print(f"\n  [{i+1:2d}] \"{comment[:50]}...\"")
        print(
            f"       Label: {sentiment['label']:8s} | Confidence: {sentiment['confidence']:.3f} | "
            f"Overall: {sentiment['overall_rating']:.2f}/5.0 | Signal S_c: {sentiment['sentiment_signal']:+.3f}"
        )
        print(
            "       Aspects: "
            f"{', '.join(list(sentiment.get('aspect_sentiments', {}).keys())[:3]) if sentiment.get('aspect_sentiments') else 'general'}"
        )
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 4: Anti-manipulation detection
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 4: Anti-Manipulation Checks")
    
    # Simulate some manipulation attempts
    manipulation_checks = []
    
    for i, comment in enumerate(sample_comments):
        result = anti_manipulation.check_submission(
            text=comment,
            ip_hash=f"ip_hash_{i % 5}",  # Simulate different IPs
            recent_submissions=sample_comments[:i],
            sentiment_confidence=sentiment_results[i]['confidence']
        )
        manipulation_checks.append(result)
        
        if result['flags']:
            print(f"  [{i+1:2d}] ⚠️  Flags: {', '.join(result['flags'])}")
            print(f"        Weight factor: {result['weight_factor']:.2f} | "
                  f"Recommendation: {result['recommendation'].upper()}")
    
    flagged_count = sum(1 for c in manipulation_checks if c['flags'])
    print(f"\n  Summary: {flagged_count} out of {len(sample_comments)} submissions flagged")
    print(f"  All-clear submissions: {len(sample_comments) - flagged_count}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 5: Compute reputation score
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 5: Reputation Score Computation")
    
    # Filter approved submissions (exclude high-risk ones)
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
    
    # Timestamps (simulate varied ages)
    now = datetime.now()
    timestamps = [
        now - timedelta(days=np.random.uniform(0, 90))
        for _ in approved_submissions
    ]
    
    # Compute reputation score
    score_5star = reputation_scorer.compute_reputation_score(
        sentiment_signals,
        submission_timestamps=timestamps,
        current_time=now,
        scale='5star'
    )
    
    score_100 = reputation_scorer.compute_reputation_score(
        sentiment_signals,
        submission_timestamps=timestamps,
        current_time=now,
        scale='100percent'
    )
    
    print(f"\n  Raw Sentiment Signal (mean): {score_5star['raw_sentiment_signal']:+.3f}")
    print(f"  Bayesian Smoothed Signal:    {score_5star['sentiment_signal']:+.3f}")
    print(f"    (Regressed toward neutral due to N={score_5star['sample_size']} comments)")
    
    print(f"\n  5-STAR SCALE:")
    print(f"    Reputation Score: {score_5star['score']:.1f} / 5.0 ⭐")
    print(f"    95% Confidence Interval: [{score_5star['ci_lower']:.1f}, {score_5star['ci_upper']:.1f}]")
    print(f"    Interpretation: The true reputation is likely between {score_5star['ci_lower']:.1f} "
          f"and {score_5star['ci_upper']:.1f} stars")
    
    print(f"\n  0-100% SCALE (Glassdoor-style):")
    print(f"    Reputation Score: {score_100['score']:.0f}%")
    print(f"    95% Confidence Interval: [{score_100['ci_lower']:.0f}%, {score_100['ci_upper']:.0f}%]")
    
    print(f"\n  SAMPLE CHARACTERISTICS:")
    print(f"    Total submissions analyzed: {len(sample_comments)}")
    print(f"    Approved (included in score): {len(approved_submissions)}")
    print(f"    Flagged/rejected: {len(sample_comments) - len(approved_submissions)}")
    print(f"    Temporal weight (total): {score_5star['temporal_weight']:.1f}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 6: Aspect-based sentiment breakdown
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 6: Aspect-Based Sentiment Analysis")
    
    # Aggregate aspect scores (5-star scale)
    aspect_scores_5star = reputation_scorer.compute_aspect_scores(
        [aspect_results[i] for i in approved_submissions],
        submission_timestamps=timestamps,
        current_time=now,
        scale='5star'
    )
    
    print(f"\n  Company Reputation by Aspect (5-star scale):")
    print(f"  {'Aspect':<20} {'Score':<15} {'Sentiment':<20} {'Assessment'}")
    print(f"  {'-' * 20} {'-' * 15} {'-' * 20} {'-' * 20}")
    
    for aspect in sorted(aspect_scores_5star.keys()):
        score = aspect_scores_5star[aspect]
        signal = (aspect_scores_5star[aspect] - 3.0) / 2.0
        
        if score >= 4.0:
            assessment = "💚 Strong"
            sentiment_word = "Very Positive"
        elif score >= 3.5:
            assessment = "✅ Good"
            sentiment_word = "Positive"
        elif score >= 2.5:
            assessment = "⚠️  Neutral"
            sentiment_word = "Mixed"
        else:
            assessment = "❌ Weak"
            sentiment_word = "Negative"
        
        print(f"  {aspect:<20} {score:>6.1f}/5.0       {signal:+.3f} ({sentiment_word:>8s})  {assessment}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 7: Final dashboard
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 7: Company Reputation Dashboard")
    
    # Create dashboard
    print(f"\n  {'COMPANY REPUTATION REPORT':^80}")
    print(f"  {'═' * 80}")
    print(f"  Company: {company_name}")
    print(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'─' * 80}")
    
    # Overall rating display
    star_count = int(round(score_5star['score']))
    star_display = '⭐' * star_count + '☆' * (5 - star_count)
    
    print(f"\n  OVERALL RATING")
    print(f"    {star_display} {score_5star['score']:.1f}/5.0")
    print(f"    {score_100['score']:.0f}% (Glassdoor-style)")
    
    # Confidence
    ci_width = score_5star['ci_upper'] - score_5star['ci_lower']
    confidence_level = "🔒 High" if ci_width < 0.5 else ("⚖️  Medium" if ci_width < 1.0 else "⚠️  Low")
    
    print(f"\n  RELIABILITY")
    print(f"    95% Confidence Interval: [{score_5star['ci_lower']:.1f}, {score_5star['ci_upper']:.1f}]")
    print(f"    Sample Size: {score_5star['sample_size']} verified reviews")
    print(f"    Confidence Level: {confidence_level} (CI width: ±{ci_width/2:.2f})")
    
    # Aspect breakdown
    print(f"\n  ASPECT BREAKDOWN")
    top_aspects = sorted(
        aspect_scores_5star.items(),
        key=lambda x: abs(x[1] - 3.0),
        reverse=True
    )[:3]
    
    for aspect, score in top_aspects:
        if score > 3.3:
            indicator = "📈"
        elif score < 2.7:
            indicator = "📉"
        else:
            indicator = "→"
        print(f"    {indicator} {aspect:15s}: {score:.1f}/5.0")
    
    # Key insights
    print(f"\n  KEY INSIGHTS")
    positive_aspects = [a for a, s in aspect_scores_5star.items() if s >= 3.5]
    negative_aspects = [a for a, s in aspect_scores_5star.items() if s < 2.5]
    
    if positive_aspects:
        print(f"    ✅ Strengths: {', '.join(positive_aspects)}")
    if negative_aspects:
        print(f"    ⚠️  Weaknesses: {', '.join(negative_aspects)}")
    
    anti_spam_rate = 1.0 - (flagged_count / len(sample_comments))
    print(f"    🛡️  Data Quality: {anti_spam_rate*100:.0f}% of submissions passed fraud checks")
    
    print(f"\n  {'═' * 80}")
    
    # ────────────────────────────────────────────────────────────────────────
    # Step 8: Recommendations
    # ────────────────────────────────────────────────────────────────────────
    
    print_subheader("STEP 8: Recommendations")
    
    if score_5star['score'] >= 4.0:
        print("  ✨ Strong Employer Brand")
        print("     - Continue focusing on what's working")
        print("     - Leverage positive reputation in recruiting")
    elif score_5star['score'] >= 3.0:
        print("  🎯 Moderate Reputation - Room for Improvement")
        print(f"     - Priority: Improve {negative_aspects[0] if negative_aspects else 'overall'} experience")
        print("     - Monitor trending issues in feedback")
    else:
        print("  ⚠️  Reputation at Risk")
        print("     - Urgent action needed on key weaknesses")
        print(f"     - Focus on: {', '.join(negative_aspects)}")
    
    if ci_width > 1.0:
        print("     - Collect more reviews to stabilize scores")
    
    # ────────────────────────────────────────────────────────────────────────
    # Conclusion
    # ────────────────────────────────────────────────────────────────────────
    
    print_header("WORKFLOW COMPLETE")
    
    print("""
    This end-to-end example demonstrated:
    
    ✅ Sentiment Analysis: Classified 10 opinions using DistilBERT
    ✅ Aspect Extraction: Linked sentiments to specific company aspects
    ✅ Anti-Manipulation: Detected suspicious patterns and fraud attempts
    ✅ Reputation Scoring: Aggregated signals into 1 stable, reliable score with confidence intervals
    ✅ Bayesian Smoothing: Applied statistical smoothing to prevent overfitting to small samples
    ✅ Dashboard: Generated human-readable reputation report
    
    KEY FEATURES:
    • Transparent & Explainable: Score traces back to source comments
    • Robust: Resistant to gaming, outlier detection, temporal decay
    • Scalable: Stateless, cacheable, suitable for production
    • Fair: Anti-bot checks, no single comment dominates
    • Accurate: Domain-specific fine-tuning recommended for Glassdoor-level accuracy
    
    NEXT STEPS FOR PRODUCTION:
    1. Fine-tune DistilBERT on 2,000-5,000 labeled job review examples
    2. Integrate with PostgreSQL database for persistence
    3. Add API rate limiting and authentication
    4. Set up Prometheus/Grafana monitoring dashboards
    5. Implement retraining pipeline (monthly) with new labeled data
    6. Deploy on Kubernetes or cloud VM with load balancing
    7. Conduct user interviews to validate fairness and interpretability
    """)
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
