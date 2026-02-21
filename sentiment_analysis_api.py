"""
Production REST API for Sentiment Analysis & Reputation Scoring
================================================================
FastAPI-based service with endpoints for:
- Comment submission and sentiment analysis
- Company reputation scores
- Batch processing
- Monitoring and health checks

Zero-cost: Uses only open-source dependencies.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import logging
import hashlib

from sentiment_analyzer import SentimentAnalyzer
from aspect_extractor import AspectExtractor, aggregate_aspect_scores
from reputation_scorer import ReputationScorer
from anti_manipulation import AntiManipulationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models (on startup)
sentiment_analyzer = None
aspect_extractor = None
reputation_scorer = None
anti_manipulation = None

# Mock database (in production, use PostgreSQL)
submission_store = {}  # {company_id: [submission_dicts]}
reputation_cache = {}  # {company_id: score_dict}
pending_batch = {}  # {company_id: [pending submissions waiting for batch]}

# ─────────────────────────────────────────────────────────────────────────
# Configuration - FYP Demo Settings
# ─────────────────────────────────────────────────────────────────────────
BATCH_THRESHOLD = 5  # For FYP demo (triggers after 5 comments)
# For production Glassdoor-style: change to 20
CACHE_TTL_HOURS = 24

# ─────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────

class SubmissionRequest(BaseModel):
    """Input: jobseeker comment submission."""
    company_id: int
    company_name: str
    comment: str
    user_ip: Optional[str] = None
    anonymous: bool = True

class SubmissionResponse(BaseModel):
    """Output: submission acceptance response."""
    submitted: bool
    status: str  # 'approved', 'review_pending', 'rejected'
    message: str
    company_reputation: Optional[float]
    confidence_interval: Optional[List[float]]
    sample_size: Optional[int]
    timestamp: str

class ReputationScoreResponse(BaseModel):
    """Company reputation score."""
    company_id: int
    company_name: str
    reputation_score: float
    scale: str  # '5star' or '100percent'
    confidence_interval: List[float]
    sample_size: int
    aspect_scores: Optional[Dict[str, float]]
    last_updated: str
    explanation: str

class HealthCheckResponse(BaseModel):
    """Service health status."""
    status: str
    models_loaded: bool
    cache_size: int
    timestamp: str

# ─────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentiment Analysis & Reputation Scoring API",
    description="Production endpoint for job review sentiment analysis",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global sentiment_analyzer, aspect_extractor, reputation_scorer, anti_manipulation
    
    logger.info("Starting up service...")
    
    try:
        logger.info("Loading sentiment analyzer...")
        sentiment_analyzer = SentimentAnalyzer(device='cpu')  # Use CPU for production stability
        
        logger.info("Loading aspect extractor...")
        aspect_extractor = AspectExtractor()
        
        logger.info("Loading reputation scorer...")
        reputation_scorer = ReputationScorer()
        
        logger.info("Loading anti-manipulation engine...")
        anti_manipulation = AntiManipulationEngine()
        
        logger.info("✅ All models loaded successfully")
    
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down service...")
    # In production, save cache, close DB connections, etc.

# ─────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    models_ready = all([
        sentiment_analyzer is not None,
        aspect_extractor is not None,
        reputation_scorer is not None,
        anti_manipulation is not None
    ])
    
    return HealthCheckResponse(
        status="healthy" if models_ready else "degraded",
        models_loaded=models_ready,
        cache_size=len(reputation_cache),
        timestamp=datetime.now().isoformat()
    )

@app.post("/submit", response_model=SubmissionResponse)
async def submit_comment(request: SubmissionRequest):
    """
    Submit a jobseeker comment and get immediate reputation update.
    
    Request:
    {
        "company_id": 123,
        "company_name": "Acme Corp",
        "comment": "Great team, but pay is low",
        "user_ip": "192.168.1.100",
        "anonymous": true
    }
    
    Response includes updated company reputation score.
    """
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        company_id = request.company_id
        comment = request.comment
        
        # Anonymize IP
        ip_hash = hashlib.sha256(request.user_ip.encode()).hexdigest() if request.user_ip else "unknown"
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.predict(comment)
        is_calibrated = sentiment_analyzer.calibration_check(sentiment_result)
        
        # Aspect extraction
        aspects_result = aspect_extractor.extract_full(
            comment,
            external_sentiment_signal=sentiment_result['sentiment_signal']
        )
        
        # Anti-manipulation check
        recent_company_comments = submission_store.get(company_id, [])
        recent_texts = [s['comment'] for s in recent_company_comments[-20:]]
        
        manipulation_check = anti_manipulation.check_submission(
            text=comment,
            ip_hash=ip_hash,
            recent_submissions=recent_texts,
            sentiment_confidence=sentiment_result['confidence']
        )
        
        # Decide on approval
        weight_factor = manipulation_check['weight_factor']
        recommendation = manipulation_check['recommendation']
        
        if recommendation == 'reject':
            status = 'rejected'
            message = "Submission rejected due to suspicious patterns"
            approved = False
        elif recommendation == 'review':
            status = 'review_pending'
            message = "Submission flagged for human review"
            approved = False
        else:
            status = 'approved'
            message = "Comment accepted and included in reputation score"
            approved = True
        
        # Store submission
        if company_id not in submission_store:
            submission_store[company_id] = []
        
        submission_store[company_id].append({
            'timestamp': datetime.now(),
            'comment': comment,
            'sentiment_signal': sentiment_result['sentiment_signal'],
            'confidence': sentiment_result['confidence'],
            'aspects': aspects_result['sentiments'],
            'approved': approved,
            'weight_factor': weight_factor,
            'flags': manipulation_check['flags']
        })
        
        # BATCH PROCESSING LOGIC
        # Add to pending batch queue
        if company_id not in pending_batch:
            pending_batch[company_id] = []
        
        if approved:
            pending_batch[company_id].append(submission_store[company_id][-1])
        
        # Check if batch threshold reached
        should_compute_batch = len(pending_batch.get(company_id, [])) >= BATCH_THRESHOLD
        
        # Update reputation score (only if batch triggered)
        if should_compute_batch and approved:
            logger.info(f"🚨 BATCH TRIGGERED for company {company_id}: {len(pending_batch[company_id])} pending comments")
            
            approved_submissions = [s for s in submission_store[company_id] if s['approved']]
            sentiment_signals = [
                {
                    'signal': s['sentiment_signal'],
                    'confidence': s['confidence']
                }
                for s in approved_submissions
            ]
            
            timestamps = [s['timestamp'] for s in approved_submissions]
            
            score_result = reputation_scorer.compute_reputation_score(
                sentiment_signals,
                submission_timestamps=timestamps,
                scale='5star'
            )
            
            reputation_cache[company_id] = {
                'score': score_result['score'],
                'ci_lower': score_result['ci_lower'],
                'ci_upper': score_result['ci_upper'],
                'sample_size': score_result['sample_size'],
                'last_updated': datetime.now().isoformat()
            }
            
            # Clear pending batch after processing
            pending_batch[company_id] = []
            
            logger.info(f"✅ Reputation score computed: {score_result['score']:.2f}/5.0")
        
        # Get current reputation (from cache or None)
        current_reputation = reputation_cache.get(company_id, {})
        
        # If no score yet, return pending status
        if not current_reputation:
            return SubmissionResponse(
                submitted=True,
                status=status,
                message=f"{message} (Pending: {len(pending_batch.get(company_id, []))} comments, triggers at {BATCH_THRESHOLD})",
                company_reputation=3.0,  # Neutral default
                confidence_interval=[1.0, 5.0],
                sample_size=0,
                timestamp=datetime.now().isoformat()
            )
        
        return SubmissionResponse(
            submitted=True,
            status=status,
            message=message,
            company_reputation=current_reputation.get('score'),
            confidence_interval=[
                current_reputation.get('ci_lower'),
                current_reputation.get('ci_upper')
            ] if current_reputation else None,
            sample_size=current_reputation.get('sample_size'),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reputation/{company_id}", response_model=ReputationScoreResponse)
async def get_reputation_score(company_id: int):
    """
    Retrieve current reputation score for a company.
    
    Returns:
    - Reputation score (1-5 scale)
    - Confidence interval
    - Aspect-based breakdown
    - Sample size
    """
    if company_id not in submission_store:
        raise HTTPException(status_code=404, detail="No submissions for this company")
    
    # Get cached score
    cached = reputation_cache.get(company_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Reputation score not computed")
    
    # Compute aspect scores
    approved_submissions = [s for s in submission_store[company_id] if s['approved']]
    aspect_sentiments = [s['aspects'] for s in approved_submissions]
    
    aspect_scores = {}
    if aspect_sentiments:
        aspect_scores = aggregate_aspect_scores(aspect_sentiments)
        # Convert to 5-star scale
        aspect_scores = {
            aspect: 3.0 + 2.0 * score
            for aspect, score in aspect_scores.items()
        }
    
    explanation = (
        f"Company reputation is {cached['score']:.1f}/5.0 based on {cached['sample_size']} "
        f"verified reviews. We are 95% confident the true reputation is between "
        f"{cached['ci_lower']:.1f} and {cached['ci_upper']:.1f} stars."
    )
    
    return ReputationScoreResponse(
        company_id=company_id,
        company_name=f"Company {company_id}",
        reputation_score=cached['score'],
        scale='5star',
        confidence_interval=[cached['ci_lower'], cached['ci_upper']],
        sample_size=cached['sample_size'],
        aspect_scores=aspect_scores if aspect_scores else None,
        last_updated=cached['last_updated'],
        explanation=explanation
    )

@app.get("/submissions/{company_id}")
async def get_submissions(company_id: int):
    """Get submission history for a company (admin only)."""
    if company_id not in submission_store:
        return {"submissions": []}
    
    submissions = submission_store[company_id]
    
    return {
        "company_id": company_id,
        "total_submissions": len(submissions),
        "approved_count": sum(1 for s in submissions if s['approved']),
        "flagged_count": sum(1 for s in submissions if not s['approved']),
        "submissions": [
            {
                "timestamp": s['timestamp'].isoformat(),
                "comment_preview": s['comment'][:100],
                "sentiment": s['sentiment_signal'],
                "confidence": s['confidence'],
                "approved": s['approved'],
                "flags": s['flags']
            }
            for s in submissions[-50:]  # Last 50
        ]
    }

@app.post("/batch_submit")
async def batch_submit(requests: List[SubmissionRequest]):
    """
    Batch submit multiple comments (for bulk operations).
    
    Returns list of SubmissionResponse objects.
    """
    results = []
    for request in requests:
        try:
            # Reuse single submit logic
            result = await submit_comment(request)
            results.append(result.dict())
        except HTTPException as e:
            results.append({
                'submitted': False,
                'status': 'error',
                'message': e.detail
            })
    
    return {"results": results, "total": len(results)}

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    total_submissions = sum(len(subs) for subs in submission_store.values())
    total_approved = sum(
        sum(1 for s in subs if s['approved'])
        for subs in submission_store.values()
    )
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_companies": len(submission_store),
        "total_submissions": total_submissions,
        "total_approved": total_approved,
        "approval_rate": total_approved / max(total_submissions, 1),
        "cached_scores": len(reputation_cache),
        "model_status": "ready" if sentiment_analyzer else "not_loaded"
    }

# ─────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """API documentation."""
    return {
        "service": "Sentiment Analysis & Reputation Scoring",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Service health check",
            "POST /submit": "Submit comment and get reputation update",
            "GET /reputation/{company_id}": "Get company reputation score",
            "GET /submissions/{company_id}": "Get submission history",
            "POST /batch_submit": "Batch submit comments",
            "GET /stats": "System statistics"
        },
        "documentation": "/docs"
    }


# ─────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
