"""
Minimal REST API for sentiment inference.

Single endpoint:
- POST /analyze

Input: list of comments
Output: sentiment scores per comment
"""

from datetime import datetime
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Inference endpoint for trained multitask model",
    version="1.0.0"
)

sentiment_analyzer = None


class AnalyzeRequest(BaseModel):
    """Input: list of comments to analyze."""

    comments: List[str]


class AnalyzeItem(BaseModel):
    """Output: sentiment score per comment."""

    comment: str
    sentiment_score: float


class AnalyzeResponse(BaseModel):
    """Batch response with sentiment scores."""

    results: List[AnalyzeItem]
    timestamp: str


@app.on_event("startup")
async def startup_event():
    global sentiment_analyzer

    try:
        logger.info("Loading sentiment analyzer...")
        sentiment_analyzer = SentimentAnalyzer(device="cpu")
        logger.info("✅ Model loaded")
    except Exception as exc:
        logger.error(f"❌ Startup failed: {exc}")
        raise


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_comments(request: AnalyzeRequest):
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.comments:
        raise HTTPException(status_code=400, detail="comments list is empty")

    results = sentiment_analyzer.batch_predict(request.comments)
    payload = [
        AnalyzeItem(comment=item["text"], sentiment_score=item["overall_rating"])
        for item in results
    ]

    return AnalyzeResponse(results=payload, timestamp=datetime.now().isoformat())


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
