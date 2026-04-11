"""REST API for five-head employee review rating inference."""

from datetime import datetime
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment_analyzer import SentimentAnalyzer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Employee Review Rating API",
    description="Predict overall, work-life balance, culture, career opportunities, and salary ratings.",
    version="2.0.0",
)

sentiment_analyzer = None


class PredictRequest(BaseModel):
    review: str = Field(..., description="Single employee review text")


class BatchRequest(BaseModel):
    reviews: List[str] = Field(..., description="Batch of employee review texts")


class RatingResponse(BaseModel):
    review: str
    overall_rating: float
    work_life_balance: float
    company_culture: float
    career_opportunities: float
    salary_benefits: float


class BatchResponse(BaseModel):
    results: List[RatingResponse]
    timestamp: str


@app.on_event("startup")
async def startup_event() -> None:
    global sentiment_analyzer

    try:
        logger.info("Loading sentiment analyzer...")
        sentiment_analyzer = SentimentAnalyzer(device="cpu")
        logger.info("Model loaded")
    except Exception as exc:
        logger.exception("Startup failed")
        raise exc


def to_response(item: dict) -> RatingResponse:
    return RatingResponse(
        review=item["text"],
        overall_rating=float(item["overall_rating"]),
        work_life_balance=float(item["work_life_balance"]),
        company_culture=float(item["company_culture"]),
        career_opportunities=float(item["career_opportunities"]),
        salary_benefits=float(item["salary_benefits"]),
    )


@app.post("/predict", response_model=RatingResponse)
async def predict_review(request: PredictRequest):
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="review is empty")
    return to_response(sentiment_analyzer.predict(request.review))


@app.post("/analyze", response_model=BatchResponse)
async def analyze_reviews(request: BatchRequest):
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.reviews:
        raise HTTPException(status_code=400, detail="reviews list is empty")

    results = [to_response(item) for item in sentiment_analyzer.batch_predict(request.reviews)]
    return BatchResponse(results=results, timestamp=datetime.now().isoformat())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
