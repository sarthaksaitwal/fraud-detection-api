"""GET /health: is the service up, and which model is it serving?"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_scorer
from src.api.schemas import HealthResponse
from src.scoring import Scorer

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(scorer: Annotated[Scorer, Depends(get_scorer)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=scorer.model_version,
        review_threshold=scorer.review_threshold,
        block_threshold=scorer.block_threshold,
    )
